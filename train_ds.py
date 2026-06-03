"""
Joint Affordance 训练脚本（新版，基于 Qwen MLLM）

主要特性：
- 使用 calculator.compute_losses 统一计算损失（返回字典）
- 使用 torchmetrics 进行 epoch 级分割指标追踪
- 使用 metrics.log_epoch_summary 统一格式化输出
- 所有指标通过字典传递，避免大量零散变量
"""

import argparse
import os
import time
from collections import OrderedDict
from functools import partial
from typing import Dict

from configs.training_config import DeepSpeedConfigs
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from utils.base_dataset import JointDataset
from utils.dataset import (
    JointAffordanceTorchDataset,
    JointAffordanceTrainDataset,
    joint_affordance_collate_fn,
    build_functional_tokens_from_samples,
    build_functional_tokens_from_sample_ids,
)
from utils.common import dict_to_cuda, setup_logger, FUNCTIONAL_TOKENS
from utils import calculator as calc
from utils.metrics import (
    build_torchmetrics_bundle,
    update_torchmetrics,
    compute_and_reset_torchmetrics,
    log_scalar_dict,
    log_epoch_summary,
)
from utils.threshold_search import (
    build_threshold_candidates,
    init_threshold_search_stats,
    update_threshold_search_stats,
    finalize_threshold_search,
)
from utils.debug import log_param_dtype_stats, count_model_params
from utils.model_io import build_portable_assets, load_checkpoint_payload, save_portable_checkpoint
from utils.trainability_summary import log_trainability_summary


ENV_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def parse_args():
    parser = argparse.ArgumentParser(description="JointAffordance (Qwen) training")
    parser.add_argument("--qwen_model", type=str, default=None, help="Qwen 模型路径或名称")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="SAM 权重路径")
    parser.add_argument("--point_backbone_pretrained", type=str, default=None, help="SONATA point backbone 预训练权重路径")
    parser.add_argument(
        "--point_backbone_pretrained_config",
        type=str,
        default=None,
        help="SONATA point backbone 预训练配置路径；不传时默认尝试使用权重同目录下的 config.json",
    )
    parser.add_argument(
        "--point_decoder_backbone_mode",
        type=str,
        default=None,
        choices=["shared", "independent"],
        help="3D decoder backbone 模式：shared 为与 encoder 共用基座，independent 为独立随机初始化 backbone",
    )
    parser.add_argument(
        "--point_decoder_decode_mode",
        type=str,
        default=None,
        choices=["prompt", "similarity"],
        help="3D decoder 后端对齐方式：prompt 为 prompt-based 解码，similarity 为逐点相似度对齐",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="每卡训练 batch size（同时覆写 val_batch_size）")
    parser.add_argument("--epochs", type=int, default=None, help="训练总 epoch 数")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集路径")
    parser.add_argument(
        "--train_json_path",
        type=str,
        default=None,
        help="训练集分割 JSON 路径；不传时使用 dataset_dir/train.json，并默认从 JSON 同目录加载原数据",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default=None,
        help="验证集分割 JSON 路径；不传时使用 dataset_dir/val.json，并默认从 JSON 同目录加载原数据",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=None,
        help="兼容旧参数：等价于 --val_json_path",
    )
    parser.add_argument(
        "--val_dataset_dir",
        type=str,
        default=None,
        help="验证集原数据根目录；不传时由 val_split_file 所在目录推导，未指定 val_split_file 时跟随 dataset_dir",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="日志与权重输出目录")
    parser.add_argument("--update_epoch", type=int, default=5, help="每隔多少个 epoch 保存 latest checkpoint")
    parser.add_argument("--fixed_save_interval", type=int, default=100, help="固定长周期保存 checkpoint 的间隔，用于保存收敛状态。")
    parser.add_argument("--lazy_load", dest="lazy_load", action="store_true", help="启用懒加载（默认启用）", default=True)
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 断点续训", default=False)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="断点续训 checkpoint 路径；支持单文件 .pth 或 HF 分片目录，为空则默认 latest_ds.pth")
    parser.add_argument("--local_rank", type=int, default=ENV_LOCAL_RANK)
    args, _ = parser.parse_known_args()
    if args.resume_ckpt is not None:
        args.resume = True
    assert args.point_decoder_backbone_mode in (None, "independent"), "建议使用独立权重的3D decoder backbone模式"
    assert args.point_decoder_decode_mode in (None, "similarity"), "请使用相似度解码的3D decoder decode模式，prompt解码效果很差"
    return args


def apply_point_decoder_backbone_mode(model_config, mode: str):
    model_config.point_decoder.backbone_mode = mode
    if mode == "shared":
        encoder_backbone_cfg = model_config.mllm.point_encoder_backbone.to_dict()
        decoder_backbone_cfg = dict(encoder_backbone_cfg)
        decoder_backbone_cfg["enc_mode"] = False
        model_config.point_decoder.backbone_kwargs = decoder_backbone_cfg
        model_config.point_decoder.backbone_out_channels = int(
            decoder_backbone_cfg.get("dec_channels", (64,))[0]
        )
        model_config.mllm.enable_point_encoder = True


# ===================== 模型初始化辅助 =====================

def apply_trainability_policy(model, training_configs, logger):
    def enable_all_trainable(model):
        """默认将全部参数设为可训练。"""
        for _, param in model.named_parameters():
            param.requires_grad = True

    def freeze_modules_by_name(model, name_filters):
        """按关键字列表冻结参数，保留 LoRA 参数可训练"""
        if not name_filters:
            return
        for name, param in model.named_parameters():
            if any(key in name for key in name_filters):
                if "lora_" in name:
                    continue
                param.requires_grad = False

    """默认全量训练；LoRA 开启时自动冻结 MLLM 原始参数；额外冻结统一走黑名单。"""
    if training_configs.lora.lora_r > 0:
        freeze_modules_by_name(model, ["mllm.model"])
    freeze_modules_by_name(model, training_configs.name_of_params_to_freeze)


def create_param_groups(model, config, logger):
    """创建分层学习率参数组（不启用分层 LR 时返回普通参数列表）"""
    if not config.use_layerwise_lr:
        return [p for p in model.parameters() if p.requires_grad]

    def _collect(module):
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad and p.is_floating_point()]

    llm_params = _collect(getattr(model, "mllm", None))
    vis2d_params = _collect(getattr(model, "image_decoder", None))
    vis3d_params = _collect(getattr(model, "point_decoder", None))
    used_ids = {id(p) for p in llm_params + vis2d_params + vis3d_params}
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.is_floating_point() and id(p) not in used_ids
    ]

    groups = []
    for params, lr, name in [
        (llm_params, config.llm_lr, "llm"),
        (vis2d_params, config.vision_2d_lr, "vision_2d"),
        (vis3d_params, config.vision_3d_lr, "vision_3d"),
        (other_params, config.lr, "other"),
    ]:
        if params:
            groups.append({"params": params, "lr": lr, "name": name})
            logger.info(f"  {name}: {len(params)} tensors, lr={lr}")
    return groups


def get_current_lr(scheduler, optimizer):
    """返回当前学习率字典，兼容分层/非分层学习率。"""
    if optimizer is None or not hasattr(optimizer, "param_groups"):
        return {}
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        lr_list = scheduler.get_last_lr()
    else:
        lr_list = [group.get("lr", 0.0) for group in optimizer.param_groups]
    lr_dict = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name")
        if not group_name:
            group_name = "lr" if len(optimizer.param_groups) == 1 else f"group_{idx}"
        lr_dict[group_name] = float(lr_list[idx] if idx < len(lr_list) else group.get("lr", 0.0))
    return lr_dict


def _to_serializable_metrics(metrics: Dict) -> Dict:
    """将指标字典转成可 JSON/ckpt 持久化的标量字典。"""
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = [float(x) for x in v.detach().cpu().view(-1).tolist()]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out


def _move_optimizer_state_to_device(optimizer, device: torch.device):
    """将 optimizer state 中的张量迁移到目标设备，避免 resume 后设备不一致。"""
    if optimizer is None or not hasattr(optimizer, "state"):
        return
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device, non_blocking=True)


def train_one_epoch(
    train_loader, model_engine, optimizer, scheduler, config,
    epoch, global_step, writer, logger, local_rank,
    loss_kwargs: Dict,
):
    """
    训练一个 epoch。

    Args:
        loss_kwargs: 传给 calc.compute_losses 的关键字参数字典
                     （包含 device, focal/bce/dice/ce 权重等）
    Returns:
        global_step: 更新后的全局步数
        train_results: epoch 指标字典
    """
    model_engine.train()
    metrics = build_torchmetrics_bundle(
        device=model_engine.device,
        threshold_2d=config.mask_threshold_2d,
        threshold_3d=config.mask_threshold_3d,
    )

    loader = (
        tqdm(train_loader, total=len(train_loader), dynamic_ncols=True,
             desc=f"Epoch {epoch + 1}/{config.epochs}")
        if local_rank == 0 else train_loader
    )

    for batch_idx, input_dict in enumerate(loader):
        input_dict = dict_to_cuda(input_dict, device=model_engine.device)
        output_dict = model_engine(**input_dict)

        # 统一计算损失（Focal+Dice for 2D, BCE+Dice for 3D, CE for LLM）
        loss_dict = calc.compute_losses(output_dict, input_dict, **loss_kwargs)
        loss = loss_dict["loss"]

        model_engine.backward(loss)
        model_engine.step()
        if not hasattr(model_engine, "is_gradient_accumulation_boundary") or model_engine.is_gradient_accumulation_boundary():
            global_step += 1

        # 一站式更新全部指标
        update_torchmetrics(
            metrics, loss_dict, output_dict, input_dict, config.batch_size,
            threshold_2d=config.mask_threshold_2d,
            threshold_3d=config.mask_threshold_3d,
            gt_threshold_2d=getattr(config, "gt_threshold_2d", 0.5),
            gt_threshold_3d=getattr(config, "gt_threshold_3d", 0.5),
        )

        # 定期打印批次进度
        if (batch_idx + 1) % config.print_freq == 0 or (batch_idx + 1) == len(train_loader):
            lr_dict = get_current_lr(scheduler, optimizer)
            lr_text = ", ".join([f"{k}={v:.2e}" for k, v in lr_dict.items()])
            route_stats = output_dict.get("route_classification_stats") or {}
            route_wrong = int(route_stats.get("route_wrong", 0) or 0)
            route_total = int(route_stats.get("route_total", 0) or 0)
            text_as_aff = int(route_stats.get("route_text_as_aff", 0) or 0)
            text_total = int(route_stats.get("route_text_total", 0) or 0)
            aff_as_text = int(route_stats.get("route_aff_as_text", 0) or 0)
            aff_total = int(route_stats.get("route_aff_total", 0) or 0)
            logger.info(
                f"  [{batch_idx + 1}/{len(train_loader)}] "
                f"loss={loss_dict['loss'].item():.6f} "
                f"(ce={loss_dict['ce_loss'].item():.6f}, "
                f"img={loss_dict['img_loss'].item():.6f}, "
                f"pc={loss_dict['pc_loss'].item():.6f}, "
                f"route_mis={route_wrong}/{route_total}, "
                f"text2aff={text_as_aff}/{text_total}, "
                f"aff2text={aff_as_text}/{aff_total})"
                + (f" lr=({lr_text})" if lr_text else "")
            )
            if local_rank == 0 and writer is not None:
                batch_log = {k: loss_dict[k].item() for k in loss_dict}
                batch_log.update(
                    calc.img_batch_metrics(
                        output_dict,
                        input_dict,
                        threshold=config.mask_threshold_2d,
                        gt_threshold=getattr(config, "gt_threshold_2d", 0.5),
                    )
                )
                # Router 统计直接记录计数，避免小 batch 下准确率波动掩盖误判规模。
                for k, v in route_stats.items():
                    batch_log[f"router/{k}"] = float(v)
                for k, v in lr_dict.items():
                    batch_log[f"lr/{k}"] = v
                if lr_dict:
                    batch_log["lr"] = next(iter(lr_dict.values()))
                log_scalar_dict(writer, "train_batch", batch_log, global_step)
            if local_rank == 0 and hasattr(loader, "set_postfix"):
                postfix = {"loss": f"{loss_dict['loss'].item():.4f}"}
                if lr_dict:
                    postfix["lr"] = f"{next(iter(lr_dict.values())):.2e}"
                loader.set_postfix(postfix, refresh=False)

    # Epoch 结束：汇总
    train_results = compute_and_reset_torchmetrics(metrics)
    lr_dict = get_current_lr(scheduler, optimizer)
    log_epoch_summary(logger, epoch + 1, config.epochs, "train", train_results, lr_dict)
    if local_rank == 0 and writer is not None:
        log_scalar_dict(writer, "train_epoch", train_results, epoch + 1)
        for k, v in lr_dict.items():
            log_scalar_dict(writer, "train_epoch", {f"lr/{k}": v}, epoch + 1)

    return global_step, train_results


@torch.no_grad()
def validate_one_epoch(
    val_loader, model_engine, config,
    epoch, writer, logger, local_rank,
    loss_kwargs: Dict,
):
    """
    验证一个 epoch。

    Args:
        loss_kwargs: 传给 calc.compute_losses 的关键字参数字典
    Returns:
        val_results: epoch 指标字典
    """
    model_engine.eval()
    metrics = build_torchmetrics_bundle(
        device=model_engine.device,
        threshold_2d=config.mask_threshold_2d,
        threshold_3d=config.mask_threshold_3d,
    )
    threshold_stats = None
    if getattr(config, "auto_select_mask_threshold", True):
        threshold_stats = init_threshold_search_stats(
            build_threshold_candidates(
                model_engine.device,
                extra_thresholds=[config.mask_threshold_2d, config.mask_threshold_3d],
            )
        )

    for val_dict in val_loader:
        val_dict = dict_to_cuda(val_dict, device=model_engine.device)
        # 训练内验证应与独立 validate.py 保持一致：只用 prompt generate，不把 GT answer 输入 MLLM。
        val_output = model_engine(
            **val_dict,
            return_hidden_states=False,
            return_mllm_output=False,
            inference_generate=True,
        )
        loss_dict = calc.compute_losses(val_output, val_dict, **loss_kwargs)
        update_torchmetrics(
            metrics, loss_dict, val_output, val_dict, config.val_batch_size,
            threshold_2d=config.mask_threshold_2d,
            threshold_3d=config.mask_threshold_3d,
            gt_threshold_2d=getattr(config, "gt_threshold_2d", 0.5),
            gt_threshold_3d=getattr(config, "gt_threshold_3d", 0.5),
        )
        if threshold_stats is not None:
            update_threshold_search_stats(threshold_stats, val_output, val_dict, config)

    val_results = compute_and_reset_torchmetrics(metrics)
    if threshold_stats is not None:
        val_results.update(finalize_threshold_search(threshold_stats))
    if local_rank == 0:
        log_epoch_summary(logger, epoch + 1, config.epochs, "val", val_results)
        if "best_mask_threshold_2d" in val_results or "best_mask_threshold_3d" in val_results:
            parts = []
            if "best_mask_threshold_2d" in val_results:
                parts.append(
                    f"2D={val_results['best_mask_threshold_2d']:.4f} "
                    f"(gIoU={val_results.get('best_giou_2d', 0.0):.4f}, "
                    f"cIoU={val_results.get('best_ciou_2d', 0.0):.4f})"
                )
            if "best_mask_threshold_3d" in val_results:
                parts.append(
                    f"3D={val_results['best_mask_threshold_3d']:.4f} "
                    f"(mIoU={val_results.get('best_miou_3d', 0.0):.4f}, "
                    f"cumIoU={val_results.get('best_cumulative_iou_3d', 0.0):.4f})"
                )
            logger.info("验证集最优预测阈值: " + ", ".join(parts))
        if val_results.get("threshold_search_2d_tie", 0.0) or val_results.get("threshold_search_3d_tie", 0.0):
            logger.info("阈值搜索出现并列或无区分结果；对应分支不会给出可写回的最佳阈值。")
        if writer is not None:
            log_scalar_dict(writer, "val_epoch", val_results, epoch + 1)

    model_engine.train()
    return val_results


def _save_model_state_to_cpu(
    model_engine,
    save_path: str,
    client_state: Dict,
    local_rank: int,
    logger,
    optimizer=None,
    scheduler=None,
    training_cfg=None,
    asset_bundle: Dict | None = None,
    zero_stage: int = 3,
) -> bool:
    """
    将模型 state_dict 逐参数汇聚到 CPU 后保存，避免 ZeRO-3 在 GPU 上全量汇聚导致 OOM。
    仅 rank 0 写入文件，其他 rank 仅参与 collective。
    若当前不是 ZeRO-3 或 GatheredParameters 不可用，返回 False，调用方应回退到 save_checkpoint。
    """
    module = model_engine.module if hasattr(model_engine, "module") else model_engine
    state_cpu = OrderedDict()

    if zero_stage == 3:
        try:
            from deepspeed.zero import GatheredParameters
        except ImportError:
            try:
                from deepspeed.runtime.zero.stage3 import GatheredParameters
            except ImportError:
                return False

        # 逐参数 gather，只在 rank 0 上拷贝到 CPU，避免 GPU 上同时存在完整模型
        for name, param in module.named_parameters():
            with GatheredParameters([param]):
                if local_rank == 0:
                    state_cpu[name] = param.detach().cpu().clone()
        for name, buf in module.named_buffers():
            if local_rank == 0:
                state_cpu[name] = buf.detach().cpu().clone()
    elif local_rank == 0:
        for name, tensor in module.state_dict().items():
            state_cpu[name] = tensor.detach().cpu().clone()

    if local_rank == 0:
        save_portable_checkpoint(
            save_path,
            model_state_dict=state_cpu,
            meta=client_state,
            training_cfg=training_cfg,
            asset_bundle=asset_bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_dict=get_current_lr(scheduler, optimizer),
            logger=logger,
        )
        logger.info(f"完整推理权重已保存到 CPU 再落盘: {save_path}")
    if dist.is_initialized():
        dist.barrier()
    return True


def main():
    args = parse_args()
    training_configs = TrainingConfig(deepspeed_config=DeepSpeedConfigs(precision='fp32'))
    model_config = training_configs.model_config

    # 命令行覆盖配置
    if args.qwen_model:
        model_config.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        model_config.image_decoder.vision_pretrained = args.vision_pretrained
    if args.point_backbone_pretrained:
        model_config.mllm.point_encoder_pretrained = args.point_backbone_pretrained
    if args.point_backbone_pretrained_config:
        model_config.mllm.point_encoder_pretrained_config = args.point_backbone_pretrained_config
    decoder_backbone_mode = (
        args.point_decoder_backbone_mode
        if args.point_decoder_backbone_mode is not None
        else str(model_config.point_decoder.backbone_mode).lower()
    )
    apply_point_decoder_backbone_mode(model_config, decoder_backbone_mode)
    decoder_decode_mode = (
        args.point_decoder_decode_mode
        if args.point_decoder_decode_mode is not None
        else str(model_config.point_decoder.decode_mode).lower()
    )
    model_config.point_decoder.decode_mode = decoder_decode_mode

    if args.batch_size is not None:
        training_configs.batch_size = int(args.batch_size)
        training_configs.val_batch_size = int(args.batch_size)
    if args.epochs is not None:
        training_configs.epochs = int(args.epochs)
    if args.dataset_dir:
        training_configs.dataset_dir = args.dataset_dir
    if args.log_dir:
        training_configs.log_dir = args.log_dir
    if args.update_epoch is not None:
        training_configs.update_epoch = max(1, int(args.update_epoch))
    train_json_path = args.train_json_path or os.path.join(training_configs.dataset_dir, "train.json")
    val_json_path = args.val_json_path or args.val_split_file or os.path.join(training_configs.dataset_dir, "val.json")

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    cpu_obj_group = dist.new_group(backend="gloo")

    # 日志系统
    logger = setup_logger(training_configs.log_dir, local_rank)
    logger.info("=" * 80)
    logger.info("Joint Affordance Model - 开始训练")
    logger.info("=" * 80)

    writer = None

    # ---------- 加载数据集 ----------
    logger.info("加载数据集...")
    if args.lazy_load:
        train_samples = JointDataset(split_file_path=train_json_path, lazy_load=True)
        val_samples = JointDataset(split_file_path=val_json_path, lazy_load=True)

        token_obj = [None]
        if local_rank == 0:
            logger.info(f"训练集路径: root={train_samples.dataset_root}, split={train_samples.split_file}")
            logger.info(f"验证集路径: root={val_samples.dataset_root}, split={val_samples.split_file}")
            merged_ids = {"ins": {}, "img": {}, "pc": {}}
            for mod_key in ("ins", "img", "pc"):
                for ds in (train_samples, val_samples):
                    for obj, aff_map in ds.sample_ids.get(mod_key, {}).items():
                        merged_ids[mod_key].setdefault(obj, {})
                        for aff, ids in aff_map.items():
                            merged_ids[mod_key][obj].setdefault(aff, [])
                            merged_ids[mod_key][obj][aff].extend(list(ids))
            token_obj[0] = build_functional_tokens_from_sample_ids(merged_ids)
            logger.info(f"训练集 {len(train_samples)} 条, 验证集 {len(val_samples)} 条")
        dist.barrier()
        dist.broadcast_object_list(token_obj, src=0, group=cpu_obj_group)
        pair_token_map = token_obj[0] or {"img": {}, "pc": {}}
    else:
        data_objects = [None, None, None]
        if local_rank == 0:
            train_data = JointDataset(split_file_path=train_json_path, lazy_load=False).load_all_data()
            val_data = JointDataset(
                dataset_root=os.path.abspath(args.val_dataset_dir) if args.val_dataset_dir else None,
                split_file_path=val_json_path,
                lazy_load=False,
            ).load_all_data()
            logger.info(f"训练集路径: root={train_data.dataset_root}, split={train_data.split_file}")
            logger.info(f"验证集路径: root={val_data.dataset_root}, split={val_data.split_file}")
            train_payload = train_data.samples
            val_payload = val_data.samples
            pair_token_map = build_functional_tokens_from_samples(train_payload + val_payload)
            data_objects = [train_payload, val_payload, pair_token_map]
            logger.info(f"训练集 {len(train_payload)} 条, 验证集 {len(val_payload)} 条")
        dist.barrier()
        dist.broadcast_object_list(data_objects, src=0, group=cpu_obj_group)
        train_samples, val_samples, pair_token_map = data_objects

    functional_tokens = {
        "img": dict(FUNCTIONAL_TOKENS.get("img", {})),
        "pc": dict(FUNCTIONAL_TOKENS.get("pc", {})),
    }
    functional_tokens["img"].update(pair_token_map.get("img", {}))
    functional_tokens["pc"].update(pair_token_map.get("pc", {}))
    model_config.mllm.functional_tokens = functional_tokens
    if local_rank == 0:
        logger.info(
            f"已注册 obj-aff 专用 token: img={len(pair_token_map.get('img', {}))}, "
            f"pc={len(pair_token_map.get('pc', {}))}"
        )

    # ---------- 初始化模型 ----------
    logger.info("正在初始化模型...")
    model = JointAffordanceModel(model_config)
    if getattr(model, "point_encoder", None) is not None and getattr(model.point_encoder, "pretrained_info", None):
        load_info = model.point_encoder.pretrained_info
        model_config.mllm.point_encoder_pretrained_config = load_info.get("config_path")
        logger.info(
            "已通过 PointCloudEncoder.from_pretrained 加载 point backbone: "
            f"{load_info.get('weight_path')} | config={load_info.get('config_path')} | "
            f"prefix={load_info['matched_prefix']} | loaded={load_info['loaded_tensors']} | "
            f"missing={len(load_info['missing_keys'])} | unexpected={len(load_info['unexpected_keys'])}"
        )
    FUNCTIONAL_TOKENS.clear()
    FUNCTIONAL_TOKENS.update(model.mllm.functional_token_ids)

    processor = model.processor
    data_collator = partial(
        joint_affordance_collate_fn,
        tokenizer=model.tokenizer,
        output_image_size=training_configs.image_size,
        output_point_nums=training_configs.num_points,
        mllm_precision=model_config.mllm.compute_dtype,
        image_precision=model_config.image_decoder.compute_dtype,
        point_precision=model_config.point_decoder.compute_dtype,
    )

    # ---------------- 配置LORA ------------------  
    if training_configs.lora.lora_r > 0:
        logger.info(f"应用 LoRA: r={training_configs.lora.lora_r}, alpha={training_configs.lora.lora_alpha}")
        model.mllm.model = get_peft_model(model.mllm.model, training_configs.lora.to_peft_config())

    # ---------- 冻结模块 统计参数 ----------
    apply_trainability_policy(model, training_configs, logger)
    portable_asset_bundle = build_portable_assets(model)
    log_param_dtype_stats(model, logger, stage="before_deepspeed")
    total_params, trainable_params = count_model_params(model)
    logger.info(
        f"参数统计: total={total_params:,}, trainable={trainable_params:,}, "
        f"ratio={100.0 * trainable_params / max(1, total_params):.2f}%"
    )
    if local_rank == 0:
        log_trainability_summary(model, logger, output_path=os.path.join(training_configs.log_dir, f"params_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"))

    # ---------- 断点续训：先恢复模型权重（DeepSpeed 包装前） ----------
    ckpt_dir = os.path.join(training_configs.log_dir, "checkpoints_ds")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_payload = None
    resume_path = args.resume_ckpt
    if args.resume or args.resume_ckpt:
        if resume_path is None:
            resume_path = os.path.join(ckpt_dir, "latest_ds.pth")
        if not os.path.isabs(resume_path):
            resume_path = os.path.abspath(resume_path)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"断点续训失败，checkpoint 不存在: {resume_path}")
        resume_payload = load_checkpoint_payload(resume_path, map_location="cpu")
        state_dict = resume_payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"checkpoint 缺少 model_state_dict: {resume_path}")
        miss, unexp = model.load_state_dict(state_dict, strict=False)
        if local_rank == 0:
            logger.info(
                f"已加载断点模型: {resume_path} | missing={len(miss)} unexpected={len(unexp)}"
            )

    # 构建 Dataset
    train_ds_cls = JointAffordanceTrainDataset if training_configs.samples_per_epoch else JointAffordanceTorchDataset
    train_ds_kwargs = dict(
        processor=processor, image_size=training_configs.image_size, num_points=training_configs.num_points,
        mllm_precision=model_config.mllm.compute_dtype,
        image_precision=model_config.image_decoder.compute_dtype,
        point_precision=model_config.point_decoder.compute_dtype,
        use_sample_cache=training_configs.use_sample_cache,
    )
    if training_configs.samples_per_epoch:
        train_ds_kwargs["samples_per_epoch"] = training_configs.samples_per_epoch
    train_dataset = train_ds_cls(train_samples, **train_ds_kwargs)

    val_dataset = JointAffordanceTorchDataset(val_samples, **{
        k: v for k, v in train_ds_kwargs.items() if k != "samples_per_epoch"
    })

    # ---------- DeepSpeed 初始化 ----------
    # ZeRO-3 时：在 CPU 上保留完整模型，由 DeepSpeed 初始化时再分片到各 GPU，避免每张卡先加载完整模型再分片导致显存峰值
    zero_stage = getattr(training_configs.deepspeed, "zero_stage", 3)
    if zero_stage >= 3:
        logger.info("将模型置于 CPU，由 DeepSpeed 初始化时再分片到 GPU...")
        model = model.cpu()
        torch.cuda.empty_cache()

    logger.info("初始化 DeepSpeed...")
    params_to_train = create_param_groups(model, training_configs, logger)
    if not params_to_train:
        raise RuntimeError("没有可训练参数，请检查配置")

    if training_configs.use_layerwise_lr:
        # 分层学习率：手动创建 optimizer + scheduler
        steps_per_epoch = training_configs.steps_per_epoch or max(
            1, len(train_dataset) // max(1, training_configs.deepspeed.train_micro_batch_size_per_gpu)
        )
        optimizer_cls = (DeepSpeedCPUAdam if training_configs.deepspeed.offload_optimizer_device == "cpu"
                         else torch.optim.AdamW)
        optimizer = optimizer_cls(
            params_to_train, weight_decay=training_configs.weight_decay, betas=(training_configs.beta1, training_configs.beta2),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_configs.warmup_num_steps,
            num_training_steps=training_configs.epochs * steps_per_epoch,
        )
        model_engine, _, train_loader, _ = deepspeed.initialize(
            model=model, model_parameters=params_to_train, training_data=train_dataset,
            optimizer=optimizer, lr_scheduler=scheduler, collate_fn=data_collator,
            config=training_configs.deepspeed.to_dict(),
        )
    else:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model, model_parameters=params_to_train, training_data=train_dataset,
            collate_fn=data_collator, config=training_configs.deepspeed.to_dict(),
        )
    optimizer = getattr(model_engine, "optimizer", optimizer)
    scheduler = getattr(model_engine, "lr_scheduler", scheduler)

    # 验证 DataLoader（在循环外创建，避免每个 epoch 重复创建）
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=training_configs.val_batch_size, shuffle=False,
        num_workers=training_configs.workers, pin_memory=True, collate_fn=data_collator,
    )
    log_param_dtype_stats(
        model_engine.module if hasattr(model_engine, "module") else model_engine,
        logger, stage="after_deepspeed",
    )

    global_step = 0
    best_metric = float("inf")
    best_epoch = -1
    start_epoch = 0
    last_val_metrics: Dict = {}
    update_epoch = max(1, int(getattr(training_configs, "update_epoch", 1)))
    config_json_path = os.path.join(ckpt_dir, "training_config.json")

    # ---------- 断点续训：恢复优化器/调度器与训练进度 ----------
    if resume_payload is not None:
        opt_state = resume_payload.get("optimizer_state_dict")
        if opt_state is not None and optimizer is not None:
            try:
                optimizer.load_state_dict(opt_state)
                _move_optimizer_state_to_device(optimizer, model_engine.device)
            except Exception as exc:
                if local_rank == 0:
                    logger.warning(f"DeepSpeed optimizer state 加载失败，将仅恢复模型权重: {exc}")
        sch_state = resume_payload.get("scheduler_state_dict")
        if sch_state is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(sch_state)
            except Exception as exc:
                if local_rank == 0:
                    logger.warning(f"scheduler state 加载失败，将重新开始学习率调度: {exc}")
        global_step = int(resume_payload.get("global_step", 0) or 0)
        best_epoch = int(resume_payload.get("best_epoch", -1) or -1)
        best_metric = float(resume_payload.get("best_val_loss", float("inf")))
        start_epoch = int(resume_payload.get("epoch", 0) or 0)
        last_val_metrics = _to_serializable_metrics(resume_payload.get("val_metrics", {}))
        if local_rank == 0:
            logger.info(
                f"断点续训状态: start_epoch={start_epoch + 1}, global_step={global_step}, "
                f"best_epoch={best_epoch}, best_val_loss={best_metric:.6f}"
            )
            if opt_state is None or sch_state is None:
                logger.warning(
                    "resume checkpoint 缺少 optimizer/scheduler 状态，"
                    "学习率曲线与优化器动量无法完全无缝衔接。建议使用新版本 latest_ds.pth 续训。"
                )

    if local_rank == 0:
        tb_dir = os.path.join(training_configs.log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir, purge_step=global_step)
        writer.add_text("config/training", str(training_configs.to_dict()))
        writer.add_text("config/model", str(model_config.to_dict()))
        if resume_payload is not None:
            writer.add_text("resume/info", f"resume_from={resume_path}, global_step={global_step}, start_epoch={start_epoch + 1}")
        training_configs.save_json(config_json_path, include_deepspeed=True)
        logger.info(f"配置已导出: {config_json_path}")
        logger.info(
            f"Checkpoint 保存策略: 直接保存完整 .pth（best/latest/fixed），latest 每 {update_epoch} epoch，固定存档每 {args.fixed_save_interval} epoch"
        )

    # ---------- 提取损失计算参数（避免每次调用重复写） ----------
    loss_kwargs = dict(
        device=model_engine.device,
        focal_loss_weight=getattr(training_configs, "focal_loss_weight", 2.0),
        dice_loss_weight=getattr(training_configs, "dice_loss_weight", 0.5),
        focal_alpha=getattr(training_configs, "focal_alpha", 0.25),
        focal_gamma=getattr(training_configs, "focal_gamma", 2.0),
        bce_loss_weight=getattr(training_configs, "bce_loss_weight", 2.0),
        ce_loss_weight=getattr(training_configs, "ce_loss_weight", 1.0),
        route_loss_weight=getattr(training_configs, "route_loss_weight", 1.0),
        route_exist_loss_weight=getattr(training_configs, "route_exist_loss_weight", 0.25),
        route_sparse_loss_weight=getattr(training_configs, "route_sparse_loss_weight", 0.05),
        route_target_present_count=getattr(training_configs, "route_target_present_count", 1.0),
    )

    logger.info(f"损失配置: {loss_kwargs}")

    # ---------- 训练循环 ----------
    logger.info("=" * 80)
    logger.info("开始训练循环")
    logger.info("=" * 80)

    def _save_full_checkpoint(filename: str, meta: Dict):
        zero_stage = getattr(training_configs.deepspeed, "zero_stage", 3)
        saved = _save_model_state_to_cpu(
            model_engine,
            os.path.join(ckpt_dir, filename),
            meta,
            local_rank,
            logger,
            optimizer=optimizer,
            scheduler=scheduler,
            training_cfg=training_configs,
            asset_bundle=portable_asset_bundle,
            zero_stage=zero_stage,
        )
        if not saved:
            logger.warning(f"保存 portable checkpoint 失败: {filename}，回退到 DeepSpeed ZeRO checkpoint")
            tag = os.path.splitext(filename)[0]
            model_engine.save_checkpoint(ckpt_dir, tag=tag, client_state=meta, save_latest=False)

    for epoch in range(start_epoch, training_configs.epochs):
        logger.info(f"----- Epoch [{epoch + 1}/{training_configs.epochs}] -----")
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        # 训练
        global_step, train_results = train_one_epoch(
            train_loader, model_engine, optimizer, scheduler, training_configs,
            epoch, global_step, writer, logger, local_rank,
            loss_kwargs=loss_kwargs,
        )

        # 验证
        val_results = validate_one_epoch(
            val_loader, model_engine, training_configs, epoch, writer, logger, local_rank,
            loss_kwargs=loss_kwargs,
        )
        # 将验证集搜索得到的预测二值化阈值写回配置；GT 阈值保持用户/benchmark 指定值不变。
        if (
            getattr(training_configs, "auto_select_mask_threshold", True)
            and getattr(training_configs, "write_selected_mask_threshold_to_config", True)
        ):
            updated_threshold = False
            if "best_mask_threshold_2d" in val_results:
                training_configs.mask_threshold_2d = float(val_results["best_mask_threshold_2d"])
                updated_threshold = True
            if "best_mask_threshold_3d" in val_results:
                training_configs.mask_threshold_3d = float(val_results["best_mask_threshold_3d"])
                updated_threshold = True
            if updated_threshold and local_rank == 0:
                training_configs.save_json(config_json_path, include_deepspeed=True)
                logger.info(
                    "已将验证集最优预测阈值写回配置: "
                    f"mask_threshold_2d={training_configs.mask_threshold_2d:.4f}, "
                    f"mask_threshold_3d={training_configs.mask_threshold_3d:.4f}, "
                    f"path={config_json_path}"
                )
        last_val_metrics = _to_serializable_metrics(val_results)
        torch.cuda.empty_cache()

        monitor = float(val_results.get("loss", train_results.get("loss", float("inf"))))
        current_epoch = epoch + 1
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor
            best_epoch = current_epoch

        should_save_latest = (current_epoch % update_epoch == 0) or (current_epoch == training_configs.epochs)
        should_save_fixed = (current_epoch % args.fixed_save_interval == 0)

        save_flags = [False, False, False]  # [best, latest, fixed]
        if local_rank == 0:
            save_flags = [bool(is_best), bool(should_save_latest), bool(should_save_fixed)]
        dist.broadcast_object_list(save_flags, src=0, group=cpu_obj_group)
        is_best_save, is_latest_save, is_fixed_save = save_flags

        did_save = False
        if is_best_save or is_latest_save or is_fixed_save:
            common_meta = {
                "epoch": current_epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_metric),
                "val_loss": float(monitor),
                "val_metrics": last_val_metrics,
            }
            if is_best_save:
                _save_full_checkpoint("best_ds.pth", common_meta)
                if local_rank == 0:
                    logger.info(f"Best checkpoint 更新: epoch={best_epoch}, val_loss={best_metric:.6f}")
                    if writer is not None:
                        log_scalar_dict(
                            writer,
                            "checkpoint",
                            {"best_val_loss": best_metric, "best_epoch": float(best_epoch)},
                            current_epoch,
                        )
                did_save = True
            if is_latest_save:
                _save_full_checkpoint("latest_ds.pth", common_meta)
                if local_rank == 0:
                    logger.info(
                        f"Latest checkpoint 已保存: "
                        f"epoch={current_epoch}, best_epoch={best_epoch}, best_val_loss={best_metric:.6f}"
                    )
                did_save = True
            if is_fixed_save:
                fixed_name = f"epoch_{current_epoch:04d}_ds.pth"
                _save_full_checkpoint(fixed_name, common_meta)
                if local_rank == 0:
                    logger.info(f"固定周期 checkpoint 已保存: {os.path.join(ckpt_dir, fixed_name)}")
                did_save = True
        if did_save:
            dist.barrier()

    # ---------- 训练结束：保存最新模型（用于断点续训）----------
    final_epoch = training_configs.epochs
    _save_full_checkpoint(
        "latest_ds.pth",
        {
            "epoch": final_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_metric),
            "global_step": global_step,
            "val_metrics": last_val_metrics,
        },
    )
    dist.barrier()
    if local_rank == 0:
        logger.info("训练结束：已额外导出完整 latest 权重 latest_ds.pth")

    logger.info("=" * 80)
    logger.info(
        f"训练完成! epochs={final_epoch}, "
        f"best_epoch={best_epoch}, best_val_loss={best_metric:.6f}"
    )
    logger.info("=" * 80)
    if local_rank == 0 and writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
