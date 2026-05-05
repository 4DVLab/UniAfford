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
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from tqdm import tqdm

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from utils.base_dataset import JointDataset
from utils.dataset import (
    JointAffordanceTorchDataset,
    JointAffordanceTrainDataset,
    joint_affordance_collate_fn,
)
from utils.common import dict_to_cuda, setup_logger, get_current_lr
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
from utils.model_io import build_portable_assets, build_portable_checkpoint_payload
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
    parser.add_argument("--log_dir", type=str, default=None, help="日志与权重输出目录")
    parser.add_argument("--local_rank", type=int, default=ENV_LOCAL_RANK)
    return parser.parse_known_args()[0]


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
        loss = loss_dict["loss"] / max(1, config.grad_accumulation_steps)

        model_engine.backward(loss)
        model_engine.step()
        model_engine.zero_grad()
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
            lr = get_current_lr(scheduler, optimizer)
            logger.info(
                f"  [{batch_idx + 1}/{len(train_loader)}] "
                f"loss={loss_dict['loss'].item():.6f} "
                f"(ce={loss_dict['ce_loss'].item():.6f}, "
                f"img={loss_dict['img_loss'].item():.6f}, "
                f"pc={loss_dict['pc_loss'].item():.6f})"
                + (f" lr={lr:.2e}" if lr else "")
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
                if lr is not None:
                    batch_log["lr"] = lr
                log_scalar_dict(writer, "train_batch", batch_log, global_step)
            if local_rank == 0 and hasattr(loader, "set_postfix"):
                postfix = {"loss": f"{loss_dict['loss'].item():.4f}"}
                if lr is not None:
                    postfix["lr"] = f"{lr:.2e}"
                loader.set_postfix(postfix, refresh=False)

    # Epoch 结束：汇总
    train_results = compute_and_reset_torchmetrics(metrics)
    lr = get_current_lr(scheduler, optimizer)
    log_epoch_summary(logger, epoch + 1, config.epochs, "train", train_results, lr)
    if local_rank == 0 and writer is not None:
        log_scalar_dict(writer, "train_epoch", train_results, epoch + 1)
        if lr is not None:
            log_scalar_dict(writer, "train_epoch", {"lr": lr}, epoch + 1)

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
            build_threshold_candidates(model_engine.device)
        )

    for val_dict in val_loader:
        val_dict = dict_to_cuda(val_dict, device=model_engine.device)
        val_output = model_engine(**val_dict)
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
            logger.info(
                "验证集最优预测阈值: "
                f"2D={val_results.get('best_mask_threshold_2d', config.mask_threshold_2d):.4f} "
                f"(gIoU={val_results.get('best_giou_2d', 0.0):.4f}), "
                f"3D={val_results.get('best_mask_threshold_3d', config.mask_threshold_3d):.4f} "
                f"(IoU={val_results.get('best_iou_3d', 0.0):.4f})"
            )
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
        ckpt = build_portable_checkpoint_payload(
            model_state_dict=state_cpu,
            meta=client_state,
            training_cfg=training_cfg,
            asset_bundle=asset_bundle,
        )
        torch.save(ckpt, save_path)
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

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    # 日志系统
    logger = setup_logger(training_configs.log_dir, local_rank)
    logger.info("=" * 80)
    logger.info("Joint Affordance Model - 开始训练")
    logger.info("=" * 80)

    writer = None
    if local_rank == 0:
        tb_dir = os.path.join(training_configs.log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("config/training", str(training_configs.to_dict()))
        writer.add_text("config/model", str(model_config.to_dict()))

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
    processor = AutoProcessor.from_pretrained(model_config.mllm.qwen_model_name_or_path)
    data_collator = partial(
        joint_affordance_collate_fn,
        tokenizer=processor.tokenizer,
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

    # ---------- 加载数据集（rank 0 读取后广播） ----------
    logger.info("加载数据集...")
    data_objects = [None, None]
    if local_rank == 0:
        train_data = JointDataset(dataset_root=training_configs.dataset_dir, split_file='train.json').load_all_data()
        val_data = JointDataset(dataset_root=training_configs.dataset_dir, split_file='val.json').load_all_data()
        data_objects = [train_data.samples, val_data.samples]
        logger.info(f"训练集 {len(data_objects[0])} 条, 验证集 {len(data_objects[1])} 条")
    dist.broadcast_object_list(data_objects, src=0)
    train_samples, val_samples = data_objects

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

    # 验证 DataLoader（在循环外创建，避免每个 epoch 重复创建）
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=training_configs.val_batch_size, shuffle=False,
        num_workers=training_configs.workers, pin_memory=True, collate_fn=data_collator,
    )
    log_param_dtype_stats(
        model_engine.module if hasattr(model_engine, "module") else model_engine,
        logger, stage="after_deepspeed",
    )

    # ---------- 提取损失计算参数（避免每次调用重复写） ----------
    loss_kwargs = dict(
        device=model_engine.device,
        # 2D: Focal + Dice
        focal_loss_weight=getattr(training_configs, "focal_loss_weight", 2.0),
        dice_loss_weight=getattr(training_configs, "dice_loss_weight", 0.5),
        focal_alpha=getattr(training_configs, "focal_alpha", 0.25),
        focal_gamma=getattr(training_configs, "focal_gamma", 2.0),
        # 3D: BCE + Dice
        bce_loss_weight=getattr(training_configs, "bce_loss_weight", 2.0),
        # LLM CE
        ce_loss_weight=getattr(training_configs, "ce_loss_weight", 1.0),
    )

    logger.info(f"损失配置: {loss_kwargs}")

    # ---------- 训练循环 ----------
    logger.info("=" * 80)
    logger.info("开始训练循环")
    logger.info("=" * 80)

    global_step = 0
    best_metric = float("inf")  # 以 val_loss 越小越好
    best_epoch = -1
    ckpt_dir = os.path.join(training_configs.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    config_json_path = os.path.join(ckpt_dir, "training_config.json")
    if local_rank == 0:
        training_configs.save_json(config_json_path, include_deepspeed=True)
        logger.info(f"配置已导出: {config_json_path}")

    for epoch in range(training_configs.epochs):
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
        # 验证阶段 torchmetrics 会暂存 pred/target，结束后释放；此处统一释放碎片显存，为保存 checkpoint 腾出空间
        torch.cuda.empty_cache()

        # Checkpoint：以 val_loss 为监控指标，保存 best
        # 优先「卸载到 CPU 再保存」避免 ZeRO-3 在 GPU 上全量汇聚导致 OOM；否则回退到 save_checkpoint
        monitor = val_results.get("loss", train_results.get("loss", float("inf")))
        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch + 1
            client_state = {
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_val_loss": best_metric,
            }
            zero_stage = getattr(training_configs.deepspeed, "zero_stage", 3)
            best_cpu_path = os.path.join(ckpt_dir, "best_cpu.pth")
            saved_to_cpu = _save_model_state_to_cpu(
                model_engine,
                best_cpu_path,
                client_state,
                local_rank,
                logger,
                training_cfg=training_configs,
                asset_bundle=portable_asset_bundle,
                zero_stage=zero_stage,
            )
            if not saved_to_cpu:
                logger.warning("Failed to save best checkpoint to CPU, falling back to ZeRO format")
                model_engine.save_checkpoint(
                    ckpt_dir, tag="best",
                    client_state=client_state,
                    save_latest=False,
                )
                if local_rank == 0:
                    logger.info(f"Best checkpoint (ZeRO 格式) 更新: epoch={best_epoch}, val_loss={best_metric:.6f}")
            if local_rank == 0:
                if writer is not None:
                    log_scalar_dict(writer, "checkpoint",
                                    {"best_val_loss": best_metric, "best_epoch": float(best_epoch)},
                                    epoch + 1)

    # ---------- 训练结束：保存最新模型（用于断点续训）----------
    final_epoch = training_configs.epochs
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.empty_cache()
    model_engine.save_checkpoint(
        ckpt_dir, tag="latest",
        client_state={
            "epoch": final_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_metric,
            "global_step": global_step,
        },
    )
    if local_rank == 0:
        logger.info(f"Latest checkpoint 已保存: epoch={final_epoch}, path={ckpt_dir}/latest")
    latest_cpu_path = os.path.join(ckpt_dir, "latest_cpu.pth")
    _save_model_state_to_cpu(
        model_engine,
        latest_cpu_path,
        client_state={
            "epoch": final_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_metric,
            "global_step": global_step,
        },
        local_rank=local_rank,
        logger=logger,
        training_cfg=training_configs,
        asset_bundle=portable_asset_bundle,
        zero_stage=zero_stage,
    )

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
