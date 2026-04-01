"""
Joint Affordance 训练脚本（FSDP 版本，基于 Qwen MLLM）

与现有的 train.py 基本一致，唯一主要区别：
- 使用 PyTorch FSDP 替代 DeepSpeed，用于模型分片（ZeRO-3 等价能力）
- 不强制统一参数 dtype，允许不同子模块使用不同精度（参数级混合精度）

说明：
- MixedPrecision 不在 FSDP 里配置 param_dtype，让各子模块在构造/初始化阶段自行决定 dtype
  （例如 Qwen / 图像分支用 bf16，PointNet++ 用 fp32），FSDP 只负责分片和通信。
"""

import argparse
import os
from functools import partial
from typing import Dict
import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
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
from utils.debug import log_param_dtype_stats, count_model_params, _collect_batch_runtime_stats


ENV_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def parse_args():
    parser = argparse.ArgumentParser(description="JointAffordance (Qwen) training with FSDP")
    parser.add_argument("--qwen_model", type=str, default=None, help="Qwen 模型路径或名称")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="SAM 权重路径")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集路径")
    parser.add_argument("--log_dir", type=str, default=None, help="日志与权重输出目录")
    parser.add_argument("--update_epoch", type=int, default=5, help="每隔多少个 epoch 保存 latest checkpoint")
    parser.add_argument("--fixed_save_interval", type=int, default=100, help="固定长周期保存 checkpoint 的间隔，用于保存收敛状态。")
    parser.add_argument("--lazy_load", dest="lazy_load", action="store_true", help="启用懒加载（默认启用）",default=True)
    parser.add_argument("--local_rank", type=int, default=ENV_LOCAL_RANK)
    return parser.parse_known_args()[0]


# ===================== 模型初始化辅助 =====================

def enable_trainable_modules(model, name_filters):
    """按关键字列表解冻参数"""
    for name, param in model.named_parameters():
        if any(key in name for key in name_filters):
            param.requires_grad = True


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
        p for _, p in model.named_parameters()
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
    lr_list = scheduler.get_last_lr()
    lr_dict = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name")
        if not group_name:
            group_name = "lr" if len(optimizer.param_groups) == 1 else f"group_{idx}"
        lr_dict[group_name] = float(lr_list[idx])
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


def train_one_epoch(
    train_loader, model_fsdp, optimizer, scheduler, config,
    epoch, global_step, writer, logger, local_rank,
    loss_kwargs: Dict,
):
    """
    训练一个 epoch。
    与 train.py 基本一致，只是模型包装为 FSDP，反向传播使用标准 .backward()。
    """
    device = torch.device("cuda", local_rank)
    model_fsdp.train()
    metrics = build_torchmetrics_bundle(device=device)

    loader = (
        tqdm(train_loader, total=len(train_loader), dynamic_ncols=True,
             desc=f"Epoch {epoch + 1}/{config.epochs}")
        if local_rank == 0 else train_loader
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    accum_steps = max(1, config.grad_accumulation_steps)
    monitor_freq = max(1, getattr(config, "monitor_freq", config.print_freq))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, input_dict in enumerate(loader):
        input_dict = dict_to_cuda(input_dict, device=device)

        # 前向
        output_dict = model_fsdp(**input_dict)

        # 统一计算损失（Focal+Dice for 2D, BCE+Dice for 3D, CE for LLM）
        loss_dict = calc.compute_losses(output_dict, input_dict, **loss_kwargs)
        loss = loss_dict["loss"] / accum_steps

        # 反向 + 梯度累积更新
        loss.backward()
        should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
        if should_step:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # 指标更新
        update_torchmetrics(
            metrics, loss_dict, output_dict, input_dict, config.batch_size,
            threshold_2d=max(config.mask_threshold_2d, 0.5),
            threshold_3d=config.mask_threshold_3d,
        )

        # 打印与 TensorBoard
        if (batch_idx + 1) % config.print_freq == 0 or (batch_idx + 1) == len(train_loader):
            lr_dict = get_current_lr(scheduler, optimizer)
            lr_text = ", ".join([f"{k}={v:.2e}" for k, v in lr_dict.items()])
            ignored_cnt = int(output_dict.get("ce_ignored_token_count", 0) or 0)
            log_msg = (
                f"  [{batch_idx + 1}/{len(train_loader)}] "
                f"loss={loss_dict['loss'].item():.6f} "
                f"(ce={loss_dict['ce_loss'].item():.6f}, "
                f"img={loss_dict['img_loss'].item():.6f}, "
                f"pc={loss_dict['pc_loss'].item():.6f}, "
                f"ce_ignore_aff_tok={ignored_cnt})"
                + (f" lr=({lr_text})" if lr_text else "")
            )
            logger.info(log_msg)
            if local_rank == 0:
                print(log_msg)
            # if (batch_idx + 1) % monitor_freq == 0 or (batch_idx + 1) == len(train_loader):
            #     runtime_stats = _collect_batch_runtime_stats(input_dict, device)
            #     if runtime_stats:
            #         logger.info(
            #             "  runtime: "
            #             f"mem={runtime_stats.get('mem_allocated_gb', 0.0):.2f}G "
            #             f"reserved={runtime_stats.get('mem_reserved_gb', 0.0):.2f}G "
            #             f"peak={runtime_stats.get('mem_peak_allocated_gb', 0.0):.2f}G "
            #             f"seq(max/mean)={runtime_stats.get('seq_len_max', 0.0):.0f}/{runtime_stats.get('seq_len_mean', 0.0):.1f} "
            #             f"vtok(max/mean)={runtime_stats.get('vision_tokens_max', 0.0):.0f}/{runtime_stats.get('vision_tokens_mean', 0.0):.1f} "
            #             f"pc(max/mean)={runtime_stats.get('pc_points_max', 0.0):.0f}/{runtime_stats.get('pc_points_mean', 0.0):.1f}"
            #         )
            if local_rank == 0 and writer is not None:
                batch_log = {k: loss_dict[k].item() for k in loss_dict}
                batch_log["ce_ignored_token_count"] = float(ignored_cnt)
                for k, v in lr_dict.items():
                    batch_log[f"lr/{k}"] = v
                if lr_dict:
                    batch_log["lr"] = next(iter(lr_dict.values()))
                # runtime_stats = _collect_batch_runtime_stats(input_dict, device)
                # for k, v in runtime_stats.items():
                #     batch_log[k] = v
                log_scalar_dict(writer, "train_batch", batch_log, global_step)
            if local_rank == 0 and hasattr(loader, "set_postfix"):
                postfix = {"loss": f"{loss_dict['loss'].item():.4f}"}
                if lr_dict:
                    postfix["lr"] = f"{next(iter(lr_dict.values())):.2e}"
                loader.set_postfix(postfix, refresh=False)

    # 汇总指标（FSDP 下 compute() 已自动在进程内同步，需要时可再做 dist.all_reduce）
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
    val_loader, model_fsdp, config,
    epoch, writer, logger, local_rank,
    loss_kwargs: Dict,
):
    """验证一个 epoch。"""
    device = torch.device("cuda", local_rank)
    model_fsdp.eval()
    metrics = build_torchmetrics_bundle(device=device)

    val_iter = (
        tqdm(val_loader, total=len(val_loader), dynamic_ncols=True,
             desc=f"Val {epoch + 1}/{config.epochs}")
        if local_rank == 0 else val_loader
    )
    for val_dict in val_iter:
        val_dict = dict_to_cuda(val_dict, device=device)
        val_output = model_fsdp(**val_dict, return_hidden_states=False, return_mllm_output=False)
        loss_dict = calc.compute_losses(val_output, val_dict, **loss_kwargs)
        update_torchmetrics(
            metrics, loss_dict, val_output, val_dict, config.val_batch_size,
            threshold_2d=max(config.mask_threshold_2d, 0.5),
            threshold_3d=config.mask_threshold_3d,
        )

    val_results = compute_and_reset_torchmetrics(metrics)
    if local_rank == 0:
        log_epoch_summary(logger, epoch + 1, config.epochs, "val", val_results)
        if writer is not None:
            log_scalar_dict(writer, "val_epoch", val_results, epoch + 1)

    model_fsdp.train()
    return val_results


def main():
    args = parse_args()
    training_configs = TrainingConfig()
    model_config = training_configs.model_config

    # 覆盖配置
    if args.qwen_model:
        model_config.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        model_config.image_decoder.vision_pretrained = args.vision_pretrained
    if args.dataset_dir:
        training_configs.dataset_dir = args.dataset_dir
    if args.log_dir:
        training_configs.log_dir = args.log_dir
    if args.update_epoch is not None:
        training_configs.update_epoch = max(1, int(args.update_epoch))

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    # object 广播走 CPU(gloo) 通道，避免 NCCL object collective 占用大量显存
    cpu_obj_group = dist.new_group(backend="gloo")

    logger = setup_logger(training_configs.log_dir, local_rank)
    logger.info("=" * 80)
    logger.info("Joint Affordance Model - FSDP 训练开始")
    logger.info("=" * 80)

    writer = None
    if local_rank == 0:
        tb_dir = os.path.join(training_configs.log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("config/training", str(training_configs.to_dict()))
        writer.add_text("config/model", str(model_config.to_dict()))

    # ---------- 加载数据集 ----------
    logger.info("加载数据集...")
    if args.lazy_load:
        # 懒加载模式：各 rank 仅构建轻量索引（不广播 dataset/samples 大对象）
        train_samples = JointDataset(dataset_root=training_configs.dataset_dir, split_file='train.json', lazy_load=True)
        val_samples = JointDataset(dataset_root=training_configs.dataset_dir, split_file='val.json', lazy_load=True)

        token_obj = [None]
        if local_rank == 0:
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
            train_data = JointDataset(dataset_root=training_configs.dataset_dir, split_file='train.json', lazy_load=False).load_all_data()
            val_data = JointDataset(dataset_root=training_configs.dataset_dir, split_file='val.json', lazy_load=False).load_all_data()
            train_payload = train_data.samples
            val_payload = val_data.samples
            pair_token_map = build_functional_tokens_from_samples(train_payload + val_payload)
            data_objects = [train_payload, val_payload, pair_token_map]
            logger.info(f"训练集 {len(train_payload)} 条, 验证集 {len(val_payload)} 条")
        dist.barrier()  # 防止加载训练数据过久导致崩溃
        # 非懒加载模式：保持“仅 rank0 加载一次 + 广播样本列表”策略
        dist.broadcast_object_list(data_objects, src=0, group=cpu_obj_group)
        train_samples, val_samples, pair_token_map = data_objects

    # 构造按模态分组的 token 注册表（先传 token 字符串给 MLLM，随后会映射到 token_id）
    functional_tokens = {
        "img": dict(FUNCTIONAL_TOKENS.get("img", {})),
        "pc": dict(FUNCTIONAL_TOKENS.get("pc", {})),
    }
    functional_tokens["img"].update(pair_token_map.get("img", {}))
    functional_tokens["pc"].update(pair_token_map.get("pc", {}))
    model_config.mllm.functional_tokens = functional_tokens
    if local_rank == 0:
        num_img = len(pair_token_map.get("img", {}))
        num_pc = len(pair_token_map.get("pc", {}))
        logger.info(
            f"已注册 obj-aff 专用 token: img={num_img}, pc={num_pc}"
        )

    # ---------- 初始化模型 ----------
    logger.info("正在初始化模型...")
    model = JointAffordanceModel(model_config)
    # MLLM 完成 tokenizer 注入后，回写 FUNCTIONAL_TOKENS 为最终双向映射（token_name <-> token_id）
    FUNCTIONAL_TOKENS.clear()
    FUNCTIONAL_TOKENS.update(model.mllm.functional_token_ids)
    device = torch.device("cuda", local_rank)
    model.to(device=device)  # 初始迁移到当前 GPU（各子模块内部再自行控制 dtype）
    if getattr(training_configs, "gradient_checkpointing", False):
        mllm_core = getattr(getattr(model, "mllm", None), "model", None)
        if mllm_core is not None and hasattr(mllm_core, "gradient_checkpointing_enable"):
            try:
                mllm_core.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # 兼容旧版 transformers：不支持 gradient_checkpointing_kwargs
                mllm_core.gradient_checkpointing_enable()
            if hasattr(mllm_core, "enable_input_require_grads"):
                mllm_core.enable_input_require_grads()
            logger.info("已启用 MLLM gradient checkpointing")

    # 直接复用模型内部的 processor（已含注入 functional tokens 的 tokenizer + image_processor），
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

    # 冻结 + LoRA + 解冻
    for p in model.parameters():
        p.requires_grad = False
    if training_configs.lora.lora_r > 0:
        logger.info(f"应用 LoRA: r={training_configs.lora.lora_r}, alpha={training_configs.lora.lora_alpha}")
        model.mllm.model = get_peft_model(model.mllm.model, training_configs.lora.to_peft_config())

    enable_trainable_modules(model, training_configs.name_of_params_to_train)
    log_param_dtype_stats(model, logger, stage="before_fsdp")
    total_params, trainable_params = count_model_params(model)
    logger.info(
        f"参数统计: total={total_params:,}, trainable={trainable_params:,}, "
        f"ratio={100.0 * trainable_params / max(1, total_params):.2f}%"
    )

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

    val_dataset = JointAffordanceTorchDataset(
        val_samples,
        **{k: v for k, v in train_ds_kwargs.items() if k != "samples_per_epoch"}
    )

    # ---------- DataLoader + Sampler ----------
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=False,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_configs.batch_size,
        sampler=train_sampler,
        num_workers=training_configs.workers,
        pin_memory=True,
        persistent_workers=training_configs.workers > 0,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_configs.val_batch_size,
        sampler=val_sampler,
        num_workers=training_configs.workers,
        pin_memory=True,
        persistent_workers=training_configs.workers > 0,
        collate_fn=data_collator,
    )

    # ---------- 根据实际 DataLoader 回填 steps_per_epoch ----------
    # 以每个 rank 的优化器更新步数为准（考虑梯度累积）。
    accum_steps = max(1, training_configs.grad_accumulation_steps)
    computed_steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)
    configured_steps = getattr(training_configs, "steps_per_epoch", None)
    training_configs.steps_per_epoch = computed_steps_per_epoch
    if configured_steps is not None and configured_steps != computed_steps_per_epoch:
        logger.warning(
            f"steps_per_epoch 配置值({configured_steps})与实际值({computed_steps_per_epoch})不一致，"
            f"已自动覆盖为实际值。"
        )
    else:
        logger.info(f"steps_per_epoch 已设置为 {computed_steps_per_epoch}（基于当前 DataLoader 自动计算）")

    # ---------- FSDP 包装模型 ----------
    model_fsdp = FSDP(model, device_id=device, use_orig_params=True)
    log_param_dtype_stats(model_fsdp, logger, stage="after_fsdp_wrap")

    # ---------- 优化器 & 调度器 ----------
    logger.info("初始化优化器与调度器（支持分层学习率）")
    params_to_train = create_param_groups(model_fsdp, training_configs, logger)
    if not params_to_train:
        raise RuntimeError("没有可训练参数，请检查配置")

    optimizer = torch.optim.AdamW(
        params_to_train,
        weight_decay=training_configs.weight_decay,
        betas=(training_configs.beta1, training_configs.beta2),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_configs.warmup_num_steps,
        num_training_steps=training_configs.epochs * training_configs.steps_per_epoch,
    )

    # ---------- 提取损失配置 ----------
    loss_kwargs = dict(
        device=device,
        focal_loss_weight=getattr(training_configs, "focal_loss_weight", 2.0),
        dice_loss_weight=getattr(training_configs, "dice_loss_weight", 0.5),
        focal_alpha=getattr(training_configs, "focal_alpha", 0.25),
        focal_gamma=getattr(training_configs, "focal_gamma", 2.0),
        bce_loss_weight=getattr(training_configs, "bce_loss_weight", 2.0),
        ce_loss_weight=getattr(training_configs, "ce_loss_weight", 1.0),
        route_loss_weight=getattr(training_configs, "route_loss_weight", 1.0),
        route_bal_loss_weight=getattr(training_configs, "route_bal_loss_weight", 0.05),
    )
    logger.info(f"损失配置: {loss_kwargs}")

    # ---------- 训练循环 ----------
    logger.info("=" * 80)
    logger.info("开始 FSDP 训练循环")
    logger.info("=" * 80)

    global_step = 0
    best_metric = float("inf")
    best_epoch = -1
    last_val_metrics: Dict = {}
    update_epoch = max(1, int(getattr(training_configs, "update_epoch", 1)))
    ckpt_dir = os.path.join(training_configs.log_dir, "checkpoints_fsdp")
    os.makedirs(ckpt_dir, exist_ok=True)
    config_json_path = os.path.join(ckpt_dir, "training_config.json")
    if local_rank == 0:
        training_configs.save_json(config_json_path)
        logger.info(f"配置已导出: {config_json_path}")
        logger.info(
            f"Checkpoint 保存策略: 直接保存完整 .pth（best/latest/fixed），latest 每 {update_epoch} epoch，固定存档每 {args.fixed_save_interval} epoch"
        )

    def _save_full_checkpoint(filename: str, meta: Dict):
        """
        标准 FSDP 完整权重保存：
        - 所有 rank 参与 state_dict 聚合（rank0_only=True）
        - 仅 rank0 落盘 .pth，可直接用于评估/推理加载
        """
        with FSDP.state_dict_type(
            model_fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            full_state = model_fsdp.state_dict()
        if local_rank == 0:
            payload = {
                **meta,
                "model_state_dict": full_state,
            }
            torch.save(payload, os.path.join(ckpt_dir, filename))

    for epoch in range(training_configs.epochs):
        logger.info(f"----- Epoch [{epoch + 1}/{training_configs.epochs}] -----")
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        global_step, train_results = train_one_epoch(
            train_loader, model_fsdp, optimizer, scheduler, training_configs,
            epoch, global_step, writer, logger, local_rank,
            loss_kwargs=loss_kwargs,
        )

        val_results = validate_one_epoch(
            val_loader, model_fsdp, training_configs, epoch, writer, logger, local_rank,
            loss_kwargs=loss_kwargs,
        )
        last_val_metrics = _to_serializable_metrics(val_results)

        monitor = float(val_results.get("loss", train_results.get("loss", float("inf"))))
        current_epoch = epoch + 1
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor
            best_epoch = current_epoch

        should_save_latest = (current_epoch % update_epoch == 0) or (current_epoch == training_configs.epochs)
        should_save_fixed = (current_epoch % args.fixed_save_interval == 0)

        # 由 rank0 决策，然后广播给所有 rank；保证保存触发条件一致
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
                # 记录完整评估指标（含 2D/3D 分支），便于横向比较 checkpoint
                "val_metrics": last_val_metrics,
            }
            if is_best_save:
                _save_full_checkpoint("best_fsdp.pth", common_meta)
                if local_rank == 0:
                    logger.info(f"Best checkpoint 更新: epoch={best_epoch}, val_loss={best_metric:.6f}")
                did_save = True
            if is_latest_save:
                _save_full_checkpoint("latest_fsdp.pth", common_meta)
                if local_rank == 0:
                    logger.info(
                        f"Latest checkpoint 已保存: "
                        f"epoch={current_epoch}, best_epoch={best_epoch}, best_val_loss={best_metric:.6f}"
                    )
                did_save = True
            if is_fixed_save:
                fixed_name = f"epoch_{current_epoch:04d}_fsdp.pth"
                _save_full_checkpoint(fixed_name, common_meta)
                if local_rank == 0:
                    logger.info(f"固定周期 checkpoint 已保存: {os.path.join(ckpt_dir, fixed_name)}")
                did_save = True
            if local_rank == 0:
                training_configs.save_json(config_json_path)
        # 仅在发生保存时同步，避免每个 epoch 都阻塞等待
        if did_save:
            dist.barrier()

    # 训练结束后可选导出一次完整 latest（与训练中 latest 保持一致，可用于最终覆盖）
    if local_rank == 0:
        with FSDP.state_dict_type(
            model_fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            final_state = model_fsdp.state_dict()
        torch.save(
            {
                "epoch": training_configs.epochs,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": best_metric,
                "val_metrics": last_val_metrics,
                "model_state_dict": final_state,
            },
            os.path.join(ckpt_dir, "latest_fsdp.pth"),
        )
        logger.info("训练结束：已额外导出完整 latest 权重 latest_fsdp.pth")

    logger.info("=" * 80)
    logger.info("FSDP 训练结束")
    logger.info("=" * 80)
    if local_rank == 0 and writer is not None:
        writer.flush()
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

