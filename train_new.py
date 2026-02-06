import argparse
import os
import logging
from contextlib import nullcontext
from functools import partial
from datetime import datetime

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
import torch.distributed as dist 
from peft import get_peft_model
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from tqdm import tqdm

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from utils.base_dataset import JointDataset
from utils.dataset import Qwen3VLDataset, Qwen3VLTrainDataset, qwen3vl_collate_fn
from utils.common import dict_to_cuda
from utils import calculator as calc

local_rank = int(os.environ.get("LOCAL_RANK", 0))


def setup_logger(log_dir, local_rank=0):
    """
    设置日志系统
    - Rank 0: 同时输出到控制台和文件
    - 其他 Rank: 只输出到控制台
    """
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s [Rank %(rank)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出（所有进程）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（仅 Rank 0）
    if local_rank == 0 and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")
    
    # 为日志记录添加 rank 信息
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = local_rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="JointAffordance (Qwen) training")
    parser.add_argument("--qwen_model", type=str, default=None, help="Qwen 模型路径或名称")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="SAM 权重路径")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集路径")
    parser.add_argument("--log_dir", type=str, default=None, help="日志与权重输出目录")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_known_args()[0]


def enable_trainable_modules(model, name_filters):
    for name, param in model.named_parameters():
        if any(key in name for key in name_filters):
            param.requires_grad = True


def create_param_groups(model, config, logger):
    if not config.use_layerwise_lr:
        return [p for p in model.parameters() if p.requires_grad]

    def _collect_params(module):
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad]

    llm_params = _collect_params(getattr(model, "mllm", None))
    vision_2d_params = _collect_params(getattr(model, "image_decoder", None))
    vision_3d_params = _collect_params(getattr(model, "point_decoder", None))

    used_ids = {id(p) for p in llm_params + vision_2d_params + vision_3d_params}
    other_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in used_ids:
            continue
        other_params.append(param)

    param_groups = []
    if llm_params:
        param_groups.append({
            "params": llm_params,
            "lr": config.llm_lr,
            "name": "llm",
        })
        logger.info(f"✓ LLM 参数组: {len(llm_params)} 个参数, lr={config.llm_lr}")
    if vision_2d_params:
        param_groups.append({
            "params": vision_2d_params,
            "lr": config.vision_2d_lr,
            "name": "vision_2d",
        })
        logger.info(f"✓ 2D 视觉参数组: {len(vision_2d_params)} 个参数, lr={config.vision_2d_lr}")
    if vision_3d_params:
        param_groups.append({
            "params": vision_3d_params,
            "lr": config.vision_3d_lr,
            "name": "vision_3d",
        })
        logger.info(f"✓ 3D 视觉参数组: {len(vision_3d_params)} 个参数, lr={config.vision_3d_lr}")
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": config.lr,
            "name": "other",
        })
        logger.info(f"✓ 其他参数组: {len(other_params)} 个参数, lr={config.lr}")

    return param_groups


def main():
    args = parse_args()
    config = TrainingConfig()
    model_config = config.model_config

    if args.qwen_model:
        model_config.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        model_config.image_decoder.vision_pretrained = args.vision_pretrained
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    if args.log_dir:
        config.log_dir = args.log_dir

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式训练
    deepspeed.init_distributed()
    
    # 设置日志系统
    logger = setup_logger(config.log_dir, local_rank)
    logger.info("=" * 80)
    logger.info("开始训练 - Joint Affordance Model")
    logger.info("=" * 80) 
    
    """ ------------------------- 初始化模型 --------------------------- """
    logger.info("正在初始化模型...")
    zero_init_context = nullcontext()
    # 和SAM有冲突，暂不使用
    # if config.deepspeed.zero_stage == 3:
    #     logger.info("启用 ZeRO-3 初始化以避免显存峰值")
    #     zero_init_context = deepspeed.zero.Init(
    #         enabled=True,
    #         config_dict_or_path=config.deepspeed.to_dict(),
    #         remote_device=config.deepspeed.offload_param_device,
    #         pin_memory=config.deepspeed.offload_param_pin_memory,
    #     )
    with zero_init_context:
        model = JointAffordanceModel(model_config)
    logger.info(f"模型初始化完成: {model_config.mllm.qwen_model_name_or_path}")
    
    processor = AutoProcessor.from_pretrained(model_config.mllm.qwen_model_name_or_path)
    data_collator = partial(
        qwen3vl_collate_fn,
        tokenizer=processor.tokenizer,
        output_image_size=config.image_size,
        output_point_nums=config.num_points,
        precision=config.precision,
    )

    """ ------------------------- 选择训练参数、应用lora --------------------------- """
    logger.info("配置训练参数...")
    for p in model.parameters():
        p.requires_grad = False

    # 使用 LoRA 包裹 MLLM 主干（qwen）
    if config.lora.lora_r > 0:
        logger.info(f"应用 LoRA: r={config.lora.lora_r}, alpha={config.lora.lora_alpha}")
        lora_config = config.lora.to_peft_config()
        model.mllm.model = get_peft_model(model.mllm.model, lora_config)
    else:
        logger.info("未使用 LoRA")

    # 解冻必要模块
    logger.info(f"训练参数模块: {', '.join(config.name_of_params_to_train)}")
    enable_trainable_modules(model, config.name_of_params_to_train)

    """ ------------------------- 加载数据集 --------------------------- """
    logger.info("=" * 80)
    logger.info("加载数据集")
    logger.info("=" * 80)
    data_objects = [None, None]

    if local_rank == 0:
        logger.info(f"主进程 (Rank {local_rank}) 开始读取磁盘数据...")
        logger.info(f"数据集路径: {config.dataset_dir}")
        train_data_manager = JointDataset(dataset_root=config.dataset_dir, dtype='train').load_all_data()
        val_data_manager = JointDataset(dataset_root=config.dataset_dir, dtype='val').load_all_data()
        
        data_objects[0] = train_data_manager.samples
        data_objects[1] = val_data_manager.samples
        logger.info(f"数据加载完成: 训练集 {len(data_objects[0])} 条, 验证集 {len(data_objects[1])} 条")
        logger.info("准备广播数据到其他 GPU...")
    else:
        logger.info(f"子进程 (Rank {local_rank}) 等待主进程广播数据...")

    # 广播数据：将 data_objects 从 src=0 发送给所有进程
    dist.broadcast_object_list(data_objects, src=0)

    # 解包数据
    train_samples = data_objects[0]
    val_samples = data_objects[1]
    
    if local_rank != 0:
        logger.info(f"已收到数据: 训练集 {len(train_samples)} 条, 验证集 {len(val_samples)} 条")
    
    logger.info(f"数据集配置: image_size={config.image_size}, num_points={config.num_points}, precision={config.precision}")
    
    if config.samples_per_epoch is not None:
        train_dataset = Qwen3VLTrainDataset(
            train_samples,
            processor=processor,
            image_size=config.image_size,
            num_points=config.num_points,
            precision=config.precision,
            samples_per_epoch=config.samples_per_epoch,
            use_sample_cache=config.use_sample_cache,
        )
    else:
        train_dataset = Qwen3VLDataset(
            train_samples,
            processor=processor,
            image_size=config.image_size,
            num_points=config.num_points,
            precision=config.precision,
            use_sample_cache=config.use_sample_cache,
        )
    val_dataset = Qwen3VLDataset(
        val_samples,
        processor=processor,
        image_size=config.image_size,
        num_points=config.num_points,
        precision=config.precision,
        use_sample_cache=config.use_sample_cache,
    )

    """ ------------------------- DeepSpeed 初始化（分层学习率） --------------------------- """
    logger.info("=" * 80)
    logger.info("初始化 DeepSpeed")
    logger.info("=" * 80)
    params_to_train = create_param_groups(model, config, logger)
    if len(params_to_train) == 0:
        logger.error("没有可训练参数，请检查 name_of_params_to_train 或 LoRA 配置")
        raise RuntimeError("没有可训练参数，请检查 name_of_params_to_train 或 LoRA 配置")
    
    logger.info(f"DeepSpeed 配置: ZeRO Stage {config.deepspeed.zero_stage}")
    logger.info(f"训练配置: epochs={config.epochs}, micro_batch_size={config.deepspeed.train_micro_batch_size_per_gpu}, "
                f"gradient_accumulation={config.deepspeed.gradient_accumulation_steps}")

    if config.deepspeed.zero_stage == 3 and config.deepspeed.offload_param_device == "cpu":
        logger.info("ZeRO-3 + CPU 参数卸载：初始化前将模型留在 CPU")
        model = model.cpu()
        torch.cuda.empty_cache()

    if config.use_layerwise_lr:
        logger.info("使用分层学习率 + 自定义优化器/调度器")
        steps_per_epoch = config.steps_per_epoch
        if steps_per_epoch is None:
            micro_bs = config.deepspeed.train_micro_batch_size_per_gpu
            steps_per_epoch = max(1, len(train_dataset) // max(1, micro_bs))
        logger.info(f"每个 epoch 步数: {steps_per_epoch}, 总步数: {config.epochs * steps_per_epoch}")

        optimizer_cls = (
            DeepSpeedCPUAdam
            if config.deepspeed.offload_optimizer_device == "cpu"
            else torch.optim.AdamW
        )
        optimizer = optimizer_cls(
            params_to_train,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        total_steps = config.epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_num_steps,
            num_training_steps=total_steps,
        )
        logger.info(f"优化器: AdamW (weight_decay={config.weight_decay}, betas=({config.beta1}, {config.beta2}))")
        logger.info(f"学习率调度器: CosineAnnealingLR (warmup_steps={config.warmup_num_steps})")
        
        model_engine, _, train_loader, _ = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            collate_fn=data_collator,
            config=config.deepspeed.to_dict(),
        )
    else:
        logger.info("使用 DeepSpeed 内置优化器/调度器")
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            collate_fn=data_collator,
            config=config.deepspeed.to_dict(),
        )
    
    logger.info(f"DeepSpeed 初始化完成，训练数据加载器大小: {len(train_loader)}")


    """ ------------------------- 训练 --------------------------- """
    logger.info("=" * 80)
    logger.info("开始训练循环")
    logger.info("=" * 80)
    model_engine.train()
    
    for epoch in range(config.epochs):
        logger.info("-" * 80)
        logger.info(f"Epoch [{epoch+1}/{config.epochs}]")
        logger.info("-" * 80)
        
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        
        epoch_img_loss = 0.0
        epoch_pc_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        if local_rank == 0:
            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                dynamic_ncols=True,
                desc=f"Epoch {epoch+1}/{config.epochs}",
            )
        else:
            train_iter = train_loader

        for batch_idx, input_dict in enumerate(train_iter):
            input_dict = dict_to_cuda(input_dict, device=model_engine.device)
            output_dict = model_engine(**input_dict)

            img_loss = torch.tensor(0.0, device=model_engine.device)
            pc_loss = torch.tensor(0.0, device=model_engine.device)

            if output_dict.get("image_logits") is not None and "img_gt_tensor" in input_dict:
                _, _, img_loss = calc.img_loss(
                    pred_masks=output_dict["image_logits"],
                    gt_masks=input_dict["img_gt_tensor"],
                    bce_loss_weight=model_engine.module.config.bce_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                    dice_loss_weight=model_engine.module.config.dice_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                )

            if output_dict.get("point_logits") is not None and "pc_gt_tensor" in input_dict:
                _, _, pc_loss = calc.pc_loss(
                    pred_3d_masks=output_dict["point_logits"],
                    gt_3d_masks=input_dict["pc_gt_tensor"],
                    bce_loss_weight=model_engine.module.config.bce_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                    dice_loss_weight=model_engine.module.config.dice_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                )

            loss = (img_loss + pc_loss) / max(1, config.grad_accumulation_steps)
            model_engine.backward(loss)
            model_engine.step()
            model_engine.zero_grad()
            
            # 累计损失
            epoch_img_loss += img_loss.item()
            epoch_pc_loss += pc_loss.item()
            epoch_total_loss += loss.item()
            num_batches += 1
            
            # 定期打印训练进度
            if (batch_idx + 1) % config.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                current_lr = scheduler.get_last_lr()[0] if config.use_layerwise_lr and scheduler else None
                log_msg = f"Epoch [{epoch+1}/{config.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] " \
                         f"Loss: {loss.item():.6f} (Img: {img_loss.item():.6f}, PC: {pc_loss.item():.6f})"
                if current_lr is not None:
                    log_msg += f" LR: {current_lr:.2e}"
                logger.info(log_msg)
                if local_rank == 0 and hasattr(train_iter, "set_postfix"):
                    postfix = {"loss": f"{loss.item():.4f}"}
                    if current_lr is not None:
                        postfix["lr"] = f"{current_lr:.2e}"
                    train_iter.set_postfix(postfix, refresh=False)
        
        # Epoch 结束统计
        avg_img_loss = epoch_img_loss / max(1, num_batches)
        avg_pc_loss = epoch_pc_loss / max(1, num_batches)
        avg_total_loss = epoch_total_loss / max(1, num_batches)
        current_lr = scheduler.get_last_lr()[0] if config.use_layerwise_lr and scheduler else None
        
        logger.info(f"Epoch [{epoch+1}/{config.epochs}] 训练完成 - "
                   f"平均 Loss: {avg_total_loss:.6f} (Img: {avg_img_loss:.6f}, PC: {avg_pc_loss:.6f})")
        if current_lr is not None:
            logger.info(f"当前学习率: {current_lr:.2e}")

        """ ------------------------- 验证 --------------------------- """
        if val_dataset is not None:
            logger.info("开始验证...")
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.val_batch_size,
                shuffle=False,
                num_workers=config.workers,
                pin_memory=True,
                collate_fn=data_collator,
            )
            model_engine.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                total_val_img_loss = 0.0
                total_val_pc_loss = 0.0
                total_batches = 0
                for val_dict in val_loader:
                    val_dict = dict_to_cuda(val_dict, device=model_engine.device)
                    val_output = model_engine(**val_dict)
                    img_loss = torch.tensor(0.0, device=model_engine.device)
                    pc_loss = torch.tensor(0.0, device=model_engine.device)
                    if val_output.get("image_logits") is not None and "img_gt_tensor" in val_dict:
                        _, _, img_loss = calc.img_loss(
                            pred_masks=val_output["image_logits"],
                            gt_masks=val_dict["img_gt_tensor"],
                            bce_loss_weight=model_engine.module.config.bce_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                            dice_loss_weight=model_engine.module.config.dice_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                        )
                    if val_output.get("point_logits") is not None and "pc_gt_tensor" in val_dict:
                        _, _, pc_loss = calc.pc_loss(
                            pred_3d_masks=val_output["point_logits"],
                            gt_3d_masks=val_dict["pc_gt_tensor"],
                            bce_loss_weight=model_engine.module.config.bce_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                            dice_loss_weight=model_engine.module.config.dice_loss_weight if hasattr(model_engine.module, "config") else 1.0,
                        )
                    total_val_loss += (img_loss + pc_loss).item()
                    total_val_img_loss += img_loss.item()
                    total_val_pc_loss += pc_loss.item()
                    total_batches += 1
                if local_rank == 0 and total_batches > 0:
                    avg_val_loss = total_val_loss / total_batches
                    avg_val_img_loss = total_val_img_loss / total_batches
                    avg_val_pc_loss = total_val_pc_loss / total_batches
                    logger.info(f"Epoch [{epoch+1}/{config.epochs}] 验证完成 - "
                               f"Loss: {avg_val_loss:.6f} (Img: {avg_val_img_loss:.6f}, PC: {avg_val_pc_loss:.6f})")
            model_engine.train()
    
    logger.info("=" * 80)
    logger.info("训练完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()