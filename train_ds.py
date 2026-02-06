import os
import shutil
import time
from functools import partial
from contextlib import nullcontext

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
from tqdm import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, collate_fn
from utils.common import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                          ProgressMeter, dict_to_cuda)
from utils.metrics import (
    AverageMeter,
    MetricsTracker,
    TensorBoardLogger,
    evaluate_segmentation_batch,
    print_validation_summary,
)
from utils import calculator as calc
from configs import TrainingConfig


# 全局变量
params_to_train = []


def create_param_groups(model, config):
    """
    创建参数组，为不同模块设置不同的学习率
    
    Args:
        model: LISA 模型
        config: 训练配置
        
    Returns:
        param_groups: 参数组列表，每组包含参数和对应的学习率
    """
    if not config.use_layerwise_lr:
        # 不使用分层学习率，返回所有可训练参数
        return [p for p in model.parameters() if p.requires_grad]
    
    # 分组参数
    llm_params = []          # LLM 部分（LoRA 层）
    vision_2d_params = []    # 2D 视觉部分（SAM mask_decoder, text_hidden_fcs）
    vision_3d_params = []    # 3D 点云部分（point_cloud_segmentor）
    other_params = []        # 其他参数（mm_projector, lm_head, embed_tokens）
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # LLM 部分（LoRA 层）
        if 'lora_' in name or 'base_layer' in name:
            llm_params.append(param)
            print(f"[LLM] {name}")
        
        # 2D 视觉部分
        elif 'mask_decoder' in name or 'text_hidden_fcs' in name:
            vision_2d_params.append(param)
            print(f"[2D Vision] {name}")
        
        # 3D 点云部分
        elif 'point_cloud_segmentor' in name:
            vision_3d_params.append(param)
            print(f"[3D Vision] {name}")
        
        # 其他参数
        else:
            other_params.append(param)
            print(f"[Other] {name}")
    
    # 创建参数组
    param_groups = []
    
    if llm_params:
        param_groups.append({
            'params': llm_params,
            'lr': config.llm_lr,
            'name': 'llm'
        })
        print(f"\n✓ LLM 参数组: {len(llm_params)} 个参数, lr={config.llm_lr}")
    
    if vision_2d_params:
        param_groups.append({
            'params': vision_2d_params,
            'lr': config.vision_2d_lr,
            'name': 'vision_2d'
        })
        print(f"✓ 2D 视觉参数组: {len(vision_2d_params)} 个参数, lr={config.vision_2d_lr}")
    
    if vision_3d_params:
        param_groups.append({
            'params': vision_3d_params,
            'lr': config.vision_3d_lr,
            'name': 'vision_3d'
        })
        print(f"✓ 3D 视觉参数组: {len(vision_3d_params)} 个参数, lr={config.vision_3d_lr}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.lr,
            'name': 'other'
        })
        print(f"✓ 其他参数组: {len(other_params)} 个参数, lr={config.lr}\n")
    
    return param_groups



def main():
    global params_to_train
    config = TrainingConfig()
    model_config = config.model_config
    mllm = model_config.mllm
    if config.local_rank == 0:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(config.log_dir)
    else:
        writer = None

    # Create model - 使用魔改后的 LISAForCausalLM 模型（已支持点云）
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        mllm.version,
        cache_dir=None,
        model_max_length=mllm.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 添加特殊标记
    tokenizer.add_tokens("[SEG]")  # 2D分割标记
    tokenizer.add_tokens("[AFF]")  # 3D affordance标记
    mllm.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    mllm.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if mllm.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # 模型参数配置
    model_args = {
        "train_mask_decoder": mllm.train_mask_decoder,
        "out_dim": mllm.out_dim,
        "ce_loss_weight": mllm.ce_loss_weight,
        "dice_loss_weight": mllm.dice_loss_weight,
        "bce_loss_weight": mllm.bce_loss_weight,
        "seg_token_idx": mllm.seg_token_idx,
        "aff_token_idx": mllm.aff_token_idx,  # 3D affordance token
        "vision_pretrained": mllm.vision_pretrained,
        "vision_tower": mllm.vision_tower,
        "use_mm_start_end": mllm.use_mm_start_end,
    }
    
    # 初始化魔改后的 LISA 模型
    zero_init_context = nullcontext()
    if config.deepspeed.zero_stage == 3:
        print("启用 ZeRO-3 初始化以避免显存峰值")
        zero_init_context = deepspeed.zero.Init(
            enabled=True,
            config_dict_or_path=config.deepspeed.to_dict(),
            remote_device=config.deepspeed.offload_param_device,
            pin_memory=config.deepspeed.offload_param_pin_memory,
        )
    with zero_init_context:
        model = LISAForCausalLM.from_pretrained(
            mllm.version, dtype=config.precision, low_cpu_mem_usage=True, **model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    if config.deepspeed.zero_stage != 3:
        vision_tower.to(dtype=config.precision, device=config.local_rank)
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        mllm.conv_type
    ]

    model.resize_token_embeddings(len(tokenizer))
    
    # 初始化 LISA 模块（包括 SAM 和 3D 点云分割器）
    if not config.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    # 先把所有参数冻结 (作为基底)
    for p in model.parameters():
        p.requires_grad = False

    # LoRA 配置（可选）
    if config.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all([x not in name for x in config.name_of_params_to_train])
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(model, config.lora_target_modules)
        lora_config = config.get_lora_config()
        lora_config.target_modules = lora_target_modules  # 使用动态查找的模块
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ✅ 在应用 LoRA 之后，重新开启非 LoRA 的关键模块
    # 这样可以确保 LoRA 层和其他模块都是可训练的
    for n, p in model.named_parameters():
        if any(t in n for t in config.name_of_params_to_train):
            p.requires_grad = True
            print(f"[Enabled] {n}")
    
    params_to_train = create_param_groups(model, config)
    
    # 打印检查
    if isinstance(params_to_train, list) and len(params_to_train) > 0:
        if isinstance(params_to_train[0], dict):
            # 参数组模式
            total_params = sum(sum(p.numel() for p in group['params']) for group in params_to_train)
            print(f"\n最终训练参数统计 (分层学习率模式):")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"Trainable parameters: {total_params} elements in {len(params_to_train)} groups")
        else:
            # 单一学习率模式
            print(f"\n最终训练参数统计 (单一学习率模式):")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"Trainable parameters: {len(params_to_train)} tensors, {sum(p.numel() for p in params_to_train)} elements")
    
    if len(params_to_train) == 0:
        raise ValueError("严重错误：没有发现可训练参数！请检查冻结逻辑。")
    world_size = torch.cuda.device_count()
    config.distributed = world_size > 1
    
    # 使用新的 DatasetManager 创建数据集
    dataset_manager = DatasetManager(
        dataset_dir=config.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=mllm.vision_tower,
        precision=config.precision,
        image_size=config.image_size,
        num_points=config.num_points,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        use_mm_start_end=mllm.use_mm_start_end,
        samples_per_epoch=config.samples_per_epoch,
        conv_type=mllm.conv_type,
        use_sample_cache=config.use_sample_cache
    )
    config.samples_per_epoch = dataset_manager.samples_per_epoch

    train_dataset = dataset_manager.get_train_dataset()

    if config.no_eval == False:
        val_dataset = dataset_manager.get_val_dataset()
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    # DeepSpeed 配置
    # 如果使用分层学习率，需要手动创建优化器和调度器
    if config.deepspeed.zero_stage == 3 and config.deepspeed.offload_param_device == "cpu":
        print("ZeRO-3 + CPU 参数卸载：初始化前将模型留在 CPU")
        model = model.cpu()
        torch.cuda.empty_cache()

    if config.use_layerwise_lr:
        # 手动创建优化器（支持参数组）
        optimizer_cls = (
            DeepSpeedCPUAdam
            if config.deepspeed.offload_optimizer_device == "cpu"
            else torch.optim.AdamW
        )
        optimizer = optimizer_cls(
            params_to_train,  # 参数组列表，每组有自己的 lr
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        
        # 手动创建学习率调度器
        total_steps = config.epochs * config.steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_num_steps,
            num_training_steps=total_steps,
        )
        
        # 初始化 DeepSpeed（不使用内置优化器和调度器）
        model_engine, _, train_loader, _ = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                output_image_size=config.image_size,
                output_point_nums=config.num_points,
                precision=config.precision,
            ),
            config=config.deepspeed.to_dict(),
        )
    else:
        # 使用 DeepSpeed 内置优化器和调度器
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                output_image_size=config.image_size,
                output_point_nums=config.num_points,
                precision=config.precision,
            ),
            config=config.deepspeed.to_dict(),
        )

    # resume deepspeed checkpoint
    if config.auto_resume and len(config.resume) == 0:
        resume = os.path.join(config.log_dir, "ckpt_model")
        if os.path.exists(resume):
            config.resume = resume

    if config.resume:
        load_path, client_state = model_engine.load_checkpoint(config.resume)
        with open(os.path.join(config.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        config.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // config.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                config.resume, config.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        # 移除 batch_size=1 的限制，支持任意 batch_size
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True,  # 优化：启用 pinned memory 加速 CPU->GPU 传输
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                output_image_size=config.image_size,
                output_point_nums=config.num_points,
                precision=config.precision,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if config.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, config)
        exit()

    if config.steps_per_epoch is None:
        config.steps_per_epoch = config.samples_per_epoch // config.batch_size

    for epoch in range(config.start_epoch, config.epochs):
        # 在每个 epoch 开始时重新生成随机采样索引
        train_dataset.set_epoch(epoch)
        
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            config,
        )

        if config.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, config)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if config.no_eval or is_best:
            save_dir = os.path.join(config.log_dir, "ckpt_model")
            
            # 方案1：保存完整的 DeepSpeed checkpoint（包含优化器状态，用于断点续训）
            if config.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            
            # 方案2：额外保存轻量级的仅模型权重 checkpoint（用于推理）
            if config.local_rank == 0:
                # 只保存可训练参数
                trainable_state_dict = {
                    name: param.cpu() 
                    for name, param in model_engine.module.named_parameters() 
                    if param.requires_grad
                }
                
                lightweight_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": trainable_state_dict,
                    "best_giou": best_score,
                    "best_ciou": cur_ciou,
                }
                
                lightweight_path = os.path.join(
                    config.log_dir,
                    "lightweight_giou{:.3f}_ciou{:.3f}.pth".format(best_score, cur_ciou)
                )
                torch.save(lightweight_ckpt, lightweight_path)
                print(f"Saved lightweight checkpoint to {lightweight_path}")
                
                # 删除旧的轻量级 checkpoint
                for old_ckpt in os.listdir(config.log_dir):
                    if old_ckpt.startswith("lightweight_") and old_ckpt != os.path.basename(lightweight_path):
                        os.remove(os.path.join(config.log_dir, old_ckpt))


def train(
    train_loader,
    model_engine,
    epoch,
    scheduler,
    writer,
    train_iter,
    config,
):
    """Main training loop."""
    global params_to_train

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    
    # 使用 MetricsTracker 管理所有指标
    metrics_tracker = MetricsTracker()
    tb_logger = TensorBoardLogger(writer) if writer is not None else None
    
    # 获取 loss meters 用于进度显示
    loss_meters = metrics_tracker.loss_meters

    progress = ProgressMeter(
        config.steps_per_epoch,
        [
            batch_time,
            loss_meters["loss"],
            loss_meters["ce_loss"],
            loss_meters["mask_loss"],
            loss_meters["mask_bce_loss"],
            loss_meters["mask_dice_loss"],
            loss_meters["mask_3d_loss"],
            loss_meters["mask_3d_bce_loss"],
            loss_meters["mask_3d_dice_loss"],
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    if config.local_rank == 0:
        step_iter = tqdm(
            range(config.steps_per_epoch),
            total=config.steps_per_epoch,
            dynamic_ncols=True,
            desc=f"Epoch {epoch+1}/{config.epochs}",
        )
    else:
        step_iter = range(config.steps_per_epoch)

    for local_step in step_iter:
        # 计算全局步数
        global_step = epoch * config.steps_per_epoch + local_step
        total_loss = 0.0  # 累积总损失，用于归一化
        # 梯度累积内层循环：仅累积梯度，不更新参数
        for i in range(config.grad_accumulation_steps):
            accum_step += 1
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
                # 重新计时，避免迭代器重置导致data_time失真
                end = time.time()

            input_dict = dict_to_cuda(input_dict, device=model_engine.device)
            data_time.update(time.time() - end)

            # 调用魔改后的 LISA 模型
            output_dict = model_engine(**input_dict)

            # ========== 在训练脚本中计算损失 ==========
            if not input_dict.get("inference", False):
                # 1. 计算语言模型损失（交叉熵）
                ce_loss = output_dict["output"].loss * model_engine.module.ce_loss_weight
                
                # 2. 计算 2D 掩码损失
                mask_bce_loss, mask_dice_loss, mask_loss = calc.img_loss(
                    pred_masks=output_dict["pred_masks"],
                    gt_masks=input_dict["img_gt_tensor"],
                    bce_loss_weight=model_engine.module.bce_loss_weight,
                    dice_loss_weight=model_engine.module.dice_loss_weight,
                ) if output_dict.get("has_valid_image") and output_dict.get("pred_masks") is not None else (
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device)
                )
                
                # 3. 计算 3D 点云掩码损失
                mask_3d_bce_loss, mask_3d_dice_loss, mask_3d_loss = calc.pc_loss(
                    pred_3d_masks=output_dict["pred_3d_masks"],
                    gt_3d_masks=input_dict["pc_gt_tensor"],
                    bce_loss_weight=model_engine.module.bce_loss_weight,
                    dice_loss_weight=model_engine.module.dice_loss_weight,
                ) if output_dict.get("has_valid_point_cloud") and output_dict.get("pred_3d_masks") is not None else (
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device)
                )
                
                # 4. 计算虚拟损失以保持所有参数连接到计算图
                dummy_loss = calc.dummy_loss(model_engine.module.model)
                
                # 5. 总损失
                loss = ce_loss + mask_loss + mask_3d_loss + dummy_loss
                

                loss = loss / config.grad_accumulation_steps

                # 构建损失字典
                output_dict.update({
                    "loss": loss, "ce_loss": ce_loss,
                    "mask_bce_loss": mask_bce_loss, "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,
                    "mask_3d_bce_loss": mask_3d_bce_loss, "mask_3d_dice_loss": mask_3d_dice_loss, "mask_3d_loss": mask_3d_loss
                })
            else:
                # 推理模式，损失置0
                output_dict["loss"] = torch.tensor(0.0, device=next(model_engine.parameters()).device, requires_grad=False)

            # 分布式先同步损失，再更新指标
            if config.distributed:
                # 所有损失张量all_reduce，保证多卡损失一致
                for k in ["loss", "ce_loss", "mask_loss", "mask_3d_loss"]:
                    if k in output_dict:
                        torch.distributed.all_reduce(output_dict[k], op=torch.distributed.ReduceOp.SUM)
                        output_dict[k] = output_dict[k] / config.world_size  # 求平均

            # 计算全局batch_size（分布式：单卡batch_size * 卡数）
            global_bs = config.batch_size * (config.world_size if config.distributed else 1)
            # 更新指标（传入全局batch_size，保证指标计算准确）
            metrics_tracker.update_loss_metrics(output_dict, global_bs)

            # 仅累积梯度，不执行step
            model_engine.backward(output_dict["loss"])
            # 重置时间，为下一个batch做准备
            end = time.time()

        # 梯度累积结束，统一更新参数 + 清空梯度（核心）
        model_engine.step()  # 更新参数
        model_engine.zero_grad()  # 清空梯度，避免污染下一轮

        # 测量整个累积批次的耗时
        batch_time.update(time.time() - end)
        end = time.time()

        if local_step % config.print_freq == 0:
            if config.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                metrics_tracker.all_reduce()

            if config.local_rank == 0:
                progress.display(local_step + 1)
                if hasattr(step_iter, "set_postfix"):
                    loss_meter = metrics_tracker.loss_meters.get("loss")
                    postfix = {"loss": f"{loss_meter.val:.4f}"} if loss_meter else {}
                    step_iter.set_postfix(postfix, refresh=False)
                
                # 使用 TensorBoardLogger 记录所有损失到 TensorBoard
                if tb_logger is not None:
                    tb_logger.log_loss_metrics(metrics_tracker, global_step, prefix="train")
                    writer.add_scalar("metrics/total_secs_per_accum_batch", batch_time.avg, global_step)
                    writer.add_scalar("metrics/data_secs_per_single_batch", data_time.avg / config.grad_accumulation_steps, global_step)

            # 重置统计量
            batch_time.reset()
            data_time.reset()
            metrics_tracker.reset()

        curr_lr = scheduler.get_last_lr()
        if config.local_rank == 0 and writer is not None:
            # 记录学习率
            if isinstance(params_to_train, list) and len(params_to_train) > 0 and isinstance(params_to_train[0], dict):
                # 分层学习率
                for idx, (lr, param_group) in enumerate(zip(curr_lr, params_to_train)):
                    group_name = param_group.get('name', f'group_{idx}')
                    writer.add_scalar(f"train/lr/{group_name}", lr, global_step)
                writer.add_scalar("train/lr/average", sum(curr_lr) / len(curr_lr), global_step)
            else:
                # 单一学习率
                writer.add_scalar("train/lr", curr_lr[0] if curr_lr else 0.0, global_step)

        # ========== 重要补充：学习率调度按有效更新次数step ==========
        scheduler.step()

    # 返回迭代器，用于续训
    return train_iter


def validate(val_loader, model_engine, epoch, writer, config):
    """
    验证函数（支持批处理）
    
    Args:
        val_loader: 验证数据加载器
        model_engine: DeepSpeed 模型引擎
        epoch: 当前 epoch
        writer: TensorBoard writer
        config: 训练配置
        
    Returns:
        giou: 2D Global IoU
        ciou: 2D Class IoU
    """
    # 使用 MetricsTracker 管理所有评估指标
    metrics_tracker = MetricsTracker()
    tb_logger = TensorBoardLogger(writer) if writer is not None else None
    
    model_engine.eval()

    for input_dict in tqdm(val_loader, desc='Validating'):
        torch.cuda.empty_cache()

        with torch.no_grad():
            input_dict = dict_to_cuda(input_dict, device=model_engine.device)
            output_dict = model_engine(**input_dict, inference=True)

        # 使用统一的评估函数（支持批处理）
        evaluate_segmentation_batch(
            input_dict,
            output_dict,
            metrics_tracker,
            mask_threshold_2d=config.mask_threshold_2d
        )

    # 分布式训练时汇总所有进程的指标
    if config.distributed:
        metrics_tracker.all_reduce()

    # 计算最终结果
    giou, ciou = metrics_tracker.compute_2d_seg_results()
    mae_3d, auc_3d, aiou_3d, sim_3d = metrics_tracker.compute_3d_seg_results()

    if config.local_rank == 0:
        # 计算当前的全局步数（用于与训练指标对齐）
        global_step = (epoch + 1) * config.steps_per_epoch
        
        # 打印验证摘要
        print_validation_summary(
            epoch, 
            metrics_tracker, 
            giou, 
            ciou, 
            mae_3d, 
            auc_3d, 
            aiou_3d, 
            sim_3d
        )
        
        # 使用 TensorBoardLogger 记录指标到 TensorBoard
        if tb_logger is not None:
            tb_logger.log_seg_2d_metrics(giou, ciou, global_step, prefix="val")
            tb_logger.log_seg_3d_metrics(mae_3d, auc_3d, aiou_3d, sim_3d, global_step, prefix="val")

    return giou, ciou


if __name__ == "__main__":
    main()
