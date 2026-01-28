import os
import shutil
import time
from functools import partial

import deepspeed
import torch
from tqdm import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, collate_fn
from utils.common import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                          ProgressMeter)
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
    if config.local_rank == 0:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(config.log_dir)
    else:
        writer = None

    # Create model - 使用魔改后的 LISAForCausalLM 模型（已支持点云）
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.version,
        cache_dir=None,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 添加特殊标记
    tokenizer.add_tokens("[SEG]")  # 2D分割标记
    tokenizer.add_tokens("[AFF]")  # 3D affordance标记
    config.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    config.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if config.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # 模型参数配置
    model_args = {
        "train_mask_decoder": config.train_mask_decoder,
        "out_dim": config.out_dim,
        "ce_loss_weight": config.ce_loss_weight,
        "dice_loss_weight": config.dice_loss_weight,
        "bce_loss_weight": config.bce_loss_weight,
        "seg_token_idx": config.seg_token_idx,
        "aff_token_idx": config.aff_token_idx,  # 3D affordance token
        "vision_pretrained": config.vision_pretrained,
        "vision_tower": config.vision_tower,
        "use_mm_start_end": config.use_mm_start_end,
    }
    
    torch_dtype = torch.float32
    if config.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif config.precision == "fp16":
        torch_dtype = torch.half
    
    # 初始化魔改后的 LISA 模型
    model = LISAForCausalLM.from_pretrained(
        config.version, dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=config.local_rank)
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        config.conv_type
    ]

    model.resize_token_embeddings(len(tokenizer))
    
    # 初始化 LISA 模块（包括 SAM 和 3D 点云分割器）
    if not config.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    # # 冻结视觉编码器和投影层
    # for p in vision_tower.parameters():
    #     p.requires_grad = False
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = False


    # 先把所有参数冻结 (作为基底)
    for p in model.parameters():
        p.requires_grad = False

    # 开启非 LoRA 的关键模块 (Projector, Mask Decoder 等)
    for n, p in model.named_parameters():
        if any(t in n for t in config.name_of_params_to_train):
            p.requires_grad = True

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
        vision_tower=config.vision_tower,
        samples_per_epoch=config.batch_size
        * config.grad_accumulation_steps
        * config.steps_per_epoch
        * world_size,
        precision=config.precision,
        image_size=config.image_size,
        num_points=config.num_points,
        num_classes_per_sample=config.num_classes_per_sample,
        exclude_val=config.exclude_val,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        use_mm_start_end=config.use_mm_start_end,  # 传递给 Dataset 用于预处理
    )
    
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
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=params_to_train,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=config.conv_type,
            use_mm_start_end=config.use_mm_start_end,
            local_rank=config.local_rank,
        ),
        config=config.get_deepspeed_config(),
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
                conv_type=config.conv_type,
                use_mm_start_end=config.use_mm_start_end,
                local_rank=config.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if config.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, config)
        exit()

    for epoch in range(config.start_epoch, config.epochs):
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
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    for local_step in range(config.steps_per_epoch):
        # 计算全局步数
        global_step = epoch * config.steps_per_epoch + local_step
        for i in range(config.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)

            # 调用魔改后的 LISA 模型
            output_dict = model_engine(**input_dict)

            # ========== 在训练脚本中计算损失 ==========
            if not output_dict.get("inference", False):
                # 1. 计算语言模型损失（交叉熵）
                ce_loss = output_dict["output"].loss * model_engine.module.ce_loss_weight
                
                # 2. 计算 2D 掩码损失
                mask_bce_loss, mask_dice_loss, mask_loss = calc.img_loss(
                    pred_masks=output_dict["pred_masks"] if output_dict["has_image"] else torch.empty(0),
                    gt_masks=output_dict["gt_masks"] if output_dict["has_image"] else torch.empty(0),
                    bce_loss_weight=model_engine.module.bce_loss_weight,
                    dice_loss_weight=model_engine.module.dice_loss_weight,
                ) if output_dict.get("has_image") and output_dict.get("pred_masks") is not None else (
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device)
                )
                
                # 3. 计算 3D 点云掩码损失
                mask_3d_bce_loss, mask_3d_dice_loss, mask_3d_loss = calc.pc_loss(
                    pred_3d_masks=output_dict["pred_3d_masks"] if output_dict["has_point_cloud"] else torch.empty(0),
                    gt_3d_masks=output_dict["gt_3d_masks"] if output_dict["has_point_cloud"] else torch.empty(0),
                    bce_loss_weight=model_engine.module.bce_loss_weight,
                    dice_loss_weight=model_engine.module.dice_loss_weight,
                ) if output_dict.get("has_point_cloud") and output_dict.get("pred_3d_masks") is not None else (
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device),
                    torch.tensor(0.0, device=ce_loss.device)
                )
                
                # 4. 计算虚拟损失以保持所有参数连接到计算图
                dummy_loss = calc.dummy_loss(model_engine.module.model)
                
                # 5. 总损失
                loss = ce_loss + mask_loss + mask_3d_loss + dummy_loss
                
                # 构建损失字典
                output_dict["loss"] = loss
                output_dict["ce_loss"] = ce_loss
                output_dict["mask_bce_loss"] = mask_bce_loss
                output_dict["mask_dice_loss"] = mask_dice_loss
                output_dict["mask_loss"] = mask_loss
                output_dict["mask_3d_bce_loss"] = mask_3d_bce_loss
                output_dict["mask_3d_dice_loss"] = mask_3d_dice_loss
                output_dict["mask_3d_loss"] = mask_3d_loss

            # 使用 MetricsTracker 更新所有损失
            metrics_tracker.update_loss_metrics(output_dict, config.batch_size)
            
            model_engine.backward(output_dict["loss"])
            model_engine.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if local_step % config.print_freq == 0:
            if config.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                metrics_tracker.all_reduce()

            if config.local_rank == 0:
                progress.display(local_step + 1)
                
                # 使用 TensorBoardLogger 记录所有损失到 TensorBoard
                if tb_logger is not None:
                    tb_logger.log_loss_metrics(metrics_tracker, global_step, prefix="train")
                    writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                    writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            batch_time.reset()
            data_time.reset()
            metrics_tracker.reset()

        if local_step != 0:
            curr_lr = scheduler.get_last_lr()
            if config.local_rank == 0 and writer is not None:
                # 记录学习率
                if isinstance(params_to_train, list) and len(params_to_train) > 0 and isinstance(params_to_train[0], dict):
                    # 分层学习率模式：记录每个参数组的学习率
                    for idx, (lr, param_group) in enumerate(zip(curr_lr, params_to_train)):
                        group_name = param_group.get('name', f'group_{idx}')
                        writer.add_scalar(f"train/lr/{group_name}", lr, global_step)
                    # 同时记录平均学习率
                    writer.add_scalar("train/lr/average", sum(curr_lr) / len(curr_lr), global_step)
                else:
                    # 单一学习率模式
                    writer.add_scalar("train/lr", curr_lr[0], global_step)

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
            output_dict = model_engine(**input_dict, inference=True)

        # 使用统一的评估函数（支持批处理）
        evaluate_segmentation_batch(
            output_dict,
            metrics_tracker,
            mask_threshold_2d=config.mask_threshold_2d,
            mask_threshold_3d=config.mask_threshold_3d
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
