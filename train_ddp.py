import argparse
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, intersectionAndUnion3D)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # 点云相关参数
    parser.add_argument("--num_points", default=2048, type=int, help="点云采样点数")
    parser.add_argument("--use_point_cloud", action="store_true", default=True, help="是否使用点云模态")
    parser.add_argument("--pc_loss_weight", default=1.0, type=float, help="点云分割损失权重")
    return parser.parse_args(args)


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def main(args):
    args = parse_args(args)
    
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size
    args.distributed = world_size > 1
    
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    # 添加 3D affordance token
    num_added_tokens = tokenizer.add_tokens("[AFF]")
    args.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "aff_token_idx": args.aff_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    # 使用 DatasetManager 统一管理数据集（只创建一次 JointDataset）
    dataset_manager = DatasetManager(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
        num_points=args.num_points,
    )
    
    # 获取训练数据集
    train_dataset = dataset_manager.get_train_dataset()

    if args.no_eval == False:
        # 获取验证数据集（共享同一个 JointDataset）
        val_dataset = dataset_manager.get_val_dataset()
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    # 将模型移动到 GPU 并包装为 DDP
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(args.beta1, args.beta2),
    )
    
    # 创建学习率调度器 (带 warmup 的余弦退火)
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = 100
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度训练 (V100 不支持 bf16，使用 fp16)
    use_amp = args.precision in ["fp16", "bf16"]
    # V100 只支持 fp16，不支持 bf16
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.float16  # V100 强制使用 fp16
    scaler = GradScaler(enabled=use_amp)
    
    if rank == 0:
        print(f"使用混合精度训练: {use_amp}, dtype: {amp_dtype}")
    
    # 创建训练数据加载器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 使用 DistributedSampler 时设为 False
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=local_rank,
        ),
        drop_last=True,
    )

    # 恢复检查点
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model.pth")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"从检查点恢复: {args.resume}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume, map_location=map_location)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model, 0, writer, args)
        cleanup_distributed()
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # 设置 epoch 以确保每个 epoch 的数据打乱不同
        train_sampler.set_epoch(epoch)
        if val_dataset is not None:
            val_sampler.set_epoch(epoch)
        
        # train for one epoch
        train_iter = train(
            train_loader,
            model,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
            optimizer,
            scaler,
            use_amp,
            amp_dtype,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model.pth")
            if rank == 0:
                # 保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'best_score': best_score,
                    'cur_ciou': cur_ciou,
                }
                torch.save(checkpoint, save_dir)
                torch.save(
                    {"epoch": epoch, "best_giou": best_score, "cur_ciou": cur_ciou},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                print(f"模型已保存到 {save_dir}")
            dist.barrier()
    
    cleanup_distributed()


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    optimizer,
    scaler,
    use_amp,
    amp_dtype,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    # 点云损失记录
    mask_3d_bce_losses = AverageMeter("Mask3DBCELoss", ":.4f")
    mask_3d_dice_losses = AverageMeter("Mask3DDICELoss", ":.4f")
    mask_3d_losses = AverageMeter("Mask3DLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            mask_3d_losses,
            mask_3d_bce_losses,
            mask_3d_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    
    # 梯度累积计数器
    optimizer.zero_grad()
    
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            # V100 使用 fp16
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                # V100 不支持 bf16，转换为 fp16
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            
            # 处理点云数据精度
            if "point_clouds" in input_dict and input_dict["point_clouds"] is not None:
                if args.precision in ["fp16", "bf16"]:
                    input_dict["point_clouds"] = input_dict["point_clouds"].half()
                else:
                    input_dict["point_clouds"] = input_dict["point_clouds"].float()

            # 使用混合精度训练
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                output_dict = model(**input_dict)
                loss = output_dict["loss"]
                # 梯度累积时需要对 loss 进行缩放
                loss = loss / args.grad_accumulation_steps

            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            
            # 获取点云损失
            mask_3d_bce_loss = output_dict.get("mask_3d_bce_loss", torch.tensor(0.0))
            mask_3d_dice_loss = output_dict.get("mask_3d_dice_loss", torch.tensor(0.0))
            mask_3d_loss = output_dict.get("mask_3d_loss", torch.tensor(0.0))

            batch_size = input_dict["images"].size(0)
            losses.update(loss.item() * args.grad_accumulation_steps, batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            mask_bce_losses.update(mask_bce_loss.item(), batch_size)
            mask_dice_losses.update(mask_dice_loss.item(), batch_size)
            mask_losses.update(mask_loss.item(), batch_size)
            # 更新点云损失
            mask_3d_bce_losses.update(mask_3d_bce_loss.item() if isinstance(mask_3d_bce_loss, torch.Tensor) else mask_3d_bce_loss, batch_size)
            mask_3d_dice_losses.update(mask_3d_dice_loss.item() if isinstance(mask_3d_dice_loss, torch.Tensor) else mask_3d_dice_loss, batch_size)
            mask_3d_losses.update(mask_3d_loss.item() if isinstance(mask_3d_loss, torch.Tensor) else mask_3d_loss, batch_size)
            
            # 反向传播
            scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 更新学习率
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                mask_3d_bce_losses.all_reduce()
                mask_3d_dice_losses.all_reduce()
                mask_3d_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                # 记录点云损失
                writer.add_scalar(
                    "train/mask_3d_bce_loss", mask_3d_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_3d_dice_loss", mask_3d_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_3d_loss", mask_3d_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            mask_3d_bce_losses.reset()
            mask_3d_dice_losses.reset()
            mask_3d_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    # 点云评估指标
    intersection_3d_meter = AverageMeter("Intersec3D", ":6.3f", Summary.SUM)
    union_3d_meter = AverageMeter("Union3D", ":6.3f", Summary.SUM)
    acc_iou_3d_meter = AverageMeter("gIoU3D", ":6.3f", Summary.SUM)

    model.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        # V100 使用 fp16
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            # V100 不支持 bf16，转换为 fp16
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        # 处理点云数据精度
        if "point_clouds" in input_dict and input_dict["point_clouds"] is not None:
            if args.precision in ["fp16", "bf16"]:
                input_dict["point_clouds"] = input_dict["point_clouds"].half()
            else:
                input_dict["point_clouds"] = input_dict["point_clouds"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        # 2D 图像分割评估
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        
        # 3D 点云分割评估
        pred_3d_masks = output_dict.get("pred_3d_masks", None)
        gt_3d_masks = output_dict.get("gt_3d_masks", None)
        
        if pred_3d_masks is not None and gt_3d_masks is not None:
            intersection_3d, union_3d, acc_iou_3d = 0.0, 0.0, 0.0
            num_3d_masks = 0
            for pred_3d, gt_3d in zip(pred_3d_masks, gt_3d_masks):
                if gt_3d is not None:
                    # 二值化预测
                    pred_binary = (pred_3d > 0.5).float()
                    gt_binary = gt_3d.float()
                    
                    # 计算交集和并集
                    inter = (pred_binary * gt_binary).sum()
                    uni = pred_binary.sum() + gt_binary.sum() - inter
                    
                    intersection_3d += inter.cpu().numpy()
                    union_3d += uni.cpu().numpy()
                    
                    # 计算 IoU
                    iou = inter / (uni + 1e-5)
                    acc_iou_3d += iou.cpu().numpy()
                    num_3d_masks += 1
            
            if num_3d_masks > 0:
                intersection_3d_meter.update(intersection_3d)
                union_3d_meter.update(union_3d)
                acc_iou_3d_meter.update(acc_iou_3d / num_3d_masks, n=num_3d_masks)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    
    intersection_3d_meter.all_reduce()
    union_3d_meter.all_reduce()
    acc_iou_3d_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    
    # 3D 点云 IoU
    ciou_3d = intersection_3d_meter.sum / (union_3d_meter.sum + 1e-10) if union_3d_meter.sum > 0 else 0.0
    giou_3d = acc_iou_3d_meter.avg if acc_iou_3d_meter.count > 0 else 0.0

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/giou_3d", giou_3d, epoch)
        writer.add_scalar("val/ciou_3d", ciou_3d, epoch)
        print("2D - giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        print("3D - giou: {:.4f}, ciou: {:.4f}".format(giou_3d, ciou_3d))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
