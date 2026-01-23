import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
from tqdm import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, debug_gradient_graph)


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
        default="llava_llama_2",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # 点云相关参数
    parser.add_argument("--num_points", default=2048, type=int, help="number of points in point cloud")
    parser.add_argument("--use_pointcloud", action="store_true", default=True, help="use point cloud modality")
    parser.add_argument("--pc_loss_weight", default=1.0, type=float, help="weight for point cloud loss")
    parser.add_argument("--train_ratio", default=0.7, type=float, help="training set ratio")
    parser.add_argument("--val_ratio", default=0.15, type=float, help="validation set ratio")
    parser.add_argument("--test_ratio", default=0.15, type=float, help="test set ratio")
    parser.add_argument("--mask_threshold_2d", default=0.0, type=float, help="threshold for 2D mask binarization")
    parser.add_argument("--mask_threshold_3d", default=0.5, type=float, help="threshold for 3D mask binarization")
    return parser.parse_args(args)

params_to_train = []


def preprocess_input_data(input_dict, precision):
    """
    预处理输入数据（图像和点云），支持动态模态输入
    
    Args:
        input_dict: 输入字典，包含 images, images_clip, point_clouds 等
        precision: 精度类型 ("fp16", "bf16", "fp32")
    
    Returns:
        batch_size: 计算出的 batch size
    
    Note:
        该函数会直接修改 input_dict，添加 batch_size 字段
    """
    # 处理图像数据
    if "images" in input_dict and input_dict["images"] is not None:
        if precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

    # 处理点云数据
    if "point_clouds" in input_dict and input_dict["point_clouds"] is not None:
        point_clouds = input_dict["point_clouds"]
        if precision == "fp16":
            point_clouds = point_clouds.half()
        elif precision == "bf16":
            point_clouds = point_clouds.bfloat16()
        else:
            point_clouds = point_clouds.float()
        # 转换为 [B, C, N] 格式（如果是 [B, N, C]）
        if point_clouds.dim() == 3 and point_clouds.size(-1) == 3:
            point_clouds = point_clouds.permute(0, 2, 1).contiguous()
        input_dict["point_clouds"] = point_clouds

    # 计算实际的 batch size（支持动态模态输入）
    if "images" in input_dict and input_dict["images"] is not None:
        batch_size = input_dict["images"].size(0)
    elif "point_clouds" in input_dict and input_dict["point_clouds"] is not None:
        batch_size = input_dict["point_clouds"].size(0)
    else:
        batch_size = 1  # 默认值
    
    # 将 batch_size 写入 input_dict 传入模型
    input_dict["batch_size"] = batch_size
    
    return batch_size


def main(args):
    global params_to_train
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model - 使用魔改后的 LISAForCausalLM 模型（已支持点云）
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 添加特殊标记
    tokenizer.add_tokens("[SEG]")  # 2D分割标记
    tokenizer.add_tokens("[AFF]")  # 3D affordance标记
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # 模型参数配置
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "aff_token_idx": args.aff_token_idx,  # 3D affordance token
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # 初始化魔改后的 LISA 模型
    model = LISAForCausalLM.from_pretrained(
        args.version, dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    model.resize_token_embeddings(len(tokenizer))
    
    # 初始化 LISA 模块（包括 SAM 和 3D 点云分割器）
    if not args.eval_only:
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
    target_modules = ["mask_decoder", "text_hidden_fcs", "mm_projector", "lm_head", "embed_tokens", "point_cloud_segmentor"]
    for n, p in model.named_parameters():
        if any(t in n for t in target_modules):
            p.requires_grad = True

    # LoRA 配置（可选）
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
                                "point_cloud_segmentor",
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

    
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    # 打印检查
    print(f"\n最终训练参数统计:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {len(params_to_train)} tensors, {sum(p.numel() for p in params_to_train)} elements")
    
    if len(params_to_train) == 0:
        raise ValueError("严重错误：没有发现可训练参数！请检查冻结逻辑。")
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    # 使用新的 DatasetManager 创建数据集
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
        num_points=args.num_points,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    train_dataset = dataset_manager.get_train_dataset()

    if args.no_eval == False:
        val_dataset = dataset_manager.get_val_dataset()
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=params_to_train,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
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
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            
            # 方案1：保存完整的 DeepSpeed checkpoint（包含优化器状态，用于断点续训）
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            
            # 方案2：额外保存轻量级的仅模型权重 checkpoint（用于推理）
            if args.local_rank == 0:
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
                    args.log_dir,
                    "lightweight_giou{:.3f}_ciou{:.3f}.pth".format(best_score, cur_ciou)
                )
                torch.save(lightweight_ckpt, lightweight_path)
                print(f"Saved lightweight checkpoint to {lightweight_path}")
                
                # 删除旧的轻量级 checkpoint
                for old_ckpt in os.listdir(args.log_dir):
                    if old_ckpt.startswith("lightweight_") and old_ckpt != os.path.basename(lightweight_path):
                        os.remove(os.path.join(args.log_dir, old_ckpt))


def train(
    train_loader,
    model_engine,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    global params_to_train

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
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
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            
            # 预处理输入数据（图像和点云）并计算 batch_size
            batch_size = preprocess_input_data(input_dict, args.precision)

            # 调用魔改后的 LISA 模型
            output_dict = model_engine(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_3d_bce_loss = output_dict.get("mask_3d_bce_loss", torch.tensor(0.0))
            mask_3d_dice_loss = output_dict.get("mask_3d_dice_loss", torch.tensor(0.0))
            mask_3d_loss = output_dict.get("mask_3d_loss", torch.tensor(0.0))

            losses.update(loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            mask_bce_losses.update(mask_bce_loss.item(), batch_size)
            mask_dice_losses.update(mask_dice_loss.item(), batch_size)
            mask_losses.update(mask_loss.item(), batch_size)
            mask_3d_bce_losses.update(mask_3d_bce_loss.item(), batch_size)
            mask_3d_dice_losses.update(mask_3d_dice_loss.item(), batch_size)
            mask_3d_losses.update(mask_3d_loss.item(), batch_size)
            
            model_engine.backward(loss)
            model_engine.step()

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
                writer.add_scalar("train/mask_3d_bce_loss", mask_3d_bce_losses.avg, global_step)
                writer.add_scalar("train/mask_3d_dice_loss", mask_3d_dice_losses.avg, global_step)
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


def validate(val_loader, model_engine, epoch, writer, args):
    # 2D 分割评估指标
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.AVERAGE)
    
    # 3D 点云评估指标
    intersection_meter_3d = AverageMeter("Intersec3D", ":6.3f", Summary.SUM)
    union_meter_3d = AverageMeter("Union3D", ":6.3f", Summary.SUM)
    acc_iou_meter_3d = AverageMeter("gIoU3D", ":6.3f", Summary.AVERAGE)
    
    # 调试计数器
    num_2d_samples = 0
    num_3d_samples = 0

    model_engine.eval()

    for input_dict in tqdm(val_loader, desc='Validating'):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        
        # 预处理输入数据（图像和点云）并计算 batch_size
        batch_size = preprocess_input_data(input_dict, args.precision)

        with torch.no_grad():
            output_dict = model_engine(**input_dict, inference=True)

        # 评估 2D 分割（支持任意 batch_size）
        if "pred_masks" in output_dict and output_dict["pred_masks"] is not None:
            pred_masks = output_dict["pred_masks"]  # List[Tensor], 长度为 batch_size
            gt_masks = output_dict["masks"]  # List[Tensor], 长度为 batch_size
            
            # 遍历 batch 中的每个样本
            for batch_idx in range(len(pred_masks)):
                num_2d_samples += 1
                masks_list = gt_masks[batch_idx].int()  # [num_masks, H, W]
                output_list = (pred_masks[batch_idx] > args.mask_threshold_2d).int()  # [num_masks, H, W]
                
                # 确保预测和真实掩码数量一致
                assert masks_list.shape[0] == output_list.shape[0], \
                    f"Mismatch: gt has {masks_list.shape[0]} masks, pred has {output_list.shape[0]} masks"

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
                intersection_meter.update(intersection)
                union_meter.update(union)
                acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        
        # 评估 3D 点云分割（支持任意 batch_size）
        if "pred_masks_3d" in output_dict and output_dict["pred_masks_3d"] is not None:
            pred_3d_masks = output_dict["pred_masks_3d"]  # List[Tensor], 长度为 batch_size
            gt_3d_masks = output_dict["masks_3d"]  # List[Tensor], 长度为 batch_size
            
            # 遍历 batch 中的每个样本
            for batch_idx in range(len(pred_3d_masks)):
                num_3d_samples += 1
                masks_3d_list = gt_3d_masks[batch_idx].int()  # [num_masks, N]
                output_3d_list = (pred_3d_masks[batch_idx] > args.mask_threshold_3d).int()  # [num_masks, N]
                
                # 确保预测和真实掩码数量一致
                assert masks_3d_list.shape[0] == output_3d_list.shape[0], \
                    f"Mismatch: gt has {masks_3d_list.shape[0]} masks, pred has {output_3d_list.shape[0]} masks"

                intersection_3d, union_3d, acc_iou_3d = 0.0, 0.0, 0.0
                for mask_3d_i, output_3d_i in zip(masks_3d_list, output_3d_list):
                    intersection_3d_i, union_3d_i, _ = intersectionAndUnionGPU(
                        output_3d_i.contiguous().clone(), mask_3d_i.contiguous(), 2, ignore_index=255
                    )
                    intersection_3d += intersection_3d_i
                    union_3d += union_3d_i
                    acc_iou_3d += intersection_3d_i / (union_3d_i + 1e-5)
                    acc_iou_3d[union_3d_i == 0] += 1.0  # no-object target
                
                intersection_3d, union_3d = intersection_3d.cpu().numpy(), union_3d.cpu().numpy()
                acc_iou_3d = acc_iou_3d.cpu().numpy() / masks_3d_list.shape[0]
                intersection_meter_3d.update(intersection_3d)
                union_meter_3d.update(union_3d)
                acc_iou_meter_3d.update(acc_iou_3d, n=masks_3d_list.shape[0])

    # 汇总 2D 分割结果
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if isinstance(iou_class, np.ndarray) and iou_class.size > 1 else 0.0
    giou = acc_iou_meter.avg[1] if isinstance(acc_iou_meter.avg, np.ndarray) and acc_iou_meter.avg.size > 1 else 0.0
    
    # 汇总 3D 点云分割结果
    intersection_meter_3d.all_reduce()
    union_meter_3d.all_reduce()
    acc_iou_meter_3d.all_reduce()

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    if isinstance(iou_class_3d, np.ndarray) and iou_class_3d.size > 1:
        ciou_3d = iou_class_3d[1]
    else:
        ciou_3d = float(iou_class_3d) if not isinstance(iou_class_3d, np.ndarray) else iou_class_3d.item()
    
    if isinstance(acc_iou_meter_3d.avg, np.ndarray) and acc_iou_meter_3d.avg.size > 1:
        giou_3d = acc_iou_meter_3d.avg[1]
    else:
        giou_3d = float(acc_iou_meter_3d.avg) if not isinstance(acc_iou_meter_3d.avg, np.ndarray) else acc_iou_meter_3d.avg.item()

    if args.local_rank == 0:
        # 打印调试信息
        print(f"\n{'='*60}")
        print(f"Validation Summary (Epoch {epoch}):")
        print(f"  2D samples evaluated: {num_2d_samples}")
        print(f"  3D samples evaluated: {num_3d_samples}")
        print(f"{'='*60}")
        
        # 记录 2D 分割指标
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("2D Segmentation - giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        
        # 记录 3D 点云分割指标
        writer.add_scalar("val/giou_3d", giou_3d, epoch)
        writer.add_scalar("val/ciou_3d", ciou_3d, epoch)
        print("3D Point Cloud - giou: {:.4f}, ciou: {:.4f}".format(giou_3d, ciou_3d))
        
        # 警告信息
        if num_3d_samples == 0:
            print(f"\n⚠️  WARNING: No 3D samples were evaluated!")
            print(f"   Possible reasons:")
            print(f"   1. Validation dataset has no point cloud data")
            print(f"   2. Point clouds not being loaded correctly")
            print(f"   3. Model not generating pred_masks_3d")
        print(f"{'='*60}\n")

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
