import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.joint_affordance import JointAff
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


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
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model - 使用新的 JointAff 模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 添加特殊标记
    num_added_tokens = tokenizer.add_tokens("[SEG]")  # 2D分割标记
    tokenizer.add_tokens("[AFF]")  # 3D affordance标记
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # 初始化 JointAff 模型
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    model = JointAff(
        pretrained_weight=None,
        use_image=True,
        use_pointcloud=args.use_pointcloud,
    )
    model = model.to(dtype=torch_dtype, device=args.local_rank)
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    # 设置可训练参数
    print("Setting trainable parameters for JointAff model...")
    for n, p in model.named_parameters():
        print(f"Parameter: {n}, shape: {p.shape}, requires_grad: {p.requires_grad}")

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
        model_parameters=model.parameters(),
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
        assert args.val_batch_size == 1
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
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    loss_2d_meter = AverageMeter("Loss2D", ":.4f")
    loss_3d_meter = AverageMeter("Loss3D", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            loss_2d_meter,
            loss_3d_meter,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
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

            # 准备模型输入
            batch_size = input_dict["input_ids"].size(0)
            
            # 处理图像数据
            images = input_dict.get("images")
            if images is not None:
                if args.precision == "fp16":
                    images = images.half()
                elif args.precision == "bf16":
                    images = images.bfloat16()
                else:
                    images = images.float()
            
            # 处理点云数据
            point_clouds = input_dict.get("point_clouds")
            if point_clouds is not None:
                if args.precision == "fp16":
                    point_clouds = point_clouds.half()
                elif args.precision == "bf16":
                    point_clouds = point_clouds.bfloat16()
                else:
                    point_clouds = point_clouds.float()
                # 转换为 [B, C, N] 格式（如果是 [B, N, C]）
                if point_clouds.dim() == 3 and point_clouds.size(-1) == 3:
                    point_clouds = point_clouds.permute(0, 2, 1).contiguous()
            
            # 准备文本输入（使用 input_ids 作为文本表示）
            text_input = input_dict["input_ids"]
            
            # 准备 GT
            gt_aff2d = None
            if "masks_list" in input_dict and len(input_dict["masks_list"]) > 0:
                # 将 masks_list 转换为批次张量
                gt_aff2d = torch.stack([m[0] if m.dim() == 3 else m for m in input_dict["masks_list"]])
                gt_aff2d = gt_aff2d.unsqueeze(1) if gt_aff2d.dim() == 3 else gt_aff2d
            
            gt_aff3d = None
            if "point_masks_list" in input_dict and input_dict["point_masks_list"] is not None:
                gt_aff3d = torch.stack(input_dict["point_masks_list"])
            
            # 获取有效性掩码
            img_mask = input_dict.get("image_valid_mask")
            pc_mask = input_dict.get("pc_valid_mask")
            
            # 前向传播
            output_dict = model(
                text=text_input,
                img=images,
                points=point_clouds,
                gt_aff2d=gt_aff2d,
                gt_aff3d=gt_aff3d,
                img_mask=img_mask,
                pc_mask=pc_mask,
                return_loss=True,
            )

            loss = output_dict.get("loss", torch.tensor(0.0, device=args.local_rank))
            loss_2d = output_dict.get("loss_2d", torch.tensor(0.0, device=args.local_rank))
            loss_3d = output_dict.get("loss_3d", torch.tensor(0.0, device=args.local_rank))

            losses.update(loss.item(), batch_size)
            if loss_2d is not None and loss_2d.item() > 0:
                loss_2d_meter.update(loss_2d.item(), batch_size)
            if loss_3d is not None and loss_3d.item() > 0:
                loss_3d_meter.update(loss_3d.item(), batch_size)
            
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                loss_2d_meter.all_reduce()
                loss_3d_meter.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/loss_2d", loss_2d_meter.avg, global_step)
                writer.add_scalar("train/loss_3d", loss_3d_meter.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            loss_2d_meter.reset()
            loss_3d_meter.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter_2d = AverageMeter("Intersec2D", ":6.3f", Summary.SUM)
    union_meter_2d = AverageMeter("Union2D", ":6.3f", Summary.SUM)
    acc_iou_meter_2d = AverageMeter("gIoU2D", ":6.3f", Summary.SUM)
    
    intersection_meter_3d = AverageMeter("Intersec3D", ":6.3f", Summary.SUM)
    union_meter_3d = AverageMeter("Union3D", ":6.3f", Summary.SUM)
    acc_iou_meter_3d = AverageMeter("gIoU3D", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        
        # 处理图像数据
        images = input_dict.get("images")
        if images is not None:
            if args.precision == "fp16":
                images = images.half()
            elif args.precision == "bf16":
                images = images.bfloat16()
            else:
                images = images.float()
        
        # 处理点云数据
        point_clouds = input_dict.get("point_clouds")
        if point_clouds is not None:
            if args.precision == "fp16":
                point_clouds = point_clouds.half()
            elif args.precision == "bf16":
                point_clouds = point_clouds.bfloat16()
            else:
                point_clouds = point_clouds.float()
            # 转换为 [B, C, N] 格式
            if point_clouds.dim() == 3 and point_clouds.size(-1) == 3:
                point_clouds = point_clouds.permute(0, 2, 1).contiguous()
        
        text_input = input_dict["input_ids"]

        with torch.no_grad():
            output_dict = model_engine(
                text=text_input,
                img=images,
                points=point_clouds,
                return_loss=False,
            )

        # 评估 2D 预测
        if "aff2d" in output_dict and "masks_list" in input_dict and len(input_dict["masks_list"]) > 0:
            pred_masks_2d = output_dict["aff2d"]
            gt_masks_2d = input_dict["masks_list"][0].int()
            
            # 确保预测和GT形状匹配
            if pred_masks_2d.dim() == 4:  # [B, 1, H, W]
                pred_masks_2d = pred_masks_2d[0, 0]  # [H, W]
            elif pred_masks_2d.dim() == 3:  # [B, H, W]
                pred_masks_2d = pred_masks_2d[0]  # [H, W]
            
            output_2d = (pred_masks_2d > 0.5).int()
            
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i in gt_masks_2d:
                if mask_i.dim() == 3:
                    mask_i = mask_i[0]
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_2d.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0
            
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / max(len(gt_masks_2d), 1)
            intersection_meter_2d.update(intersection)
            union_meter_2d.update(union)
            acc_iou_meter_2d.update(acc_iou, n=len(gt_masks_2d))
        
        # 评估 3D 预测
        if "aff3d" in output_dict and "point_masks_list" in input_dict and input_dict["point_masks_list"] is not None:
            pred_masks_3d = output_dict["aff3d"]  # [B, N, 1]
            gt_masks_3d = input_dict["point_masks_list"][0]  # [N] or [N, 1]
            
            if pred_masks_3d.dim() == 3:
                pred_masks_3d = pred_masks_3d[0].squeeze(-1)  # [N]
            
            if gt_masks_3d.dim() == 2:
                gt_masks_3d = gt_masks_3d.squeeze(-1)  # [N]
            
            output_3d = (pred_masks_3d > 0.5).int()
            gt_3d = gt_masks_3d.int()
            
            # 计算 3D IoU
            intersection_3d = (output_3d & gt_3d).sum().float()
            union_3d = (output_3d | gt_3d).sum().float()
            iou_3d = intersection_3d / (union_3d + 1e-5)
            
            intersection_meter_3d.update(intersection_3d.cpu().numpy())
            union_meter_3d.update(union_3d.cpu().numpy())
            acc_iou_meter_3d.update(iou_3d.cpu().numpy(), n=1)

    # 汇总 2D 指标
    intersection_meter_2d.all_reduce()
    union_meter_2d.all_reduce()
    acc_iou_meter_2d.all_reduce()
    
    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    ciou_2d = iou_class_2d[1] if len(iou_class_2d) > 1 else 0.0
    giou_2d = acc_iou_meter_2d.avg[1] if len(acc_iou_meter_2d.avg) > 1 else 0.0
    
    # 汇总 3D 指标
    intersection_meter_3d.all_reduce()
    union_meter_3d.all_reduce()
    acc_iou_meter_3d.all_reduce()
    
    ciou_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    giou_3d = acc_iou_meter_3d.avg

    if args.local_rank == 0:
        writer.add_scalar("val/giou_2d", giou_2d, epoch)
        writer.add_scalar("val/ciou_2d", ciou_2d, epoch)
        writer.add_scalar("val/giou_3d", giou_3d, epoch)
        writer.add_scalar("val/ciou_3d", ciou_3d, epoch)
        print("2D - giou: {:.4f}, ciou: {:.4f}".format(giou_2d, ciou_2d))
        print("3D - giou: {:.4f}, ciou: {:.4f}".format(giou_3d, ciou_3d))

    # 返回综合指标（2D和3D的平均）
    avg_giou = (giou_2d + giou_3d) / 2.0
    avg_ciou = (ciou_2d + ciou_3d) / 2.0
    
    return avg_giou, avg_ciou


if __name__ == "__main__":
    main(sys.argv[1:])
