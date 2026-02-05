import argparse
from functools import partial

import deepspeed
import torch
from peft import get_peft_model
from transformers import AutoProcessor

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from utils.base_dataset import JointDataset
from utils.dataset import Qwen3VLDataset, Qwen3VLTrainDataset, qwen3vl_collate_fn
from utils.common import dict_to_cuda
from utils import calculator as calc


def parse_args():
    parser = argparse.ArgumentParser(description="JointAffordance (Qwen) training")
    parser.add_argument("--qwen_model", type=str, default=None, help="Qwen 模型路径或名称")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="SAM 权重路径")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集路径")
    parser.add_argument("--log_dir", type=str, default=None, help="日志与权重输出目录")
    return parser.parse_args()


def enable_trainable_modules(model, name_filters):
    for name, param in model.named_parameters():
        if any(key in name for key in name_filters):
            param.requires_grad = True


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

    model = JointAffordanceModel(model_config)
    processor = AutoProcessor.from_pretrained(model_config.mllm.qwen_model_name_or_path)
    data_collator = partial(
        qwen3vl_collate_fn,
        tokenizer=processor.tokenizer,
        output_image_size=config.image_size,
        output_point_nums=config.num_points,
        precision=config.precision,
    )

    # 先冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    # 使用 LoRA 包裹 MLLM 主干（qwen）
    if config.lora_r > 0:
        lora_config = config.get_lora_config()
        model.mllm.model = get_peft_model(model.mllm.model, lora_config)

    # 解冻必要模块
    enable_trainable_modules(model, config.name_of_params_to_train)

    train_data_manager = JointDataset(dataset_root=config.dataset_dir, dtype='train').load_all_data()
    val_data_manager = JointDataset(dataset_root=config.dataset_dir, dtype='val').load_all_data()
    if config.samples_per_epoch is not None:
        train_dataset = Qwen3VLTrainDataset(
            train_data_manager.samples,
            processor=processor,
            image_size=config.image_size,
            num_points=config.num_points,
            precision=config.precision,
            samples_per_epoch=config.samples_per_epoch,
            use_sample_cache=config.use_sample_cache,
        )
    else:
        train_dataset = Qwen3VLDataset(
            train_data_manager.samples,
            processor=processor,
            image_size=config.image_size,
            num_points=config.num_points,
            precision=config.precision,
            use_sample_cache=config.use_sample_cache,
        )
    val_dataset = Qwen3VLDataset(
        val_data_manager.samples,
        processor=processor,
        image_size=config.image_size,
        num_points=config.num_points,
        precision=config.precision,
        use_sample_cache=config.use_sample_cache,
    )

    # DeepSpeed 初始化
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset,
        collate_fn=data_collator,
        config=config.deepspeed.to_dict(),
    )

    model_engine.train()
    for epoch in range(config.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        for input_dict in train_loader:
            input_dict = dict_to_cuda(input_dict)
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

        if val_dataset is not None:
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
                total_batches = 0
                for val_dict in val_loader:
                    val_dict = dict_to_cuda(val_dict)
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
                    total_batches += 1
                if config.local_rank == 0 and total_batches > 0:
                    print(f"[val] epoch={epoch} loss={total_val_loss / total_batches:.6f}")
            model_engine.train()


if __name__ == "__main__":
    main()
