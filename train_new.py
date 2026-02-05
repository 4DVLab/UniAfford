import argparse
from functools import partial

import deepspeed
import torch
from peft import get_peft_model
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

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


def create_param_groups(model, config):
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
        print(f"\n✓ LLM 参数组: {len(llm_params)} 个参数, lr={config.llm_lr}")
    if vision_2d_params:
        param_groups.append({
            "params": vision_2d_params,
            "lr": config.vision_2d_lr,
            "name": "vision_2d",
        })
        print(f"✓ 2D 视觉参数组: {len(vision_2d_params)} 个参数, lr={config.vision_2d_lr}")
    if vision_3d_params:
        param_groups.append({
            "params": vision_3d_params,
            "lr": config.vision_3d_lr,
            "name": "vision_3d",
        })
        print(f"✓ 3D 视觉参数组: {len(vision_3d_params)} 个参数, lr={config.vision_3d_lr}")
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": config.lr,
            "name": "other",
        })
        print(f"✓ 其他参数组: {len(other_params)} 个参数, lr={config.lr}\n")

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


    """ ------------------------- 初始化模型 --------------------------- """
    model = JointAffordanceModel(model_config)
    processor = AutoProcessor.from_pretrained(model_config.mllm.qwen_model_name_or_path)
    data_collator = partial(
        qwen3vl_collate_fn,
        tokenizer=processor.tokenizer,
        output_image_size=config.image_size,
        output_point_nums=config.num_points,
        precision=config.precision,
    )

    """ ------------------------- 选择训练参数、应用lora --------------------------- """
    for p in model.parameters():
        p.requires_grad = False

    # 使用 LoRA 包裹 MLLM 主干（qwen）
    if config.lora.lora_r > 0:
        lora_config = config.lora.to_peft_config()
        model.mllm.model = get_peft_model(model.mllm.model, lora_config)

    # 解冻必要模块
    enable_trainable_modules(model, config.name_of_params_to_train)

    """ ------------------------- 加载数据集 --------------------------- """
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

    """ ------------------------- DeepSpeed 初始化（分层学习率） --------------------------- """
    params_to_train = create_param_groups(model, config)
    if len(params_to_train) == 0:
        raise RuntimeError("没有可训练参数，请检查 name_of_params_to_train 或 LoRA 配置")

    if config.use_layerwise_lr:
        steps_per_epoch = config.steps_per_epoch
        if steps_per_epoch is None:
            micro_bs = config.deepspeed.train_micro_batch_size_per_gpu
            steps_per_epoch = max(1, len(train_dataset) // max(1, micro_bs))

        optimizer = torch.optim.AdamW(
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
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            collate_fn=data_collator,
            config=config.deepspeed.to_dict(),
        )


    """ ------------------------- 训练 --------------------------- """
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

        """ ------------------------- 验证 --------------------------- """
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
