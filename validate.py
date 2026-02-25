"""
Joint Affordance 验证脚本（新版，适配 Qwen + JointAffordanceModel）

相对 validate_old.py 的主要变化：
- 使用新的 JointAffordanceModel（Qwen3-VL + SAM + PointNet++），不再依赖 LISAForCausalLM / deepspeed.init_inference
- 使用 JointDataset + JointAffordanceTorchDataset 作为数据来源（与 train.py 一致）
- 使用 utils.metrics 中的 torchmetrics 方案统一统计 2D / 3D 指标
- 输出字典字段为 "image_logits" / "point_logits"，并据此保存预测结果
"""

import argparse
import os
from functools import partial
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from peft import get_peft_model
import cv2

from configs import TrainingConfig
from configs.inference_config import InferenceConfig
from model.joint_affordance import JointAffordanceModel
from utils.base_dataset import JointDataset
from utils.dataset import (
    JointAffordanceTorchDataset,
    joint_affordance_collate_fn,
)
from utils.common import dict_to_cuda
from utils.checkpoint_utils import load_checkpoint_to_model
from utils import calculator as calc
from utils.metrics import (
    build_torchmetrics_bundle,
    update_torchmetrics,
    compute_and_reset_torchmetrics,
    log_epoch_summary,
)
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="验证 JointAffordance 模型（新版）")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的模型 checkpoint 路径（包含 model_state_dict 或直接为 state_dict）")
    parser.add_argument("--config_json", type=str, default=None,
                        help="训练配置 JSON 路径（默认自动在 checkpoint 同目录查找 training_config.json）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集目录（默认使用 TrainingConfig 中的设置）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="验证 batch 大小")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="要评估的数据集分割（默认：test）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="使用的设备（默认：cuda）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数量（默认：4）")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存预测结果（2D mask PNG + 3D CSV）")
    parser.add_argument("--output_dir", type=str, default="./validation_output",
                        help="预测结果保存目录（默认：./validation_output）")
    parser.add_argument("--qwen_model", type=str, default=None,
                        help="Qwen 模型路径或名称（覆盖 TrainingConfig.model_config.mllm）")
    parser.add_argument("--vision_pretrained", type=str, default=None,
                        help="SAM 权重路径（覆盖 TrainingConfig.model_config.image_decoder.vision_pretrained）")
    parser.add_argument("--log_name", type=str, default="validate",
                        help="日志名/实验名（用于输出目录命名）")
    return parser.parse_args()


def _resolve_config_json_path(checkpoint_path: str, config_json_arg: Optional[str]) -> Optional[str]:
    """优先使用显式参数，否则自动在 checkpoint 同目录查找 training_config.json。"""
    if config_json_arg:
        if not os.path.exists(config_json_arg):
            raise FileNotFoundError(f"指定的配置 JSON 不存在: {config_json_arg}")
        return config_json_arg
    ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    candidate = os.path.join(ckpt_dir, "training_config.json")
    return candidate if os.path.exists(candidate) else None


def build_dataloader_for_split(
    training_cfg: TrainingConfig,
    model_cfg,
    infer_cfg: InferenceConfig,
):
    """根据 split（train/val/test）构建对应的 DataLoader。"""
    processor = AutoProcessor.from_pretrained(model_cfg.mllm.qwen_model_name_or_path)
    collator = partial(
        joint_affordance_collate_fn,
        tokenizer=processor.tokenizer,
        output_image_size=training_cfg.image_size,
        output_point_nums=training_cfg.num_points,
        mllm_precision=model_cfg.mllm.compute_dtype,
        image_precision=model_cfg.image_decoder.compute_dtype,
        point_precision=model_cfg.point_decoder.compute_dtype,
    )

    joint_dataset = JointDataset(
        dataset_root=training_cfg.dataset_dir,
        dtype=infer_cfg.split,
    ).load_all_data()
    torch_dataset = JointAffordanceTorchDataset(
        joint_dataset.samples,
        processor=processor,
        image_size=training_cfg.image_size,
        num_points=training_cfg.num_points,
        mllm_precision=model_cfg.mllm.compute_dtype,
        image_precision=model_cfg.image_decoder.compute_dtype,
        point_precision=model_cfg.point_decoder.compute_dtype,
        use_sample_cache=training_cfg.use_sample_cache,
    )

    loader = DataLoader(
        torch_dataset,
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=infer_cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return loader, torch_dataset


def _normalize_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    mask = mask_tensor.detach().float()
    if mask.dim() > 2:
        mask = mask.squeeze()
    if mask.max() > 1.0 or mask.min() < 0.0:
        mask = mask.sigmoid()
    return mask.clamp(0.0, 1.0)


def _to_uint8_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = _normalize_mask(mask_tensor)
    return mask.mul(255.0).round().to(torch.uint8).cpu().numpy()


def _to_float_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    return _normalize_mask(mask_tensor).cpu().numpy()


def _save_pointcloud_csv(file_path: str, points: np.ndarray, mask: np.ndarray, label: str):
    header = ["x", "y", "z", label]
    data = np.concatenate([points, mask[:, None]], axis=1)
    with open(file_path, "w") as f:
        np.savetxt(f, data, delimiter=",", header=",".join(header))


def save_batch_predictions(
    input_dict: Dict,
    output_dict: Dict,
    batch_idx: int,
    output_dir: str,
    dataset=None,
    batch_start: Optional[int] = None,
):
    """
    保存一个 batch 的预测结果（2D PNG + 3D CSV），适配新版输出字段：
        - 2D: output_dict["image_logits"]
        - 3D: output_dict["point_logits"]
    """

    def _get_batch_size() -> int:
        for key in ("input_ids", "images", "point_clouds", "img_gt_tensor", "pc_gt_tensor"):
            value = input_dict.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        img_logits = output_dict.get("image_logits")
        if isinstance(img_logits, torch.Tensor):
            return int(img_logits.shape[0])
        pt_logits = output_dict.get("point_logits")
        if isinstance(pt_logits, torch.Tensor):
            return int(pt_logits.shape[0])
        return 0

    def _extract_pred_mask(key: str, index: int):
        masks = output_dict.get(key)
        if masks is None:
            return None
        if isinstance(masks, list):
            return masks[index] if index < len(masks) else None
        if isinstance(masks, torch.Tensor):
            if masks.dim() == 2:
                return masks
            return masks[index] if masks.shape[0] > index else None
        return None

    batch_size = _get_batch_size()
    if batch_size <= 0:
        return

    if batch_start is None:
        batch_start = batch_idx * batch_size

    samples = getattr(dataset, "samples", None) if dataset is not None else None

    for i in range(batch_size):
        sample = None
        if samples is not None:
            sample_index = batch_start + i
            if 0 <= sample_index < len(samples):
                sample = samples[sample_index]
        if sample is None:
            continue

        obj_type = sample.obj_type
        aff_type = sample.aff_type
        sample_id = sample.id

        # 2D mask 保存
        pred_mask_2d = _extract_pred_mask("image_logits", i)
        if pred_mask_2d is not None and sample.img is not None and sample.img.img is not None:
            mask_2d = _to_uint8_mask(pred_mask_2d)
            img_dir = os.path.join(output_dir, obj_type, "Image")
            rgb_dir = os.path.join(img_dir, "rgb")
            mask_dir = os.path.join(img_dir, "mask", aff_type)
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            img_path = os.path.join(rgb_dir, f"{obj_type}_{sample_id}.png")
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, sample.img.img)
            mask_path = os.path.join(mask_dir, f"{obj_type}_{sample_id}_{aff_type}.png")
            cv2.imwrite(mask_path, mask_2d)

        # 3D mask 保存
        pred_mask_3d = _extract_pred_mask("point_logits", i)
        if pred_mask_3d is not None and sample.pc is not None:
            points = None
            pc_tensor = input_dict.get("point_clouds")
            if isinstance(pc_tensor, torch.Tensor) and pc_tensor.shape[0] > i:
                points = pc_tensor[i].detach().cpu().numpy()
            if points is None and sample.pc is not None:
                points = sample.pc.points
            if points is None:
                continue

            if points.ndim == 3 and points.shape[0] == 3:
                points = np.transpose(points, (1, 0))

            mask_3d = _to_float_mask(pred_mask_3d).reshape(-1)
            pc_lengths = input_dict.get("pc_lengths")
            if isinstance(pc_lengths, torch.Tensor) and pc_lengths.shape[0] > i:
                num_points = int(pc_lengths[i].item())
            else:
                num_points = min(points.shape[0], mask_3d.shape[0])

            num_points = max(0, min(num_points, points.shape[0], mask_3d.shape[0]))
            if num_points == 0:
                continue

            points = points[:num_points]
            mask_3d = mask_3d[:num_points]

            pc_dir = os.path.join(output_dir, obj_type, "PointCloud")
            os.makedirs(pc_dir, exist_ok=True)
            pc_path = os.path.join(pc_dir, f"{obj_type}_{sample_id}.csv")
            _save_pointcloud_csv(pc_path, points, mask_3d, aff_type)


def main():
    args = parse_args()

    # 训练 & 推理配置
    cfg_json_path = _resolve_config_json_path(args.checkpoint_path, args.config_json)
    if cfg_json_path is not None and os.path.exists(cfg_json_path):
        training_cfg = TrainingConfig.from_json(cfg_json_path)
        print(f"已加载训练配置: {cfg_json_path}")
    else:
        training_cfg = TrainingConfig()
        print("未找到训练配置 JSON，使用 TrainingConfig 默认值。")
    infer_cfg = InferenceConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    if args.dataset_dir:
        training_cfg.dataset_dir = args.dataset_dir
    training_cfg.val_batch_size = infer_cfg.batch_size
    training_cfg.workers = infer_cfg.num_workers

    device = torch.device(infer_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 初始化模型
    model_cfg = training_cfg.model_config
    if args.qwen_model:
        model_cfg.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        model_cfg.image_decoder.vision_pretrained = args.vision_pretrained

    model = JointAffordanceModel(model_cfg).to(device)
    # 应用lora
    if training_cfg.lora.lora_r > 0:
        model.mllm.model = get_peft_model(model.mllm.model, training_cfg.lora.to_peft_config())

    load_checkpoint_to_model(model, args.checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()

    # DataLoader
    val_loader, torch_dataset = build_dataloader_for_split(training_cfg, model_cfg, infer_cfg)

    # 指标 & 验证循环
    metrics = build_torchmetrics_bundle(
        device=device,
        threshold_2d=max(training_cfg.mask_threshold_2d, 0.5),
        threshold_3d=training_cfg.mask_threshold_3d,
    )

    # 用于构造 loss_dict 时的配置（与 train.py 保持一致）
    loss_kwargs = dict(
        device=device,
        focal_loss_weight=getattr(training_cfg, "focal_loss_weight", 2.0),
        dice_loss_weight=getattr(training_cfg, "dice_loss_weight", 0.5),
        focal_alpha=getattr(training_cfg, "focal_alpha", 0.25),
        focal_gamma=getattr(training_cfg, "focal_gamma", 2.0),
        bce_loss_weight=getattr(training_cfg, "bce_loss_weight", 2.0),
        ce_loss_weight=getattr(training_cfg, "ce_loss_weight", 1.0),
    )

    if infer_cfg.save_predictions and infer_cfg.output_dir:
        os.makedirs(infer_cfg.output_dir, exist_ok=True)
        print(f"预测结果将保存到: {infer_cfg.output_dir}")

    print("开始验证...")
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(tqdm(val_loader, desc="验证中")):
            input_dict = dict_to_cuda(input_dict, device=device)
            output_dict = model(**input_dict)

            # 计算损失 + 更新指标（包括 IoU/MAE/AUROC/AUC/SIM）
            loss_dict = calc.compute_losses(output_dict, input_dict, **loss_kwargs)
            update_torchmetrics(metrics, loss_dict, output_dict, input_dict, infer_cfg.batch_size)

            # 保存预测（可选）
            if infer_cfg.save_predictions and infer_cfg.output_dir:
                save_batch_predictions(
                    input_dict,
                    output_dict,
                    batch_idx,
                    infer_cfg.output_dir,
                    dataset=torch_dataset,
                )
            # 释放当前 batch 的 GPU 张量引用，避免长验证阶段显存碎片累积
            del output_dict, loss_dict, input_dict
            if device.type == "cuda" and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

    results = compute_and_reset_torchmetrics(metrics)

    # 打印摘要（重用 log_epoch_summary 的输出格式）
    log_epoch_summary(
        logger=type("DummyLogger", (), {"info": print})(),
        epoch=1,
        total_epochs=1,
        phase="val",
        results=results,
        lr=None,
    )

    # 保存评估结果
    out_dir = infer_cfg.output_dir if infer_cfg.save_predictions else "."
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "validation_results.pt")
    torch.save(results, save_path)
    print(f"\n评估结果已保存到: {save_path}")


if __name__ == "__main__":
    main()

