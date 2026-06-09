"""
Joint Affordance 验证脚本（新版，适配 Qwen + JointAffordanceModel）

相对 validate_old.py 的主要变化：
- 使用新的 JointAffordanceModel（Qwen3-VL + SAM + PointNet++），不再依赖 LISAForCausalLM / deepspeed.init_inference
- 使用 JointDataset + JointAffordanceTorchDataset 作为数据来源（与 train.py 一致）
- 使用 utils.metrics 中的 torchmetrics 方案统一统计 2D / 3D 指标
- 输出字典字段为 "image_logits" / "point_logits"，并据此保存预测结果
"""

import argparse
import csv
import json
import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2

from configs import TrainingConfig
from configs.inference_config import InferenceConfig
from utils.base_dataset import JointDataset
from utils.dataset import (
    JointAffordanceTorchDataset,
    joint_affordance_collate_fn,
)
from utils.common import dict_to_cuda, resolve_dtype
from utils.model_io import (
    load_checkpoint_payload,
    load_portable_model,
    resolve_training_config_from_payload,
)
from utils import calculator as calc
from utils.metrics import (
    build_torchmetrics_bundle,
    update_torchmetrics,
    compute_and_reset_torchmetrics,
    compute_sample_metrics,
    log_epoch_summary,
)
from utils.threshold_search import (
    build_threshold_candidates,
    init_threshold_search_stats,
    update_threshold_search_stats,
    finalize_threshold_search,
)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="验证 JointAffordance 模型（新版）")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的模型 checkpoint 路径；支持单文件 .pth 或 HF 分片目录")
    parser.add_argument("--config_json", type=str, default=None,
                        help="训练配置 JSON 路径（默认自动在 checkpoint 同目录查找 training_config.json）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集目录（默认使用 TrainingConfig 中的设置）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="验证 batch 大小（默认使用 TrainingConfig.val_batch_size）")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"],
                        help="要评估的数据集分割（默认：test）")
    parser.add_argument("--inference_mode", type=str, default="generate", choices=["generate", "forward"],
                        help="推理模式：generate 为 prompt-only 自回归推理（默认）；forward 为 teacher-forcing 前向诊断。")
    parser.add_argument("--generate_query_fallback", action="store_true",
                        help="generate 模式下若缺少 img/pc query，则用对应 route 概率最高的 hidden state 兜底；默认关闭以真实评估 router。")
    parser.add_argument("--device", type=str, default=None,
                        help="使用的设备（默认：cuda，若不可用则 CPU）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker 数量（默认使用 TrainingConfig.workers）")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存预测结果（2D mask PNG + 3D CSV）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="预测结果保存目录（默认：./validation_output）")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16", "fp16"],
                        help="推理精度覆盖项（默认不覆盖 TrainingConfig）")
    parser.add_argument("--mask_threshold_2d", type=float, default=None,
                        help="2D 预测概率二值化阈值覆盖项（默认使用 TrainingConfig）")
    parser.add_argument("--mask_threshold_3d", type=float, default=None,
                        help="3D 预测概率二值化阈值覆盖项（默认使用 TrainingConfig）")
    parser.add_argument("--gt_threshold_2d", type=float, default=None,
                        help="2D GT 二值化阈值覆盖项，用于对齐不同 benchmark（默认使用 TrainingConfig）")
    parser.add_argument("--gt_threshold_3d", type=float, default=None,
                        help="3D GT 二值化阈值覆盖项，用于对齐不同 benchmark（默认使用 TrainingConfig）")
    parser.add_argument("--qwen_model", type=str, default=None,
                        help="Qwen 模型路径或名称（覆盖 TrainingConfig.model_config.mllm）")
    parser.add_argument("--vision_pretrained", type=str, default=None,
                        help="SAM 权重路径（覆盖 TrainingConfig.model_config.image_decoder.vision_pretrained）")
    parser.add_argument("--log_name", type=str, default="validate",
                        help="日志名/实验名（用于输出目录命名）")
    parser.add_argument("--lazy_load", dest="lazy_load", action="store_true",
                        help="启用数据懒加载（默认启用）", default=True)
    parser.add_argument("--save_tsne", action="store_true",
                        help="记录 task token hidden/vocab embedding，并在验证结束后尝试生成 t-SNE 可视化")
    parser.add_argument("--tsne_max_points", type=int, default=4000,
                        help="t-SNE 最多采样点数，避免验证集过大导致内存和耗时过高")
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
    processor,
    lazy_load: bool = True,
):
    """根据 split（train/val/test）构建对应的 DataLoader。"""
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
        split_file=f"{infer_cfg.split}.json",
        lazy_load=lazy_load,
    )
    samples = joint_dataset if lazy_load else joint_dataset.load_all_data().samples
    torch_dataset = JointAffordanceTorchDataset(
        samples,
        processor=processor,
        image_size=training_cfg.image_size,
        num_points=training_cfg.num_points,
        mllm_precision=model_cfg.mllm.compute_dtype,
        image_precision=model_cfg.image_decoder.compute_dtype,
        point_precision=model_cfg.point_decoder.compute_dtype,
        use_sample_cache=training_cfg.use_sample_cache,
        use_simple_answer_template=training_cfg.use_simple_answer_template,
    )

    loader = DataLoader(
        torch_dataset,
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=infer_cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return loader, torch_dataset, processor


def run_model_inference(
    model,
    input_dict: Dict,
    inference_mode: str,
    return_hidden_states: bool = False,
    generate_query_fallback: bool = False,
) -> Dict:
    """根据验证模式执行模型推理。

    Args:
        model: JointAffordanceModel 实例。
        input_dict: 已经移动到目标设备上的 batch 输入。
        inference_mode: ``"generate"`` 使用 prompt-only 自回归推理；``"forward"`` 使用
            teacher-forcing 前向，仅用于诊断 forward 上限。
        return_hidden_states: 是否返回 hidden states，用于 t-SNE 等可视化。

    Returns:
        Dict: 模型输出字典，字段与指标计算逻辑保持兼容。
    """
    if inference_mode == "generate":
        # 独立验证默认真实评估 router；仅显式打开开关时启用 query fallback。
        return model.generate_forward(
            **input_dict,
            return_hidden_states=return_hidden_states,
            generate_query_fallback=generate_query_fallback,
        )
        # 默认真实推理路径：不把 GT answer 输入 MLLM。
        return model.generate_forward(**input_dict, return_hidden_states=return_hidden_states)
    if inference_mode == "forward":
        # 诊断路径：GT answer 会进入 MLLM，只用于定位 generate/forward 差异。
        return model(**input_dict, return_hidden_states=return_hidden_states)
        # 诊断路径：GT answer 会进入 MLLM，只用于定位 generate/forward 差异。
        return model(**input_dict, return_hidden_states=return_hidden_states)
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def save_batch_predictions(
    input_dict: Dict,
    output_dict: Dict,
    batch_idx: int,
    output_dir: str,
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
                return masks[index] if masks.shape[0] > index else None
            if masks.shape[0] <= index:
                return None
            mask = masks[index]
            if key == "image_logits" and mask.dim() == 3:
                return mask[0]
            if key == "point_logits" and mask.dim() == 2:
                return mask[0]
            return mask
        return None

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

    batch_size = _get_batch_size()
    if batch_size <= 0:
        return

    if batch_start is None:
        batch_start = batch_idx * batch_size

    for i in range(batch_size):
        sample_index = batch_start + i
        obj_type = input_dict.get("obj_type")[i]
        aff_type = input_dict.get("aff_type")[i]
        sample_id = input_dict.get("sample_id")[i]

        # 2D mask 保存：仅当该样本有有效图像输入时保存
        has_img = False
        img_valid = input_dict.get("img_valid_mask")
        if isinstance(img_valid, torch.Tensor) and img_valid.shape[0] > i:
            has_img = bool(img_valid[i].item())
        pred_mask_2d = _extract_pred_mask("image_logits", i)
        images_tensor = input_dict.get("images")
        if has_img and pred_mask_2d is not None and isinstance(images_tensor, torch.Tensor) and images_tensor.shape[0] > i:
            mask_2d = _to_uint8_mask(pred_mask_2d)

            # 从 input_dict["images"] 还原 RGB，按 original_size_list 缩放到原始尺寸
            # original_size_list 为 _build_sample 中 resize 前记录的原图 (H, W)，保证准确未篡改
            img_tensor = images_tensor[i].detach().float().cpu().clamp(0.0, 1.0)
            model_img = (img_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            original_size_list = input_dict.get("original_size_list")
            if isinstance(original_size_list, (list, tuple)) and i < len(original_size_list):
                orig_h, orig_w = map(int, original_size_list[i])
                orig_h, orig_w = max(1, orig_h), max(1, orig_w)
                if (orig_h, orig_w) != model_img.shape[:2]:
                    orig_img = cv2.resize(model_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    orig_img = model_img
            else:
                orig_h, orig_w = model_img.shape[0], model_img.shape[1]
                orig_img = model_img

            if mask_2d.ndim == 3:
                # 如果是 (C, H, W) 或 (H, W, C)，先取单通道
                if mask_2d.shape[0] in (1, 3) and mask_2d.shape[1] == orig_h and mask_2d.shape[2] == orig_w:
                    mask_2d = mask_2d[0]
                elif mask_2d.shape[-1] in (1, 3) and mask_2d.shape[0] == orig_h and mask_2d.shape[1] == orig_w:
                    mask_2d = mask_2d[..., 0]
                else:
                    # 退化为取第一个通道再缩放
                    mask_2d = mask_2d[0] if mask_2d.shape[0] in (1, 3) else mask_2d[..., 0]

            if mask_2d.shape[:2] != (orig_h, orig_w):
                mask_2d = cv2.resize(mask_2d, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            img_dir = os.path.join(output_dir, obj_type, "Image")
            rgb_dir = os.path.join(img_dir, "rgb")
            mask_dir = os.path.join(img_dir, "mask", aff_type)
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            img_path = os.path.join(rgb_dir, f"{obj_type}_{sample_id}.png")
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
            mask_path = os.path.join(mask_dir, f"{obj_type}_{sample_id}_{aff_type}.png")
            cv2.imwrite(mask_path, mask_2d)

        # 3D mask 保存：仅当该样本有有效点云输入时保存
        has_pc = False
        pc_lengths = input_dict.get("pc_valid_lengths")
        if isinstance(pc_lengths, torch.Tensor) and pc_lengths.shape[0] > i:
            has_pc = int(pc_lengths[i].item()) > 0
        pred_mask_3d = _extract_pred_mask("point_logits", i)
        if has_pc and pred_mask_3d is not None:
            points = None
            pc_tensor = input_dict.get("point_clouds")
            if isinstance(pc_tensor, torch.Tensor) and pc_tensor.shape[0] > i:
                points = pc_tensor[i].detach().cpu().numpy()
            if points is None:
                continue

            if points.ndim == 3 and points.shape[0] == 3:
                points = np.transpose(points, (1, 0))

            mask_3d = _to_float_mask(pred_mask_3d).reshape(-1)
            pc_lengths = input_dict.get("pc_valid_lengths")
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


def _parse_metric_val(v) -> Optional[float]:
    """将样本记录中的指标值解析为 float，无效则返回 None。"""
    if v is None or v == "":
        return None
    try:
        value = float(v)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _aggregate_by_label(sample_records: List[Dict]) -> Dict:
    """
    从逐样本记录聚合出按 obj-aff、按 obj、按 aff、总体的指标均值，便于分析各标签表现。

    Returns:
        {
            "by_obj_aff": { "obj_type": { "aff_type": {"giou_2d": ..., "kld_2d": ..., "sim_2d": ..., "nss_2d": ..., "iou_3d": ..., "mae_3d": ..., "sim_3d": ..., "n_2d": int, "n_3d": int} } },
            "by_obj": { "obj_type": {"giou_2d": ..., "kld_2d": ..., "sim_2d": ..., "nss_2d": ..., "iou_3d": ..., "mae_3d": ..., "sim_3d": ..., "n_2d": int, "n_3d": int} },
            "by_aff": { "aff_type": {"giou_2d": ..., "kld_2d": ..., "sim_2d": ..., "nss_2d": ..., "iou_3d": ..., "mae_3d": ..., "sim_3d": ..., "n_2d": int, "n_3d": int} },
            "overall": {"giou_2d": ..., "kld_2d": ..., "sim_2d": ..., "nss_2d": ..., "iou_3d": ..., "mae_3d": ..., "sim_3d": ..., "n_2d": int, "n_3d": int},
        }
    """
    metric_keys = ("giou_2d", "ciou_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d", "iou_3d", "auc_3d", "mae_3d", "sim_3d")

    # 按 (obj, aff) 收集有效值
    obj_aff_vals: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    obj_aff_n2d: Dict[Tuple[str, str], int] = defaultdict(int)
    obj_aff_n3d: Dict[Tuple[str, str], int] = defaultdict(int)

    for r in sample_records:
        obj, aff = r.get("obj_type", ""), r.get("aff_type", "")
        if not obj or not aff:
            continue
        giou_2d = _parse_metric_val(r.get("giou_2d"))
        inter_2d = _parse_metric_val(r.get("inter_2d"))
        union_2d = _parse_metric_val(r.get("union_2d"))
        p50_2d = _parse_metric_val(r.get("p50_2d"))
        p50_95_2d = _parse_metric_val(r.get("p50_95_2d"))
        kld_2d = _parse_metric_val(r.get("kld_2d"))
        sim_2d = _parse_metric_val(r.get("sim_2d"))
        nss_2d = _parse_metric_val(r.get("nss_2d"))
        iou_3d = _parse_metric_val(r.get("iou_3d"))
        auc_3d = _parse_metric_val(r.get("auc_3d"))
        mae_3d = _parse_metric_val(r.get("mae_3d"))
        sim_3d = _parse_metric_val(r.get("sim_3d"))
        if giou_2d is not None:
            obj_aff_vals[obj][aff]["giou_2d"].append(giou_2d)
            obj_aff_n2d[(obj, aff)] += 1
        if inter_2d is not None and union_2d is not None:
            obj_aff_vals[obj][aff]["inter_2d"].append(inter_2d)
            obj_aff_vals[obj][aff]["union_2d"].append(union_2d)
        if p50_2d is not None:
            obj_aff_vals[obj][aff]["p50_2d"].append(p50_2d)
        if p50_95_2d is not None:
            obj_aff_vals[obj][aff]["p50_95_2d"].append(p50_95_2d)
        if kld_2d is not None:
            obj_aff_vals[obj][aff]["kld_2d"].append(kld_2d)
        if sim_2d is not None:
            obj_aff_vals[obj][aff]["sim_2d"].append(sim_2d)
        if nss_2d is not None:
            obj_aff_vals[obj][aff]["nss_2d"].append(nss_2d)
        if iou_3d is not None:
            obj_aff_vals[obj][aff]["iou_3d"].append(iou_3d)
            obj_aff_n3d[(obj, aff)] += 1
        if auc_3d is not None:
            obj_aff_vals[obj][aff]["auc_3d"].append(auc_3d)
        if mae_3d is not None:
            obj_aff_vals[obj][aff]["mae_3d"].append(mae_3d)
        if sim_3d is not None:
            obj_aff_vals[obj][aff]["sim_3d"].append(sim_3d)

    def _mean(lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 6) if lst else None

    def _to_summary(vals: Dict[str, List[float]], n2d: int, n3d: int) -> Dict:
        out = {}
        for k in metric_keys:
            if k == "ciou_2d":
                inter_lst = vals.get("inter_2d", [])
                union_lst = vals.get("union_2d", [])
                if inter_lst and union_lst and sum(union_lst) > 0:
                    out[k] = round(sum(inter_lst) / sum(union_lst), 6)
                else:
                    out[k] = None
            else:
                out[k] = _mean(vals.get(k, []))
        out["n_2d"] = n2d
        out["n_3d"] = n3d
        return out

    # 1) by_obj_aff
    by_obj_aff = {}
    for obj in sorted(obj_aff_vals.keys()):
        by_obj_aff[obj] = {}
        for aff in sorted(obj_aff_vals[obj].keys()):
            vals = obj_aff_vals[obj][aff]
            by_obj_aff[obj][aff] = _to_summary(vals, obj_aff_n2d[(obj, aff)], obj_aff_n3d[(obj, aff)])

    # 2) by_obj：该 obj 下所有 aff 的平均（按样本数加权）
    by_obj = {}
    for obj in sorted(obj_aff_vals.keys()):
        agg = defaultdict(list)
        n2d, n3d = 0, 0
        for aff, vals in obj_aff_vals[obj].items():
            for k, lst in vals.items():
                agg[k].extend(lst)
            n2d += obj_aff_n2d[(obj, aff)]
            n3d += obj_aff_n3d[(obj, aff)]
        by_obj[obj] = _to_summary(dict(agg), n2d, n3d)

    # 3) by_aff：该 aff 下所有 obj 的平均（按样本数加权）
    aff_vals: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    aff_n2d: Dict[str, int] = defaultdict(int)
    aff_n3d: Dict[str, int] = defaultdict(int)
    for obj in obj_aff_vals:
        for aff in obj_aff_vals[obj]:
            for k, lst in obj_aff_vals[obj][aff].items():
                aff_vals[aff][k].extend(lst)
            aff_n2d[aff] += obj_aff_n2d[(obj, aff)]
            aff_n3d[aff] += obj_aff_n3d[(obj, aff)]

    by_aff = {}
    for aff in sorted(aff_vals.keys()):
        by_aff[aff] = _to_summary(aff_vals[aff], aff_n2d[aff], aff_n3d[aff])

    # 4) overall
    all_vals = defaultdict(list)
    total_n2d = sum(obj_aff_n2d.values())
    total_n3d = sum(obj_aff_n3d.values())
    for obj in obj_aff_vals:
        for aff in obj_aff_vals[obj]:
            for k, lst in obj_aff_vals[obj][aff].items():
                all_vals[k].extend(lst)
    overall = _to_summary(dict(all_vals), total_n2d, total_n3d)

    return {
        "by_obj_aff": by_obj_aff,
        "by_obj": by_obj,
        "by_aff": by_aff,
        "overall": overall,
    }


def main():
    args = parse_args()

    # 训练 & 推理配置
    cfg_json_path = _resolve_config_json_path(args.checkpoint_path, args.config_json)
    ckpt_payload = load_checkpoint_payload(args.checkpoint_path, map_location="cpu")
    if cfg_json_path is not None and os.path.exists(cfg_json_path):
        training_cfg = TrainingConfig.from_json(cfg_json_path)
        print(f"已加载训练配置: {cfg_json_path}")
    else:
        training_cfg = resolve_training_config_from_payload(ckpt_payload)
        if isinstance(ckpt_payload.get("training_config"), dict):
            print("已从 checkpoint 元信息加载训练配置。")
        else:
            print("未找到训练配置 JSON 或 checkpoint 内嵌配置，使用 TrainingConfig 默认值。")
    # 覆盖优先级：命令行参数（非 None） > InferenceConfig.defaults > training_config.json。
    infer_cfg = InferenceConfig(
        precision=args.precision,
        mask_threshold_2d=args.mask_threshold_2d,
        mask_threshold_3d=args.mask_threshold_3d,
    )

    # InferenceConfig 只表达覆盖项；运行所需默认值从 TrainingConfig 回退。
    infer_cfg.batch_size = args.batch_size if args.batch_size is not None else training_cfg.val_batch_size
    infer_cfg.num_workers = args.num_workers if args.num_workers is not None else training_cfg.workers
    infer_cfg.split = args.split or "test"
    infer_cfg.device = args.device or "cuda"
    infer_cfg.save_predictions = args.save_predictions
    infer_cfg.output_dir = args.output_dir or "./validation_output"

    if args.dataset_dir:
        training_cfg.dataset_dir = args.dataset_dir
    training_cfg.val_batch_size = infer_cfg.batch_size
    training_cfg.workers = infer_cfg.num_workers

    if infer_cfg.precision is not None:
        infer_dtype = resolve_dtype(infer_cfg.precision)
        training_cfg.model_config.mllm.compute_dtype = infer_dtype
        training_cfg.model_config.image_decoder.compute_dtype = infer_dtype
        training_cfg.model_config.point_decoder.compute_dtype = infer_dtype
    if infer_cfg.mask_threshold_2d is not None:
        training_cfg.mask_threshold_2d = float(infer_cfg.mask_threshold_2d)
    if infer_cfg.mask_threshold_3d is not None:
        training_cfg.mask_threshold_3d = float(infer_cfg.mask_threshold_3d)
    if args.gt_threshold_2d is not None:
        training_cfg.gt_threshold_2d = float(args.gt_threshold_2d)
    if args.gt_threshold_3d is not None:
        training_cfg.gt_threshold_3d = float(args.gt_threshold_3d)

    device = torch.device(infer_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    if args.qwen_model:
        training_cfg.model_config.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        training_cfg.model_config.image_decoder.vision_pretrained = args.vision_pretrained

    model, training_cfg, _ = load_portable_model(
        args.checkpoint_path,
        config_json_path=cfg_json_path,
        training_cfg=training_cfg,
        map_location="cpu",
        device=device,
        strict=False,
        ckpt_payload=ckpt_payload,
    )
    model_cfg = training_cfg.model_config
    model.eval()

    # DataLoader
    val_loader, torch_dataset, processor = build_dataloader_for_split(
        training_cfg,
        model_cfg,
        infer_cfg,
        processor=model.mllm.processor,
        lazy_load=args.lazy_load,
    )
    print(f"数据加载模式: {'lazy load' if args.lazy_load else 'eager load'}")
    tokenizer = processor.tokenizer
    lm_head = model.mllm.model.get_output_embeddings()
    IGNORE_INDEX = -100

    # 指标 & 验证循环
    metrics = build_torchmetrics_bundle(device=device)

    # 用于构造 loss_dict 时的配置（与 train.py 保持一致）
    loss_kwargs = dict(
        device=device,
        focal_loss_weight=getattr(training_cfg, "focal_loss_weight", 2.0),
        dice_loss_weight=getattr(training_cfg, "dice_loss_weight", 0.5),
        focal_alpha=getattr(training_cfg, "focal_alpha", 0.25),
        focal_gamma=getattr(training_cfg, "focal_gamma", 2.0),
        bce_loss_weight=getattr(training_cfg, "bce_loss_weight", 2.0),
        ce_loss_weight=getattr(training_cfg, "ce_loss_weight", 1.0),
        route_loss_weight=getattr(training_cfg, "route_loss_weight", 1.0),
        route_exist_loss_weight=getattr(training_cfg, "route_exist_loss_weight", 0.25),
        route_sparse_loss_weight=getattr(training_cfg, "route_sparse_loss_weight", 0.05),
        route_target_present_count=getattr(training_cfg, "route_target_present_count", 1.0),
    )

    threshold_2d = training_cfg.mask_threshold_2d
    threshold_3d = training_cfg.mask_threshold_3d
    gt_threshold_2d = getattr(training_cfg, "gt_threshold_2d", 0.5)
    gt_threshold_3d = getattr(training_cfg, "gt_threshold_3d", 0.5)

    if infer_cfg.save_predictions and infer_cfg.output_dir:
        os.makedirs(infer_cfg.output_dir, exist_ok=True)
        print(f"预测结果将保存到: {infer_cfg.output_dir}")

    sample_records: List[Dict] = []
    tsne_vectors: List[np.ndarray] = []
    tsne_records: List[Dict] = []
    if args.save_tsne:
        from utils.tsne_visualize import _append_task_vocab_embeddings, _collect_tsne_batch, _save_tsne_artifacts
        _append_task_vocab_embeddings(
            tsne_vectors,
            tsne_records,
            model,
            tokenizer,
            max_points=max(1, int(args.tsne_max_points)),
        )
    threshold_stats = None
    if getattr(training_cfg, "auto_select_mask_threshold", True):
        threshold_stats = init_threshold_search_stats(
            build_threshold_candidates(
                device,
                extra_thresholds=[threshold_2d, threshold_3d],
            )
        )

    print("开始验证...")
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(tqdm(val_loader, desc="验证中")):
            input_dict = dict_to_cuda(input_dict, device=device)
            # validate 推理必须只使用 prompt 进行 generate，GT answer 仅用于指标与文本对齐记录。
            output_dict = run_model_inference(
                model,
                input_dict,
                inference_mode=args.inference_mode,
                return_hidden_states=args.save_tsne,
                generate_query_fallback=args.generate_query_fallback,
            )

            # 计算损失 + 更新指标（包括 2D IoU/KLD/SIM/NSS 与 3D IoU/MAE/AUC/SIM）
            loss_dict = calc.compute_losses(output_dict, input_dict, **loss_kwargs)
            update_torchmetrics(
                metrics, loss_dict, output_dict, input_dict, infer_cfg.batch_size,
                threshold_2d=threshold_2d, threshold_3d=threshold_3d,
                gt_threshold_2d=gt_threshold_2d, gt_threshold_3d=gt_threshold_3d,
            )
            if threshold_stats is not None:
                update_threshold_search_stats(threshold_stats, output_dict, input_dict, training_cfg)
            if args.save_tsne:
                _collect_tsne_batch(
                    tsne_vectors,
                    tsne_records,
                    output_dict,
                    input_dict,
                    model,
                    tokenizer,
                    max_points=max(1, int(args.tsne_max_points)),
                    ignore_index=IGNORE_INDEX,
                )

            pred_token_ids_batch = output_dict.get("generated_token_ids")
            if pred_token_ids_batch is None:
                pred_token_ids_batch = output_dict.get("token_ids")
            if pred_token_ids_batch is not None:
                pred_token_ids_batch = pred_token_ids_batch.detach().cpu()
            aligned_labels_batch = output_dict.get("labels")
            if isinstance(aligned_labels_batch, torch.Tensor):
                aligned_labels_batch = aligned_labels_batch.detach().cpu()
            aligned_attention_batch = output_dict.get("attention_mask")
            if isinstance(aligned_attention_batch, torch.Tensor):
                aligned_attention_batch = aligned_attention_batch.detach().cpu()

            # ---- 逐样本记录 ----
            batch_size = input_dict["input_ids"].shape[0]
            for i in range(batch_size):
                sample_idx = batch_idx * infer_cfg.batch_size + i
                if sample_idx >= len(torch_dataset):
                    break
                src_ids_batch = input_dict.get("data_source_id", [])
                src_ids = src_ids_batch[i] if isinstance(src_ids_batch, (list, tuple)) and i < len(src_ids_batch) else {}

                record: Dict = {
                    "sample_id": input_dict.get("sample_id")[i],
                    "obj_type": input_dict.get("obj_type")[i],
                    "aff_type": input_dict.get("aff_type")[i],
                    "text_id": src_ids.get("ins_id", ""),
                    "img_id": src_ids.get("img_id", ""),
                    "pc_id": src_ids.get("pc_id", ""),
                }

                # GT 文本：从 labels 中提取非 IGNORE 的 token ids 解码
                # 关键：优先使用模型返回的对齐 labels（已包含点云动态注入后的位点变化）
                if isinstance(aligned_labels_batch, torch.Tensor) and aligned_labels_batch.shape[0] > i:
                    labels_i = aligned_labels_batch[i]
                else:
                    labels_i = input_dict["labels"][i].cpu()
                answer_mask = labels_i != IGNORE_INDEX
                if isinstance(aligned_attention_batch, torch.Tensor) and aligned_attention_batch.shape[0] > i:
                    answer_mask = answer_mask & aligned_attention_batch[i].bool()
                supervised_pos = torch.nonzero(answer_mask, as_tuple=False).squeeze(-1)
                gt_ids = labels_i[supervised_pos].tolist()
                record["gt_text"] = tokenizer.decode(gt_ids, skip_special_tokens=False)

                # 预测文本：validate 现在使用 prompt-only generate，直接解码 generated_token_ids。
                # 这里不再按 labels 左移取 forward logits，避免 GT answer 泄露后的 teacher-forcing 诊断值混入推理结果。
                if pred_token_ids_batch is not None:
                    pred_ids_i = pred_token_ids_batch[i]
                    pred_answer_ids = pred_ids_i.tolist()
                    pad_id = getattr(tokenizer, "pad_token_id", None)
                    eos_id = getattr(tokenizer, "eos_token_id", None)
                    if pad_id is not None:
                        pred_answer_ids = [tid for tid in pred_answer_ids if int(tid) != int(pad_id)]
                    if eos_id is not None:
                        eos_ids = {int(eos_id)} if isinstance(eos_id, int) else {int(tid) for tid in eos_id}
                        trimmed = []
                        for tid in pred_answer_ids:
                            trimmed.append(tid)
                            if int(tid) in eos_ids:
                                break
                        pred_answer_ids = trimmed
                    record["pred_token_ids"] = json.dumps(pred_answer_ids)
                    record["pred_text"] = tokenizer.decode(pred_answer_ids, skip_special_tokens=False)
                else:
                    record["pred_token_ids"] = "[]"
                    record["pred_text"] = ""

                # ---- 路由到 <aff> token 的 hidden state 通过文本解码头反投影为 token ----
                # 用于观察这些跨模态路由位置是否可以被解释为“文本语义”。
                aff_pairs = output_dict.get("aff_token_pairs")
                if aff_pairs is not None and i < len(aff_pairs) and lm_head is not None:
                    pairs_i = aff_pairs[i]
                    if pairs_i:
                        emb_stack = torch.stack([p[1] for p in pairs_i], dim=0)
                        emb_stack = emb_stack.to(device=next(lm_head.parameters()).device, dtype=next(lm_head.parameters()).dtype)
                        aff_logits = lm_head(emb_stack)  # [K, V]
                        aff_ids = aff_logits.argmax(dim=-1).detach().cpu().tolist()
                        aff_text_tokens = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in aff_ids]
                        # aff_token_names 中同时写入 token 与 id，便于和 pred_token_ids 对齐排查
                        aff_token_infos = [
                            {"token": tok, "id": int(tid)}
                            for tok, tid in zip(aff_text_tokens, aff_ids)
                        ]
                        record["aff_token_names"] = json.dumps(aff_token_infos, ensure_ascii=False)[1:-1]
                    else:
                        record["aff_token_names"] = ""
                else:
                    record["aff_token_names"] = ""

                # ---- 逐样本 2D/3D 指标（与总体一致：giou_2d, ciou_2d, kld/sim/nss_2d, iou_3d=aiou20 等）----
                # 使用实际输入 decoder 的 query 覆盖旧的 hard-route aff token 记录。
                record["img_query_count"] = 0
                record["pc_query_count"] = 0
                record["img_query_fallback_count"] = 0
                record["pc_query_fallback_count"] = 0
                decoder_query_pairs = output_dict.get("decoder_query_pairs")
                if decoder_query_pairs is not None and i < len(decoder_query_pairs) and lm_head is not None:
                    query_pairs_i = decoder_query_pairs[i]
                    if query_pairs_i:
                        emb_stack = torch.stack([p[1] for p in query_pairs_i], dim=0)
                        emb_stack = emb_stack.to(device=next(lm_head.parameters()).device, dtype=next(lm_head.parameters()).dtype)
                        query_logits = lm_head(emb_stack)
                        query_ids = query_logits.argmax(dim=-1).detach().cpu().tolist()
                        query_tokens = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in query_ids]
                        query_infos = []
                        for (branch, _emb, is_fallback), tok, tid in zip(query_pairs_i, query_tokens, query_ids):
                            branch = str(branch)
                            is_fallback = bool(is_fallback)
                            if branch == "img":
                                record["img_query_count"] += 1
                                record["img_query_fallback_count"] += int(is_fallback)
                            elif branch == "pc":
                                record["pc_query_count"] += 1
                                record["pc_query_fallback_count"] += int(is_fallback)
                            query_infos.append(
                                {
                                    "branch": branch,
                                    "token": tok,
                                    "id": int(tid),
                                    "fallback": is_fallback,
                                }
                            )
                        record["aff_token_names"] = json.dumps(query_infos, ensure_ascii=False)
                    else:
                        record["aff_token_names"] = ""

                sample_metrics = compute_sample_metrics(
                    output_dict, input_dict, i,
                    threshold_2d=threshold_2d, threshold_3d=threshold_3d,
                    gt_threshold_2d=gt_threshold_2d, gt_threshold_3d=gt_threshold_3d,
                )
                for mk in (
                    "giou_2d", "inter_2d", "union_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d",
                    "iou_3d", "auc_3d", "mae_3d", "sim_3d",
                ):
                    record[mk] = sample_metrics[mk] if sample_metrics[mk] is not None else ""
                # 逐样本 ciou_2d = inter/union（与总体 ciou 定义一致）
                inter_2d, union_2d = sample_metrics.get("inter_2d"), sample_metrics.get("union_2d")
                if inter_2d is not None and union_2d is not None and union_2d > 0:
                    record["ciou_2d"] = round(inter_2d / union_2d, 6)
                else:
                    record["ciou_2d"] = ""

                sample_records.append(record)

            # 保存预测（可选）
            if infer_cfg.save_predictions and infer_cfg.output_dir:
                save_batch_predictions(
                    input_dict,
                    output_dict,
                    batch_idx,
                    infer_cfg.output_dir,
                )
            # 释放当前 batch 的 GPU 张量引用，避免长验证阶段显存碎片累积
            del output_dict, loss_dict, input_dict, pred_token_ids_batch
            if device.type == "cuda" and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

    results = compute_and_reset_torchmetrics(metrics)
    threshold_search_results = {}
    if threshold_stats is not None:
        threshold_search_results = finalize_threshold_search(threshold_stats)

    # 打印摘要（重用 log_epoch_summary 的输出格式）
    log_epoch_summary(
        logger=type("DummyLogger", (), {"info": print})(),
        epoch=1,
        total_epochs=1,
        phase="val",
        results=results,
        lr_dict=None,
    )
    if threshold_search_results:
        parts = []
        if "best_mask_threshold_2d" in threshold_search_results:
            parts.append(
                f"2D={threshold_search_results['best_mask_threshold_2d']:.4f} "
                f"(gIoU={threshold_search_results.get('best_giou_2d', 0.0):.4f}, "
                f"cIoU={threshold_search_results.get('best_ciou_2d', 0.0):.4f})"
            )
        if "best_mask_threshold_3d" in threshold_search_results:
            parts.append(
                f"3D={threshold_search_results['best_mask_threshold_3d']:.4f} "
                    f"(mIoU={threshold_search_results.get('best_miou_3d', 0.0):.4f}, "
                    f"cumIoU={threshold_search_results.get('best_cumulative_iou_3d', 0.0):.4f})"
            )
        if parts:
            print("验证集参考最优预测阈值（不写回配置）: " + ", ".join(parts))
        if threshold_search_results.get("threshold_search_2d_tie", 0.0) or threshold_search_results.get("threshold_search_3d_tie", 0.0):
            print("阈值搜索出现并列或无区分结果；对应分支没有可靠的参考最佳阈值。")

    # ---- 保存评估结果（人类可读格式）----
    out_dir = infer_cfg.output_dir if (infer_cfg.save_predictions or args.save_tsne) else "."
    os.makedirs(out_dir, exist_ok=True)
    if args.save_tsne:
        _save_tsne_artifacts(tsne_vectors, tsne_records, out_dir)

    # 1) 逐样本 CSV（按 sample_id 升序）
    sample_records.sort(key=lambda r: r["sample_id"])
    csv_fields = [
        "sample_id", "obj_type", "aff_type",
        "text_id", "img_id", "pc_id",
        "pred_token_ids", "pred_text", "gt_text",
        "aff_token_names",
        "img_query_count", "pc_query_count", "img_query_fallback_count", "pc_query_fallback_count",
        "giou_2d", "ciou_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d",
        "iou_3d", "auc_3d", "mae_3d", "sim_3d",
    ]
    csv_path = os.path.join(out_dir, "validation_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample_records)
    print(f"\n逐样本评估结果已保存到: {csv_path}")

    # 2) 按 obj-aff / obj / aff / 总体 聚合指标，便于分析各标签表现
    label_agg = _aggregate_by_label(sample_records)

    def _row_for_csv(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v if v is not None else "") for k, v in d.items()}

    # 2a) 按 obj-aff 的 CSV（便于表格分析）
    obj_aff_rows = []
    for obj in sorted(label_agg["by_obj_aff"].keys()):
        for aff in sorted(label_agg["by_obj_aff"][obj].keys()):
            row = {"obj_type": obj, "aff_type": aff, **_row_for_csv(label_agg["by_obj_aff"][obj][aff])}
            obj_aff_rows.append(row)
    obj_aff_csv = os.path.join(out_dir, "validation_by_obj_aff.csv")
    if obj_aff_rows:
        with open(obj_aff_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["obj_type", "aff_type", "giou_2d", "ciou_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d", "iou_3d", "auc_3d", "mae_3d", "sim_3d", "n_2d", "n_3d"])
            writer.writeheader()
            writer.writerows(obj_aff_rows)
        print(f"按 obj-aff 聚合结果已保存到: {obj_aff_csv}")

    # 2b) 按 obj 的 CSV
    obj_rows = [{"obj_type": k, **_row_for_csv(v)} for k, v in sorted(label_agg["by_obj"].items())]
    obj_csv = os.path.join(out_dir, "validation_by_obj.csv")
    if obj_rows:
        with open(obj_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["obj_type", "giou_2d", "ciou_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d", "iou_3d", "auc_3d", "mae_3d", "sim_3d", "n_2d", "n_3d"])
            writer.writeheader()
            writer.writerows(obj_rows)
        print(f"按 obj 聚合结果已保存到: {obj_csv}")

    # 2c) 按 aff 的 CSV
    aff_rows = [{"aff_type": k, **_row_for_csv(v)} for k, v in sorted(label_agg["by_aff"].items())]
    aff_csv = os.path.join(out_dir, "validation_by_aff.csv")
    if aff_rows:
        with open(aff_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["aff_type", "giou_2d", "ciou_2d", "p50_2d", "p50_95_2d", "kld_2d", "sim_2d", "nss_2d", "iou_3d", "auc_3d", "mae_3d", "sim_3d", "n_2d", "n_3d"])
            writer.writeheader()
            writer.writerows(aff_rows)
        print(f"按 aff 聚合结果已保存到: {aff_csv}")

    # 3) 汇总指标 JSON（含 label 聚合）
    json_path = os.path.join(out_dir, "validation_results.json")
    results_with_labels = {**results, "by_label": label_agg}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_with_labels, f, indent=2, ensure_ascii=False)
    print(f"汇总评估指标（含 label 聚合）已保存到: {json_path}")


if __name__ == "__main__":
    main()

