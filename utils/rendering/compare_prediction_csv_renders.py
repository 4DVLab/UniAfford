"""
根据多个 validation_samples.csv 对齐预测结果，并按输入样本逐行保存渲染对比图。

对齐规则：
- 只能使用 CSV 中的原始 ``img_id`` / ``pc_id`` 关联不同 run。
- 预测文件仍按 validate.py 保存时的 ``sample_id`` 查找。
- 原始 RGB、原始点云和 GT mask 均从 ``--dataset-root`` 指定的数据集目录读取。
"""

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.base_dataset import Modality  # noqa: E402
from utils.rendering.batch_render import (  # noqa: E402
    _fit_image_to_cell,
    _render_image_overlay_affordance_r1,
    _render_point_cloud_static,
)


@dataclass(frozen=True)
class PredictionRun:
    """描述一个预测 run。

    Args:
        name: 对比图中的列名。
        csv_path: validation_samples.csv 路径。
        prediction_root: validate.py 保存预测结果的目录。
        threshold_2d: 当前 run 的 2D 预测二值化阈值。
        threshold_3d: 当前 run 的 3D 预测二值化阈值。
    """

    name: str
    csv_path: str
    prediction_root: str
    threshold_2d: float = 0.5
    threshold_3d: float = 0.5


@dataclass(frozen=True)
class RenderKey:
    """跨 run 对齐用的原始数据键。

    Args:
        obj_type: 归一化后的 object 类型。
        aff_type: 归一化后的 affordance 类型。
        source_id: 原始 img_id 或 pc_id。
    """

    obj_type: str
    aff_type: str
    source_id: int


def _normalize_label(value: Any) -> str:
    """归一化 object/affordance 名称。

    Args:
        value: 待归一化值。

    Returns:
        归一化后的标签。
    """

    return Modality._normalize_label(str(value or "").strip())


def _safe_name(value: str) -> str:
    """生成可用于文件名的安全字符串。

    Args:
        value: 原始字符串。

    Returns:
        文件名安全字符串。
    """

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe.strip("_") or "item"


def _is_missing(value: Any) -> bool:
    """判断 CSV 单元格是否为空。

    Args:
        value: CSV 单元格内容。

    Returns:
        是否为空值。
    """

    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "null", "nan", "na"}


def _parse_int_id(value: Any, *, csv_path: str, row_idx: int, column: str) -> Optional[int]:
    """解析原始 img_id/pc_id，缺失时只报警告，不回退 sample_id。

    Args:
        value: CSV 单元格内容。
        csv_path: 当前 CSV 路径。
        row_idx: CSV 行号。
        column: 列名。

    Returns:
        解析得到的整数 ID；失败返回 None。
    """

    if _is_missing(value):
        warnings.warn(f"{csv_path}: row={row_idx} 缺少原始 {column}，无法跨 run 对齐，已跳过。")
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            warnings.warn(f"{csv_path}: row={row_idx} 的 {column}={text!r} 不是有效整数 ID，已跳过。")
            return None
        if not number.is_integer():
            warnings.warn(f"{csv_path}: row={row_idx} 的 {column}={text!r} 不是整数 ID，已跳过。")
            return None
        return int(number)


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    """读取 validation_samples.csv。

    Args:
        csv_path: CSV 文件路径。

    Returns:
        CSV 行字典列表。
    """

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {csv_path}")
        return [dict(row) for row in reader]


def _build_row_index(csv_path: str, modality: str, rows: Sequence[Dict[str, str]]) -> Dict[RenderKey, Dict[str, str]]:
    """用原始 ID 建立 CSV 行索引。

    Args:
        csv_path: CSV 文件路径。
        modality: ``image`` 或 ``point``。
        rows: CSV 行列表。

    Returns:
        RenderKey 到 CSV 行的映射。
    """

    id_column = "img_id" if modality == "image" else "pc_id"
    index: Dict[RenderKey, Dict[str, str]] = {}
    for row_idx, row in enumerate(rows, start=2):
        obj_type = _normalize_label(row.get("obj_type"))
        aff_type = _normalize_label(row.get("aff_type"))
        if not obj_type or not aff_type:
            warnings.warn(f"{csv_path}: row={row_idx} 缺少 obj_type 或 aff_type，已跳过。")
            continue
        source_id = _parse_int_id(row.get(id_column), csv_path=csv_path, row_idx=row_idx, column=id_column)
        if source_id is None:
            continue
        if _is_missing(row.get("sample_id")):
            warnings.warn(f"{csv_path}: row={row_idx} 缺少 sample_id，无法定位预测文件，已跳过。")
            continue
        key = RenderKey(obj_type=obj_type, aff_type=aff_type, source_id=source_id)
        if key in index:
            warnings.warn(f"{csv_path}: row={row_idx} 出现重复原始键 {key}，保留首次出现记录。")
            continue
        index[key] = row
    return index


def _choose_keys(indices: Sequence[Dict[RenderKey, Dict[str, str]]], join: str) -> List[RenderKey]:
    """按 join 策略选择要输出的原始样本。

    Args:
        indices: 每个 run 的索引。
        join: ``reference``、``inner`` 或 ``outer``。

    Returns:
        RenderKey 列表。
    """

    if not indices:
        return []
    if join == "reference":
        return list(indices[0].keys())
    key_sets = [set(index.keys()) for index in indices]
    if join == "inner":
        keys = set.intersection(*key_sets) if key_sets else set()
    elif join == "outer":
        keys = set.union(*key_sets) if key_sets else set()
    else:
        raise ValueError(f"未知 join 策略: {join}")
    return sorted(keys, key=lambda item: (item.obj_type, item.aff_type, item.source_id))


def _find_real_obj_dir(root: str, obj_type: str) -> str:
    """在 root 下找到与 obj_type 归一化后匹配的真实目录名。

    Args:
        root: 数据集或预测结果根目录。
        obj_type: object 类型。

    Returns:
        真实目录名；找不到时返回 obj_type。
    """

    obj_norm = _normalize_label(obj_type)
    if not os.path.isdir(root):
        return obj_type
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and _normalize_label(name) == obj_norm:
            return name
    return obj_type


def _find_original_rgb_path(dataset_root: str, obj_type: str, img_id: int) -> Optional[str]:
    """查找原始 RGB 图片路径。

    Args:
        dataset_root: 原始数据集目录。
        obj_type: object 类型。
        img_id: 原始图片 ID。

    Returns:
        RGB 文件路径，找不到则返回 None。
    """

    real_obj = _find_real_obj_dir(dataset_root, obj_type)
    rgb_dir = os.path.join(dataset_root, real_obj, "Image", "rgb")
    prefixes = list(dict.fromkeys([obj_type, real_obj, _normalize_label(obj_type)]))
    for prefix in prefixes:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(rgb_dir, f"{prefix}_{img_id}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _find_original_mask_path(dataset_root: str, obj_type: str, aff_type: str, img_id: int) -> Optional[str]:
    """查找原始 2D GT mask 路径。

    Args:
        dataset_root: 原始数据集目录。
        obj_type: object 类型。
        aff_type: affordance 类型。
        img_id: 原始图片 ID。

    Returns:
        GT mask 路径，找不到则返回 None。
    """

    real_obj = _find_real_obj_dir(dataset_root, obj_type)
    aff_norm = _normalize_label(aff_type)
    mask_dir = os.path.join(dataset_root, real_obj, "Image", "mask", aff_norm)
    prefixes = list(dict.fromkeys([obj_type, real_obj, _normalize_label(obj_type)]))
    for prefix in prefixes:
        candidate = os.path.join(mask_dir, f"{prefix}_{img_id}_{aff_norm}.png")
        if os.path.exists(candidate):
            return candidate
    return None


def _find_prediction_mask_path(prediction_root: str, obj_type: str, aff_type: str, sample_id: str) -> Optional[str]:
    """按 simple sample_id 查找 validate.py 保存的 2D 预测 mask。

    Args:
        prediction_root: 预测结果根目录。
        obj_type: object 类型。
        aff_type: affordance 类型。
        sample_id: 当前 run CSV 中的 simple sample_id。

    Returns:
        预测 mask 路径，找不到则返回 None。
    """

    real_obj = _find_real_obj_dir(prediction_root, obj_type)
    aff_norm = _normalize_label(aff_type)
    mask_dir = os.path.join(prediction_root, real_obj, "Image", "mask", aff_norm)
    sample_id = str(sample_id).strip()
    prefixes = list(dict.fromkeys([obj_type, real_obj, _normalize_label(obj_type)]))
    for prefix in prefixes:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(mask_dir, f"{prefix}_{sample_id}_{aff_norm}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _find_point_csv_path(root: str, obj_type: str, item_id: Any) -> Optional[str]:
    """查找点云 CSV 路径。

    Args:
        root: 原始数据集目录或预测根目录。
        obj_type: object 类型。
        item_id: 原始 pc_id 或 simple sample_id。

    Returns:
        点云 CSV 路径，找不到则返回 None。
    """

    real_obj = _find_real_obj_dir(root, obj_type)
    pc_dir = os.path.join(root, real_obj, "PointCloud")
    item_id = str(item_id).strip()
    prefixes = list(dict.fromkeys([obj_type, real_obj, _normalize_label(obj_type)]))
    for prefix in prefixes:
        candidate = os.path.join(pc_dir, f"{prefix}_{item_id}.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def _read_point_csv(path: str, aff_type: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读取点云 CSV，并可选读取某个 affordance mask 列。

    Args:
        path: 点云 CSV 路径。
        aff_type: 需要读取的 affordance 列；为 None 时只返回点坐标。

    Returns:
        ``(points, mask)``，mask 可能为 None。
    """

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().lstrip("#").split(",")
        data = np.loadtxt(f, delimiter=",")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    if data.shape[1] < 3:
        raise ValueError(f"点云 CSV 至少需要 x,y,z 三列: {path}")
    if aff_type is None:
        return data[:, :3], None
    aff_norm = _normalize_label(aff_type)
    labels = [_normalize_label(label) for label in header[3:]]
    if aff_norm not in labels:
        raise ValueError(f"点云 CSV 中找不到 aff 列: aff={aff_norm}, file={path}")
    mask_idx = labels.index(aff_norm)
    return data[:, :3], data[:, 3 + mask_idx]


def _mask_to_prob(mask: np.ndarray) -> np.ndarray:
    """将 PNG/CSV mask 统一归一化到 [0, 1] 概率范围。

    Args:
        mask: 输入 mask 数组。

    Returns:
        float32 概率 mask。
    """

    prob = np.asarray(mask, dtype=np.float32)
    if prob.size > 0 and np.nanmax(prob) > 1.0:
        prob = prob / 255.0
    return np.clip(prob, 0.0, 1.0)


def _binarize_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    """按阈值将预测概率 mask 二值化。

    Args:
        mask: 概率或 uint8 mask。
        threshold: 二值化阈值。

    Returns:
        float32 二值 mask，取值为 0/1。
    """

    return (_mask_to_prob(mask) > float(threshold)).astype(np.float32)


def _binary_seg_metrics(pred_binary: np.ndarray, gt_mask: np.ndarray, gt_threshold: float = 0.5) -> Dict[str, float]:
    """计算二值预测与 GT 的通用分割指标。

    Args:
        pred_binary: 已二值化的预测 mask。
        gt_mask: GT mask，可为概率或 uint8。
        gt_threshold: GT 二值化阈值。

    Returns:
        指标字典，包含 IoU/intersection/union/MAE/SIM。
    """

    pred = np.asarray(pred_binary, dtype=np.float32).reshape(-1) > 0.5
    gt_prob = _mask_to_prob(gt_mask).reshape(-1)
    gt = gt_prob > float(gt_threshold)
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    iou = intersection / (union + 1e-8) if union > 0 else 0.0
    pred_float = pred.astype(np.float32)
    gt_float = gt.astype(np.float32)
    mae = float(np.mean(np.abs(pred_float - gt_float))) if pred_float.size else 0.0
    pred_sum = float(pred_float.sum())
    gt_sum = float(gt_float.sum())
    if pred_sum > 1e-12 and gt_sum > 1e-12:
        sim = float(np.minimum(pred_float / pred_sum, gt_float / gt_sum).sum())
    else:
        sim = 0.0
    return {
        "iou": round(iou, 6),
        "intersection": round(intersection, 6),
        "union": round(union, 6),
        "mae": round(mae, 6),
        "sim": round(sim, 6),
    }


def _image_saliency_metrics(pred_binary: np.ndarray, gt_mask: np.ndarray, gt_threshold: float = 0.5) -> Dict[str, float]:
    """计算 2D 二值预测的补充 saliency 指标。

    Args:
        pred_binary: 已二值化的 2D 预测 mask。
        gt_mask: 2D GT mask。
        gt_threshold: GT 二值化阈值。

    Returns:
        KLD、NSS 和 P@IoU 指标。
    """

    pred = np.asarray(pred_binary, dtype=np.float32)
    gt = (_mask_to_prob(gt_mask) > float(gt_threshold)).astype(np.float32)
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    pred_sum = pred_flat.sum()
    gt_sum = gt_flat.sum()
    eps = 1e-12
    pred_norm = pred_flat / (pred_sum + eps)
    gt_norm = gt_flat / (gt_sum + eps)
    kld = float((gt_norm * (np.log(gt_norm + eps) - np.log(pred_norm + eps))).sum())
    pred_std = float(pred_flat.std())
    if pred_std > 1e-8 and gt_flat.sum() > 0:
        pred_z = (pred_flat - float(pred_flat.mean())) / (pred_std + 1e-8)
        nss = float((pred_z * gt_flat).sum() / (gt_flat.sum() + 1e-8))
    else:
        nss = 0.0
    base = _binary_seg_metrics(pred, gt, gt_threshold=0.5)
    iou = base["iou"]
    p_thresholds = np.arange(0.5, 0.96, 0.05)
    p_values = (iou > p_thresholds).astype(np.float32)
    return {
        "kld": round(kld, 6),
        "nss": round(nss, 6),
        "p50": round(float(p_values[0]), 6),
        "p50_95": round(float(p_values.mean()), 6),
    }


def _load_render_config(config_path: Optional[str]) -> Dict[str, Any]:
    """加载 batch_render 风格的渲染配置。

    Args:
        config_path: 可选 JSON 配置路径。

    Returns:
        渲染配置字典。
    """

    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "docs", "render_manifest_example.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"output": data.get("output", {}), "render": data.get("render", {})}


def _placeholder(text: str, width: int, height: int, background_rgb: Sequence[int]) -> np.ndarray:
    """生成缺失项占位图。

    Args:
        text: 占位文本。
        width: 目标宽度。
        height: 目标高度。
        background_rgb: RGB 背景色。

    Returns:
        BGR 占位图。
    """

    bg_bgr = tuple(int(v) for v in reversed(background_rgb))
    image = np.full((height, width, 3), bg_bgr, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (190, 190, 190), 2)
    y = max(28, height // 2 - 12)
    for line in str(text).split("\n"):
        cv2.putText(image, line[:80], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
        y += 26
    return image


def _add_label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """给渲染单元格添加顶部标签。

    Args:
        image: BGR 图像。
        title: 主标题。
        subtitle: 副标题。

    Returns:
        带标签的 BGR 图像。
    """

    label_h = 58 if subtitle else 36
    height, width = image.shape[:2]
    canvas = np.full((height + label_h, width, 3), 255, dtype=np.uint8)
    canvas[label_h:, :] = image
    cv2.putText(canvas, title[:90], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:110], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def _render_image_cells(
    dataset_root: str,
    runs: Sequence[PredictionRun],
    indices: Sequence[Dict[RenderKey, Dict[str, str]]],
    key: RenderKey,
    render_cfg: Dict[str, Any],
    cell_size: Tuple[int, int],
    background_rgb: Sequence[int],
    gt_threshold_2d: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """渲染一个 2D 样本对应的一排单元格。

    Args:
        dataset_root: 原始数据集目录。
        runs: 预测 run 列表。
        indices: 每个 run 的 CSV 索引。
        key: 当前原始样本键。
        render_cfg: 渲染配置。
        cell_size: 占位图尺寸。
        background_rgb: RGB 背景色。
        gt_threshold_2d: 2D GT 二值化阈值。

    Returns:
        ``(cells, manifest_row)``。
    """

    cell_width, cell_height = cell_size
    image_cfg = render_cfg.get("image", {})
    alpha = float(image_cfg.get("alpha", 0.5))
    color_rgb = tuple(image_cfg.get("color_rgb", [255, 0, 0]))

    rgb_path = _find_original_rgb_path(dataset_root, key.obj_type, key.source_id)
    gt_mask_path = _find_original_mask_path(dataset_root, key.obj_type, key.aff_type, key.source_id)
    row_info: Dict[str, Any] = {
        "modality": "image",
        "obj_type": key.obj_type,
        "aff_type": key.aff_type,
        "img_id": key.source_id,
        "rgb_path": rgb_path,
        "gt_mask_path": gt_mask_path,
        "predictions": [],
    }

    if rgb_path is None:
        warnings.warn(f"image: 找不到原始 RGB: {key}")
        base_img = _placeholder("missing RGB", cell_width, cell_height, background_rgb)
    else:
        base_img = cv2.imread(rgb_path)
        if base_img is None:
            warnings.warn(f"image: 读取原始 RGB 失败: {rgb_path}")
            base_img = _placeholder("bad RGB", cell_width, cell_height, background_rgb)

    cells = [_add_label(base_img, "Input", f"{key.obj_type} | img_id={key.source_id}")]
    if gt_mask_path is None or rgb_path is None:
        if gt_mask_path is None:
            warnings.warn(f"image: 找不到 GT mask: {key}")
        gt_cell = _placeholder("missing GT", cell_width, cell_height, background_rgb)
    else:
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_cell = _render_image_overlay_affordance_r1(base_img, gt_mask, alpha=alpha, color_rgb=color_rgb, threshold=gt_threshold_2d)
    cells.append(_add_label(gt_cell, "GT", f"{key.aff_type}"))

    for run, index in zip(runs, indices):
        row = index.get(key)
        pred_path = None
        sample_id = None
        metrics: Dict[str, float] = {}
        if row is None:
            warnings.warn(f"image: {run.name} 缺少对齐记录 {key}")
            pred_cell = _placeholder("missing row", cell_width, cell_height, background_rgb)
        else:
            sample_id = str(row.get("sample_id", "")).strip()
            pred_path = _find_prediction_mask_path(run.prediction_root, row.get("obj_type") or key.obj_type, row.get("aff_type") or key.aff_type, sample_id)
            if pred_path is None or rgb_path is None:
                warnings.warn(f"image: {run.name} 找不到预测 mask: key={key}, sample_id={sample_id}")
                pred_cell = _placeholder("missing pred", cell_width, cell_height, background_rgb)
            else:
                pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                pred_binary = _binarize_mask(pred_mask, run.threshold_2d)
                pred_cell = _render_image_overlay_affordance_r1(
                    base_img,
                    pred_binary,
                    alpha=alpha,
                    color_rgb=color_rgb,
                    threshold=0.5,
                )
                if gt_mask_path is not None:
                    gt_for_metrics = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    metrics = _binary_seg_metrics(pred_binary, gt_for_metrics, gt_threshold=gt_threshold_2d)
                    metrics.update(_image_saliency_metrics(pred_binary, gt_for_metrics, gt_threshold=gt_threshold_2d))
                else:
                    metrics = {}
        row_info["predictions"].append(
            {
                "run": run.name,
                "sample_id": sample_id,
                "mask_path": pred_path,
                "threshold_2d": run.threshold_2d,
                "metrics": metrics if row is not None and pred_path is not None else {},
            }
        )
        title = run.name
        if row is not None and pred_path is not None and metrics:
            title = f"{run.name} IoU={metrics['iou']:.3f}"
        cells.append(_add_label(pred_cell, title, f"sample_id={sample_id or 'missing'} | th={run.threshold_2d:.3f}"))
    return cells, row_info


def _render_point_cells(
    dataset_root: str,
    runs: Sequence[PredictionRun],
    indices: Sequence[Dict[RenderKey, Dict[str, str]]],
    key: RenderKey,
    render_cfg: Dict[str, Any],
    cell_size: Tuple[int, int],
    background_rgb: Sequence[int],
    gt_threshold_3d: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """渲染一个 3D 样本对应的一排单元格。

    Args:
        dataset_root: 原始数据集目录。
        runs: 预测 run 列表。
        indices: 每个 run 的 CSV 索引。
        key: 当前原始样本键。
        render_cfg: 渲染配置。
        cell_size: 占位图尺寸。
        background_rgb: RGB 背景色。
        gt_threshold_3d: 3D GT 二值化阈值。

    Returns:
        ``(cells, manifest_row)``。
    """

    cell_width, cell_height = cell_size
    point_cfg = render_cfg.get("point_cloud", {})
    point_backend = str(point_cfg.get("backend", "matplotlib")).lower()
    if point_backend in {"iagnet", "mitsuba", "mitsuba_iagnet"}:
        # 这里按“已配对路径 -> 直接渲染单元格”的方式工作，无法复用 IAGNet 的 manifest 批处理后端。
        point_backend = "matplotlib"
    point_kwargs = {
        "size": int(point_cfg.get("size", 800)),
        "elev": float(point_cfg.get("elev", 30)),
        "azim": float(point_cfg.get("azim", 45)),
        "point_size": float(point_cfg.get("point_size", 10)),
        "max_points": point_cfg.get("max_points", 6000),
        "backend": point_backend,
        "sphere_radius": float(point_cfg.get("sphere_radius", 0.024)),
        "sphere_resolution": int(point_cfg.get("sphere_resolution", 8)),
        "background_rgb": tuple(point_cfg.get("background", [255, 255, 255])),
        "ground_plane": bool(point_cfg.get("ground_plane", True)),
        "base_color_rgb": tuple(point_cfg.get("base_color_rgb", [190, 190, 190])),
        "affordance_color_rgb": tuple(point_cfg.get("affordance_color_rgb", [255, 0, 0])),
        "fov": float(point_cfg.get("fov", 25)),
        "camera_origin": tuple(point_cfg.get("camera_origin", [3, 3, 3])),
    }

    original_pc_path = _find_point_csv_path(dataset_root, key.obj_type, key.source_id)
    row_info: Dict[str, Any] = {
        "modality": "point",
        "obj_type": key.obj_type,
        "aff_type": key.aff_type,
        "pc_id": key.source_id,
        "gt_point_path": original_pc_path,
        "predictions": [],
    }

    if original_pc_path is None:
        warnings.warn(f"point: 找不到原始点云 CSV: {key}")
        input_cell = _placeholder("missing point", cell_width, cell_height, background_rgb)
        gt_cell = _placeholder("missing GT", cell_width, cell_height, background_rgb)
    else:
        try:
            points, gt_mask = _read_point_csv(original_pc_path, key.aff_type)
            zero_mask = np.zeros(points.shape[0], dtype=np.float32)
            input_cell = _render_point_cloud_static(points, zero_mask, **point_kwargs)
            gt_cell = _render_point_cloud_static(points, gt_mask, **point_kwargs)
        except Exception as exc:
            warnings.warn(f"point: 渲染原始点云/GT 失败 {key}: {exc}")
            input_cell = _placeholder("bad point", cell_width, cell_height, background_rgb)
            gt_cell = _placeholder("bad GT", cell_width, cell_height, background_rgb)

    cells = [
        _add_label(input_cell, "Input", f"{key.obj_type} | pc_id={key.source_id}"),
        _add_label(gt_cell, "GT", f"{key.aff_type}"),
    ]
    for run, index in zip(runs, indices):
        row = index.get(key)
        pred_path = None
        sample_id = None
        metrics: Dict[str, float] = {}
        if row is None:
            warnings.warn(f"point: {run.name} 缺少对齐记录 {key}")
            pred_cell = _placeholder("missing row", cell_width, cell_height, background_rgb)
        else:
            sample_id = str(row.get("sample_id", "")).strip()
            pred_path = _find_point_csv_path(run.prediction_root, row.get("obj_type") or key.obj_type, sample_id)
            if pred_path is None:
                warnings.warn(f"point: {run.name} 找不到预测点云 CSV: key={key}, sample_id={sample_id}")
                pred_cell = _placeholder("missing pred", cell_width, cell_height, background_rgb)
            else:
                try:
                    pred_points, pred_mask = _read_point_csv(pred_path, row.get("aff_type") or key.aff_type)
                    pred_binary = _binarize_mask(pred_mask, run.threshold_3d)
                    pred_cell = _render_point_cloud_static(pred_points, pred_binary, **point_kwargs)
                    if original_pc_path is not None:
                        _, gt_for_metrics = _read_point_csv(original_pc_path, key.aff_type)
                        if gt_for_metrics is not None and gt_for_metrics.shape[0] == pred_binary.shape[0]:
                            metrics = _binary_seg_metrics(pred_binary, gt_for_metrics, gt_threshold=gt_threshold_3d)
                        else:
                            warnings.warn(
                                f"point: {run.name} 预测点数与 GT 点数不一致，跳过指标: "
                                f"key={key}, sample_id={sample_id}"
                            )
                except Exception as exc:
                    warnings.warn(f"point: {run.name} 渲染预测失败 {key}, sample_id={sample_id}: {exc}")
                    pred_cell = _placeholder("bad pred", cell_width, cell_height, background_rgb)
        row_info["predictions"].append(
            {
                "run": run.name,
                "sample_id": sample_id,
                "point_path": pred_path,
                "threshold_3d": run.threshold_3d,
                "metrics": metrics,
            }
        )
        title = run.name
        if metrics:
            title = f"{run.name} IoU={metrics['iou']:.3f}"
        cells.append(_add_label(pred_cell, title, f"sample_id={sample_id or 'missing'} | th={run.threshold_3d:.3f}"))
    return cells, row_info


def _save_row_image(cells: Sequence[np.ndarray], output_path: str, grid_cfg: Dict[str, Any]) -> None:
    """将一组单元格横向拼成一排并保存。

    Args:
        cells: BGR 单元格图像。
        output_path: 输出路径。
        grid_cfg: batch_render 风格 grid 配置。
    """

    cell_width = int(grid_cfg.get("cell_width", 800))
    cell_height = int(grid_cfg.get("cell_height", 800))
    padding = int(grid_cfg.get("padding", 12))
    bg_rgb = tuple(int(v) for v in grid_cfg.get("background", [255, 255, 255]))
    background = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
    canvas_h = cell_height + padding * 2
    canvas_w = len(cells) * cell_width + (len(cells) + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)
    for col_idx, image in enumerate(cells):
        x0 = padding + col_idx * (cell_width + padding)
        cell = _fit_image_to_cell(image, cell_width, cell_height, background)
        canvas[padding:padding + cell_height, x0:x0 + cell_width] = cell
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def _save_metrics_csv(rows: Sequence[Dict[str, Any]], output_path: str, modality: str) -> None:
    """将 paired manifest 中的逐 run 指标展开为 CSV。

    Args:
        rows: paired_paths JSON 中的行记录。
        output_path: CSV 输出路径。
        modality: ``image`` 或 ``point``。
    """

    id_key = "img_id" if modality == "image" else "pc_id"
    metric_keys = ["iou", "intersection", "union", "mae", "sim", "kld", "nss", "p50", "p50_95"]
    path_key = "mask_path" if modality == "image" else "point_path"
    threshold_key = "threshold_2d" if modality == "image" else "threshold_3d"
    fieldnames = [
        "modality",
        "obj_type",
        "aff_type",
        id_key,
        "run",
        "sample_id",
        threshold_key,
        path_key,
        "output_path",
    ] + metric_keys
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for pred in row.get("predictions", []):
                out = {
                    "modality": modality,
                    "obj_type": row.get("obj_type"),
                    "aff_type": row.get("aff_type"),
                    id_key: row.get(id_key),
                    "run": pred.get("run"),
                    "sample_id": pred.get("sample_id"),
                    threshold_key: pred.get(threshold_key),
                    path_key: pred.get(path_key),
                    "output_path": row.get("output_path"),
                }
                metrics = pred.get("metrics") or {}
                for key in metric_keys:
                    out[key] = metrics.get(key)
                writer.writerow(out)


def _split_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔参数。

    Args:
        value: 参数字符串。

    Returns:
        字符串列表或 None。
    """

    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_sequence(value: Optional[str], count: int, default: float, name: str) -> List[float]:
    """解析逗号分隔的浮点序列。

    Args:
        value: 参数字符串。
        count: 期望数量。
        default: 未提供参数时使用的默认值。
        name: 参数名，用于报错。

    Returns:
        浮点数列表。
    """

    if not value:
        return [float(default)] * count
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != count:
        raise ValueError(f"{name} 数量必须与 CSV 数量一致: got={len(items)}, expected={count}")
    return [float(item) for item in items]


def _build_runs(
    csv_files: Sequence[str],
    method_names: Optional[Sequence[str]],
    roots: Optional[Sequence[str]],
    thresholds_2d: Sequence[float],
    thresholds_3d: Sequence[float],
) -> List[PredictionRun]:
    """构建预测 run 列表。

    Args:
        csv_files: CSV 文件列表。
        method_names: 可选显示名。
        roots: 可选预测根目录。
        thresholds_2d: 每个 run 的 2D 二值化阈值。
        thresholds_3d: 每个 run 的 3D 二值化阈值。

    Returns:
        PredictionRun 列表。
    """

    if method_names is not None and len(method_names) != len(csv_files):
        raise ValueError("--method-names 数量必须与 CSV 数量一致")
    if roots is not None and len(roots) != len(csv_files):
        raise ValueError("--prediction-roots 数量必须与 CSV 数量一致")
    if len(thresholds_2d) != len(csv_files) or len(thresholds_3d) != len(csv_files):
        raise ValueError("thresholds_2d/thresholds_3d 数量必须与 CSV 数量一致")

    runs: List[PredictionRun] = []
    for idx, csv_path in enumerate(csv_files):
        abs_csv = os.path.abspath(csv_path)
        name = method_names[idx] if method_names else os.path.splitext(os.path.basename(os.path.dirname(abs_csv)))[0]
        root = os.path.abspath(roots[idx]) if roots else os.path.dirname(abs_csv)
        runs.append(
            PredictionRun(
                name=name or f"run_{idx}",
                csv_path=abs_csv,
                prediction_root=root,
                threshold_2d=float(thresholds_2d[idx]),
                threshold_3d=float(thresholds_3d[idx]),
            )
        )
    return runs


def render_comparison(
    dataset_root: str,
    runs: Sequence[PredictionRun],
    output_dir: str,
    modality: str,
    render_config: Dict[str, Any],
    join: str = "reference",
    max_rows: Optional[int] = None,
    gt_threshold_2d: float = 0.5,
    gt_threshold_3d: float = 0.5,
) -> Dict[str, List[str]]:
    """渲染并保存对比结果。

    Args:
        dataset_root: 原始数据集目录。
        runs: 预测 run 列表。
        output_dir: 输出目录。
        modality: ``image``、``point`` 或 ``both``。
        render_config: 渲染配置。
        join: 对齐策略。
        max_rows: 最大输出样本数。
        gt_threshold_2d: 2D GT 二值化阈值。
        gt_threshold_3d: 3D GT 二值化阈值。

    Returns:
        模态到输出图片列表的映射。
    """

    dataset_root = os.path.abspath(dataset_root)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    modalities = ["image", "point"] if modality == "both" else [modality]
    rows_by_run = {run.name: _read_csv_rows(run.csv_path) for run in runs}
    grid_cfg = dict(render_config.get("output", {}).get("grid", {}))
    render_cfg = render_config.get("render", {})
    bg_rgb = tuple(int(v) for v in grid_cfg.get("background", [255, 255, 255]))
    cell_size = (int(grid_cfg.get("cell_width", 800)), int(grid_cfg.get("cell_height", 800)))

    saved: Dict[str, List[str]] = {}
    manifests: Dict[str, List[Dict[str, Any]]] = {}
    for one_modality in modalities:
        indices = [_build_row_index(run.csv_path, one_modality, rows_by_run[run.name]) for run in runs]
        keys = _choose_keys(indices, join=join)
        if max_rows is not None:
            keys = keys[:max(0, int(max_rows))]
        if not keys:
            warnings.warn(f"{one_modality}: 没有可对齐的原始 ID 记录。")
            continue

        saved[one_modality] = []
        manifests[one_modality] = []
        row_dir = os.path.join(output_dir, f"{one_modality}_rows")
        for key in keys:
            if one_modality == "image":
                cells, row_info = _render_image_cells(
                    dataset_root,
                    runs,
                    indices,
                    key,
                    render_cfg,
                    cell_size,
                    bg_rgb,
                    gt_threshold_2d=gt_threshold_2d,
                )
                name = _safe_name(f"{key.obj_type}_{key.aff_type}_img{key.source_id}")
            else:
                cells, row_info = _render_point_cells(
                    dataset_root,
                    runs,
                    indices,
                    key,
                    render_cfg,
                    cell_size,
                    bg_rgb,
                    gt_threshold_3d=gt_threshold_3d,
                )
                name = _safe_name(f"{key.obj_type}_{key.aff_type}_pc{key.source_id}")
            output_path = os.path.join(row_dir, f"{name}.jpg")
            _save_row_image(cells, output_path, grid_cfg)
            row_info["output_path"] = output_path
            saved[one_modality].append(output_path)
            manifests[one_modality].append(row_info)

        manifest_path = os.path.join(output_dir, f"paired_paths_{one_modality}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifests[one_modality], f, ensure_ascii=False, indent=2)
        _save_metrics_csv(
            manifests[one_modality],
            os.path.join(output_dir, f"metrics_{one_modality}.csv"),
            one_modality,
        )
    return saved


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace。
    """

    parser = argparse.ArgumentParser(description="Compare prediction masks from validation_samples.csv files.")
    parser.add_argument("csv_files", nargs="+", help="多个 validation_samples.csv 路径。")
    parser.add_argument("--dataset-root", required=True, help="原始数据集目录，用于读取 RGB/点云和 GT mask。")
    parser.add_argument("--output-dir", required=True, help="对比图输出目录。")
    parser.add_argument("--method-names", default=None, help="逗号分隔的列名，数量需与 CSV 一致。")
    parser.add_argument("--prediction-roots", default=None, help="逗号分隔的预测根目录；默认使用每个 CSV 的父目录。")
    parser.add_argument("--thresholds-2d", default=None, help="逗号分隔的 2D 预测二值化阈值，数量需与 CSV 一致；默认全为 0.5。")
    parser.add_argument("--thresholds-3d", default=None, help="逗号分隔的 3D 预测二值化阈值，数量需与 CSV 一致；默认全为 0.5。")
    parser.add_argument("--gt-threshold-2d", type=float, default=0.5, help="2D GT 二值化阈值。")
    parser.add_argument("--gt-threshold-3d", type=float, default=0.5, help="3D GT 二值化阈值。")
    parser.add_argument("--modality", default="both", choices=("image", "point", "both"), help="渲染模态。")
    parser.add_argument(
        "--join",
        default="reference",
        choices=("reference", "inner", "outer"),
        help="对齐策略：reference 使用第一个 CSV 的原始 ID 顺序；inner 只保留交集；outer 输出并集。",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="最多输出多少个原始 ID 样本。")
    parser.add_argument("--render-config", default=None, help="复用 batch_render 风格的渲染 JSON 配置。")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    thresholds_2d = _parse_float_sequence(args.thresholds_2d, len(args.csv_files), 0.5, "--thresholds-2d")
    thresholds_3d = _parse_float_sequence(args.thresholds_3d, len(args.csv_files), 0.5, "--thresholds-3d")
    runs = _build_runs(
        csv_files=args.csv_files,
        method_names=_split_csv_arg(args.method_names),
        roots=_split_csv_arg(args.prediction_roots),
        thresholds_2d=thresholds_2d,
        thresholds_3d=thresholds_3d,
    )
    render_config = _load_render_config(args.render_config)
    saved = render_comparison(
        dataset_root=args.dataset_root,
        runs=runs,
        output_dir=args.output_dir,
        modality=args.modality,
        render_config=render_config,
        join=args.join,
        max_rows=args.max_rows,
        gt_threshold_2d=args.gt_threshold_2d,
        gt_threshold_3d=args.gt_threshold_3d,
    )
    for mode, paths in saved.items():
        print(f"{mode}: saved {len(paths)} row images")
        if paths:
            print(f"  first: {paths[0]}")


if __name__ == "__main__":
    main()
