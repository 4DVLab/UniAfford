"""Utilities for selecting prediction binarization thresholds on validation data."""

from typing import Dict, Iterable

import torch
import torch.distributed as dist

from utils import calculator as calc


def build_threshold_candidates(
    device: torch.device,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.05,
    extra_thresholds: Iterable[float] = None,
) -> torch.Tensor:
    """Build candidate prediction thresholds. GT thresholds are controlled separately."""
    start = float(start)
    end = float(end)
    step = float(step)
    if step <= 0:
        step = 0.05
    if end < start:
        start, end = end, start
    count = int(round((end - start) / step)) + 1
    values = start + torch.arange(count, device=device, dtype=torch.float32) * step
    if extra_thresholds is not None:
        extra = [
            float(v) for v in extra_thresholds
            if v is not None and 0.0 <= float(v) <= 1.0
        ]
        if extra:
            values = torch.cat([values, torch.tensor(extra, device=device, dtype=torch.float32)])
    return values.clamp(0.0, 1.0).unique(sorted=True)


def init_threshold_search_stats(thresholds: torch.Tensor) -> Dict[str, torch.Tensor]:
    zeros = torch.zeros_like(thresholds, dtype=torch.float32)
    return {
        "thresholds": thresholds,
        "sum_iou_2d": zeros.clone(),
        "sum_inter_2d": zeros.clone(),
        "sum_union_2d": zeros.clone(),
        "count_2d": torch.zeros((), device=thresholds.device, dtype=torch.float32),
        "sum_iou_3d": zeros.clone(),
        "sum_inter_3d": zeros.clone(),
        "sum_union_3d": zeros.clone(),
        "count_3d": torch.zeros((), device=thresholds.device, dtype=torch.float32),
    }


def sync_threshold_search_stats(stats: Dict[str, torch.Tensor]) -> None:
    """Synchronize accumulated threshold-search stats across distributed ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for key, value in stats.items():
        if key == "thresholds":
            continue
        if torch.is_tensor(value):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)


@torch.no_grad()
def update_threshold_search_stats(
    stats: Dict[str, torch.Tensor],
    output_dict: Dict,
    input_dict: Dict,
    config,
) -> None:
    """Accumulate validation IoU under each candidate prediction threshold."""
    thresholds = stats["thresholds"]

    image_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    if image_logits is not None and img_gt is not None:
        aligned_logits, aligned_gt = calc._align_ordered_query_masks(
            image_logits.detach(),
            img_gt,
            sample_valid_mask=input_dict.get("img_valid_mask"),
            gt_valid_mask=input_dict.get("img_gt_valid_mask"),
        )
        if aligned_logits is not None:
            preds = aligned_logits.sigmoid()
            target = aligned_gt.float()
            gt_threshold = float(getattr(config, "gt_threshold_2d", 0.5))
            batch_size = preds.shape[0]
            for idx, threshold in enumerate(thresholds):
                inter, union = calc.img_I_and_U(
                    preds,
                    target,
                    threshold=float(threshold.item()),
                    gt_threshold=gt_threshold,
                )
                iou = inter / (union + 1e-8)
                stats["sum_iou_2d"][idx] += iou.sum()
                stats["sum_inter_2d"][idx] += inter.sum()
                stats["sum_union_2d"][idx] += union.sum()
            stats["count_2d"] += batch_size

    point_logits = output_dict.get("point_logits")
    pc_gt = input_dict.get("pc_gt_tensor")
    if point_logits is not None and pc_gt is not None:
        valid_lengths = input_dict.get("pc_valid_lengths")
        aligned_logits, aligned_gt = calc._align_ordered_query_masks(
            point_logits.detach(),
            pc_gt,
            sample_valid_mask=None if valid_lengths is None else valid_lengths > 0,
            gt_valid_mask=input_dict.get("pc_gt_valid_mask"),
        )
        if aligned_logits is not None:
            preds = aligned_logits.sigmoid()
            target = aligned_gt.float()
            gt_threshold = float(getattr(config, "gt_threshold_3d", 0.5))
            batch_size = preds.shape[0]
            for idx, threshold in enumerate(thresholds):
                inter, union = calc.pc_I_and_U(
                    preds,
                    target,
                    threshold=float(threshold.item()),
                    gt_threshold=gt_threshold,
                )
                iou = inter / (union + 1e-8)
                stats["sum_iou_3d"][idx] += iou.sum()
                stats["sum_inter_3d"][idx] += inter.sum()
                stats["sum_union_3d"][idx] += union.sum()
            stats["count_3d"] += batch_size


def finalize_threshold_search(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return best thresholds by validation mean per-sample IoU."""
    sync_threshold_search_stats(stats)
    results = {}
    thresholds = stats["thresholds"]
    if stats["count_2d"].item() > 0:
        mean_iou = stats["sum_iou_2d"] / stats["count_2d"].clamp_min(1.0)
        if mean_iou.numel() == 1 or (mean_iou.max() - mean_iou.min()).item() > 1e-8:
            best_idx = int(torch.argmax(mean_iou).item())
            results["best_mask_threshold_2d"] = float(thresholds[best_idx].item())
            results["best_giou_2d"] = float(mean_iou[best_idx].item())
            inter = stats["sum_inter_2d"][best_idx]
            union = stats["sum_union_2d"][best_idx]
            results["best_ciou_2d"] = float((inter / (union + 1e-8)).item()) if union.item() > 0 else 0.0
        else:
            results["threshold_search_2d_tie"] = 1.0
    if stats["count_3d"].item() > 0:
        mean_iou = stats["sum_iou_3d"] / stats["count_3d"].clamp_min(1.0)
        if mean_iou.numel() == 1 or (mean_iou.max() - mean_iou.min()).item() > 1e-8:
            best_idx = int(torch.argmax(mean_iou).item())
            results["best_mask_threshold_3d"] = float(thresholds[best_idx].item())
            # This is the 3D mIoU used for threshold selection:
            # compute point-wise IoU per sample, then average over samples.
            results["best_miou_3d"] = float(mean_iou[best_idx].item())
            inter = stats["sum_inter_3d"][best_idx]
            union = stats["sum_union_3d"][best_idx]
            results["best_cumulative_iou_3d"] = float((inter / (union + 1e-8)).item()) if union.item() > 0 else 0.0
        else:
            results["threshold_search_3d_tie"] = 1.0
    return results
