"""Utilities for selecting prediction binarization thresholds on validation data."""

from typing import Dict

import torch

from utils import calculator as calc


def build_threshold_candidates(
    device: torch.device,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.05,
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
    return values.clamp(0.0, 1.0).unique(sorted=True)


def init_threshold_search_stats(thresholds: torch.Tensor) -> Dict[str, torch.Tensor]:
    zeros = torch.zeros_like(thresholds, dtype=torch.float32)
    return {
        "thresholds": thresholds,
        "sum_iou_2d": zeros.clone(),
        "count_2d": torch.zeros((), device=thresholds.device, dtype=torch.float32),
        "sum_iou_3d": zeros.clone(),
        "count_3d": torch.zeros((), device=thresholds.device, dtype=torch.float32),
    }


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
            batch_size = preds.shape[0]
            gt_threshold = float(getattr(config, "gt_threshold_2d", 0.5))
            for idx, threshold in enumerate(thresholds):
                iou = calc.img_IoU(
                    preds,
                    target,
                    threshold=float(threshold.item()),
                    gt_threshold=gt_threshold,
                )
                stats["sum_iou_2d"][idx] += iou.sum()
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
            batch_size = preds.shape[0]
            gt_threshold = float(getattr(config, "gt_threshold_3d", 0.5))
            for idx, threshold in enumerate(thresholds):
                iou = calc.pc_IoU(
                    preds,
                    target,
                    threshold=float(threshold.item()),
                    gt_threshold=gt_threshold,
                )
                stats["sum_iou_3d"][idx] += iou.sum()
            stats["count_3d"] += batch_size


def finalize_threshold_search(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return best 2D/3D prediction thresholds and their validation IoU."""
    results = {}
    thresholds = stats["thresholds"]
    if stats["count_2d"].item() > 0:
        mean_iou = stats["sum_iou_2d"] / stats["count_2d"].clamp_min(1.0)
        best_idx = int(torch.argmax(mean_iou).item())
        results["best_mask_threshold_2d"] = float(thresholds[best_idx].item())
        results["best_giou_2d"] = float(mean_iou[best_idx].item())
    if stats["count_3d"].item() > 0:
        mean_iou = stats["sum_iou_3d"] / stats["count_3d"].clamp_min(1.0)
        best_idx = int(torch.argmax(mean_iou).item())
        results["best_mask_threshold_3d"] = float(thresholds[best_idx].item())
        results["best_iou_3d"] = float(mean_iou[best_idx].item())
    return results
