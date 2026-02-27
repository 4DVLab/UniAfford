"""
评估指标状态累计模块（纯状态层）

职责：
- 构建 epoch 级指标容器（MeanMetric 包装）
- 每 batch 从 output_dict / input_dict 中提取 pred / gt，调用 calculator.py 计算，再更新到容器
- epoch 结束后 compute + reset，返回 {name: float} 字典
- 日志格式化输出

所有指标的数学定义统一在 calculator.py 中，本文件不重复实现。
"""

import torch
from typing import Dict, Optional
from torchmetrics import MeanMetric, SumMetric
from utils import calculator as calc


# ===================== 指标名称常量 =====================

LOSS_KEYS = [
    "loss", "ce_loss",
    "img_focal_loss", "img_dice_loss", "img_loss",
    "pc_bce_loss", "pc_dice_loss", "pc_loss",
]

SEG_2D_KEYS = ["giou_2d", "ciou_2d"]
# giou_2d: 逐样本 IoU 的平均（generalized IoU）
# ciou_2d: 累积 intersection / 累积 union（class IoU）
_CIOU_ACCUM_KEYS = ["_ciou_2d_intersection", "_ciou_2d_union"]

SEG_3D_KEYS = ["iou_3d", "mae_3d", "auc_3d", "sim_3d"]


# ===================== 构建 / 更新 / 汇总 =====================

def build_torchmetrics_bundle(
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Dict[str, MeanMetric]:
    """
    构建一组 epoch 级 MeanMetric 指标容器。

    所有指标统一使用 MeanMetric（标量均值累积），
    具体的数学计算由 update_torchmetrics 调用 calculator.py 完成后再 .update()。
    这样增删指标只需改 KEYS 和 update 里的一行调用，不涉及 torchmetrics 类型适配。

    Args:
        device: 指标状态所在设备（推荐 CPU，避免 GPU 显存累积）。
        **kwargs: 保留兼容（threshold_2d / threshold_3d 等由 update 时使用，不影响容器构建）。
    """
    all_keys = LOSS_KEYS + SEG_2D_KEYS + SEG_3D_KEYS
    bundle = {}
    for key in all_keys:
        bundle[key] = MeanMetric().to(device)
    for key in _CIOU_ACCUM_KEYS:
        bundle[key] = SumMetric().to(device)
    return bundle


def update_torchmetrics(
    metrics: Dict[str, MeanMetric],
    loss_dict: Dict[str, torch.Tensor],
    output_dict: Dict[str, torch.Tensor],
    input_dict: Dict[str, torch.Tensor],
    batch_size: int = 1,
    threshold_2d: float = 0.5,
    threshold_3d: float = 0.5,
):
    """
    每 batch 更新所有指标。

    数学计算全部调用 calculator.py，本函数只负责：
    1. 从 dict 中取出 pred / gt
    2. 过滤无效样本
    3. 调用 calc.xxx 得到标量
    4. .update() 到对应 MeanMetric
    """
    # ---- 损失 ----
    for key in LOSS_KEYS:
        val = loss_dict.get(key)
        if val is not None and key in metrics:
            metrics[key].update(val.detach().cpu().item(), weight=batch_size)

    # ---- 2D 分割指标 ----
    image_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    if image_logits is not None and img_gt is not None:
        preds_2d = image_logits.detach().sigmoid()
        target_2d = img_gt.float()
        img_valid = input_dict.get("img_valid_mask")
        if img_valid is not None:
            valid = img_valid.bool()
            if not valid.any():
                preds_2d = target_2d = None
            else:
                preds_2d = preds_2d[valid]
                target_2d = target_2d[valid]
        if preds_2d is not None:
            bs_2d = preds_2d.shape[0]
            iou_2d = calc.img_IoU(preds_2d, target_2d, threshold=threshold_2d)  # [B']
            metrics["giou_2d"].update(iou_2d.mean().item(), weight=bs_2d)
            inter_2d, union_2d = calc.img_I_and_U(preds_2d, target_2d, threshold=threshold_2d)
            metrics["_ciou_2d_intersection"].update(inter_2d.sum().item())
            metrics["_ciou_2d_union"].update(union_2d.sum().item())

    # ---- 3D 分割指标 ----
    point_logits = output_dict.get("point_logits")
    pc_gt = input_dict.get("pc_gt_tensor")
    if point_logits is not None and pc_gt is not None:
        preds_3d = point_logits.detach().sigmoid()
        target_3d = pc_gt.float()
        valid_lengths = input_dict.get("pc_valid_lengths")
        if valid_lengths is not None:
            valid = (valid_lengths > 0).bool()
            if not valid.any():
                return
            preds_3d = preds_3d[valid]
            target_3d = target_3d[valid]
        bs = preds_3d.shape[0]

        iou_3d = calc.pc_IoU(preds_3d, target_3d, threshold=threshold_3d)  # [B']
        metrics["iou_3d"].update(iou_3d.mean().item(), weight=bs)

        mae_3d = calc.pc_MAE(preds_3d, target_3d)
        metrics["mae_3d"].update(mae_3d.item(), weight=bs)

        try:
            auc_3d = calc.pc_AUC(preds_3d, target_3d, num_thresholds=50)
            metrics["auc_3d"].update(auc_3d.item(), weight=bs)
        except Exception:
            pass

        sim_vals = calc.pc_SIM(preds_3d, target_3d)  # [B']
        metrics["sim_3d"].update(sim_vals.mean().item(), weight=bs)


def compute_sample_metrics(
    output_dict: Dict[str, torch.Tensor],
    input_dict: Dict[str, torch.Tensor],
    sample_idx: int,
    threshold_2d: float = 0.5,
    threshold_3d: float = 0.5,
) -> Dict[str, float]:
    """
    计算单个样本的 2D/3D 指标（供 validate.py 逐样本记录使用）。

    统一调用 calculator.py，避免在 validate.py 的循环中重复手写指标计算。

    Returns:
        {"iou_2d": float|None, "iou_3d": float|None, "mae_3d": float|None, "sim_3d": float|None}
    """
    record = {}
    i = sample_idx

    # 2D
    img_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    img_valid = input_dict.get("img_valid_mask")
    if (img_logits is not None and img_gt is not None
            and (img_valid is None or img_valid[i].bool())):
        pred_2d = img_logits[i].detach().sigmoid().unsqueeze(0)
        gt_2d = img_gt[i].float().unsqueeze(0)
        record["iou_2d"] = round(calc.img_IoU(pred_2d, gt_2d, threshold=threshold_2d)[0].item(), 6)
    else:
        record["iou_2d"] = None

    # 3D
    pt_logits = output_dict.get("point_logits")
    pc_gt = input_dict.get("pc_gt_tensor")
    pc_valid = input_dict.get("pc_valid_lengths")
    if (pt_logits is not None and pc_gt is not None
            and (pc_valid is None or pc_valid[i] > 0)):
        pred_3d = pt_logits[i].detach().sigmoid().unsqueeze(0)
        gt_3d = pc_gt[i].float().unsqueeze(0)
        record["iou_3d"] = round(calc.pc_IoU(pred_3d, gt_3d, threshold=threshold_3d)[0].item(), 6)
        record["mae_3d"] = round(calc.pc_MAE(pred_3d, gt_3d).item(), 6)
        record["sim_3d"] = round(calc.pc_SIM(pred_3d, gt_3d)[0].item(), 6)
    else:
        record["iou_3d"] = None
        record["mae_3d"] = None
        record["sim_3d"] = None

    return record


def compute_and_reset_torchmetrics(metrics: Dict[str, MeanMetric]) -> Dict[str, float]:
    """汇总并重置所有指标，返回 {name: float} 字典。cIoU 从累积的 I/U 中计算。"""
    results = {}
    for name, metric in metrics.items():
        value = metric.compute()
        value = value.detach().float().cpu().item() if isinstance(value, torch.Tensor) else float(value)
        if value != value:
            value = 0.0
        results[name] = value
        metric.reset()

    # cIoU = 累积 intersection / 累积 union（MeanMetric 记录的是 weighted_sum / weight_sum，
    # 但我们用 weight=1 累积 sum，所以 compute() 返回的就是 sum / count，
    # 需要用 weighted_sum 而非 compute 值来算比值）
    total_inter = results.pop("_ciou_2d_intersection", 0.0)
    total_union = results.pop("_ciou_2d_union", 0.0)
    results["ciou_2d"] = total_inter / (total_union + 1e-8) if total_union > 0 else 0.0

    return results


# ===================== 日志工具 =====================

def log_scalar_dict(writer, prefix: str, metrics_dict: Dict[str, float], step: int):
    """批量写入 TensorBoard 标量。"""
    if writer is None:
        return
    for key, value in metrics_dict.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def log_epoch_summary(
    logger,
    epoch: int,
    total_epochs: int,
    phase: str,
    results: Dict[str, float],
    lr_dict: Optional[Dict[str, float]] = None,
):
    """格式化打印 epoch 摘要。"""
    tag = "训练" if phase == "train" else "验证"
    pfx = f"Epoch [{epoch}/{total_epochs}] {tag}"

    loss_str = (
        f"Loss: {results.get('loss', 0):.6f} "
        f"(CE: {results.get('ce_loss', 0):.6f}, "
        f"Img: {results.get('img_loss', 0):.6f} "
        f"[focal={results.get('img_focal_loss', 0):.6f}, dice={results.get('img_dice_loss', 0):.6f}], "
        f"PC: {results.get('pc_loss', 0):.6f} "
        f"[bce={results.get('pc_bce_loss', 0):.6f}, dice={results.get('pc_dice_loss', 0):.6f}])"
    )
    seg2d_str = f"gIoU2D: {results.get('giou_2d', 0):.4f}, cIoU2D: {results.get('ciou_2d', 0):.4f}"
    seg3d_str = (
        f"IoU3D: {results.get('iou_3d', 0):.4f}, "
        f"MAE3D: {results.get('mae_3d', 0):.4f}, "
        f"AUC3D: {results.get('auc_3d', 0):.4f}, "
        f"SIM3D: {results.get('sim_3d', 0):.4f}"
    )
    logger.info(f"{pfx} - {loss_str}")
    logger.info(f"{pfx} 2D - {seg2d_str}")
    logger.info(f"{pfx} 3D - {seg3d_str}")
    if lr_dict is not None:
        lr_text = ", ".join([f"{k}={v:.2e}" for k, v in lr_dict.items()])
        logger.info(f"当前学习率: {lr_text}")
