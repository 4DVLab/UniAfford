"""
评估指标记录模块

本模块提供两套指标管理方案，根据需要选用：
1. torchmetrics 方案（推荐）：基于 torchmetrics 的 epoch 级指标，自带分布式同步，
   适合新版训练循环（try.py / train_new.py）。
2. AverageMeter 方案（兼容）：手动累积的轻量级指标，适合旧版训练循环（train.py）。

主要对外接口：
- build_torchmetrics_bundle / update_torchmetrics / compute_and_reset_torchmetrics
- log_scalar_dict / log_epoch_summary
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from enum import Enum
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryJaccardIndex
from torchmetrics.regression import MeanAbsoluteError
from utils import calculator as calc


# ===================== 基础枚举/工具类 =====================

class Summary(Enum):
    """统计汇总类型"""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """计算并存储平均值和当前值（兼容旧版训练代码）"""
    
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE, device="cuda"):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.device = device if not isinstance(device, str) else torch.device(device)
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def all_reduce(self):
        """分布式训练时同步所有进程的统计值"""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        
        if isinstance(self.sum, (np.ndarray, list)):
            sum_array = np.array(self.sum) if isinstance(self.sum, list) else self.sum
            sum_tensor = torch.tensor(sum_array, dtype=torch.float32, device=self.device)
            count_tensor = torch.tensor([self.count], dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(sum_tensor, torch.distributed.ReduceOp.SUM, async_op=False)
            torch.distributed.all_reduce(count_tensor, torch.distributed.ReduceOp.SUM, async_op=False)
            self.sum = sum_tensor.cpu().numpy()
            self.count = count_tensor.item()
            self.avg = self.sum / self.count if self.count > 0 else self.sum
        else:
            total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM, async_op=False)
            self.sum, self.count = total.tolist()
            self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return fmtstr.format(**self.__dict__)


# ===================== 旧版跟踪器（兼容 train.py）=====================

class MetricsTracker:
    """指标跟踪器：负责存储、更新、汇总Loss和Segmentation指标（旧版 AverageMeter 方案）"""
    
    def __init__(self):
        # 损失指标
        self.loss_meters = {
            "loss": AverageMeter("Loss", ":.4f"),
            "ce_loss": AverageMeter("CeLoss", ":.4f"),
            "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
            "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
            "mask_loss": AverageMeter("MaskLoss", ":.4f"),
            "mask_3d_bce_loss": AverageMeter("Mask3DBCELoss", ":.4f"),
            "mask_3d_dice_loss": AverageMeter("Mask3DDICELoss", ":.4f"),
            "mask_3d_loss": AverageMeter("Mask3DLoss", ":.4f"),
        }
        # 2D 分割指标
        self.seg_2d_meters = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "giou": AverageMeter("gIoU", ":6.3f", Summary.AVERAGE),
        }
        # 3D 分割指标
        self.seg_3d_meters = {
            "mae": AverageMeter("MAE3D", ":6.4f", Summary.AVERAGE),
            "auc": AverageMeter("AUC3D", ":6.4f", Summary.AVERAGE),
            "aiou": AverageMeter("aIoU3D", ":6.4f", Summary.AVERAGE),
            "sim": AverageMeter("SIM3D", ":6.4f", Summary.AVERAGE),
        }
        self.num_2d_samples = 0
        self.num_3d_samples = 0

    def reset(self):
        for meter in self.loss_meters.values():
            meter.reset()
        for meter in self.seg_2d_meters.values():
            meter.reset()
        for meter in self.seg_3d_meters.values():
            meter.reset()
        self.num_2d_samples = 0
        self.num_3d_samples = 0

    def update_loss_metrics(self, output_dict: Dict[str, torch.Tensor], batch_size: int):
        for key, meter in self.loss_meters.items():
            val = output_dict.get(key, torch.tensor(0.0))
            meter.update(val.item() if isinstance(val, torch.Tensor) else float(val), batch_size)

    def update_2d_seg_metrics(self, pred_mask, gt_mask, threshold=0.0):
        if pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(-1)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(-1)
        elif pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(0) if pred_mask.shape[-1] == 1 else pred_mask.unsqueeze(-1)
            gt_mask = gt_mask.unsqueeze(0) if gt_mask.shape[-1] == 1 else gt_mask.unsqueeze(-1)
        
        batch_size = pred_mask.shape[0]
        self.num_2d_samples += batch_size
        
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(-1)
            gt_mask = gt_mask.unsqueeze(-1)
        
        mask_pred = (pred_mask > threshold).float()
        mask_gt = gt_mask.float()
        intersection_batch, union_batch = calc.BA_I_and_U(mask_pred, mask_gt)
        
        for b in range(batch_size):
            i_val = intersection_batch[b].item()
            u_val = union_batch[b].item()
            iou_val = i_val / (u_val + 1e-6) if u_val > 0 else 1.0
            intersection_2c = np.array([0.0, i_val])
            union_2c = np.array([0.0, u_val])
            iou_2c = np.array([0.0, iou_val])
            self.seg_2d_meters["intersection"].update(intersection_2c, n=1)
            self.seg_2d_meters["union"].update(union_2c, n=1)
            self.seg_2d_meters["giou"].update(iou_2c, n=1)

    def update_3d_seg_metrics(self, pred_mask, gt_mask):
        if pred_mask.dim() == 1:
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(-1)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(-1)
        elif pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0)
        
        batch_size = pred_mask.shape[0]
        self.num_3d_samples += batch_size
        pred_norm = pred_mask.clamp(0, 1)
        gt_norm = gt_mask.float().clamp(0, 1)
        
        mae_val = calc.pc_MAE(pred_norm, gt_norm)
        sim_val = calc.pc_SIM(pred_norm, gt_norm)
        aiou_val = calc.pc_aIOU(pred_norm, gt_norm, num_thresholds=50)
        self.seg_3d_meters["mae"].update(mae_val.item(), n=batch_size)
        self.seg_3d_meters["sim"].update(sim_val.item(), n=batch_size)
        self.seg_3d_meters["aiou"].update(aiou_val.item(), n=batch_size)
        try:
            auc_val = calc.pc_AUC(pred_norm, gt_norm, num_thresholds=50)
            self.seg_3d_meters["auc"].update(auc_val.item(), n=batch_size)
        except Exception:
            pass

    def all_reduce(self):
        for meter in self.loss_meters.values():
            meter.all_reduce()
        for meter in self.seg_2d_meters.values():
            meter.all_reduce()
        for meter in self.seg_3d_meters.values():
            meter.all_reduce()

    def compute_2d_seg_results(self) -> Tuple[float, float]:
        intersection = self.seg_2d_meters["intersection"].sum
        union = self.seg_2d_meters["union"].sum
        iou_class = intersection / (union + 1e-10)
        ciou = iou_class[1] if isinstance(iou_class, np.ndarray) and iou_class.size > 1 else 0.0
        giou = self.seg_2d_meters["giou"].avg[1] if isinstance(self.seg_2d_meters["giou"].avg, np.ndarray) and self.seg_2d_meters["giou"].avg.size > 1 else 0.0
        return giou, ciou

    def compute_3d_seg_results(self) -> Tuple[float, float, float, float]:
        mae_3d = float(self.seg_3d_meters["mae"].avg) if self.seg_3d_meters["mae"].count > 0 else 0.0
        auc_3d = float(self.seg_3d_meters["auc"].avg) if self.seg_3d_meters["auc"].count > 0 else 0.0
        aiou_3d = float(self.seg_3d_meters["aiou"].avg) if self.seg_3d_meters["aiou"].count > 0 else 0.0
        sim_3d = float(self.seg_3d_meters["sim"].avg) if self.seg_3d_meters["sim"].count > 0 else 0.0
        return mae_3d, auc_3d, aiou_3d, sim_3d

    def get_sample_counts(self) -> Tuple[int, int]:
        return self.num_2d_samples, self.num_3d_samples


# ===================== 旧版日志类（兼容 train.py）=====================

class TensorBoardLogger:
    """TensorBoard日志写入类（兼容旧版 train.py）"""
    
    def __init__(self, writer):
        self.writer = writer

    def log_loss_metrics(self, metrics_tracker: MetricsTracker, global_step: int, prefix: str = "train"):
        for key, meter in metrics_tracker.loss_meters.items():
            self.writer.add_scalar(f"{prefix}/loss/{key}", meter.avg, global_step)

    def log_seg_2d_metrics(self, giou: float, ciou: float, global_step: int, prefix: str = "val"):
        self.writer.add_scalar(f"{prefix}/seg_2d/giou", giou, global_step)
        self.writer.add_scalar(f"{prefix}/seg_2d/ciou", ciou, global_step)

    def log_seg_3d_metrics(self, mae_3d, auc_3d, aiou_3d, sim_3d, global_step, prefix="val"):
        self.writer.add_scalar(f"{prefix}/seg_3d/mae", mae_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/auc", auc_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/aiou", aiou_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/sim", sim_3d, global_step)

    def log_custom_metrics(self, metrics_dict: Dict[str, float], global_step: int, prefix: str = "custom"):
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, global_step)


# ===================== 旧版辅助函数（兼容 train.py）=====================

def evaluate_segmentation_batch(
    input_dict,
    output_dict: Dict[str, any],
    metrics_tracker: MetricsTracker,
    mask_threshold_2d: float = 0.0,
):
    """评估一个批次的分割结果（兼容旧版 train.py）"""
    # 2D分割评估
    if "pred_masks" in output_dict and output_dict["pred_masks"] is not None:
        pred_masks = output_dict["pred_masks"]
        gt_masks = input_dict["img_gt_tensor"]
        if isinstance(pred_masks, list):
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                metrics_tracker.update_2d_seg_metrics(pred_mask, gt_mask, mask_threshold_2d)
        else:
            metrics_tracker.update_2d_seg_metrics(pred_masks, gt_masks, mask_threshold_2d)
    
    # 3D分割评估
    if "pred_3d_masks" in output_dict and output_dict["pred_3d_masks"] is not None:
        pred_3d_masks = output_dict["pred_3d_masks"]
        gt_3d_masks = input_dict["pc_gt_tensor"]
        if isinstance(pred_3d_masks, list):
            for pred_3d_mask, gt_3d_mask in zip(pred_3d_masks, gt_3d_masks):
                if gt_3d_mask is not None:
                    metrics_tracker.update_3d_seg_metrics(pred_3d_mask, gt_3d_mask)
        else:
            metrics_tracker.update_3d_seg_metrics(pred_3d_masks, gt_3d_masks)


def print_validation_summary(epoch, metrics_tracker, giou, ciou, mae_3d=0.0, auc_3d=0.0, aiou_3d=0.0, sim_3d=0.0):
    """打印验证结果摘要（兼容旧版 train.py）"""
    num_2d, num_3d = metrics_tracker.get_sample_counts()
    print(f"\n{'='*70}")
    print(f"Validation Summary (Epoch {epoch}):")
    print(f"  2D samples: {num_2d},  3D samples: {num_3d}")
    print(f"{'='*70}")
    print(f"  2D -> gIoU: {giou:.4f}, cIoU: {ciou:.4f}")
    print(f"  3D -> MAE: {mae_3d:.4f}, AUC: {auc_3d:.4f}, aIoU: {aiou_3d:.4f}, SIM: {sim_3d:.4f}")
    if num_3d == 0:
        print(f"  WARNING: No 3D samples were evaluated!")
    print(f"{'='*70}\n")


# ===================== torchmetrics 方案（推荐，新版训练代码使用） =====================

# ---------- 名称常量 ----------
LOSS_KEYS = [
    "loss", "ce_loss",
    "img_focal_loss", "img_dice_loss", "img_loss",
    "pc_bce_loss", "pc_dice_loss", "pc_loss",
]
SEG_2D_KEYS = ["iou_2d"]
SEG_3D_KEYS = ["iou_3d", "mae_3d", "auroc_3d", "auc_3d", "sim_3d"]


def build_torchmetrics_bundle(
    device,
    threshold_2d: float = 0.5,
    threshold_3d: float = 0.5,
    auroc_num_thresholds: int = 256,
) -> Dict[str, torch.nn.Module]:
    """
    构建一组 epoch 级 torchmetrics 指标（支持多卡 compute 时自动同步）。

    包含三类指标:
      1. 损失追踪 (MeanMetric): loss / ce_loss / img_focal / img_dice / img_loss / pc_bce / pc_dice / pc_loss
      2. 2D 分割: iou_2d (BinaryJaccardIndex)
      3. 3D 分割: iou_3d, mae_3d, auroc_3d, auc_3d (自定义), sim_3d (自定义)
    """
    bundle = {}
    # ---- 损失追踪 ----
    for key in LOSS_KEYS:
        bundle[key] = MeanMetric(sync_on_compute=True).to(device)
    # ---- 2D 分割 ----
    bundle["iou_2d"] = BinaryJaccardIndex(threshold=threshold_2d).to(device)
    # ---- 3D 分割（标准 torchmetrics）----
    bundle["iou_3d"] = BinaryJaccardIndex(threshold=threshold_3d).to(device)
    bundle["mae_3d"] = MeanAbsoluteError().to(device)
    # BinaryAUROC 默认会缓存整轮所有预测/标签，数据集大时显存线性增长。
    # 指定 thresholds 后改为分桶统计，显存占用与样本数解耦（固定上界）。
    bundle["auroc_3d"] = BinaryAUROC(thresholds=auroc_num_thresholds).to(device)
    # ---- 3D 分割（自定义指标，用 MeanMetric 包装）----
    bundle["auc_3d"] = MeanMetric(sync_on_compute=True).to(device)   # calc.pc_AUC 的多阈值 AUC
    bundle["sim_3d"] = MeanMetric(sync_on_compute=True).to(device)   # calc.pc_SIM 相似度
    return bundle


def _align_dims(preds: torch.Tensor, targets: torch.Tensor):
    """对齐 pred/target 的尾部维度（去掉尾部的 size-1 维）"""
    if targets.dim() == preds.dim() + 1 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    if preds.dim() == targets.dim() + 1 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    return preds, targets


def update_torchmetrics(
    metrics: Dict,
    loss_dict: Dict[str, torch.Tensor],
    output_dict: Dict[str, torch.Tensor],
    input_dict: Dict[str, torch.Tensor],
    batch_size: int = 1,
):
    """
    一站式更新所有 torchmetrics（损失 + 2D/3D 分割指标）。

    Args:
        metrics: build_torchmetrics_bundle 返回的指标字典
        loss_dict: compute_losses 返回的损失字典
        output_dict: 模型输出字典（含 image_logits / point_logits）
        input_dict: 模型输入字典（含 img_gt_tensor / pc_gt_tensor）
        batch_size: 当前 batch 大小
    """
    # ---- 更新损失 ----
    for key in LOSS_KEYS:
        val = loss_dict.get(key)
        if val is not None and key in metrics:
            metrics[key].update(val.detach(), weight=batch_size)

    # ---- 更新 2D 分割指标 ----
    image_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    if image_logits is not None and img_gt is not None:
        preds_2d = image_logits.detach()
        target_2d = img_gt.float()
        preds_2d, target_2d = _align_dims(preds_2d, target_2d)
        # 过滤无效样本
        img_valid = input_dict.get("img_valid_mask")
        if img_valid is not None:
            valid = img_valid.bool()
            if valid.any():
                metrics["iou_2d"].update(preds_2d[valid], target_2d[valid].int())
        else:
            metrics["iou_2d"].update(preds_2d, target_2d.int())

    # ---- 更新 3D 分割指标 ----
    point_logits = output_dict.get("point_logits")
    pc_gt = input_dict.get("pc_gt_tensor")
    if point_logits is not None and pc_gt is not None:
        preds_3d = point_logits.detach()
        target_3d = pc_gt.float()
        preds_3d, target_3d = _align_dims(preds_3d, target_3d)
        # 过滤无效样本
        valid_lengths = input_dict.get("pc_valid_lengths")
        if valid_lengths is not None:
            valid = (valid_lengths > 0).bool()
            if valid.any():
                preds_3d = preds_3d[valid]
                target_3d = target_3d[valid]
            else:
                return  # 全部无效，跳过
        bs = preds_3d.shape[0]
        # 标准 torchmetrics
        metrics["iou_3d"].update(preds_3d, target_3d.int())
        metrics["mae_3d"].update(preds_3d, target_3d)
        # 自定义指标：需要概率值 [0,1]（logits → sigmoid）
        preds_prob = preds_3d.sigmoid()
        target_01 = target_3d.clamp(0, 1)
        metrics["auroc_3d"].update(preds_prob.reshape(-1), target_01.reshape(-1).int())
        try:
            auc_val = calc.pc_AUC(preds_prob, target_01, num_thresholds=50)
            metrics["auc_3d"].update(auc_val.item(), weight=bs)
        except Exception:
            pass
        sim_vals = calc.pc_SIM(preds_prob, target_01)  # [Batch]
        metrics["sim_3d"].update(sim_vals.mean().item(), weight=bs)


def compute_and_reset_torchmetrics(metrics: Dict) -> Dict[str, float]:
    """
    汇总并重置所有 torchmetrics 指标，返回 {name: float} 字典。
    NaN 值自动替换为 0.0，compute 失败的指标记为 0.0。
    """
    results = {}
    for name, metric in metrics.items():
        try:
            value = metric.compute()
            value = value.detach().float().cpu().item() if isinstance(value, torch.Tensor) else float(value)
            if value != value:  # NaN check
                value = 0.0
        except Exception:
            value = 0.0
        results[name] = value
        metric.reset()
    return results


# ===================== 通用日志工具 =====================

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
    lr_dict: Optional[float] = None,
):
    """
    格式化打印一个阶段（训练/验证）的 epoch 摘要，输出所有已计算的指标。

    Args:
        logger: logging.Logger 实例
        epoch: 当前 epoch（1-based）
        total_epochs: 总 epoch 数
        phase: "train" 或 "val"
        results: compute_and_reset_torchmetrics 返回的字典
        lr: 当前学习率（可选）
    """
    tag = "训练" if phase == "train" else "验证"
    pfx = f"Epoch [{epoch}/{total_epochs}] {tag}"

    # 损失汇总
    loss_str = (
        f"Loss: {results.get('loss', 0):.6f} "
        f"(CE: {results.get('ce_loss', 0):.6f}, "
        f"Img: {results.get('img_loss', 0):.6f} [focal={results.get('img_focal_loss', 0):.6f}, dice={results.get('img_dice_loss', 0):.6f}], "
        f"PC: {results.get('pc_loss', 0):.6f} [bce={results.get('pc_bce_loss', 0):.6f}, dice={results.get('pc_dice_loss', 0):.6f}])"
    )
    # 2D 分割指标
    seg2d_str = f"IoU2D: {results.get('iou_2d', 0):.4f}"
    # 3D 分割指标（完整 5 项）
    seg3d_str = (
        f"IoU3D: {results.get('iou_3d', 0):.4f}, "
        f"MAE3D: {results.get('mae_3d', 0):.4f}, "
        f"AUROC3D: {results.get('auroc_3d', 0):.4f}, "
        f"AUC3D: {results.get('auc_3d', 0):.4f}, "
        f"SIM3D: {results.get('sim_3d', 0):.4f}"
    )
    logger.info(f"{pfx} - {loss_str}")
    logger.info(f"{pfx} 2D - {seg2d_str}")
    logger.info(f"{pfx} 3D - {seg3d_str}")
    if lr_dict is not None:
        lr_text = ", ".join([f"{k}={v:.2e}" for k, v in lr_dict.items()])
        logger.info(f"当前学习率: {lr_text}")
