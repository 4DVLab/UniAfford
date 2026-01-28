"""
评估指标记录模块
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from enum import Enum
from utils import calculator as calc


# ===================== 基础枚举/工具类 =====================
class Summary(Enum):
    """统计汇总类型"""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter:
    """计算并存储平均值和当前值"""
    
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
        
        # 向量情况
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
            # 标量情况
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


# ===================== 跟踪类：指标存储/更新/汇总 =====================
class MetricsTracker:
    """指标跟踪器：负责存储、更新、汇总Loss和Segmentation指标（整合原LossMetrics和SegmentationMetrics）"""
    
    def __init__(self):
        # ---------------- Loss 指标 ----------------
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

        # ---------------- 2D 分割指标 ----------------
        self.seg_2d_meters = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "giou": AverageMeter("gIoU", ":6.3f", Summary.AVERAGE),
        }

        # ---------------- 3D 分割指标 ----------------
        self.seg_3d_meters = {
            "mae": AverageMeter("MAE3D", ":6.4f", Summary.AVERAGE),
            "auc": AverageMeter("AUC3D", ":6.4f", Summary.AVERAGE),
            "aiou": AverageMeter("aIoU3D", ":6.4f", Summary.AVERAGE),
            "sim": AverageMeter("SIM3D", ":6.4f", Summary.AVERAGE),
        }

        # 样本计数器
        self.num_2d_samples = 0
        self.num_3d_samples = 0

    def reset(self):
        """重置所有指标"""
        # 重置Loss指标
        for meter in self.loss_meters.values():
            meter.reset()
        # 重置2D分割指标
        for meter in self.seg_2d_meters.values():
            meter.reset()
        # 重置3D分割指标
        for meter in self.seg_3d_meters.values():
            meter.reset()
        # 重置计数器
        self.num_2d_samples = 0
        self.num_3d_samples = 0

    def update_loss_metrics(self, output_dict: Dict[str, torch.Tensor], batch_size: int):
        """更新Loss指标"""
        for key, meter in self.loss_meters.items():
            val = output_dict.get(key, torch.tensor(0.0))
            meter.update(val.item(), batch_size)

    def update_2d_seg_metrics(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.0
    ):
        """
        更新2D分割指标（支持批处理）
        
        Args:
            pred_mask: [Batch, H, W, 1] 或 [H, W] 预测掩码
            gt_mask: [Batch, H, W, 1] 或 [H, W] 真实掩码
            threshold: 二值化阈值
        """
        # 处理单样本情况（向后兼容）
        if pred_mask.dim() == 2:  # [H, W]
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(-1)
        elif pred_mask.dim() == 3:  # [H, W, 1]
            pred_mask = pred_mask.unsqueeze(0)  # [1, H, W, 1]
            gt_mask = gt_mask.unsqueeze(0)
        
        batch_size = pred_mask.shape[0]
        self.num_2d_samples += batch_size
        
        # 确保形状为 [Batch, H, W, 1]
        if pred_mask.dim() == 3:  # [Batch, H, W]
            pred_mask = pred_mask.unsqueeze(-1)
            gt_mask = gt_mask.unsqueeze(-1)
        
        # 二值化
        mask_pred = (pred_mask > threshold).float()
        mask_gt = gt_mask.float()
        
        assert mask_pred.shape == mask_gt.shape, f"Shape mismatch: pred {mask_pred.shape}, gt {mask_gt.shape}"
        
        # 使用 calculator.BA_I_and_U 批量计算交并集
        intersection_batch, union_batch = calc.BA_I_and_U(mask_pred, mask_gt)  # [Batch], [Batch]
        
        # 逐样本更新 meter
        for b in range(batch_size):
            intersection_val = intersection_batch[b].item()
            union_val = union_batch[b].item()
            
            # 计算当前样本的 IoU
            iou_val = intersection_val / (union_val + 1e-6) if union_val > 0 else 1.0
            
            # 对于二分类，构造 [background, foreground] 的交并集
            # background 类别的交并集设为 0（我们只关注 foreground）
            intersection_2class = np.array([0.0, intersection_val])
            union_2class = np.array([0.0, union_val])
            iou_2class = np.array([0.0, iou_val])
            
            # 更新 meter
            # intersection 和 union 使用累加（SUM），用于计算 cIoU
            # giou 使用平均（AVERAGE），用于计算 gIoU
            self.seg_2d_meters["intersection"].update(intersection_2class, n=1)
            self.seg_2d_meters["union"].update(union_2class, n=1)
            self.seg_2d_meters["giou"].update(iou_2class, n=1)

    def update_3d_seg_metrics(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ):
        """
        更新3D分割指标（支持批处理）
        
        Args:
            pred_mask: [Batch, N, 1] 或 [N] 预测掩码，值域 [0, 1]
            gt_mask: [Batch, N, 1] 或 [N] 真实掩码，值域 {0, 1}
        """
        # 处理单样本情况（向后兼容）
        if pred_mask.dim() == 1:  # [N]
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(-1)
        elif pred_mask.dim() == 2:  # [N, 1]
            pred_mask = pred_mask.unsqueeze(0)  # [1, N, 1]
            gt_mask = gt_mask.unsqueeze(0)
        
        batch_size = pred_mask.shape[0]
        self.num_3d_samples += batch_size
        
        assert pred_mask.shape == gt_mask.shape, f"Shape mismatch: pred {pred_mask.shape}, gt {gt_mask.shape}"
        
        # 确保值域在 [0, 1]
        pred_norm = pred_mask.clamp(0, 1)
        gt_norm = gt_mask.float().clamp(0, 1)
        
        # 使用 calculator 中的批量计算函数
        mae_val = calc.pc_MAE(pred_norm, gt_norm)  # 标量（平均值）
        sim_val = calc.pc_SIM(pred_norm, gt_norm)  # 标量（平均值）
        aiou_val = calc.pc_aIOU(pred_norm, gt_norm, num_thresholds=50)  # 标量（平均值）
        
        # 更新 meter
        self.seg_3d_meters["mae"].update(mae_val.item(), n=batch_size)
        self.seg_3d_meters["sim"].update(sim_val.item(), n=batch_size)
        self.seg_3d_meters["aiou"].update(aiou_val.item(), n=batch_size)
        
        # AUC 计算（可能失败，需要容错）
        try:
            auc_val = calc.pc_AUC(pred_norm, gt_norm, num_thresholds=50)  # 标量（平均值）
            self.seg_3d_meters["auc"].update(auc_val.item(), n=batch_size)
        except Exception:
            pass

    def all_reduce(self):
        """分布式训练时汇总所有指标"""
        # Loss指标汇总
        for meter in self.loss_meters.values():
            meter.all_reduce()
        # 2D分割指标汇总
        for meter in self.seg_2d_meters.values():
            meter.all_reduce()
        # 3D分割指标汇总
        for meter in self.seg_3d_meters.values():
            meter.all_reduce()

    def compute_2d_seg_results(self) -> Tuple[float, float]:
        """计算2D分割最终结果（gIoU/cIoU）"""
        intersection = self.seg_2d_meters["intersection"].sum
        union = self.seg_2d_meters["union"].sum
        
        iou_class = intersection / (union + 1e-10)
        ciou = iou_class[1] if isinstance(iou_class, np.ndarray) and iou_class.size > 1 else 0.0
        giou = self.seg_2d_meters["giou"].avg[1] if isinstance(self.seg_2d_meters["giou"].avg, np.ndarray) and self.seg_2d_meters["giou"].avg.size > 1 else 0.0
        
        return giou, ciou

    def compute_3d_seg_results(self) -> Tuple[float, float, float, float]:
        """
        计算3D分割最终结果
        
        Returns:
            mae_3d: Mean Absolute Error
            auc_3d: Area Under Curve
            aiou_3d: average IoU
            sim_3d: Similarity
        """
        mae_3d = float(self.seg_3d_meters["mae"].avg) if self.seg_3d_meters["mae"].count > 0 else 0.0
        auc_3d = float(self.seg_3d_meters["auc"].avg) if self.seg_3d_meters["auc"].count > 0 else 0.0
        aiou_3d = float(self.seg_3d_meters["aiou"].avg) if self.seg_3d_meters["aiou"].count > 0 else 0.0
        sim_3d = float(self.seg_3d_meters["sim"].avg) if self.seg_3d_meters["sim"].count > 0 else 0.0
        
        return mae_3d, auc_3d, aiou_3d, sim_3d

    def get_sample_counts(self) -> Tuple[int, int]:
        """获取2D/3D样本数"""
        return self.num_2d_samples, self.num_3d_samples


# ===================== 日志类：TensorBoard 写入 =====================
class TensorBoardLogger:
    """TensorBoard日志写入类：专门处理指标的TensorBoard记录"""
    
    def __init__(self, writer):
        """
        Args:
            writer: torch.utils.tensorboard.SummaryWriter 实例
        """
        self.writer = writer

    def log_loss_metrics(self, metrics_tracker: MetricsTracker, global_step: int, prefix: str = "train"):
        """记录Loss指标到TensorBoard"""
        for key, meter in metrics_tracker.loss_meters.items():
            self.writer.add_scalar(f"{prefix}/loss/{key}", meter.avg, global_step)

    def log_seg_2d_metrics(self, giou: float, ciou: float, global_step: int, prefix: str = "val"):
        """记录2D分割指标到TensorBoard"""
        self.writer.add_scalar(f"{prefix}/seg_2d/giou", giou, global_step)
        self.writer.add_scalar(f"{prefix}/seg_2d/ciou", ciou, global_step)

    def log_seg_3d_metrics(
        self,
        mae_3d: float,
        auc_3d: float,
        aiou_3d: float,
        sim_3d: float,
        global_step: int,
        prefix: str = "val"
    ):
        """记录3D分割指标到TensorBoard"""
        self.writer.add_scalar(f"{prefix}/seg_3d/mae", mae_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/auc", auc_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/aiou", aiou_3d, global_step)
        self.writer.add_scalar(f"{prefix}/seg_3d/sim", sim_3d, global_step)

    def log_custom_metrics(self, metrics_dict: Dict[str, float], global_step: int, prefix: str = "custom"):
        """记录自定义指标到TensorBoard"""
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, global_step)


# ===================== 辅助函数 =====================
def evaluate_segmentation_batch(
    output_dict: Dict[str, any],
    metrics_tracker: MetricsTracker,
    mask_threshold_2d: float = 0.0,
):
    """
    评估一个批次的分割结果（支持批处理）
    
    Args:
        output_dict: 模型输出字典
        metrics_tracker: 指标跟踪器
        mask_threshold_2d: 2D mask 二值化阈值
        mask_threshold_3d: 3D mask 二值化阈值
    """
    # 2D分割评估
    if "pred_masks" in output_dict and output_dict["pred_masks"] is not None:
        pred_masks = output_dict["pred_masks"]  # List[Tensor] 或 Tensor
        gt_masks = output_dict["gt_masks"]
        
        # 如果是列表，逐个处理（向后兼容）
        if isinstance(pred_masks, list):
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                metrics_tracker.update_2d_seg_metrics(
                    pred_mask, gt_mask, mask_threshold_2d
                )
        # 如果是批量张量 [Batch, H, W, 1]
        else:
            metrics_tracker.update_2d_seg_metrics(
                pred_masks, gt_masks, mask_threshold_2d
            )
    
    # 3D分割评估
    if "pred_3d_masks" in output_dict and output_dict["pred_3d_masks"] is not None:
        pred_3d_masks = output_dict["pred_3d_masks"]  # List[Tensor] 或 Tensor
        gt_3d_masks = output_dict["gt_3d_masks"]
        
        # 如果是列表，逐个处理（向后兼容）
        if isinstance(pred_3d_masks, list):
            for pred_3d_mask, gt_3d_mask in zip(pred_3d_masks, gt_3d_masks):
                if gt_3d_mask is not None:
                    metrics_tracker.update_3d_seg_metrics(
                        pred_3d_mask, gt_3d_mask
                    )
        # 如果是批量张量 [Batch, N] 或 [Batch, N, 1]
        else:
            metrics_tracker.update_3d_seg_metrics(
                pred_3d_masks, gt_3d_masks
            )


def print_validation_summary(
    epoch: int,
    metrics_tracker: MetricsTracker,
    giou: float,
    ciou: float,
    mae_3d: float = 0.0,
    auc_3d: float = 0.0,
    aiou_3d: float = 0.0,
    sim_3d: float = 0.0,
):
    """
    打印验证结果摘要
    
    Args:
        epoch: 当前 epoch
        metrics_tracker: 指标跟踪器
        giou: 2D Global IoU
        ciou: 2D Class IoU
        mae_3d: 3D Mean Absolute Error
        auc_3d: 3D Area Under Curve
        aiou_3d: 3D average IoU
        sim_3d: 3D Similarity
    """
    num_2d_samples, num_3d_samples = metrics_tracker.get_sample_counts()
    
    print(f"\n{'='*70}")
    print(f"Validation Summary (Epoch {epoch}):")
    print(f"  2D samples evaluated: {num_2d_samples}")
    print(f"  3D samples evaluated: {num_3d_samples}")
    print(f"{'='*70}")
    
    # 2D指标
    print(f"📊 2D Segmentation Metrics:")
    print(f"  ├─ gIoU: {giou:.4f}")
    print(f"  └─ cIoU: {ciou:.4f}")
    
    # 3D指标
    print(f"\n📊 3D Point Cloud Metrics:")
    print(f"  ├─ MAE:  {mae_3d:.4f}  (Mean Absolute Error)")
    print(f"  ├─ AUC:  {auc_3d:.4f}  (Area Under Curve)")
    print(f"  ├─ aIoU: {aiou_3d:.4f}  (average IoU)")
    print(f"  └─ SIM:  {sim_3d:.4f}  (Similarity)")
    
    if num_3d_samples == 0:
        print(f"\n⚠️  WARNING: No 3D samples were evaluated!")
        print(f"   Possible reasons:")
        print(f"   1. Validation dataset has no point cloud data")
        print(f"   2. Point clouds not being loaded correctly")
        print(f"   3. Model not generating pred_masks_3d")
    print(f"{'='*70}\n")
