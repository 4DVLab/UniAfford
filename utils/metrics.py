"""
评估指标计算模块

包含用于计算分割任务的各种评估指标，如 IoU、损失等。
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .common import AverageMeter, Summary, intersectionAndUnionGPU


''' ------------------------------------------- loss ------------------------------------------- '''

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    计算 Dice 损失，类似于掩码的广义 IoU
    
    Dice 系数衡量两个集合的重叠程度，Dice 损失 = 1 - Dice 系数
    
    Args:
        inputs: 任意形状的浮点张量，每个样本的预测值
        targets: 与 inputs 相同形状的浮点张量，存储二元分类标签
                (0 表示负类，1 表示正类)
        num_masks: 掩码数量，用于归一化
        scale: 缩放因子，用于数值稳定性
        eps: 防止除零的小常数
        
    Returns:
        Dice 损失值
    """
    inputs = inputs.sigmoid()  # 将预测值转换为概率
    inputs = inputs.flatten(1, 2)  # 展平空间维度
    targets = targets.flatten(1, 2)
    # 计算 Dice 系数：2 * |A ∩ B| / (|A| + |B|)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    计算 sigmoid 交叉熵损失（二元交叉熵）
    
    Args:
        inputs: 任意形状的浮点张量，每个样本的预测值（logits）
        targets: 与 inputs 相同形状的浮点张量，存储二元分类标签
                (0 表示负类，1 表示正类)
        num_masks: 掩码数量，用于归一化
        
    Returns:
        损失张量
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def compute_2d_mask_loss(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    bce_loss_weight: float,
    dice_loss_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 2D 掩码损失
    
    Args:
        pred_masks: 预测掩码列表，每个元素形状为 [num_masks, H, W]
        gt_masks: 真实掩码列表，每个元素形状为 [num_masks, H, W]
        bce_loss_weight: BCE 损失权重
        dice_loss_weight: Dice 损失权重
        device: 设备
        
    Returns:
        mask_bce_loss: BCE 损失
        mask_dice_loss: Dice 损失
        mask_loss: 总掩码损失
    """
    mask_bce_loss = torch.tensor(0.0, device=device)
    mask_dice_loss = torch.tensor(0.0, device=device)
    
    if len(pred_masks) > 0:
        mask_bce_loss_sum = 0
        mask_dice_loss_sum = 0
        num_masks = 0
        
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], \
                f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            
            mask_bce_loss_sum += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss_sum += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        if num_masks > 0:
            mask_bce_loss = bce_loss_weight * mask_bce_loss_sum / num_masks
            mask_dice_loss = dice_loss_weight * mask_dice_loss_sum / num_masks
    
    mask_loss = mask_bce_loss + mask_dice_loss
    return mask_bce_loss, mask_dice_loss, mask_loss


def compute_3d_mask_loss(
    pred_3d_masks: List[torch.Tensor],
    gt_3d_masks: List[torch.Tensor],
    bce_loss_weight: float,
    dice_loss_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 3D 点云掩码损失
    
    Args:
        pred_3d_masks: 预测 3D 掩码列表，每个元素形状为 [N]
        gt_3d_masks: 真实 3D 掩码列表，每个元素形状为 [N]
        bce_loss_weight: BCE 损失权重
        dice_loss_weight: Dice 损失权重
        device: 设备
        
    Returns:
        mask_3d_bce_loss: 3D BCE 损失
        mask_3d_dice_loss: 3D Dice 损失
        mask_3d_loss: 总 3D 掩码损失
    """
    mask_3d_bce_loss = torch.tensor(0.0, device=device)
    mask_3d_dice_loss = torch.tensor(0.0, device=device)
    
    if len(pred_3d_masks) > 0 and gt_3d_masks is not None:
        mask_3d_bce_loss_sum = 0
        mask_3d_dice_loss_sum = 0
        num_3d_masks = 0
        
        for batch_idx in range(len(pred_3d_masks)):
            if gt_3d_masks[batch_idx] is not None:
                gt_3d_mask = gt_3d_masks[batch_idx]  # [N]
                pred_3d_mask = pred_3d_masks[batch_idx]  # [N]
                
                if gt_3d_mask.shape == pred_3d_mask.shape:
                    # BCE 损失（pred_3d_mask 已经经过 sigmoid）
                    bce = F.binary_cross_entropy(
                        pred_3d_mask.clamp(1e-6, 1-1e-6), 
                        gt_3d_mask.float(), 
                        reduction='mean'
                    )
                    mask_3d_bce_loss_sum += bce
                    
                    # Dice 损失
                    intersection = (pred_3d_mask * gt_3d_mask.float()).sum()
                    union = pred_3d_mask.sum() + gt_3d_mask.float().sum()
                    dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
                    mask_3d_dice_loss_sum += dice
                    num_3d_masks += 1
        
        if num_3d_masks > 0:
            mask_3d_bce_loss = bce_loss_weight * mask_3d_bce_loss_sum / num_3d_masks
            mask_3d_dice_loss = dice_loss_weight * mask_3d_dice_loss_sum / num_3d_masks
    
    mask_3d_loss = mask_3d_bce_loss + mask_3d_dice_loss
    return mask_3d_bce_loss, mask_3d_dice_loss, mask_3d_loss


def compute_dummy_loss(model, device: torch.device) -> torch.Tensor:
    """
    计算虚拟损失以保持所有参数连接到计算图
    
    使用1e-8而不是0避免torch梯度图优化掉，确保即使某些模块在当前批次中未使用，
    它们的参数仍然连接到 Loss
    
    Args:
        model: LISA 模型
        device: 设备
        
    Returns:
        dummy_loss: 虚拟损失
    """
    dummy_loss = torch.tensor(0.0, device=device)
    
    # 1. 确保 point_cloud_segmentor 的所有参数连接到计算图
    if hasattr(model, 'point_cloud_segmentor'):
        for param in model.point_cloud_segmentor.parameters():
            if param.requires_grad:
                dummy_loss = dummy_loss + (param ** 2).sum() * 1e-8
    
    # 2. 确保 SAM mask_decoder 的所有参数连接到计算图
    if hasattr(model, 'visual_model') and hasattr(model.visual_model, 'mask_decoder'):
        for param in model.visual_model.mask_decoder.parameters():
            if param.requires_grad:
                dummy_loss = dummy_loss + (param ** 2).sum() * 1e-8
    
    return dummy_loss




class LossMetrics:
    """损失指标跟踪器"""
    
    def __init__(self):
        """初始化损失指标跟踪器"""
        self.losses = AverageMeter("Loss", ":.4f")
        self.ce_losses = AverageMeter("CeLoss", ":.4f")
        self.mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
        self.mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
        self.mask_losses = AverageMeter("MaskLoss", ":.4f")
        self.mask_3d_bce_losses = AverageMeter("Mask3DBCELoss", ":.4f")
        self.mask_3d_dice_losses = AverageMeter("Mask3DDICELoss", ":.4f")
        self.mask_3d_losses = AverageMeter("Mask3DLoss", ":.4f")
    
    def update(self, output_dict: Dict[str, torch.Tensor], batch_size: int):
        """
        更新损失指标
        
        Args:
            output_dict: 模型输出字典，包含各种损失
            batch_size: 批次大小
        """
        self.losses.update(output_dict["loss"].item(), batch_size)
        self.ce_losses.update(output_dict["ce_loss"].item(), batch_size)
        self.mask_bce_losses.update(output_dict["mask_bce_loss"].item(), batch_size)
        self.mask_dice_losses.update(output_dict["mask_dice_loss"].item(), batch_size)
        self.mask_losses.update(output_dict["mask_loss"].item(), batch_size)
        
        # 3D 损失可能不存在
        mask_3d_bce_loss = output_dict.get("mask_3d_bce_loss", torch.tensor(0.0))
        mask_3d_dice_loss = output_dict.get("mask_3d_dice_loss", torch.tensor(0.0))
        mask_3d_loss = output_dict.get("mask_3d_loss", torch.tensor(0.0))
        
        self.mask_3d_bce_losses.update(mask_3d_bce_loss.item(), batch_size)
        self.mask_3d_dice_losses.update(mask_3d_dice_loss.item(), batch_size)
        self.mask_3d_losses.update(mask_3d_loss.item(), batch_size)
    
    def reset(self):
        """重置所有损失指标"""
        self.losses.reset()
        self.ce_losses.reset()
        self.mask_bce_losses.reset()
        self.mask_dice_losses.reset()
        self.mask_losses.reset()
        self.mask_3d_bce_losses.reset()
        self.mask_3d_dice_losses.reset()
        self.mask_3d_losses.reset()
    
    def all_reduce(self):
        """在分布式训练中汇总所有损失"""
        self.losses.all_reduce()
        self.ce_losses.all_reduce()
        self.mask_bce_losses.all_reduce()
        self.mask_dice_losses.all_reduce()
        self.mask_losses.all_reduce()
        self.mask_3d_bce_losses.all_reduce()
        self.mask_3d_dice_losses.all_reduce()
        self.mask_3d_losses.all_reduce()
    
    def get_meters(self) -> Dict[str, AverageMeter]:
        """
        获取所有损失 meter
        
        Returns:
            包含所有损失 meter 的字典
        """
        return {
            "loss": self.losses,
            "ce_loss": self.ce_losses,
            "mask_bce_loss": self.mask_bce_losses,
            "mask_dice_loss": self.mask_dice_losses,
            "mask_loss": self.mask_losses,
            "mask_3d_bce_loss": self.mask_3d_bce_losses,
            "mask_3d_dice_loss": self.mask_3d_dice_losses,
            "mask_3d_loss": self.mask_3d_losses,
        }




''' ------------------------------------------- eval ------------------------------------------- '''


class SegmentationMetrics:
    """分割任务评估指标计算器"""
    
    def __init__(self):
        """初始化评估指标计算器"""
        # 2D 分割评估指标
        self.intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        self.union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        self.acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.AVERAGE)
        
        # 3D 点云评估指标
        self.intersection_meter_3d = AverageMeter("Intersec3D", ":6.3f", Summary.SUM)
        self.union_meter_3d = AverageMeter("Union3D", ":6.3f", Summary.SUM)
        self.acc_iou_meter_3d = AverageMeter("gIoU3D", ":6.3f", Summary.AVERAGE)
        
        # 样本计数器
        self.num_2d_samples = 0
        self.num_3d_samples = 0
    
    def reset(self):
        """重置所有指标"""
        self.intersection_meter.reset()
        self.union_meter.reset()
        self.acc_iou_meter.reset()
        self.intersection_meter_3d.reset()
        self.union_meter_3d.reset()
        self.acc_iou_meter_3d.reset()
        self.num_2d_samples = 0
        self.num_3d_samples = 0
    
    def update_2d_metrics(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.0,
        num_classes: int = 2,
        ignore_index: int = 255
    ):
        """
        更新 2D 分割指标
        
        Args:
            pred_mask: 预测 mask，形状为 [H, W]
            gt_mask: 真实 mask，形状为 [H, W]
            threshold: 二值化阈值
            num_classes: 类别数
            ignore_index: 忽略的索引
        """
        self.num_2d_samples += 1
        
        # 二值化和类型转换
        mask_gt = gt_mask.int()
        mask_pred = (pred_mask > threshold).int()
        
        # 确保形状一致
        assert mask_gt.shape == mask_pred.shape, \
            f"Shape mismatch: gt {mask_gt.shape}, pred {mask_pred.shape}"
        
        # 计算 intersection 和 union
        intersection_i, union_i, _ = intersectionAndUnionGPU(
            mask_pred.contiguous().clone(),
            mask_gt.contiguous(),
            num_classes,
            ignore_index=ignore_index
        )
        
        # 计算 IoU
        acc_iou = intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0  # no-object target
        
        # 转换为 numpy 并更新
        intersection = intersection_i.cpu().numpy()
        union = union_i.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.acc_iou_meter.update(acc_iou, n=1)
    
    def update_3d_metrics(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        num_classes: int = 2,
        ignore_index: int = 255
    ):
        """
        更新 3D 点云分割指标
        
        Args:
            pred_mask: 预测 mask，形状为 [N]
            gt_mask: 真实 mask，形状为 [N]
            threshold: 二值化阈值
            num_classes: 类别数
            ignore_index: 忽略的索引
        """
        self.num_3d_samples += 1
        
        # 二值化和类型转换
        mask_gt = gt_mask.int()
        mask_pred = (pred_mask > threshold).int()
        
        # 确保形状一致
        assert mask_gt.shape == mask_pred.shape, \
            f"Shape mismatch: gt {mask_gt.shape}, pred {mask_pred.shape}"
        
        # 计算 intersection 和 union
        intersection_i, union_i, _ = intersectionAndUnionGPU(
            mask_pred.contiguous().clone(),
            mask_gt.contiguous(),
            num_classes,
            ignore_index=ignore_index
        )
        
        # 计算 IoU
        acc_iou = intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0  # no-object target
        
        # 转换为 numpy 并更新
        intersection = intersection_i.cpu().numpy()
        union = union_i.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy()
        
        self.intersection_meter_3d.update(intersection)
        self.union_meter_3d.update(union)
        self.acc_iou_meter_3d.update(acc_iou, n=1)
    
    def compute_2d_results(self) -> Tuple[float, float]:
        """
        计算 2D 分割的最终结果
        
        Returns:
            giou: Global IoU
            ciou: Class IoU
        """
        # 汇总结果
        self.intersection_meter.all_reduce()
        self.union_meter.all_reduce()
        self.acc_iou_meter.all_reduce()
        
        # 计算 IoU
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        ciou = iou_class[1] if isinstance(iou_class, np.ndarray) and iou_class.size > 1 else 0.0
        giou = self.acc_iou_meter.avg[1] if isinstance(self.acc_iou_meter.avg, np.ndarray) and self.acc_iou_meter.avg.size > 1 else 0.0
        
        return giou, ciou
    
    def compute_3d_results(self) -> Tuple[float, float]:
        """
        计算 3D 点云分割的最终结果
        
        Returns:
            giou_3d: Global IoU for 3D
            ciou_3d: Class IoU for 3D
        """
        # 汇总结果
        self.intersection_meter_3d.all_reduce()
        self.union_meter_3d.all_reduce()
        self.acc_iou_meter_3d.all_reduce()
        
        # 计算 IoU
        iou_class_3d = self.intersection_meter_3d.sum / (self.union_meter_3d.sum + 1e-10)
        
        if isinstance(iou_class_3d, np.ndarray) and iou_class_3d.size > 1:
            ciou_3d = iou_class_3d[1]
        else:
            ciou_3d = float(iou_class_3d) if not isinstance(iou_class_3d, np.ndarray) else iou_class_3d.item()
        
        if isinstance(self.acc_iou_meter_3d.avg, np.ndarray) and self.acc_iou_meter_3d.avg.size > 1:
            giou_3d = self.acc_iou_meter_3d.avg[1]
        else:
            giou_3d = float(self.acc_iou_meter_3d.avg) if not isinstance(self.acc_iou_meter_3d.avg, np.ndarray) else self.acc_iou_meter_3d.avg.item()
        
        return giou_3d, ciou_3d
    
    def get_sample_counts(self) -> Tuple[int, int]:
        """
        获取样本计数
        
        Returns:
            num_2d_samples: 2D 样本数
            num_3d_samples: 3D 样本数
        """
        return self.num_2d_samples, self.num_3d_samples

def evaluate_segmentation_batch(
    output_dict: Dict[str, any],
    metrics: SegmentationMetrics,
    mask_threshold_2d: float = 0.0,
    mask_threshold_3d: float = 0.5,
):
    """
    评估一个 batch 的分割结果
    
    Args:
        output_dict: 模型输出字典
        metrics: 评估指标计算器
        mask_threshold_2d: 2D mask 二值化阈值
        mask_threshold_3d: 3D mask 二值化阈值
    """
    # 评估 2D 分割
    if "pred_masks" in output_dict and output_dict["pred_masks"] is not None:
        pred_masks = output_dict["pred_masks"]  # List[Tensor]
        gt_masks = output_dict["gt_masks"]  # List[Tensor]
        
        for batch_idx in range(len(pred_masks)):
            metrics.update_2d_metrics(
                pred_masks[batch_idx],
                gt_masks[batch_idx],
                threshold=mask_threshold_2d
            )
    
    # 评估 3D 点云分割
    if "pred_3d_masks" in output_dict and output_dict["pred_3d_masks"] is not None:
        pred_3d_masks = output_dict["pred_3d_masks"]  # List[Tensor]
        gt_3d_masks = output_dict["gt_3d_masks"]  # List[Tensor]
        
        for batch_idx in range(len(pred_3d_masks)):
            metrics.update_3d_metrics(
                pred_3d_masks[batch_idx],
                gt_3d_masks[batch_idx],
                threshold=mask_threshold_3d
            )


def print_validation_summary(
    epoch: int,
    metrics: SegmentationMetrics,
    giou: float,
    ciou: float,
    giou_3d: float,
    ciou_3d: float,
):
    """
    打印验证摘要
    
    Args:
        epoch: 当前 epoch
        metrics: 评估指标计算器
        giou: 2D Global IoU
        ciou: 2D Class IoU
        giou_3d: 3D Global IoU
        ciou_3d: 3D Class IoU
    """
    num_2d_samples, num_3d_samples = metrics.get_sample_counts()
    
    print(f"\n{'='*60}")
    print(f"Validation Summary (Epoch {epoch}):")
    print(f"  2D samples evaluated: {num_2d_samples}")
    print(f"  3D samples evaluated: {num_3d_samples}")
    print(f"{'='*60}")
    print(f"2D Segmentation - giou: {giou:.4f}, ciou: {ciou:.4f}")
    print(f"3D Point Cloud - giou: {giou_3d:.4f}, ciou: {ciou_3d:.4f}")
    
    # 警告信息
    if num_3d_samples == 0:
        print(f"\n⚠️  WARNING: No 3D samples were evaluated!")
        print(f"   Possible reasons:")
        print(f"   1. Validation dataset has no point cloud data")
        print(f"   2. Point clouds not being loaded correctly")
        print(f"   3. Model not generating pred_masks_3d")
    print(f"{'='*60}\n")


def log_metrics_to_tensorboard(
    writer,
    metrics_dict: Dict[str, float],
    global_step: int,
    prefix: str = "train"
):
    """
    将指标记录到 TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics_dict: 指标字典
        global_step: 全局步数
        prefix: 指标前缀（如 "train" 或 "val"）
    """
    for key, value in metrics_dict.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step)
