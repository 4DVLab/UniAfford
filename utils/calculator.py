import torch
import torch.nn.functional as F

"""
默认输入形状：
    img_pred/gt.shape = [Batch, height, width, 1]
    pc_pred/gt.shape = [Batch, num_points, 1]
"""

""" -------------------------------------- 辅助函数 ------------------------------------- """
# Blue Archwve I and you ~ QAQ
def BA_I_and_U(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
    """
    批量计算交集和并集
    Returns:
        intersection: [Batch] 每个样本的交集
        union: [Batch] 每个样本的并集
    """
    gt_mask_float = gt_mask.to(pred_mask.dtype)
    
    # 展平除了batch维度外的所有维度
    pred_flat = pred_mask.flatten(1)
    gt_flat = gt_mask_float.flatten(1)
    
    intersection = (pred_flat * gt_flat).sum(dim=1)  # [Batch]
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)  # [Batch]
    
    return intersection, union

""" -------------------------------------- img Loss ------------------------------------- """

def img_DICE_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scale: float = 1000,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    批量计算2D Dice损失（1 - Dice系数）
    Returns:
        loss: 标量，所有样本的平均损失
    """
    batch_size = inputs.shape[0]
    
    inputs = inputs.sigmoid()  # [Batch, H, W, 1]
    inputs = inputs.flatten(1)  # [Batch, H*W*1]
    targets = targets.flatten(1)  # [Batch, H*W*1]
    
    # 逐样本计算 Dice 系数
    numerator = 2 * (inputs / scale * targets).sum(dim=1)  # [Batch]
    denominator = (inputs / scale).sum(dim=1) + (targets / scale).sum(dim=1)  # [Batch]
    dice = (numerator + eps) / (denominator + eps)  # [Batch]
    
    loss = 1 - dice  # [Batch]
    return loss.mean()  # 返回平均损失


def sigmoid_CE_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    批量计算Sigmoid交叉熵损失（二元交叉熵）
    Returns:
        loss: 标量，所有样本的平均损失
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1).mean(dim=1)  # [Batch]
    return loss.mean()  # 返回平均损失


def img_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    bce_loss_weight: float,
    dice_loss_weight: float,
) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    """
    批量计算2D掩码总损失（BCE + Dice）
    Returns:
        mask_bce_loss: BCE损失
        mask_dice_loss: Dice损失
        mask_loss: 总损失
    """
    mask_bce_loss = sigmoid_CE_loss(pred_masks, gt_masks)
    mask_dice_loss = img_DICE_loss(pred_masks, gt_masks)
    
    mask_loss =  bce_loss_weight * mask_bce_loss + dice_loss_weight * mask_dice_loss
    
    return mask_bce_loss, mask_dice_loss, mask_loss


""" -------------------------------------- pc Loss ------------------------------------- """
def pc_DICE_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    批量计算3D Dice损失（1 - Dice系数）
    Returns:
        loss: 标量，所有样本的平均损失
    """
    intersection, union = BA_I_and_U(pred_mask, gt_mask)
    dice = (2 * intersection + eps) / (union + eps)  # [Batch]
    loss = 1 - dice  # [Batch]
    return loss.mean()


def pc_loss(
    pred_3d_masks: torch.Tensor,
    gt_3d_masks: torch.Tensor,
    bce_loss_weight: float,
    dice_loss_weight: float,
) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    """
    批量计算3D点云掩码总损失（BCE + Dice）
    Returns:
        mask_3d_bce_loss: BCE损失
        mask_3d_dice_loss: Dice损失
        mask_3d_loss: 总损失
    """
    mask_3d_bce_loss = sigmoid_CE_loss(pred_3d_masks, gt_3d_masks)
    mask_3d_dice_loss = pc_DICE_loss(pred_3d_masks, gt_3d_masks)
    
    mask_3d_loss = bce_loss_weight * mask_3d_bce_loss + dice_loss_weight * mask_3d_dice_loss
    
    return mask_3d_bce_loss, mask_3d_dice_loss, mask_3d_loss


""" -------------------------------------- 热力图损失 ------------------------------------- """

def MSE_loss(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor) -> torch.Tensor:
    """
    批量计算热力图MSE损失
    Returns:
        loss: 标量，所有样本的平均损失
    """
    pred_flat = pred_heatmap.flatten(1)
    target_flat = target_heatmap.flatten(1)
    
    mse = F.mse_loss(pred_flat, target_flat, reduction="none").mean(dim=1)  # [Batch]
    return mse.mean()


def smooth_L1_loss_heatmap(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor) -> torch.Tensor:
    """
    批量计算热力图Smooth L1损失
    Returns:
        loss: 标量，所有样本的平均损失
    """
    pred_flat = pred_heatmap.flatten(1)
    target_flat = target_heatmap.flatten(1)
    
    smooth_l1 = F.smooth_l1_loss(pred_flat, target_flat, reduction="none").mean(dim=1)  # [Batch]
    return smooth_l1.mean()


def dice_loss_heatmap(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    批量计算热力图连续版Dice Loss
    Returns:
        loss: 标量，所有样本的平均损失
    """
    pred_flat = pred_heatmap.flatten(1)
    target_flat = target_heatmap.flatten(1)
    
    # 连续值交集（乘积和）
    intersection = (pred_flat * target_flat).sum(dim=1)  # [Batch]
    # 连续值"并集"（平方和）
    union = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)  # [Batch]
    
    dice = (2 * intersection + eps) / (union + eps)  # [Batch]
    loss = 1 - dice  # [Batch]
    return loss.mean()


""" -------------------------------------- 虚拟损失 ------------------------------------- """

def dummy_loss(model) -> torch.Tensor:
    """计算虚拟损失（保持参数连接到计算图）"""
    dummy_loss_val = 0
    
    if hasattr(model, 'point_cloud_segmentor'):
        for param in model.point_cloud_segmentor.parameters():
            if param.requires_grad:
                dummy_loss_val = dummy_loss_val + (param ** 2).sum() * 1e-8

    if hasattr(model, 'visual_model') and hasattr(model.visual_model, 'mask_decoder'):
        for param in model.visual_model.mask_decoder.parameters():
            if param.requires_grad:
                dummy_loss_val = dummy_loss_val + (param ** 2).sum() * 1e-8
    
    return dummy_loss_val


""" -------------------------------------- 3D 评估指标 ------------------------------------- """

def pc_MAE(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    批量计算3D平均绝对误差
    Returns:
        mae: [Batch] 个样本的平均 MAE
    """
    mae = torch.abs(pred_mask - gt_mask).flatten(1).mean(dim=1)  # [Batch]
    return mae.mean()


def pc_AUC(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    num_thresholds: int = 100
) -> torch.Tensor:
    """
    批量计算3D ROC曲线下面积（AUC）
    Args:
        num_thresholds: 阈值数量
    Returns:
        auc: [Batch] 个样本的平均 AUC
    """
    batch_size = pred_mask.shape[0]
    pred_flat = pred_mask.flatten(1)  # [Batch, N]
    gt_flat = gt_mask.flatten(1).bool()  # [Batch, N]
    
    thresholds = torch.linspace(0, 1, num_thresholds, device=pred_mask.device)
    
    auc_list = []
    for b in range(batch_size):
        pred_b = pred_flat[b]  # [N]
        gt_b = gt_flat[b]  # [N]
        
        gt_positive = gt_b.sum().float()
        gt_negative = (~gt_b).sum().float()
        
        if gt_positive == 0 or gt_negative == 0:
            auc_list.append(torch.tensor(0.0, device=pred_mask.device))
            continue
        
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            pred_positive = pred_b >= threshold
            tp = (pred_positive & gt_b).sum().float()
            fp = (pred_positive & ~gt_b).sum().float()
            
            tpr = tp / (gt_positive + 1e-8)
            fpr = fp / (gt_negative + 1e-8)
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        tpr_tensor = torch.stack(tpr_list)
        fpr_tensor = torch.stack(fpr_list)
        sorted_indices = torch.argsort(fpr_tensor)
        fpr_sorted = fpr_tensor[sorted_indices]
        tpr_sorted = tpr_tensor[sorted_indices]
        
        auc = torch.trapz(tpr_sorted, fpr_sorted)
        auc_list.append(auc)
    
    return torch.stack(auc_list).mean()


def pc_aIOU(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    num_thresholds: int = 100
) -> torch.Tensor:
    """
    批量计算点云的多阈值平均IoU（aIoU）
    Args:
        num_thresholds: 阈值数量
    Returns:
        aiou: [Batch] 个样本的平均 aIoU
    """
    batch_size = pred_mask.shape[0]
    pred_flat = pred_mask.flatten(1)  # [Batch, N]
    gt_flat = gt_mask.flatten(1).bool()  # [Batch, N]
    
    thresholds = torch.linspace(0, 1, num_thresholds, device=pred_mask.device)
    
    aiou_list = []
    for b in range(batch_size):
        pred_b = pred_flat[b]  # [N]
        gt_b = gt_flat[b]  # [N]
        
        iou_list = []
        for threshold in thresholds:
            pred_b_bool = pred_b >= threshold
            intersection = (pred_b_bool & gt_b).sum().float()
            union = (pred_b_bool | gt_b).sum().float()
            iou = intersection / (union + 1e-8)
            iou_list.append(iou)
        
        aiou = torch.stack(iou_list).mean()
        aiou_list.append(aiou)
    
    return torch.stack(aiou_list).mean()


def pc_SIM(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    批量计算点云预测的相似度（SIM）
    Returns:
        sim: [Batch] 每个样本的SIM
    """
    pred_flat = pred_mask.flatten(1)  # [Batch, N]
    gt_flat = gt_mask.flatten(1)  # [Batch, N]
    
    min_mask = torch.min(pred_flat, gt_flat)  # [Batch, N]
    numerator = min_mask.sum(dim=1)  # [Batch]
    
    pred_sum = pred_flat.sum(dim=1)  # [Batch]
    gt_sum = gt_flat.sum(dim=1)  # [Batch]
    denominator = torch.min(pred_sum, gt_sum)  # [Batch]
    
    sim = numerator / (denominator + 1e-8)  # [Batch]
    return sim


def pc_IoU(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    批量计算点云IoU
    
    Args:
        pred_mask: [Batch, N, 1] 预测掩码，值域 [0, 1]
        gt_mask: [Batch, N, 1] 真实掩码，值域 {0, 1}
        threshold: 二值化阈值
        
    Returns:
        iou: [Batch] 每个样本的IoU
    """
    pred_flat = pred_mask.flatten(1)  # [Batch, N]
    gt_flat = gt_mask.flatten(1).bool()  # [Batch, N]
    
    pred_bool = pred_flat >= threshold  # [Batch, N]
    
    intersection = (pred_bool & gt_flat).sum(dim=1).float()  # [Batch]
    union = (pred_bool | gt_flat).sum(dim=1).float()  # [Batch]
    
    iou = intersection / (union + 1e-8)  # [Batch]
    return iou


""" -------------------------------------- 2D 评估指标 ------------------------------------- """

def img_IoU(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    批量计算2D图像IoU
    
    Args:
        pred_mask: [Batch, H, W, 1] 预测掩码，值域 [0, 1]
        gt_mask: [Batch, H, W, 1] 真实掩码，值域 {0, 1}
        threshold: 二值化阈值
        
    Returns:
        iou: [Batch] 每个样本的IoU
    """
    pred_flat = pred_mask.flatten(1)  # [Batch, H*W]
    gt_flat = gt_mask.flatten(1).bool()  # [Batch, H*W]
    
    pred_bool = pred_flat >= threshold  # [Batch, H*W]
    
    intersection = (pred_bool & gt_flat).sum(dim=1).float()  # [Batch]
    union = (pred_bool | gt_flat).sum(dim=1).float()  # [Batch]
    
    iou = intersection / (union + 1e-8)  # [Batch]
    return iou
