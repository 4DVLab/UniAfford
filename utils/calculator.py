import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.metrics import roc_auc_score

"""
默认输入形状：
    img_pred/gt.shape = [Batch, height, width]
    pc_pred/gt.shape = [Batch, num_points]
"""
IGNORE_INDEX = -100


def _align_ordered_query_masks(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    sample_valid_mask: Optional[torch.Tensor] = None,
    gt_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    将 token 级预测按 GT 数量做顺序对齐。

    支持旧形状 [B, ...]，也支持新形状 pred=[B, K, ...] / gt=[B, M, ...]。
    无 GT query 维时默认每个有效样本只有 1 个 GT，只监督 pred[:, 0]。
    """
    if pred_masks is None or gt_masks is None:
        return None, None
    gt_masks = gt_masks.to(pred_masks.dtype)

    pred_has_query_dim = pred_masks.dim() == gt_masks.dim() + 1
    both_have_query_dim = (
        pred_masks.dim() == gt_masks.dim()
        and pred_masks.dim() >= 3
        and pred_masks.shape[0] == gt_masks.shape[0]
        and pred_masks.shape[2:] == gt_masks.shape[2:]
        and (pred_masks.shape != gt_masks.shape or gt_valid_mask is not None)
    )
    if not pred_has_query_dim and not both_have_query_dim:
        if pred_masks.shape != gt_masks.shape:
            raise ValueError(
                f"pred/gt shape mismatch: pred={tuple(pred_masks.shape)}, gt={tuple(gt_masks.shape)}"
            )
        valid = torch.ones(pred_masks.shape[0], dtype=torch.bool, device=pred_masks.device)
        if sample_valid_mask is not None:
            valid = valid & sample_valid_mask.bool().to(pred_masks.device)
        if not valid.any():
            return None, None
        return pred_masks[valid], gt_masks[valid]

    if pred_has_query_dim:
        if pred_masks.shape[0] != gt_masks.shape[0] or pred_masks.shape[2:] != gt_masks.shape[1:]:
            raise ValueError(
                f"pred/gt spatial shape mismatch: pred={tuple(pred_masks.shape)}, gt={tuple(gt_masks.shape)}"
            )
        gt_masks = gt_masks.unsqueeze(1)
    elif pred_masks.shape[0] != gt_masks.shape[0] or pred_masks.shape[2:] != gt_masks.shape[2:]:
        raise ValueError(
            f"pred/gt spatial shape mismatch: pred={tuple(pred_masks.shape)}, gt={tuple(gt_masks.shape)}"
        )

    max_pairs = min(pred_masks.shape[1], gt_masks.shape[1])
    if max_pairs <= 0:
        return None, None
    pred_masks = pred_masks[:, :max_pairs]
    gt_masks = gt_masks[:, :max_pairs]

    valid = torch.ones(pred_masks.shape[:2], dtype=torch.bool, device=pred_masks.device)
    if sample_valid_mask is not None:
        valid = valid & sample_valid_mask.bool().to(pred_masks.device).unsqueeze(1)
    if gt_valid_mask is not None:
        valid = valid & gt_valid_mask.bool().to(pred_masks.device)[:, :max_pairs]
    if not valid.any():
        return None, None

    return pred_masks[valid], gt_masks[valid]

""" -------------------------------------- 辅助函数 ------------------------------------- """
# Blue Archwve I and you ~ QAQ
def BA_I_and_U(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    批量计算2D Dice损失（1 - Dice系数）。
    输入为 logits（内部自动 sigmoid）。
    Returns:
        loss: 标量，所有样本的平均损失
    """
    inputs = inputs.sigmoid()  # logits → [0, 1]
    inputs = inputs.flatten(1)  # [Batch, H*W]
    targets = targets.flatten(1)  # [Batch, H*W]
    
    numerator = 2 * (inputs / scale * targets).sum(dim=1)
    denominator = (inputs / scale).sum(dim=1) + (targets / scale).sum(dim=1)
    dice = (numerator + eps) / (denominator + eps)
    
    loss = 1 - dice
    return loss.mean()


def sigmoid_CE_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    批量计算 Sigmoid 交叉熵损失（标准 BCE）。
    输入为 logits（F.binary_cross_entropy_with_logits 内部自动 sigmoid）。
    Returns:
        loss: 标量，所有样本的平均损失
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1).mean(dim=1)
    return loss.mean()


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Sigmoid Focal Loss —— 对 BCE 的改进，降低已分类正确（易分类）像素的权重，
    迫使模型关注困难样本（如物体上的非 affordance 像素）。

    核心公式：FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - gamma=0 时退化为标准 BCE
    - gamma>0 时，对 p_t 接近 1（易分类）的像素大幅降低权重

    适用于 2D affordance 分割：当模型预测覆盖了整个物体时，
    物体上非 affordance 区域的 false positive 像素将获得更大的梯度信号，
    推动模型精化到真正的 affordance 区域。

    Args:
        inputs: [Batch, H, W] raw logits（未经 sigmoid）
        targets: [Batch, H, W] GT 标签，值域 {0, 1}
        alpha: 正/负样本平衡因子，alpha 用于正样本，(1-alpha) 用于负样本
        gamma: 聚焦参数，越大越聚焦于困难样本
    Returns:
        loss: 标量，所有样本的平均 Focal Loss
    """
    p = inputs.sigmoid()
    # 标准 BCE（per-pixel）
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t: 正确类别的预测概率（p if target=1, 1-p if target=0）
    p_t = p * targets + (1 - p) * (1 - targets)
    # focal 调制因子：(1 - p_t)^gamma, 易分类样本权重趋近 0
    focal_weight = (1 - p_t) ** gamma
    # alpha 平衡因子
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight
    loss = focal_weight * ce_loss
    return loss.flatten(1).mean(dim=1).mean()


def img_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    focal_loss_weight: float = 2.0,
    dice_loss_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量计算 2D 掩码总损失（Focal + Dice）。
    用 Focal Loss 替代标准 BCE，解决 affordance 区域是物体子集时梯度信号不足的问题。

    Args:
        pred_masks: [Batch, H, W] 预测 logits
        gt_masks: [Batch, H, W] GT 掩码
        focal_loss_weight: Focal Loss 权重（对应旧版 bce_loss_weight 的位置）
        dice_loss_weight: Dice Loss 权重
        focal_alpha: Focal Loss 的 alpha 参数
        focal_gamma: Focal Loss 的 gamma 参数（0 = 退化为标准 BCE）
    Returns:
        img_focal_loss: Focal 损失
        img_dice_loss: Dice 损失
        img_total_loss: 加权总损失
    """
    img_focal = sigmoid_focal_loss(pred_masks, gt_masks, alpha=focal_alpha, gamma=focal_gamma)
    img_dice = img_DICE_loss(pred_masks, gt_masks)
    img_total = focal_loss_weight * img_focal + dice_loss_weight * img_dice
    return img_focal, img_dice, img_total


""" -------------------------------------- pc Loss ------------------------------------- """
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


def dice_loss_heatmap(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    批量计算热力图连续版Dice Loss
    Args:
        pred_heatmap: [Batch, N] 预测值（logits 或概率，由 from_logits 控制）
        target_heatmap: [Batch, N] 目标值，值域 [0, 1]
        from_logits: 若为 True，先对 pred 做 sigmoid 映射到 [0,1]；
                     若为 False，假设 pred 已经是概率值。
        eps: 防止除零的小常数
    Returns:
        loss: 标量，所有样本的平均 Dice Loss
    """
    pred_flat = pred_heatmap.flatten(1)
    target_flat = target_heatmap.flatten(1)

    # 与 img_DICE_loss 保持一致：logits 输入时先做 sigmoid
    if from_logits:
        pred_flat = pred_flat.sigmoid()
    
    # 连续值交集（乘积和）
    intersection = (pred_flat * target_flat).sum(dim=1)  # [Batch]
    # 连续值"并集"（平方和）
    union = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)  # [Batch]
    
    dice = (2 * intersection + eps) / (union + eps)  # [Batch]
    loss = 1 - dice  # [Batch]
    return loss.mean()


def pc_loss(
    pred_3d_masks: torch.Tensor,
    gt_3d_masks: torch.Tensor,
    bce_loss_weight: float,
    dice_loss_weight: float,
    from_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量计算3D点云掩码总损失（BCE + Dice）
    Args:
        pred_3d_masks: [Batch, N] 预测 logits（raw model output）
        gt_3d_masks: [Batch, N] GT 标签，值域 [0, 1]
        bce_loss_weight: BCE 损失权重
        dice_loss_weight: Dice 损失权重
        from_logits: 输入是否为 logits（默认 True，与 sigmoid_CE_loss 保持一致）
    Returns:
        mask_3d_bce_loss: BCE损失
        mask_3d_dice_loss: Dice损失
        mask_3d_loss: 加权总损失
    """
    # sigmoid_CE_loss 内部使用 binary_cross_entropy_with_logits，自带 sigmoid
    mask_3d_bce_loss = sigmoid_CE_loss(pred_3d_masks, gt_3d_masks)
    # dice_loss_heatmap 现在也会在 from_logits=True 时先做 sigmoid，保持一致
    mask_3d_dice_loss = dice_loss_heatmap(pred_3d_masks, gt_3d_masks, from_logits=from_logits)
    
    mask_3d_loss = bce_loss_weight * mask_3d_bce_loss + dice_loss_weight * mask_3d_dice_loss
    
    return mask_3d_bce_loss, mask_3d_dice_loss, mask_3d_loss


def _build_route_targets(
    labels: torch.Tensor,
    img_placeholder_id: int,
    pc_placeholder_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    由语言监督标签构造 router 的 token-level 监督：
    - 默认 text 类（0）
    - label == <img_aff> -> img 类（1）
    - label == <pc_aff>  -> pc 类（2）
    - label == IGNORE_INDEX 的位置不参与监督
    """
    targets = torch.zeros_like(labels, dtype=torch.long)
    valid_mask = labels.ne(IGNORE_INDEX)
    targets = torch.where(labels.eq(int(img_placeholder_id)), torch.ones_like(targets), targets)
    targets = torch.where(labels.eq(int(pc_placeholder_id)), torch.full_like(targets, 2), targets)
    return targets, valid_mask


def route_supervision_loss(
    route_logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    img_placeholder_id: Optional[int],
    pc_placeholder_id: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    L_route: token-level 路由监督损失（CE）。
    """
    zero = torch.tensor(0.0, device=device)
    if (
        route_logits is None
        or labels is None
        or img_placeholder_id is None
        or pc_placeholder_id is None
    ):
        return zero, None, None

    bsz, seq_len, num_routes = route_logits.shape
    if num_routes != 3:
        return zero, None, None
    if labels.dim() != 2:
        return zero, None, None

    # 与 Causal LM 对齐：hidden_states[t] / logits[t] 预测 labels[t+1]
    # 因此路由监督也应基于 shift 后的 labels（而非同位 labels）。
    if labels.shape[1] <= 1:
        return zero, None, None
    pred_len = min(seq_len, labels.shape[1] - 1)
    route_logits = route_logits[:, :pred_len, :]
    labels_shift = labels[:, 1 : 1 + pred_len]
    targets, valid_mask = _build_route_targets(labels_shift, img_placeholder_id, pc_placeholder_id)
    if not valid_mask.any():
        return zero, targets, valid_mask

    token_loss = F.cross_entropy(
        route_logits.reshape(-1, num_routes),
        targets.reshape(-1),
        reduction="none",
    )
    valid_flat = valid_mask.reshape(-1).to(route_logits.dtype)
    loss = (token_loss * valid_flat).sum() / valid_flat.sum().clamp_min(1.0)
    return loss, targets, valid_mask


def route_structure_loss(
    img_any_prob: Optional[torch.Tensor],
    pc_any_prob: Optional[torch.Tensor],
    img_expected_count: Optional[torch.Tensor],
    pc_expected_count: Optional[torch.Tensor],
    img_available: Optional[torch.Tensor],
    pc_available: Optional[torch.Tensor],
    device: torch.device,
    target_present_count: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于连续概率构造样本级结构约束。

    目标分两部分：
    1) existence loss:
       - 有图样本：至少存在一个 img_aff token
       - 无图样本：不应存在 img_aff token
       - 3D 同理
    2) sparse loss:
       - affordance token 应尽量少
       - 对有该模态的样本，期望个数接近 target_present_count（默认 1）
       - 对无该模态的样本，期望个数接近 0

    注意：
    - 这里全部作用在 soft 概率统计量上，因此对 router 参数是可微的；
    - 不依赖 argmax 结果，也不使用 MoE 式负载均衡。
    """
    zero = torch.tensor(0.0, device=device)
    required = [img_any_prob, pc_any_prob, img_expected_count, pc_expected_count]
    if any(x is None for x in required):
        return zero, zero

    if img_available is None:
        img_available = torch.zeros_like(img_any_prob, dtype=torch.bool)
    else:
        img_available = img_available.bool().to(img_any_prob.device)
    if pc_available is None:
        pc_available = torch.zeros_like(pc_any_prob, dtype=torch.bool)
    else:
        pc_available = pc_available.bool().to(pc_any_prob.device)

    img_target = img_available.to(img_any_prob.dtype)
    pc_target = pc_available.to(pc_any_prob.dtype)

    exist_img = F.binary_cross_entropy(img_any_prob.clamp(1e-6, 1 - 1e-6), img_target)
    exist_pc = F.binary_cross_entropy(pc_any_prob.clamp(1e-6, 1 - 1e-6), pc_target)
    exist_loss = exist_img + exist_pc

    target_img_count = torch.where(
        img_available,
        torch.full_like(img_expected_count, float(target_present_count)),
        torch.zeros_like(img_expected_count),
    )
    target_pc_count = torch.where(
        pc_available,
        torch.full_like(pc_expected_count, float(target_present_count)),
        torch.zeros_like(pc_expected_count),
    )
    sparse_img = F.smooth_l1_loss(img_expected_count, target_img_count)
    sparse_pc = F.smooth_l1_loss(pc_expected_count, target_pc_count)
    sparse_loss = sparse_img + sparse_pc
    return exist_loss, sparse_loss


def compute_losses(
    output_dict: Dict,
    input_dict: Dict,
    device: torch.device,
    # ---- 2D (Focal + Dice) ----
    focal_loss_weight: float = 2.0,
    dice_loss_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    # ---- 3D (BCE + Dice) ----
    bce_loss_weight: float = 2.0,
    pc_dice_loss_weight: Optional[float] = None,
    # ---- LLM CE ----
    ce_loss_weight: float = 1.0,
    # ---- Router losses ----
    route_loss_weight: float = 1.0,
    route_exist_loss_weight: float = 0.25,
    route_sparse_loss_weight: float = 0.05,
    route_target_present_count: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    统一计算所有损失，返回损失字典。训练与验证共用。

    2D 分支使用 Focal Loss + Dice（解决 affordance ⊂ 物体时的梯度不足问题）；
    3D 分支使用 BCE + Dice（保持原有方案）。

    Args:
        output_dict: 模型输出字典，应包含:
            - "image_logits": [B, H, W] 2D 预测 logits（可选）
            - "point_logits": [B, N] 3D 预测 logits（可选）
            - "ce_loss": 语言模型交叉熵损失（可选，由模型内部计算）
        input_dict: 输入字典，应包含:
            - "img_gt_tensor": [B, H, W] 2D GT 掩码（可选）
            - "pc_gt_tensor": [B, N] 3D GT 掩码（可选）
        focal_loss_weight: 2D Focal Loss 权重
        dice_loss_weight: 2D / 3D Dice Loss 权重（3D 可单独用 pc_dice_loss_weight 覆盖）
        focal_alpha: Focal Loss alpha
        focal_gamma: Focal Loss gamma（0 = 退化为标准 BCE）
        bce_loss_weight: 3D BCE 权重
        pc_dice_loss_weight: 3D Dice 权重（None 时复用 dice_loss_weight）
        ce_loss_weight: 语言模型 CE 权重
        route_loss_weight: token-level 路由监督损失权重
        route_exist_loss_weight: 样本级存在性约束权重
        route_sparse_loss_weight: affordance token 稀疏/数量约束权重
        route_target_present_count: 有效模态样本中 aff token 的目标期望个数

    Returns:
        Dict[str, Tensor]: 统一损失字典，始终包含以下键（缺失的分支为 0）:
            "loss", "ce_loss",
            "img_focal_loss", "img_dice_loss", "img_loss",
            "pc_bce_loss", "pc_dice_loss", "pc_loss",
            "route_loss", "route_exist_loss", "route_sparse_loss"
    """
    zero = torch.tensor(0.0, device=device)
    pc_dice_w = pc_dice_loss_weight if pc_dice_loss_weight is not None else dice_loss_weight

    # ---------- 2D 图像掩码损失（Focal + Dice）----------
    img_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    if img_logits is not None and img_gt is not None:
        aligned_img_logits, aligned_img_gt = _align_ordered_query_masks(
            pred_masks=img_logits,
            gt_masks=img_gt,
            sample_valid_mask=input_dict.get("img_valid_mask"),
            gt_valid_mask=input_dict.get("img_gt_valid_mask"),
        )
        if aligned_img_logits is not None:
            img_focal, img_dice, img_total = img_loss(
                pred_masks=aligned_img_logits,
                gt_masks=aligned_img_gt,
                focal_loss_weight=focal_loss_weight,
                dice_loss_weight=dice_loss_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
        else:
            img_focal = img_dice = img_total = zero
    else:
        img_focal = img_dice = img_total = zero

    # ---------- 3D 点云掩码损失（BCE + Dice）----------
    pc_logits = output_dict.get("point_logits")
    pc_gt = input_dict.get("pc_gt_tensor")
    if pc_logits is not None and pc_gt is not None:
        pc_valid_lengths = input_dict.get("pc_valid_lengths")
        pc_sample_valid = None if pc_valid_lengths is None else pc_valid_lengths > 0
        aligned_pc_logits, aligned_pc_gt = _align_ordered_query_masks(
            pred_masks=pc_logits,
            gt_masks=pc_gt,
            sample_valid_mask=pc_sample_valid,
            gt_valid_mask=input_dict.get("pc_gt_valid_mask"),
        )
        if aligned_pc_logits is not None:
            pc_bce, pc_dice, pc_total = pc_loss(
                pred_3d_masks=aligned_pc_logits,
                gt_3d_masks=aligned_pc_gt,
                bce_loss_weight=bce_loss_weight,
                dice_loss_weight=pc_dice_w,
            )
        else:
            pc_bce = pc_dice = pc_total = zero
    else:
        pc_bce = pc_dice = pc_total = zero

    # ---------- 语言模型 CE 损失 ----------
    ce = output_dict.get("ce_loss", zero)
    if not isinstance(ce, torch.Tensor):
        ce = torch.tensor(float(ce), device=device)

    # ---------- Router 损失 ----------
    route_logits = output_dict.get("route_logits")
    labels = output_dict.get("labels", input_dict.get("labels"))
    img_placeholder_id = output_dict.get("img_placeholder_id")
    pc_placeholder_id = output_dict.get("pc_placeholder_id")
    route_ce, route_targets, route_valid = route_supervision_loss(
        route_logits=route_logits,
        labels=labels,
        img_placeholder_id=img_placeholder_id,
        pc_placeholder_id=pc_placeholder_id,
        device=device,
    )

    route_exist, route_sparse = route_structure_loss(
        img_any_prob=output_dict.get("img_any_prob"),
        pc_any_prob=output_dict.get("pc_any_prob"),
        img_expected_count=output_dict.get("img_expected_count"),
        pc_expected_count=output_dict.get("pc_expected_count"),
        img_available=input_dict.get("img_valid_mask"),
        pc_available=input_dict.get("pc_valid_lengths",0) > 0,
        device=device,
        target_present_count=route_target_present_count,
    )

    # ---------- 总损失 ----------
    total_loss = (
        ce_loss_weight * ce
        + img_total
        + pc_total
        + route_loss_weight * route_ce
        + route_exist_loss_weight * route_exist
        + route_sparse_loss_weight * route_sparse
    )

    return {
        "loss": total_loss,
        "ce_loss": ce,
        # 2D 分项（Focal 替代了旧版 BCE）
        "img_focal_loss": img_focal,
        "img_dice_loss": img_dice,
        "img_loss": img_total,
        # 3D 分项
        "pc_bce_loss": pc_bce,
        "pc_dice_loss": pc_dice,
        "pc_loss": pc_total,
        # Router 分项
        "route_loss": route_ce,
        "route_exist_loss": route_exist,
        "route_sparse_loss": route_sparse,
    }


""" -------------------------------------- 3D 评估指标 ------------------------------------- """

def pc_MAE(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    批量计算3D平均绝对误差
    Returns:
        mae: [Batch] 个样本的平均 MAE
    """
    # GREAT: sum(abs(pred - label)) / total_points
    return torch.abs(pred_mask - gt_mask).mean()


def pc_AUC(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_threshold: float = 0.5,
) -> torch.Tensor:
    """
    批量计算3D ROC曲线下面积（AUC），与 GREAT 对齐使用 sklearn.metrics.roc_auc_score。
    Args:
        num_thresholds: 保留参数以兼容旧调用；sklearn 实现不使用手动阈值扫描。
    Returns:
        auc: [Batch] 个样本的平均 AUC
    """
    device = pred_mask.device
    dtype = pred_mask.dtype

    pred_np = pred_mask.detach().flatten(1).cpu().numpy()
    # GREAT: targets = targets >= 0.5; targets = targets.astype(int)
    gt_np = (gt_mask.detach().flatten(1).cpu().numpy() >= gt_threshold).astype(np.int32)

    auc_list = []
    for pred_b, gt_b in zip(pred_np, gt_np):
        # roc_auc_score requires both positive and negative labels.
        if gt_b.max() == gt_b.min():
            auc_list.append(np.nan)
            continue
        auc_list.append(roc_auc_score(gt_b, pred_b))

    if len(auc_list) == 0 or np.all(np.isnan(auc_list)):
        auc = 0.0
    else:
        auc = float(np.nanmean(auc_list))
    return torch.tensor(auc, device=device, dtype=dtype)


def pc_aIOU(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    num_thresholds: int = 20
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
    # GREAT: targets are binarized with >= 0.5 before IoU/AUC.
    gt_flat = gt_mask.flatten(1) >= 0.5  # [Batch, N]
    
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


def pc_SIM(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    批量计算点云预测的相似度（SIM / Histogram Intersection Similarity）。

    标准做法：先将 pred 和 gt 各自 L1 归一化为概率分布，再计算直方图交集。
    若某样本的 pred 或 gt 全零，SIM 定义为 0。

    Args:
        pred_mask: [Batch, N] 预测概率，值域 [0, 1]
        gt_mask: [Batch, N] GT 概率/标签，值域 [0, 1]
    Returns:
        sim: [Batch] 每个样本的 SIM
    """
    pred_flat = pred_mask.flatten(1).clamp(min=0)  # [Batch, N]
    gt_flat = gt_mask.flatten(1).clamp(min=0)

    pred_sum = pred_flat.sum(dim=1, keepdim=True)  # [Batch, 1]
    gt_sum = gt_flat.sum(dim=1, keepdim=True)

    pred_norm = pred_flat / (pred_sum + eps)
    gt_norm = gt_flat / (gt_sum + eps)

    sim = torch.min(pred_norm, gt_norm).sum(dim=1)  # [Batch]

    both_nonzero = (pred_sum.squeeze(-1) > eps) & (gt_sum.squeeze(-1) > eps)
    sim = sim * both_nonzero.float()
    return sim


def pc_IoU(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    批量计算点云IoU
    
    Args:
        pred_mask: [Batch, N] 预测掩码，值域 [0, 1]
        gt_mask: [Batch, N] 真实掩码，值域 {0, 1}
        threshold: 二值化阈值
        
    Returns:
        iou: [Batch] 每个样本的IoU
    """
    pred_flat = pred_mask.flatten(1)  # [Batch, N]
    gt_flat = gt_mask.flatten(1) >= 0.5  # [Batch, N]
    
    pred_bool = pred_flat >= threshold  # [Batch, N]
    
    intersection = (pred_bool & gt_flat).sum(dim=1).float()  # [Batch]
    union = (pred_bool | gt_flat).sum(dim=1).float()  # [Batch]
    
    iou = intersection / (union + 1e-8)  # [Batch]
    return iou


""" -------------------------------------- 2D 评估指标 ------------------------------------- """

def img_I_and_U(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算 2D 图像的交集和并集（用于累积后计算 cIoU）。

    Args:
        pred_mask: [Batch, H, W] 预测概率，值域 [0, 1]
        gt_mask: [Batch, H, W] GT 掩码，值域 {0, 1}
        threshold: 二值化阈值
    Returns:
        intersection: [Batch]
        union: [Batch]
    """
    pred_bool = (pred_mask.flatten(1) >= threshold)
    gt_bool = gt_mask.flatten(1).bool()
    intersection = (pred_bool & gt_bool).sum(dim=1).float()
    union = (pred_bool | gt_bool).sum(dim=1).float()
    return intersection, union


def img_IoU(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    批量计算 2D 图像 IoU（逐样本，gIoU = mean(img_IoU)）。

    Args:
        pred_mask: [Batch, H, W] 预测概率，值域 [0, 1]
        gt_mask: [Batch, H, W] GT 掩码，值域 {0, 1}
        threshold: 二值化阈值
    Returns:
        iou: [Batch] 每个样本的 IoU
    """
    intersection, union = img_I_and_U(pred_mask, gt_mask, threshold)
    return intersection / (union + 1e-8)


def img_P_at_IoU_thresholds(iou_scores: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    根据逐样本 IoU 计算 P@thresholds。

    与 Affordance-R1 的 calculate_iou.py 对齐：使用 ``iou > threshold`` 作为命中条件。
    """
    if iou_scores.numel() == 0:
        return torch.zeros_like(thresholds, dtype=torch.float32)
    iou_scores = iou_scores.float().view(-1, 1)
    thresholds = thresholds.to(device=iou_scores.device, dtype=iou_scores.dtype).view(1, -1)
    return (iou_scores > thresholds).float().mean(dim=0)


def img_KLD(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    批量计算 2D Kullback-Leibler Divergence（KLD）。

    先将预测和 GT 按样本 L1 归一化为概率分布，再计算 KL(gt || pred)。

    Args:
        pred_mask: [Batch, H, W] 预测概率，值域 [0, 1]
        gt_mask: [Batch, H, W] GT 概率/标签，值域 [0, 1]
    Returns:
        kld: [Batch] 每个样本的 KLD，越低越好
    """
    pred_flat = pred_mask.flatten(1).clamp(min=0)
    gt_flat = gt_mask.flatten(1).clamp(min=0)

    pred_norm = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + eps)
    gt_norm = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + eps)

    return (gt_norm * ((gt_norm + eps).log() - (pred_norm + eps).log())).sum(dim=1)


def img_SIM(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    批量计算 2D 相似度（SIM / Histogram Intersection Similarity）。

    标准做法：pred 和 gt 分别 L1 归一化后取逐像素最小值求和。
    若某样本 pred 或 gt 全零，SIM 定义为 0。
    """
    pred_flat = pred_mask.flatten(1).clamp(min=0)
    gt_flat = gt_mask.flatten(1).clamp(min=0)

    pred_sum = pred_flat.sum(dim=1, keepdim=True)
    gt_sum = gt_flat.sum(dim=1, keepdim=True)
    pred_norm = pred_flat / (pred_sum + eps)
    gt_norm = gt_flat / (gt_sum + eps)

    sim = torch.min(pred_norm, gt_norm).sum(dim=1)
    both_nonzero = (pred_sum.squeeze(-1) > eps) & (gt_sum.squeeze(-1) > eps)
    return sim * both_nonzero.float()


def img_NSS(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    fixation_threshold: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    批量计算 2D Normalized Scanpath Saliency（NSS）。

    预测图按样本做 z-score；GT 先 min-max 归一化，再以 fixation_threshold
    二值化作为 fixation map。
    """
    pred_flat = pred_mask.flatten(1).float()
    gt_flat = gt_mask.flatten(1).float()

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    pred_std = pred_flat.std(dim=1, keepdim=True, unbiased=False)
    pred_z = (pred_flat - pred_mean) / (pred_std + eps)

    gt_min = gt_flat.min(dim=1, keepdim=True).values
    gt_max = gt_flat.max(dim=1, keepdim=True).values
    gt_norm = (gt_flat - gt_min) / (gt_max - gt_min + eps)
    fixation = (gt_norm > fixation_threshold).to(pred_z.dtype)

    fixation_count = fixation.sum(dim=1)
    nss = (pred_z * fixation).sum(dim=1) / (fixation_count + eps)
    return torch.where(fixation_count > 0, nss, torch.zeros_like(nss))


def img_batch_metrics(
    output_dict: Dict,
    input_dict: Dict,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    从模型输出和 batch 输入中计算 2D 指标标量，用于训练过程日志记录。

    复用 img_IoU / img_I_and_U / img_KLD / img_SIM / img_NSS；若当前 batch
    没有有效图像监督，则返回空字典。
    """
    image_logits = output_dict.get("image_logits")
    img_gt = input_dict.get("img_gt_tensor")
    if image_logits is None or img_gt is None:
        return {}

    aligned_logits, aligned_gt = _align_ordered_query_masks(
        pred_masks=image_logits.detach(),
        gt_masks=img_gt,
        sample_valid_mask=input_dict.get("img_valid_mask"),
        gt_valid_mask=input_dict.get("img_gt_valid_mask"),
    )
    if aligned_logits is None:
        return {}
    preds_2d = aligned_logits.sigmoid()
    target_2d = aligned_gt.float()

    inter_2d, union_2d = img_I_and_U(preds_2d, target_2d, threshold=threshold)
    total_union = union_2d.sum()
    iou_2d = img_IoU(preds_2d, target_2d, threshold=threshold)
    p_thresholds = img_P_at_IoU_thresholds(
        iou_2d,
        torch.arange(0.5, 0.96, 0.05, device=iou_2d.device),
    )
    ciou_2d = (inter_2d.sum() / (total_union + 1e-8)).item() if total_union > 0 else 0.0
    return {
        "giou_2d": iou_2d.mean().item(),
        "ciou_2d": ciou_2d,
        "p50_2d": p_thresholds[0].item(),
        "p50_95_2d": p_thresholds.mean().item(),
        "kld_2d": img_KLD(preds_2d, target_2d).mean().item(),
        "sim_2d": img_SIM(preds_2d, target_2d).mean().item(),
        "nss_2d": img_NSS(preds_2d, target_2d).mean().item(),
    }
