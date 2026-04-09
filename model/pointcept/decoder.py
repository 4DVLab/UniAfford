from configs.base_config import PointDecoderConfigs
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PointCloudHiddenStateDecoder(nn.Module):
    """使用 encoder 输出的逐点特征与 aff token 做逐点相似度对齐的解码器。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int,
        point_feature_size: int,
    ):
        super().__init__()
        self.config = config

        text_fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(text_hidden_size, 2 * text_hidden_size)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(2 * text_hidden_size, config.hidden_size)),
                    # ("dropout", nn.Dropout(0.0)),
                ]
            )
        )
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # 逐点特征维度由 point encoder backbone 的 dec_point 输出通道决定。
        # 使用显式 Linear，避免 LazyLinear 在首次前向前参数未初始化，进而影响参数统计/FSDP 初始化。
        self.point_proj = nn.Linear(point_feature_size, config.hidden_size)
        # FSDP requires parameter tensors to be at least 1D.
        self.logit_scale = nn.Parameter(torch.ones(1))

        for param in self.point_proj.parameters():
            param.requires_grad = True

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到点云分割器的嵌入空间。"""
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        return projected

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        批量生成 3D 分割掩码（目标文本概率）。

        Args:
            pred_embeddings: [B, C] — 每个样本的文本嵌入（可为 text_fc 前或后）
            per_point_features: [B, N, C]，encoder 输出的逐点特征
            per_point_mask: [B, N]，逐点特征有效位

        Returns:
            pred_masks: [B, N]，每个点属于目标文本的概率
        """
        if per_point_features is None or per_point_mask is None:
            raise ValueError("Decoder requires per_point_features/per_point_mask.")

        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        point_feat = per_point_features.to(self.config.compute_dtype)
        point_mask = per_point_mask.bool()

        text_feat = F.normalize(self.project_hidden_states(pred_embeddings), p=2, dim=-1)  # [B, H]
        point_feat = self.point_proj(point_feat)  # [B, K, H]
        point_feat = F.normalize(point_feat, p=2, dim=-1)
        point_feat = torch.where(point_mask.unsqueeze(-1), point_feat, torch.zeros_like(point_feat))

        # 逐点响应场：每个点特征直接与 aff query 做相似度。
        logits = (point_feat * text_feat.unsqueeze(1)).sum(dim=-1)  # [B, K]
        logits = logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)
        logits = torch.where(point_mask, logits, torch.zeros_like(logits))
        return logits

    def forward_with_loss(
        self,
        pred_embeddings: torch.Tensor,
        gt_masks: torch.Tensor,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional helper for training:
        - pred_embeddings: [B, C]
        - gt_masks: [B, N] (0/1)
        """
        if loss_fn is None:
            loss_fn = nn.BCELoss()
        pred_masks = self.forward(
            pred_embeddings=pred_embeddings,
            per_point_features=per_point_features,
            per_point_mask=per_point_mask,
        )
        loss = loss_fn(pred_masks, gt_masks.to(pred_masks.dtype))
        return dict(loss=loss, pred_masks=pred_masks)

