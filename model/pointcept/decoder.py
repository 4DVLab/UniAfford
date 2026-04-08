from configs.base_config import PointDecoderConfigs
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PointCloudHiddenStateDecoder(nn.Module):
    """使用共享点特征与 aff token 做逐点相似度对齐的解码器。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int,
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

        # 共享 encoder 的原始点特征维度在不同骨干配置下可能变化，使用 LazyLinear 自动适配。
        self.point_proj = nn.LazyLinear(config.hidden_size)
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
        point_clouds: torch.Tensor,
        shared_point_features: Optional[torch.Tensor] = None,
        shared_point_coords: Optional[torch.Tensor] = None,
        shared_point_mask: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        批量生成 3D 分割掩码（目标文本概率）。

        Args:
            pred_embeddings: [B, C] — 每个样本的文本嵌入（可为 text_fc 前或后）
            point_clouds: [B, N, 3] 或 [B, 3, N]

        Returns:
            pred_masks: [B, N]，每个点属于目标文本的概率
        """
        if point_clouds.shape[-1] != 3 and point_clouds.shape[1] == 3:
            point_clouds = point_clouds.permute(0, 2, 1)
        elif point_clouds.shape[-1] != 3:
            raise ValueError(f"point_clouds last dim must be 3, got {tuple(point_clouds.shape)}")

        if shared_point_features is None or shared_point_coords is None or shared_point_mask is None:
            raise ValueError("Decoder requires shared_point_features/shared_point_coords/shared_point_mask.")

        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        point_clouds = point_clouds.to(self.config.compute_dtype)
        point_feat = shared_point_features.to(self.config.compute_dtype)
        point_coords = shared_point_coords.to(self.config.compute_dtype)
        point_mask = shared_point_mask.bool()

        text_feat = F.normalize(self.project_hidden_states(pred_embeddings), p=2, dim=-1)  # [B, H]
        point_feat = self.point_proj(point_feat)  # [B, K, H]
        point_feat = F.normalize(point_feat, p=2, dim=-1)
        point_feat = torch.where(point_mask.unsqueeze(-1), point_feat, torch.zeros_like(point_feat))

        # token-level 响应场：每个点 token 对 aff query 的相似度
        token_logits = (point_feat * text_feat.unsqueeze(1)).sum(dim=-1)  # [B, K]
        token_logits = token_logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)
        token_logits = torch.where(point_mask, token_logits, torch.zeros_like(token_logits))

        # 还原到原始点云 N 点。
        return self._map_token_logits_to_points(
            token_logits=token_logits,
            token_coords=point_coords,
            point_clouds=point_clouds,
            token_mask=point_mask,
            pc_valid_lengths=pc_valid_lengths,
        )

    def _map_token_logits_to_points(
        self,
        token_logits: torch.Tensor,
        token_coords: torch.Tensor,
        point_clouds: torch.Tensor,
        token_mask: torch.Tensor,
        pc_valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if point_clouds.shape[-1] != 3 and point_clouds.shape[1] == 3:
            point_clouds = point_clouds.permute(0, 2, 1)
        bsz, num_points, _ = point_clouds.shape
        out = token_logits.new_zeros((bsz, num_points))
        for i in range(bsz):
            k = int(token_mask[i].sum().item())
            n = num_points if pc_valid_lengths is None else int(pc_valid_lengths[i].item())
            if k <= 0 or n <= 0:
                continue
            src_xyz = token_coords[i, :k]
            src_logit = token_logits[i, :k]
            tgt_xyz = point_clouds[i, :n]
            dist = torch.cdist(tgt_xyz.unsqueeze(0), src_xyz.unsqueeze(0)).squeeze(0)
            nn_idx = dist.argmin(dim=-1)
            out[i, :n] = src_logit[nn_idx]
        return out

    def forward_with_loss(
        self,
        pred_embeddings: torch.Tensor,
        point_clouds: torch.Tensor,
        gt_masks: torch.Tensor,
        shared_point_features: Optional[torch.Tensor] = None,
        shared_point_coords: Optional[torch.Tensor] = None,
        shared_point_mask: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional helper for training:
        - pred_embeddings: [B, C]
        - point_clouds: [B, N, 3] or [B, 3, N]
        - gt_masks: [B, N] (0/1)
        """
        if loss_fn is None:
            loss_fn = nn.BCELoss()
        pred_masks = self.forward(
            pred_embeddings=pred_embeddings,
            point_clouds=point_clouds,
            shared_point_features=shared_point_features,
            shared_point_coords=shared_point_coords,
            shared_point_mask=shared_point_mask,
            pc_valid_lengths=pc_valid_lengths,
        )
        loss = loss_fn(pred_masks, gt_masks.to(pred_masks.dtype))
        return dict(loss=loss, pred_masks=pred_masks)

