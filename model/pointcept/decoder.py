from configs.base_config import PointDecoderConfigs
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.point_prompt_training.point_transformer_v3m2_sonata import PointTransformerV3

class PointCloudHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询点云特征的解码器。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.point_backbone = PointTransformerV3(**config.backbone_kwargs)

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

        # v1m3-style point projection + normalized similarity scoring.
        self.point_proj = nn.Linear(config.backbone_out_channels, config.hidden_size)
        # FSDP requires parameter tensors to be at least 1D.
        self.logit_scale = nn.Parameter(torch.ones(1))

        for param in self.point_backbone.parameters():
            param.requires_grad = True
        for param in self.point_proj.parameters():
            param.requires_grad = True

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到点云分割器的嵌入空间。"""
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        return projected

    @staticmethod
    def _build_pointcept_batch(
        point_clouds: torch.Tensor, grid_size: float
    ) -> Dict[str, torch.Tensor]:
        """
        Build Pointcept-compatible input dict from [B, 3, N] points.
        """
        bsz, _, num_points = point_clouds.shape
        coord = point_clouds.permute(0, 2, 1).reshape(-1, 3).contiguous()
        feat = coord
        batch = (
            torch.arange(bsz, device=point_clouds.device)
            .unsqueeze(1)
            .repeat(1, num_points)
            .reshape(-1)
            .long()
        )
        offset = (
            torch.arange(1, bsz + 1, device=point_clouds.device, dtype=torch.long)
            * num_points
        )
        return dict(
            coord=coord,
            feat=feat,
            batch=batch,
            offset=offset,
            grid_size=grid_size,
        )

    @staticmethod
    def _unflatten_by_batch(feat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened [sum(N), C] to dense [B, N, C].
        Assumes each batch item has identical point count.
        """
        bsz = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=bsz)
        assert torch.all(
            counts == counts[0]
        ), "All samples must share the same point count in this adapter."
        num_points = int(counts[0].item())
        return feat.reshape(bsz, num_points, feat.shape[-1])

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        point_clouds: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量生成 3D 分割掩码（目标文本概率）。

        Args:
            pred_embeddings: [B, C] — 每个样本的文本嵌入（可为 text_fc 前或后）
            point_clouds: [B, N, 3] 或 [B, 3, N]

        Returns:
            pred_masks: [B, N]，每个点属于目标文本的概率
        """
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)

        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        point_clouds = point_clouds.to(self.config.compute_dtype)

        text_feat = F.normalize(pred_embeddings, p=2, dim=-1)

        data_dict = self._build_pointcept_batch(point_clouds, self.config.grid_size)
        point = self.point_backbone(data_dict)
        point_feat = self._unflatten_by_batch(point.feat, point.batch)  # [B, N, Cb]
        point_feat = self.point_proj(point_feat)  # [B, N, H]
        point_feat = F.normalize(point_feat, p=2, dim=-1)

        # v1m3-style similarity, now single-target text per sample.
        logits = (point_feat * text_feat.unsqueeze(1)).sum(dim=-1)
        logits = logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)
        # pred_masks = torch.sigmoid(logits)
        return logits

    def forward_with_loss(
        self,
        pred_embeddings: torch.Tensor,
        point_clouds: torch.Tensor,
        gt_masks: torch.Tensor,
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
        pred_masks = self.forward(pred_embeddings, point_clouds)
        loss = loss_fn(pred_masks, gt_masks.to(pred_masks.dtype))
        return dict(loss=loss, pred_masks=pred_masks)

