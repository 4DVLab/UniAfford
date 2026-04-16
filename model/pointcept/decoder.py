from configs.base_config import PointDecoderConfigs
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .models.point_prompt_training.point_transformer_v3m2_sonata import PointTransformerV3


class _PromptMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PromptAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class _PointCloudTwoWayAttentionBlock(nn.Module):
    """
    3D 版轻量 Two-Way block：
    - sparse prompt tokens 自注意力
    - prompt -> point cross-attn
    - prompt MLP
    - point -> prompt cross-attn
    """

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.self_attn = _PromptAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_prompt_to_point = _PromptAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.prompt_mlp = _PromptMLP(embedding_dim, mlp_dim, embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_point_to_prompt = _PromptAttention(embedding_dim, num_heads)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self.norm1(prompt_tokens + self.self_attn(prompt_tokens, prompt_tokens, prompt_tokens))

        prompt_tokens = self.norm2(
            prompt_tokens
            + self.cross_attn_prompt_to_point(
                prompt_tokens,
                point_tokens,
                point_tokens,
                key_padding_mask=None if point_mask is None else (~point_mask.bool()),
            )
        )

        prompt_tokens = self.norm3(prompt_tokens + self.prompt_mlp(prompt_tokens))

        point_delta = self.cross_attn_point_to_prompt(point_tokens, prompt_tokens, prompt_tokens)
        point_tokens = self.norm4(point_tokens + point_delta)
        if point_mask is not None:
            point_tokens = torch.where(point_mask.unsqueeze(-1), point_tokens, torch.zeros_like(point_tokens))
        return prompt_tokens, point_tokens


class _PointCloudTwoWayTransformer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _PointCloudTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_attn_prompt_to_point = _PromptAttention(embedding_dim, num_heads)
        self.norm_final = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = prompt_tokens
        keys = point_tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_mask=point_mask)
        queries = self.norm_final(
            queries
            + self.final_attn_prompt_to_point(
                queries,
                keys,
                keys,
                key_padding_mask=None if point_mask is None else (~point_mask.bool()),
            )
        )
        return queries, keys


class PointCloudSharedBackboneSimilarityDecoder(nn.Module):
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

class PointCloudIndependentDecoder(nn.Module):
    """使用与 encoder 独立的随机初始化 point backbone 的解码器。"""

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

        # 与 shared decoder 保持一致：先把 LLM hidden 投影到 point decoder 的隐空间，
        # 再与逐点特征做相似度，避免 text/point 维度不一致。
        text_feat = F.normalize(self.project_hidden_states(pred_embeddings), p=2, dim=-1)

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


class PointCloudSharedBackbonePromptDecoder(nn.Module):
    """
    迁移 SAM 核心思路到 3D：
    - 将文本/路由 token 视作 sparse prompt token
    - 将 per-point feature 视作 dense point tokens
    - 使用轻量 two-way transformer 做 prompt<->point 双向交互
    - 使用 mask token 的动态权重生成逐点 mask logits
    """

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
                ]
            )
        )
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        self.point_proj = nn.Linear(point_feature_size, config.hidden_size)
        self.mask_token = nn.Embedding(1, config.hidden_size)
        self.prompt_transformer = _PointCloudTwoWayTransformer(
            depth=int(getattr(config, "prompt_decoder_depth", 2)),
            embedding_dim=int(config.hidden_size),
            num_heads=int(config.num_heads),
            mlp_dim=int(getattr(config, "prompt_decoder_mlp_dim", 4 * config.hidden_size)),
        )
        self.output_hypernet = _PromptMLP(
            input_dim=int(config.hidden_size),
            hidden_dim=int(config.hidden_size),
            output_dim=int(config.hidden_size),
        )
        self.output_point_refine = nn.Sequential(
            nn.LayerNorm(int(config.hidden_size)),
            nn.Linear(int(config.hidden_size), int(config.hidden_size)),
        )
        # 没有显式 3D 坐标 PE 时，使用一个可学习的 dense bias 充当点域 prompt prior。
        self.no_dense_prompt = nn.Embedding(1, config.hidden_size)

        for module in (
            self.text_hidden_fcs,
            self.point_proj,
            self.mask_token,
            self.prompt_transformer,
            self.output_hypernet,
            self.output_point_refine,
            self.no_dense_prompt,
        ):
            for param in module.parameters():
                param.requires_grad = True

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.config.compute_dtype)
        return self.text_hidden_fcs[0](hidden_states)

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if per_point_features is None or per_point_mask is None:
            raise ValueError("Prompt decoder requires per_point_features/per_point_mask.")

        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        point_feat = per_point_features.to(self.config.compute_dtype)
        point_mask = per_point_mask.bool()

        prompt_tokens = self.project_hidden_states(pred_embeddings).unsqueeze(1)  # [B, 1, H]
        mask_token = self.mask_token.weight.unsqueeze(0).expand(prompt_tokens.shape[0], -1, -1)
        sparse_prompt_tokens = torch.cat([mask_token, prompt_tokens], dim=1)  # [B, 2, H]

        point_tokens = self.point_proj(point_feat)
        dense_prompt = self.no_dense_prompt.weight.view(1, 1, -1).to(point_tokens.dtype)
        point_tokens = point_tokens + dense_prompt
        point_tokens = torch.where(point_mask.unsqueeze(-1), point_tokens, torch.zeros_like(point_tokens))

        prompt_out, point_out = self.prompt_transformer(
            prompt_tokens=sparse_prompt_tokens,
            point_tokens=point_tokens,
            point_mask=point_mask,
        )

        mask_token_out = prompt_out[:, 0, :]
        dynamic_mask_weight = self.output_hypernet(mask_token_out)  # [B, H]
        refined_point = self.output_point_refine(point_out)
        logits = (refined_point * dynamic_mask_weight.unsqueeze(1)).sum(dim=-1)
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
        if loss_fn is None:
            loss_fn = nn.BCELoss()
        pred_masks = self.forward(
            pred_embeddings=pred_embeddings,
            per_point_features=per_point_features,
            per_point_mask=per_point_mask,
        )
        loss = loss_fn(pred_masks, gt_masks.to(pred_masks.dtype))
        return dict(loss=loss, pred_masks=pred_masks)
