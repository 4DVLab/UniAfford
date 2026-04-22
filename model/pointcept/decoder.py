from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import PointDecoderConfigs

from .models.point_prompt_training.point_transformer_v3m2_sonata import PointTransformerV3


POINT_XYZ_CHANNELS = 3


def _build_text_projection(text_hidden_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(text_hidden_size, 2 * text_hidden_size)),
                ("relu", nn.ReLU(inplace=True)),
                ("fc2", nn.Linear(2 * text_hidden_size, hidden_size)),
            ]
        )
    )
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
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self.norm1(
            prompt_tokens
            + self.self_attn(
                prompt_tokens,
                prompt_tokens,
                prompt_tokens,
                key_padding_mask=None if prompt_mask is None else (~prompt_mask.bool()),
            )
        )
        if prompt_mask is not None:
            prompt_tokens = torch.where(prompt_mask.unsqueeze(-1), prompt_tokens, torch.zeros_like(prompt_tokens))

        prompt_tokens = self.norm2(
            prompt_tokens
            + self.cross_attn_prompt_to_point(
                prompt_tokens,
                point_tokens,
                point_tokens,
                key_padding_mask=None if point_mask is None else (~point_mask.bool()),
            )
        )
        if prompt_mask is not None:
            prompt_tokens = torch.where(prompt_mask.unsqueeze(-1), prompt_tokens, torch.zeros_like(prompt_tokens))

        prompt_tokens = self.norm3(prompt_tokens + self.prompt_mlp(prompt_tokens))
        if prompt_mask is not None:
            prompt_tokens = torch.where(prompt_mask.unsqueeze(-1), prompt_tokens, torch.zeros_like(prompt_tokens))

        point_delta = self.cross_attn_point_to_prompt(
            point_tokens,
            prompt_tokens,
            prompt_tokens,
            key_padding_mask=None if prompt_mask is None else (~prompt_mask.bool()),
        )
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
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = prompt_tokens
        keys = point_tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_mask=point_mask, prompt_mask=prompt_mask)
        queries = self.norm_final(
            queries
            + self.final_attn_prompt_to_point(
                queries,
                keys,
                keys,
                key_padding_mask=None if point_mask is None else (~point_mask.bool()),
            )
        )
        if prompt_mask is not None:
            queries = torch.where(prompt_mask.unsqueeze(-1), queries, torch.zeros_like(queries))
        return queries, keys


class _IndependentPointFeatureEncoder(nn.Module):
    """独立模式下的 point backbone，负责从原始点云生成逐点特征。"""

    def __init__(self, config: PointDecoderConfigs):
        super().__init__()
        self.config = config
        self.point_backbone = PointTransformerV3(**config.backbone_kwargs)
        self.in_channels = int(config.backbone_kwargs.get("in_channels", POINT_XYZ_CHANNELS))

    @staticmethod
    def _normalize_point_clouds(point_clouds: torch.Tensor) -> torch.Tensor:
        if point_clouds.dim() != 3:
            raise ValueError(f"point_clouds should be [B, N, 3] or [B, 3, N], got {tuple(point_clouds.shape)}")
        if point_clouds.shape[-1] == 3:
            return point_clouds.contiguous()
        if point_clouds.shape[1] == 3:
            return point_clouds.permute(0, 2, 1).contiguous()
        raise ValueError(f"point_clouds last dim must be 3, got {tuple(point_clouds.shape)}")

    @staticmethod
    def _build_pointcept_batch(
        point_clouds: torch.Tensor,
        grid_size: float,
        in_channels: int,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_points, _ = point_clouds.shape
        coord = point_clouds.reshape(-1, 3).contiguous()
        if in_channels < POINT_XYZ_CHANNELS:
            raise ValueError(f"point backbone in_channels must be >= 3, got {in_channels}")
        feat = coord.new_zeros((coord.shape[0], in_channels))
        feat[:, :POINT_XYZ_CHANNELS] = coord
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
        bsz = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=bsz)
        if not torch.all(counts == counts[0]):
            raise ValueError("All samples must share the same point count in the independent decoder adapter.")
        num_points = int(counts[0].item())
        return feat.reshape(bsz, num_points, feat.shape[-1])

    def forward(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        point_clouds = self._normalize_point_clouds(point_clouds).to(self.config.compute_dtype)
        data_dict = self._build_pointcept_batch(
            point_clouds=point_clouds,
            grid_size=self.config.grid_size,
            in_channels=self.in_channels,
        )
        point = self.point_backbone(data_dict)
        point_feat = self._unflatten_by_batch(point.feat, point.batch).to(self.config.compute_dtype)
        point_mask = torch.ones(
            point_feat.shape[:2],
            dtype=torch.bool,
            device=point_feat.device,
        )
        return point_feat, point_mask


class _SimilarityDecoderHead(nn.Module):
    """使用逐点特征与文本隐状态做相似度对齐的解码头。"""

    def __init__(self, config: PointDecoderConfigs, point_feature_size: int):
        super().__init__()
        self.config = config
        self.point_proj = nn.Linear(point_feature_size, config.hidden_size)
        self.logit_scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def _pool_query_tokens(
        projected_text: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if projected_text.dim() == 2:
            return projected_text
        if projected_text.dim() != 3:
            raise ValueError(
                "Similarity decoder expects [B, H] or [B, K, H] query embeddings, "
                f"got {tuple(projected_text.shape)}"
            )
        if query_mask is None:
            return projected_text.mean(dim=1)
        query_mask = query_mask.bool().to(projected_text.device)
        if query_mask.shape != projected_text.shape[:2]:
            raise ValueError(
                "query_mask shape mismatch: "
                f"expected {tuple(projected_text.shape[:2])}, got {tuple(query_mask.shape)}"
            )
        weight = query_mask.unsqueeze(-1).to(projected_text.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (projected_text * weight).sum(dim=1) / denom

    def forward(
        self,
        query_embeddings: torch.Tensor,
        per_point_features: torch.Tensor,
        per_point_mask: torch.Tensor,
        projected_text: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del query_embeddings
        point_feat = per_point_features.to(self.config.compute_dtype)
        point_mask = per_point_mask.bool()
        pooled_text = self._pool_query_tokens(
            projected_text.to(self.config.compute_dtype),
            query_mask=query_mask,
        )
        text_feat = F.normalize(pooled_text, p=2, dim=-1)
        point_feat = self.point_proj(point_feat)
        point_feat = F.normalize(point_feat, p=2, dim=-1)
        point_feat = torch.where(point_mask.unsqueeze(-1), point_feat, torch.zeros_like(point_feat))

        logits = (point_feat * text_feat.unsqueeze(1)).sum(dim=-1)
        logits = logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)
        logits = torch.where(point_mask, logits, torch.zeros_like(logits))
        return logits


class _PromptDecoderHead(nn.Module):
    """
    受 SeqAfford / SAM 思路启发的 3D prompt 解码头：
    - 将 routed 3D token 序列视作 sparse prompt tokens
    - 将 per-point feature 视作 dense point tokens
    - 使用轻量 two-way transformer 做多 query token 与逐点特征的交互
    - 由交互后的 mask token 汇总全局条件，再通过 point-wise MLP 生成逐点 logits
    """

    def __init__(self, config: PointDecoderConfigs, point_feature_size: int):
        super().__init__()
        self.config = config
        self.point_proj = nn.Linear(point_feature_size, config.hidden_size)
        self.mask_token = nn.Embedding(1, config.hidden_size)
        self.prompt_transformer = _PointCloudTwoWayTransformer(
            depth=int(getattr(config, "prompt_decoder_depth", 2)),
            embedding_dim=int(config.hidden_size),
            num_heads=int(config.num_heads),
            mlp_dim=int(getattr(config, "prompt_decoder_mlp_dim", 4 * config.hidden_size)),
        )
        self.output_point_refine = nn.Sequential(
            nn.LayerNorm(int(config.hidden_size)),
            nn.Linear(int(config.hidden_size), int(config.hidden_size)),
            nn.GELU(),
        )
        self.prompt_context_proj = nn.Sequential(
            nn.LayerNorm(int(config.hidden_size)),
            nn.Linear(int(config.hidden_size), int(config.hidden_size)),
        )
        self.output_mask_head = nn.Sequential(
            nn.LayerNorm(int(config.hidden_size)),
            nn.Linear(int(config.hidden_size), int(config.hidden_size)),
            nn.GELU(),
            nn.Linear(int(config.hidden_size), 1),
        )
        self.no_dense_prompt = nn.Embedding(1, config.hidden_size)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        per_point_features: torch.Tensor,
        per_point_mask: torch.Tensor,
        projected_text: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del query_embeddings
        point_feat = per_point_features.to(self.config.compute_dtype)
        point_mask = per_point_mask.bool()
        prompt_tokens = projected_text.to(self.config.compute_dtype)
        if prompt_tokens.dim() == 2:
            prompt_tokens = prompt_tokens.unsqueeze(1)
        elif prompt_tokens.dim() != 3:
            raise ValueError(f"Prompt decoder expects [B, H] or [B, K, H] projected_text, got {tuple(prompt_tokens.shape)}")

        if query_mask is None:
            query_mask = torch.ones(
                prompt_tokens.shape[:2],
                dtype=torch.bool,
                device=prompt_tokens.device,
            )
        else:
            query_mask = query_mask.bool().to(prompt_tokens.device)
            if query_mask.shape != prompt_tokens.shape[:2]:
                raise ValueError(
                    "query_mask shape mismatch: "
                    f"expected {tuple(prompt_tokens.shape[:2])}, got {tuple(query_mask.shape)}"
                )

        mask_token = self.mask_token.weight.unsqueeze(0).expand(prompt_tokens.shape[0], -1, -1)
        sparse_prompt_tokens = torch.cat([mask_token, prompt_tokens], dim=1)
        sparse_prompt_mask = torch.cat(
            [
                torch.ones(
                    (prompt_tokens.shape[0], 1),
                    dtype=torch.bool,
                    device=prompt_tokens.device,
                ),
                query_mask,
            ],
            dim=1,
        )

        point_tokens = self.point_proj(point_feat)
        dense_prompt = self.no_dense_prompt.weight.view(1, 1, -1).to(point_tokens.dtype)
        point_tokens = point_tokens + dense_prompt
        point_tokens = torch.where(point_mask.unsqueeze(-1), point_tokens, torch.zeros_like(point_tokens))

        prompt_out, point_out = self.prompt_transformer(
            prompt_tokens=sparse_prompt_tokens,
            point_tokens=point_tokens,
            point_mask=point_mask,
            prompt_mask=sparse_prompt_mask,
        )

        mask_token_out = prompt_out[:, 0, :]
        refined_point = self.output_point_refine(point_out)
        prompt_context = self.prompt_context_proj(mask_token_out).unsqueeze(1)
        conditioned_point = refined_point + prompt_context
        logits = self.output_mask_head(conditioned_point).squeeze(-1)
        logits = torch.where(point_mask, logits, torch.zeros_like(logits))
        return logits


class PointCloudHiddenStateDecoder(nn.Module):
    """统一的 3D hidden-state decoder，按配置切换共享/独立 backbone 与 prompt/相似度解码。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int,
        point_feature_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.backbone_mode = str(config.backbone_mode).lower()
        self.decode_mode = str(config.decode_mode).lower()

        self.text_hidden_fcs = nn.ModuleList(
            [_build_text_projection(text_hidden_size, int(config.hidden_size))]
        )

        if self.backbone_mode == "shared":
            if point_feature_size is None:
                raise ValueError("Shared point decoder requires `point_feature_size` from the point encoder.")
            self.point_feature_encoder = None
            head_input_dim = int(point_feature_size)
        elif self.backbone_mode == "independent":
            self.point_feature_encoder = _IndependentPointFeatureEncoder(config)
            head_input_dim = int(config.backbone_out_channels)
        else:
            raise ValueError(f"Unsupported point decoder backbone mode: {self.backbone_mode}")

        if self.decode_mode == "similarity":
            self.decoder_head = _SimilarityDecoderHead(config, head_input_dim)
        elif self.decode_mode == "prompt":
            self.decoder_head = _PromptDecoderHead(config, head_input_dim)
        else:
            raise ValueError(f"Unsupported point decoder decode mode: {self.decode_mode}")

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.config.compute_dtype)
        return self.text_hidden_fcs[0](hidden_states)

    def _prepare_point_features(
        self,
        point_clouds: Optional[torch.Tensor] = None,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_mode == "shared":
            if per_point_features is None or per_point_mask is None:
                raise ValueError("Shared point decoder requires per_point_features/per_point_mask.")
            return per_point_features, per_point_mask

        if point_clouds is None:
            raise ValueError("Independent point decoder requires raw point_clouds.")
        if self.point_feature_encoder is None:
            raise RuntimeError("Independent point decoder is missing its point feature encoder.")
        return self.point_feature_encoder(point_clouds)

    def forward(
        self,
        point_clouds: Optional[torch.Tensor] = None,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回逐点 mask logits，值域为未归一化实数。"""
        point_feat, point_mask = self._prepare_point_features(
            point_clouds=point_clouds,
            per_point_features=per_point_features,
            per_point_mask=per_point_mask,
        )
        assert query_embeddings is not None, "Point decoder requires token-level query_embeddings."
        
        projected_text = self.project_hidden_states(query_embeddings)
        return self.decoder_head(
            query_embeddings=query_embeddings,
            per_point_features=point_feat,
            per_point_mask=point_mask,
            projected_text=projected_text,
            query_mask=query_mask,
        )

    def forward_with_loss(
        self,
        gt_masks: torch.Tensor,
        point_clouds: Optional[torch.Tensor] = None,
        per_point_features: Optional[torch.Tensor] = None,
        per_point_mask: Optional[torch.Tensor] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        pred_logits = self.forward(
            point_clouds=point_clouds,
            per_point_features=per_point_features,
            per_point_mask=per_point_mask,
            query_embeddings=query_embeddings,
            query_mask=query_mask,
        )
        loss = loss_fn(pred_logits, gt_masks.to(pred_logits.dtype))
        return dict(loss=loss, pred_masks=pred_logits)
