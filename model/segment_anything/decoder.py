from configs import ImageDecoderConfigs
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional
from .build_sam import build_sam_vit_h


class ImageHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询图像特征的解码器。"""

    def __init__(
        self,
        config: ImageDecoderConfigs,
        text_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.visual_model = build_sam_vit_h(getattr(config, "vision_pretrained", None))

        text_fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(text_hidden_size, 2*text_hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2*text_hidden_size, self.config.hidden_size)),
            # ("dropout", nn.Dropout(0.0)),
        ]))
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        self.image_feature_proj = nn.Conv2d(self.config.hidden_size, self.config.hidden_size, kernel_size=1)
        self.logit_scale = nn.Parameter(torch.ones(1))
        for param in self.parameters():
            param.requires_grad = True

        self.image_encoder = self.visual_model.image_encoder

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到 SAM prompt 空间。"""
        # assert len(self.text_hidden_fcs) == 1
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        return projected

    @staticmethod
    def _pool_query_tokens(
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """将 token 级 query 按 mask 聚合成 SAM 所需的样本级 prompt 向量。"""
        if query_tokens.dim() == 2:
            return query_tokens
        if query_tokens.dim() != 3:
            raise ValueError(f"Image decoder expects [B, C] or [B, K, C] query tokens, got {tuple(query_tokens.shape)}")
        if query_mask is None:
            return query_tokens.mean(dim=1)
        query_mask = query_mask.bool().to(query_tokens.device)
        if query_mask.shape != query_tokens.shape[:2]:
            raise ValueError(
                "query_mask shape mismatch: "
                f"expected {tuple(query_tokens.shape[:2])}, got {tuple(query_mask.shape)}"
            )
        weight = query_mask.unsqueeze(-1).to(query_tokens.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (query_tokens * weight).sum(dim=1) / denom

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """使用 SAM image_encoder 提取图像特征，不计算梯度。"""
        pixel_values = pixel_values.to(dtype=self.config.compute_dtype)
        with torch.no_grad():
            return self.visual_model.image_encoder(pixel_values)

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        input_size: tuple,
        original_size: tuple,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 routed query token 生成空间相似度热图，再作为 mask prompt 交给 SAM decoder。

        Args:
            pred_embeddings: [B, K, C] 或 [B, C] — token 级 query 序列或已聚合 query
            image_embeddings: [B, C_sam, H_emb, W_emb] — SAM 图像编码特征
            input_size: (H, W) — 输入图像尺寸（batch 内统一）
            original_size: (H, W) — 原始图像尺寸（batch 内统一）

        Returns:
            pred_masks: [B, H_orig, W_orig]
        """
        pred_embeddings = self._pool_query_tokens(pred_embeddings, query_mask=query_mask)
        pred_embeddings = self.project_hidden_states(pred_embeddings).to(self.config.compute_dtype)
        image_embeddings = image_embeddings.to(self.config.compute_dtype)

        image_features = self.image_feature_proj(image_embeddings)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(pred_embeddings, p=2, dim=-1)

        low_res_logits = (image_features * text_features[:, :, None, None]).sum(dim=1, keepdim=True)
        low_res_logits = low_res_logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)

        mask_prompt = F.interpolate(
            low_res_logits,
            size=self.visual_model.prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
        )

        sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt,
            text_embeds=None,
        )
        low_res_masks, _ = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        return pred_masks[:, 0]

