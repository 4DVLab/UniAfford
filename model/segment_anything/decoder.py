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
        [B, K, C] 输入会保留 K 个 query 的顺序，返回 [B, K, H, W]。

        Args:
            pred_embeddings: [B, K, C] 或 [B, C] — token 级 query 序列或单 query
            image_embeddings: [B, C_sam, H_emb, W_emb] — SAM 图像编码特征
            input_size: (H, W) — 输入图像尺寸（batch 内统一）
            original_size: (H, W) — 原始图像尺寸（batch 内统一）

        Returns:
            pred_masks: [B, K, H_orig, W_orig] 或 [B, H_orig, W_orig]
        """
        single_query = pred_embeddings.dim() == 2
        if single_query:
            pred_embeddings = pred_embeddings.unsqueeze(1)
        elif pred_embeddings.dim() != 3:
            raise ValueError(f"Image decoder expects [B, C] or [B, K, C] query tokens, got {tuple(pred_embeddings.shape)}")
        if query_mask is None:
            query_mask = torch.ones(
                pred_embeddings.shape[:2],
                dtype=torch.bool,
                device=pred_embeddings.device,
            )
        else:
            query_mask = query_mask.bool().to(pred_embeddings.device)
            if query_mask.shape != pred_embeddings.shape[:2]:
                raise ValueError(
                    "query_mask shape mismatch: "
                    f"expected {tuple(pred_embeddings.shape[:2])}, got {tuple(query_mask.shape)}"
                )

        bsz, num_queries, _ = pred_embeddings.shape
        pred_embeddings = self.project_hidden_states(pred_embeddings).to(self.config.compute_dtype)
        image_embeddings = image_embeddings.to(self.config.compute_dtype)

        image_features = self.image_feature_proj(image_embeddings)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(pred_embeddings, p=2, dim=-1)

        low_res_logits = (image_features.unsqueeze(1) * text_features[:, :, :, None, None]).sum(dim=2)
        low_res_logits = low_res_logits.unsqueeze(2)
        low_res_logits = low_res_logits * self.logit_scale.exp().clamp(min=1e-4, max=100.0)
        low_res_logits = torch.where(
            query_mask[:, :, None, None, None],
            low_res_logits,
            torch.zeros_like(low_res_logits),
        )

        mask_prompt = F.interpolate(
            low_res_logits.flatten(0, 1),
            size=self.visual_model.prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
        )
        flat_image_embeddings = image_embeddings.repeat_interleave(num_queries, dim=0)

        sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt,
            text_embeds=None,
        )
        low_res_masks, _ = self.visual_model.mask_decoder(
            image_embeddings=flat_image_embeddings,
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
        pred_masks = pred_masks[:, 0].view(bsz, num_queries, *pred_masks.shape[-2:])
        pred_masks = torch.where(
            query_mask[:, :, None, None],
            pred_masks,
            torch.zeros_like(pred_masks),
        )
        return pred_masks[:, 0] if single_query else pred_masks

