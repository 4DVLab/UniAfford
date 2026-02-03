"""
联合可供性模型骨架。

本文件定义一个高层模型管理基座，包含三个主要模块：
- VLM 主干（分词器 + 视觉语言主干；当前为占位实现）
- 图像隐藏状态解码器（用文本隐藏状态查询图像特征）
- 点云隐藏状态解码器（用文本隐藏状态查询点云特征）

实现刻意轻量，仅聚焦结构。
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from configs import JointAffordanceConfig


class SimpleImageEncoder(nn.Module):
    """轻量级图像编码器占位实现（可替换为 SAM-ViT 等）。"""

    def __init__(self, in_channels: int = 3, hidden_size: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()  # 形状: (B, HW, C)


class SimplePointEncoder(nn.Module):
    """轻量级点云编码器占位实现（可替换为 PointNet2/Sonata/FTv3）。"""

    def __init__(self, input_dim: int = 3, hidden_size: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError("points 必须是 3D 张量")
        if points.shape[1] == 3:  # 形状变换: (B, 3, N) -> (B, N, 3)
            points = points.transpose(1, 2).contiguous()
        return self.proj(points)


class VLMBackbone(nn.Module):
    """VLM 主干占位实现（参考 LLaVA 的最小流程）。"""

    def __init__(self, config: JointAffordanceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.tokenizer = config.tokenizer
        self.text_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.text_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.text_norm = nn.LayerNorm(self.hidden_size)
        self.image_encoder = SimpleImageEncoder(in_channels=3, hidden_size=self.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, Optional[torch.Tensor]]:
        hidden_states = None
        if input_ids is not None:
            hidden_states = self.text_norm(self.text_proj(self.text_embed(input_ids)))
        image_features = None
        if images is not None:
            image_features = self.image_encoder(images)
        return {"hidden_states": hidden_states, "image_features": image_features}


class ImageHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询图像特征的解码器。"""

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, out_dim: int = 1):
        super().__init__()
        self.image_encoder = SimpleImageEncoder(in_channels=3, hidden_size=hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.out_proj = nn.Linear(hidden_size, out_dim)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        images: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None
        if image_features is None:
            if images is None:
                return None
            image_features = self.image_encoder(images)
        attn_out, _ = self.cross_attn(hidden_states, image_features, image_features)
        x = self.ffn(attn_out)
        return self.out_proj(x)


class PointCloudHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询点云特征的解码器。"""

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, out_dim: int = 1):
        super().__init__()
        self.point_encoder = SimplePointEncoder(input_dim=3, hidden_size=hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.out_proj = nn.Linear(hidden_size, out_dim)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        point_clouds: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None
        if point_features is None:
            if point_clouds is None:
                return None
            point_features = self.point_encoder(point_clouds)
        attn_out, _ = self.cross_attn(hidden_states, point_features, point_features)
        x = self.ffn(attn_out)
        return self.out_proj(x)


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()
        self.vision_pretrained = self.config.vision_pretrained
        self.ce_loss_weight = self.config.ce_loss_weight
        self.dice_loss_weight = self.config.dice_loss_weight
        self.bce_loss_weight = self.config.bce_loss_weight
        self.seg_token_idx = self.config.seg_token_idx
        self.aff_token_idx = self.config.aff_token_idx

        self.vlm = VLMBackbone(self.config)
        self.image_decoder = ImageHiddenStateDecoder(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            out_dim=self.config.image_out_dim,
        )
        self.point_decoder = PointCloudHiddenStateDecoder(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            out_dim=self.config.point_out_dim,
        )

    @property
    def tokenizer(self):
        return self.vlm.tokenizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[torch.Tensor]]:
        vlm_out = self.vlm(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = vlm_out["hidden_states"]
        image_logits = self.image_decoder(
            hidden_states, images=images, image_features=image_features
        )
        point_logits = self.point_decoder(
            hidden_states, point_clouds=point_clouds, point_features=point_features
        )
        return {
            "hidden_states": hidden_states,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "vlm_image_features": vlm_out.get("image_features"),
        }


__all__ = [
    "JointAffordanceModel",
    "VLMBackbone",
    "ImageHiddenStateDecoder",
    "PointCloudHiddenStateDecoder",
    "SimpleImageEncoder",
    "SimplePointEncoder",
]
