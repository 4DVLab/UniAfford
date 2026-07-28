# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM (Segment Anything Model) 模型构建函数
提供不同规模的 SAM 模型构建接口
"""
from functools import partial

import torch

from .modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,
                       TwoWayTransformer)


def build_sam_vit_h(checkpoint=None):
    """
    构建 SAM ViT-H（超大模型）版本
    
    Args:
        checkpoint: 预训练权重路径（可选）
        
    Returns:
        构建好的 SAM 模型
    """
    return _build_sam(
        encoder_embed_dim=1280,  # 编码器嵌入维度
        encoder_depth=32,  # 编码器深度（Transformer 层数）
        encoder_num_heads=16,  # 注意力头数
        encoder_global_attn_indexes=[7, 15, 23, 31],  # 全局注意力层索引
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h  # 默认使用 ViT-H


def build_sam_vit_l(checkpoint=None):
    """
    构建 SAM ViT-L（大模型）版本
    
    Args:
        checkpoint: 预训练权重路径（可选）
        
    Returns:
        构建好的 SAM 模型
    """
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    """
    构建 SAM ViT-B（基础模型）版本
    
    Args:
        checkpoint: 预训练权重路径（可选）
        
    Returns:
        构建好的 SAM 模型
    """
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


# SAM 模型注册表
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    """
    内部函数：构建 SAM 模型
    
    Args:
        encoder_embed_dim: 编码器嵌入维度
        encoder_depth: 编码器深度（Transformer 层数）
        encoder_num_heads: 注意力头数
        encoder_global_attn_indexes: 全局注意力层索引列表
        checkpoint: 预训练权重路径（可选）
        
    Returns:
        构建好的 SAM 模型
    """
    prompt_embed_dim = 256  # Prompt 嵌入维度
    image_size = 1024  # 输入图像尺寸
    vit_patch_size = 16  # ViT patch 大小
    image_embedding_size = image_size // vit_patch_size  # 图像嵌入尺寸
    # 构建 SAM 模型
    sam = Sam(
        # 图像编码器：ViT 架构
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        # Prompt 编码器：编码点、框、掩码等提示
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        # 掩码解码器：从图像嵌入和提示生成分割掩码
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,  # 多掩码输出数量
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,  # IoU 预测头深度
            iou_head_hidden_dim=256,
        ),
        # 像素归一化参数（ImageNet 统计值）
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()  # 设置为评估模式
    # 验证/便携 checkpoint 场景下，SAM 权重通常已在 UniAfford state_dict 中；
    # 本地 sam_vit_*.pth 缺失时不应阻断模型构建。
    if checkpoint is not None:
        import os
        if not os.path.exists(checkpoint):
            print(
                f"[Warning] SAM 预训练权重不存在: {checkpoint}；"
                "将跳过本地加载，等待后续 UniAfford checkpoint 覆盖。"
            )
        else:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            sam.load_state_dict(state_dict, strict=False)
    return sam
