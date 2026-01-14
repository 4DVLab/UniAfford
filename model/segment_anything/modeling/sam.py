# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    """
    SAM (Segment Anything Model) 主模型类
    
    从图像和输入提示预测对象掩码
    架构包括：
    1. 图像编码器：将图像编码为嵌入
    2. Prompt 编码器：编码各种类型的输入提示（点、框、掩码、文本）
    3. 掩码解码器：从图像嵌入和编码的提示生成分割掩码
    """
    mask_threshold: float = 0.0  # 掩码阈值
    image_format: str = "RGB"  # 图像格式

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        初始化 SAM 模型
        
        Arguments:
          image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的主干网络
          prompt_encoder (PromptEncoder): 编码各种类型输入提示的编码器
          mask_decoder (MaskDecoder): 从图像嵌入和编码提示预测掩码的解码器
          pixel_mean (list(float)): 输入图像像素归一化的均值
          pixel_std (list(float)): 输入图像像素归一化的标准差
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # 注册像素均值和标准差为缓冲区（不参与梯度更新）
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        """获取模型所在设备"""
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        端到端预测掩码：从提供的图像和提示预测掩码
        如果提示不是预先知道的，建议使用 SamPredictor 而不是直接调用模型
        
        Arguments:
          batched_input (list(dict)): 输入图像列表，每个是一个字典，包含以下键。
            如果某个提示键不存在，可以省略。
              'image': 图像张量，格式为 3xHxW，已经转换为模型输入格式
              'original_size': (tuple(int, int)) 转换前图像的原始尺寸 (H, W)
              'point_coords': (torch.Tensor) 该图像的点提示批次，形状 BxNx2
              'point_labels': (torch.Tensor) 点提示的批次标签，形状 BxN
              'boxes': (torch.Tensor) 框输入批次，形状 Bx4
              'mask_inputs': (torch.Tensor) 掩码输入批次，格式 Bx1xHxW
          multimask_output (bool): 模型是否应该预测多个消歧掩码，或返回单个掩码
        
        Returns:
          (list(dict)): 输入图像列表，每个元素是一个字典，包含以下键：
              'masks': (torch.Tensor) 批次二元掩码预测，形状 BxCxHxW
              'iou_predictions': (torch.Tensor) 模型对掩码质量的预测，形状 BxC
              'low_res_logits': (torch.Tensor) 低分辨率 logits，形状 BxCxHxW，H=W=256
        """
        # 预处理图像并堆叠
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        # 使用图像编码器提取嵌入
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        # 对每个图像进行处理
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # 准备点提示
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            # 编码提示（点、框、掩码）
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            # 使用掩码解码器生成掩码
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # 后处理掩码：调整到原始图像尺寸
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # 应用阈值
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        后处理掩码：移除填充并将掩码上采样到原始图像尺寸
        
        Arguments:
          masks (torch.Tensor): 来自 mask_decoder 的批次掩码，格式 BxCxHxW
          input_size (tuple(int, int)): 输入模型的图像尺寸 (H, W)，用于移除填充
          original_size (tuple(int, int)): 输入模型前图像的原始尺寸 (H, W)
        
        Returns:
          (torch.Tensor): 批次掩码，格式 BxCxHxW，其中 (H, W) 由 original_size 给出
        """

        dtype = masks.dtype

        # 先上采样到编码器输入尺寸
        masks = F.interpolate(
            masks.float(),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # 移除填充
        masks = masks[..., : input_size[0], : input_size[1]]
        # 上采样到原始图像尺寸
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        预处理图像：归一化像素值并填充为正方形输入
        
        Args:
            x: 输入图像张量
            
        Returns:
            预处理后的图像张量
        """
        # 归一化颜色
        x = (x - self.pixel_mean) / self.pixel_std

        # 填充为正方形
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
