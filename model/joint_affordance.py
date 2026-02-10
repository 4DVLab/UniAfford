"""
联合可供性模型骨架。

本文件定义一个高层模型管理基座，包含三个主要模块：
- MLLM 主干（分词器 + 视觉语言主干；当前为占位实现）
- 图像隐藏状态解码器（用文本隐藏状态查询图像特征）
- 点云隐藏状态解码器（用文本隐藏状态查询点云特征）

实现刻意轻量，仅聚焦结构。
"""
from typing import Optional, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from collections import OrderedDict

from configs import JointAffordanceConfig, ImageDecoderConfigs, PointDecoderConfigs, MLLMConfigs
from model.segment_anything import build_sam_vit_h
from model.pointnet2_utils import PointCloud3DSegmentor, PointCloudEncoder
from utils.common import resolve_dtype


class MLLMBackbone(nn.Module):
    """MLLM 主干实现（Qwen3-VL）。"""

    def __init__(self, config: MLLMConfigs):
        super().__init__()
        self.config = config
        self.model = self._build_qwen_model(config)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.vocab_size = self.model.config.text_config.vocab_size

        if self.config.hidden_size != self.hidden_size:
            self.config.hidden_size = self.hidden_size

        if self.config.vocab_size != self.vocab_size:
            self.config.vocab_size = self.vocab_size

        if self.config.tokenizer is None:
            self.config.tokenizer = AutoTokenizer.from_pretrained(
                self.config.qwen_model_name_or_path,
                model_max_length=self.config.model_max_length,
                padding_side="right",
                use_fast=False,
            )
        self.tokenizer = self.config.tokenizer
    

    def _build_qwen_model(self, config: MLLMConfigs):
        model_name = config.qwen_model_name_or_path
        model_name_lower = model_name.lower()
        dtype = config.qwen_dtype

        # choose QwenVL version
        if "qwen3" in model_name_lower and "a" in Path(model_name.rstrip("/")).name.lower():
            from transformers import Qwen3VLMoeForConditionalGeneration
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        elif "qwen3" in model_name_lower:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        elif "qwen2.5" in model_name_lower:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )

        model.config.use_cache = False
        
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        # 构建 Qwen 模型输入，自动过滤 None 值
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "position_ids": position_ids,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        model_inputs["output_hidden_states"] = True
        model_inputs["return_dict"] = True
        outputs = self.model(**model_inputs)

        # TODO: check outputs
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # 确保输出为 [B, L, C]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        return {"hidden_states": hidden_states, "output": outputs}


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

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if getattr(config, "train_mask_decoder", False):
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        self.image_encoder = self.visual_model.image_encoder

        text_fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(text_hidden_size, 2*text_hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2*text_hidden_size, self.config.hidden_size)),
            # ("dropout", nn.Dropout(0.0)),
        ]))
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        if self.config.compute_dtype is not None:
            self.visual_model = self.visual_model.to(dtype=self.config.compute_dtype)
            self.text_hidden_fcs = self.text_hidden_fcs.to(dtype=self.config.compute_dtype)


    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到 SAM prompt 空间。"""
        assert len(self.text_hidden_fcs) == 1
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
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
    ) -> torch.Tensor:
        """
        批量生成 2D 分割掩码。

        Args:
            pred_embeddings: [B, C] — 每个样本的 SEG token 投影嵌入（已 mean-pool）
            image_embeddings: [B, C_sam, H_emb, W_emb] — SAM 图像编码特征
            input_size: (H, W) — 输入图像尺寸（batch 内统一）
            original_size: (H, W) — 原始图像尺寸（batch 内统一）

        Returns:
            pred_masks: [B, H_orig, W_orig]
        """
        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        image_embeddings = image_embeddings.to(self.config.compute_dtype)

        # [B, C] → [B, 1, C]，作为 prompt_encoder 的 text_embeds 输入
        text_embeds = pred_embeddings.unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
            points=None, boxes=None, masks=None,
            text_embeds=text_embeds,
        )
        sparse_embeddings = sparse_embeddings.to(pred_embeddings.dtype)

        # mask_decoder 已支持 batch：image_embeddings [B,...] 与 sparse [B,...] 一一对应
        low_res_masks, _ = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: [B, 1, H_low, W_low]

        pred_masks = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        # [B, 1, H, W] → [B, H, W]
        return pred_masks[:, 0]


class PointCloudHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询点云特征的解码器。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int
    ):
        super().__init__()
        self.config = config
        self.point_encoder = PointCloudEncoder(out_dim=text_hidden_size)

        text_fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(text_hidden_size, 2*text_hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2*text_hidden_size, config.hidden_size)),
            # ("dropout", nn.Dropout(0.0)),
        ]))
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.point_cloud_segmentor = PointCloud3DSegmentor(
            embed_dim=config.hidden_size,
            num_heads=8,
            num_decoder_layers=3,
            max_text_len=77,
        )
        for param in self.point_cloud_segmentor.parameters():
            param.requires_grad = True

        if self.config.compute_dtype is not None:
            self.point_encoder = self.point_encoder.to(dtype=self.config.compute_dtype)
            self.text_hidden_fcs = self.text_hidden_fcs.to(dtype=self.config.compute_dtype)
            self.point_cloud_segmentor = self.point_cloud_segmentor.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到点云分割器的嵌入空间。"""
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
        return projected

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        point_clouds: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量生成 3D 分割掩码。

        Args:
            pred_embeddings: [B, C] — 每个样本的 SEG token 投影嵌入（已 mean-pool）
            point_clouds: [B, N, 3] 或 [B, 3, N]

        Returns:
            pred_masks: [B, N]
        """
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)

        pred_embeddings = pred_embeddings.to(self.config.compute_dtype)
        point_clouds = point_clouds.to(self.config.compute_dtype)

        # [B, C] → [B, 1, C]，PointCloud3DSegmentor 接受 [B, L, C] 的 text_feat
        text_feat = pred_embeddings.unsqueeze(1)
        text_mask = torch.ones(
            text_feat.shape[0], 1,
            dtype=torch.bool, device=text_feat.device,
        )

        return self.point_cloud_segmentor(point_clouds, text_feat, text_mask)


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()
        self.seg_token_idx = self.config.seg_token_idx
        self.aff_token_idx = self.config.aff_token_idx

        self.mllm = MLLMBackbone(self.config.mllm)
        self.image_decoder = ImageHiddenStateDecoder(
            self.config.image_decoder, self.config.mllm.hidden_size, compute_dtype=self.config.compute_dtype
        )
        self.point_decoder = PointCloudHiddenStateDecoder(
            self.config.point_decoder, self.config.mllm.hidden_size, compute_dtype=self.config.compute_dtype
        )


    @property
    def tokenizer(self): return self.mllm.tokenizer

    def _extract_token_embeddings(
        self,
        input_ids: Optional[torch.Tensor],
        last_hidden_state: torch.Tensor,
        token_idx: Optional[int],
    ) -> torch.Tensor:
        """
        从隐藏状态中提取**第一个**匹配特殊 token（如 [SEG]）位置的嵌入。

        当前假设每个样本恰好有 1 个 [SEG] token。
        TODO: 若后续支持多 [SEG]（一个 SEG 对应一组图像/点云），
              应返回 [B, N_seg, C] 并在下游逐 SEG 生成 mask。

        Args:
            input_ids: [B, L] 输入 token IDs。
            last_hidden_state: [B, L', C] 投影后的隐藏状态。
            token_idx: 特殊 token 的词汇表索引。

        Returns:
            token_embeddings: [B, C] — 每个样本中第一个匹配 token 的嵌入。
                若某样本无匹配 token，对应行为零向量。
        """
        B, _, C = last_hidden_state.shape
        if input_ids is None or token_idx is None:
            return last_hidden_state.new_zeros(B, C)

        token_mask = input_ids == token_idx  # [B, L]

        # 若长度不一致，截断到较短的公共长度
        if last_hidden_state.shape[1] != token_mask.shape[1]:
            min_len = min(last_hidden_state.shape[1], token_mask.shape[1])
            token_mask = token_mask[:, :min_len]
            last_hidden_state = last_hidden_state[:, :min_len, :]

        # 取每个样本中第一个匹配位置的嵌入（无匹配时返回零向量）
        # has_token: [B], first_idx: [B]（无匹配时 first_idx 为 0，但会被 has_token 置零）
        has_token = token_mask.any(dim=1)                                    # [B]
        first_idx = token_mask.to(torch.long).argmax(dim=1)                  # [B]
        # 用 gather 提取：[B, L, C] → [B, 1, C] → [B, C]
        embeddings = last_hidden_state.gather(
            1, first_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, C)
        ).squeeze(1)                                                         # [B, C]
        # 无匹配 token 的样本置零
        embeddings = embeddings * has_token.unsqueeze(-1).to(embeddings.dtype)
        return embeddings

    def forward(
        self,
        # Qwen 推理所需
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        # 图像分割所需
        images: Optional[torch.Tensor] = None,
        original_size_list: Optional[List] = None,
        img_valid_mask: Optional[torch.Tensor] = None,
        img_gt_tensor: Optional[torch.Tensor] = None,  # 后续可能支持，暂且保留
        # 点云分割所需
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        pc_gt_tensor: Optional[torch.Tensor] = None,  # 后续可能支持，暂且保留
    ) -> Dict[str, Optional[torch.Tensor]]:
        B = input_ids.shape[0] if input_ids is not None else 1

        # ---- 1. MLLM 前向 ----
        mllm_out = self.mllm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )
        hidden_states = mllm_out["hidden_states"]  # [B, L, C]

        image_logits = None
        point_logits = None

        if hidden_states is not None:
            # ---- 2. 投影 + 提取 SEG token 嵌入 ----
            image_hidden = self.image_decoder.project_hidden_states(hidden_states)
            point_hidden = self.point_decoder.project_hidden_states(hidden_states)

            image_pred_emb = self._extract_token_embeddings(
                input_ids, image_hidden, self.seg_token_idx
            )  # [B, C']
            point_pred_emb = self._extract_token_embeddings(
                input_ids, point_hidden, self.seg_token_idx
            )  # [B, C']

            # ---- 3. 2D 图像分割 ----
            # 默认 images 非空（由数据集与 collate 保证），所有 rank 统一前向。
            image_embeddings = self.image_decoder.get_visual_embs(images)
            H, W = images.shape[-2], images.shape[-1]
            input_size = (H, W)
            original_size = tuple(original_size_list[0]) if original_size_list else (H, W)

            all_image_logits = self.image_decoder(
                image_pred_emb, image_embeddings, input_size, original_size
            )
            all_image_logits = all_image_logits.sigmoid_()

            # 将无效样本的输出置零（不影响 loss 计算）
            if img_valid_mask is not None:
                mask_2d = img_valid_mask.bool().view(B, 1, 1).to(all_image_logits.dtype)
                image_logits = all_image_logits * mask_2d
            else:
                image_logits = all_image_logits

            # ---- 4. 3D 点云分割 ----
            # 默认 point_clouds 非空（由数据集与 collate 保证），所有 rank 统一前向。
            all_point_logits = self.point_decoder(point_pred_emb, point_clouds)

            # 将无效样本的输出置零
            if pc_valid_lengths is not None:
                mask_3d = (pc_valid_lengths > 0).to(all_point_logits.dtype).unsqueeze(-1)
                point_logits = all_point_logits * mask_3d
            else:
                point_logits = all_point_logits

        return {
            "hidden_states": hidden_states,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "labels": labels,
            "output": mllm_out.get("output"),
        }


__all__ = [
    "JointAffordanceModel",
    "MLLMBackbone",
    "ImageHiddenStateDecoder",
    "PointCloudHiddenStateDecoder",
]
