"""
联合可供性模型骨架。

本文件定义一个高层模型管理基座，包含三个主要模块：
- MLLM 主干（分词器 + 视觉语言主干；当前为占位实现）
- 图像隐藏状态解码器（用文本隐藏状态查询图像特征）
- 点云隐藏状态解码器（用文本隐藏状态查询点云特征）

实现刻意轻量，仅聚焦结构。
"""
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from collections import OrderedDict

from configs import JointAffordanceConfig, ImageDecoderConfigs, PointDecoderConfigs, MLLMConfigs
from model.segment_anything import build_sam_vit_h
from model.pointnet2_utils import PointCloud3DSegmentor, PointCloudEncoder


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
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if labels is not None:
            model_inputs["labels"] = labels
        if images is not None:
            model_inputs["pixel_values"] = images
        model_inputs.update(kwargs)
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
        text_hidden_size: int = 2048,
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

        # 使用 SAM 的 image_encoder 作为图像特征提取器
        self.image_encoder = self.visual_model.image_encoder
        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=text_hidden_size, num_heads=config.num_heads, batch_first=True
        # )
        # self.ffn = nn.Sequential(
        #     nn.Linear(text_hidden_size,text_hidden_size * 4),
        #     nn.GELU(),
        #     nn.Linear(text_hidden_size * 4, text_hidden_size),
        # )
        # self.out_proj = nn.Linear(text_hidden_size, config.out_dim)

        
        text_fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(text_hidden_size, 2*text_hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2*text_hidden_size, self.config.hidden_size)),
            # ("dropout", nn.Dropout(0.0)),
        ]))
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


    # def forward(
    #     self,
    #     hidden_states: Optional[torch.Tensor],
    #     images: Optional[torch.Tensor] = None,
    #     image_features: Optional[torch.Tensor] = None,
    # ) -> Optional[torch.Tensor]:
    #     if hidden_states is None:
    #         return None
    #     if image_features is None:
    #         if images is None:
    #             return None
    #         image_features = self.get_visual_embs(images)
    #     attn_out, _ = self.cross_attn(hidden_states, image_features, image_features)
    #     x = self.ffn(attn_out)
    #     return self.out_proj(x)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert len(self.text_hidden_fcs) == 1
        projected = self.text_hidden_fcs[0](hidden_states)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
        return projected

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return self.visual_model.image_encoder(pixel_values)

    def generate_2d_masks(
        self,
        pred_embeddings_list: List[torch.Tensor],
        image_embeddings: torch.Tensor,
        resize_list: List[List],
        original_size_list: List[List],
    ) -> Optional[torch.Tensor]:
        batch_size = len(pred_embeddings_list)
        valid_indices = [i for i in range(batch_size) if len(pred_embeddings_list[i]) > 0]
        if len(valid_indices) == 0:
            return None

        # HACK: 逐个推理预测掩码，后续优化
        pred_masks_list = []
        for i in valid_indices:
            (sparse_embeddings, dense_embeddings) = self.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embeddings_list[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings_list[i].dtype)
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=original_size_list[i],
            )
            pred_masks_list.append(pred_mask[:, 0])

        batch_pred_masks = []
        for pred_mask in pred_masks_list:
            if pred_mask.shape[0] > 1:
                pred_mask = pred_mask.mean(dim=0, keepdim=True)
            batch_pred_masks.append(pred_mask[0])
        batch_pred_masks = torch.stack(batch_pred_masks, dim=0)

        if len(valid_indices) < batch_size:
            height, width = batch_pred_masks.shape[1], batch_pred_masks.shape[2]
            full_pred_masks = torch.zeros(
                batch_size, height, width,
                dtype=batch_pred_masks.dtype,
                device=batch_pred_masks.device,
            )
            for idx, valid_idx in enumerate(valid_indices):
                full_pred_masks[valid_idx] = batch_pred_masks[idx]
            return full_pred_masks
        return batch_pred_masks


class PointCloudHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询点云特征的解码器。"""

    def __init__(
        self,
        config: PointDecoderConfigs,
        text_hidden_size: int = 2048,
    ):
        super().__init__()
        self.config = config
        # 使用 PointNet++ 编码器提取点云特征
        self.point_encoder = PointCloudEncoder(out_dim=text_hidden_size)
        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=text_hidden_size, num_heads=config.num_heads, batch_first=True
        # )
        # self.ffn = nn.Sequential(
        #     nn.Linear(text_hidden_size, text_hidden_size * 4),
        #     nn.GELU(),
        #     nn.Linear(text_hidden_size * 4, text_hidden_size),
        # )
        # self.out_proj = nn.Linear(text_hidden_size, config.out_dim)

        # TODO: 移动到JointAff中
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

    # def forward(
    #     self,
    #     hidden_states: Optional[torch.Tensor],
    #     point_clouds: Optional[torch.Tensor] = None,
    #     point_features: Optional[torch.Tensor] = None,
    # ) -> Optional[torch.Tensor]:
    #     if hidden_states is None:
    #         return None
    #     if point_features is None:
    #         if point_clouds is None:
    #             return None
    #         if point_clouds.dim() == 3 and point_clouds.shape[1] != 3:
    #             point_clouds = point_clouds.permute(0, 2, 1).contiguous()
    #         point_features = self.point_encoder(point_clouds)
    #         if point_features.dim() == 2:
    #             point_features = point_features.unsqueeze(1)
    #     attn_out, _ = self.cross_attn(hidden_states, point_features, point_features)
    #     x = self.ffn(attn_out)
    #     return self.out_proj(x)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert len(self.text_hidden_fcs) == 1
        projected = self.text_hidden_fcs[0](hidden_states)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
        return projected

    def generate_3d_masks(
        self,
        pred_embeddings_list: List[torch.Tensor],
        point_clouds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)

        batch_size = point_clouds.shape[0]
        num_points = point_clouds.shape[2]
        valid_indices = [i for i in range(batch_size) if len(pred_embeddings_list[i]) > 0]
        if len(valid_indices) == 0:
            return None

        valid_point_clouds = point_clouds[valid_indices]
        max_text_len = max(len(pred_embeddings_list[i]) for i in valid_indices)
        embed_dim = pred_embeddings_list[valid_indices[0]].shape[-1]

        batch_text_feat = torch.zeros(
            len(valid_indices), max_text_len, embed_dim,
            dtype=pred_embeddings_list[valid_indices[0]].dtype,
            device=pred_embeddings_list[valid_indices[0]].device,
        )
        batch_text_mask = torch.zeros(
            len(valid_indices), max_text_len,
            dtype=torch.bool,
            device=pred_embeddings_list[valid_indices[0]].device,
        )
        for batch_idx, orig_idx in enumerate(valid_indices):
            text_len = len(pred_embeddings_list[orig_idx])
            batch_text_feat[batch_idx, :text_len] = pred_embeddings_list[orig_idx]
            batch_text_mask[batch_idx, :text_len] = True

        batch_pred_3d_masks = self.point_cloud_segmentor(
            valid_point_clouds,
            batch_text_feat,
            batch_text_mask,
        )

        if len(valid_indices) < batch_size:
            full_pred_3d_masks = torch.zeros(
                batch_size, num_points,
                dtype=batch_pred_3d_masks.dtype,
                device=batch_pred_3d_masks.device,
            )
            for idx, valid_idx in enumerate(valid_indices):
                full_pred_3d_masks[valid_idx] = batch_pred_3d_masks[idx]
            return full_pred_3d_masks
        return batch_pred_3d_masks


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()
        # self.vision_pretrained = self.config.vision_pretrained
        # self.ce_loss_weight = self.config.ce_loss_weight
        # self.dice_loss_weight = self.config.dice_loss_weight
        # self.bce_loss_weight = self.config.bce_loss_weight
        self.seg_token_idx = self.config.seg_token_idx
        self.aff_token_idx = self.config.aff_token_idx

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder)
        self.point_decoder = PointCloudHiddenStateDecoder(self.config.point_decoder)
        self.mllm = MLLMBackbone(self.config.mllm)
        
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.config.mm_vision_tower)


    @property
    def tokenizer(self): return self.mllm.tokenizer

    def _normalize_size_lists(
        self,
        images: torch.Tensor,
        original_size_list: Optional[List[List]],
        resize_list: Optional[List[List]],
    ):
        if images is None:
            return original_size_list, resize_list
        if original_size_list is None or resize_list is None:
            height, width = images.shape[-2], images.shape[-1]
            batch_size = images.shape[0]
            original_size_list = original_size_list or [[height, width] for _ in range(batch_size)]
            resize_list = resize_list or [[height, width] for _ in range(batch_size)]
        return original_size_list, resize_list

    def _extract_token_embeddings(
        self,
        input_ids: Optional[torch.Tensor],
        last_hidden_state: torch.Tensor,
        token_idx: Optional[int],
    ):
        """
        从语言模型隐藏状态中提取指定特殊 token（如 [SEG]）位置的嵌入，并按样本分组。

        Args:
            input_ids: 输入 token IDs，形状 [B, L]。
            last_hidden_state: 投影后的隐藏状态，形状 [B, L', C]（L' 可能因图像 token 插入而大于 L）。
            token_idx: 要提取的特殊 token 的词汇表索引（如 [SEG] 的 id）。

        Returns:
            token_embeddings_list: 长度为 B 的列表，第 i 个元素为第 i 个样本中该 token 的嵌入，形状 [num_tokens_i, C]。
        """
        if input_ids is None or token_idx is None:
            return [last_hidden_state.new_empty((0, last_hidden_state.shape[-1])) for _ in range(last_hidden_state.shape[0])]

        # Qwen 输入已包含视觉 token，占位符与 hidden_states 位置一致
        token_mask = input_ids == token_idx

        # 若长度不一致，截断到较短的公共长度，避免手动对齐
        if last_hidden_state.shape[1] != token_mask.shape[1]:
            min_len = min(last_hidden_state.shape[1], token_mask.shape[1])
            token_mask = token_mask[:, :min_len]
            last_hidden_state = last_hidden_state[:, :min_len, :]

        # 按 mask 取出对应位置的嵌入，得到一维张量 [总 token 数, C]
        token_embeddings = last_hidden_state[token_mask]
        # 每个样本中该 token 的数量，用于后续按样本切分
        token_counts = token_mask.int().sum(-1)
        token_offset = token_counts.cumsum(-1)
        token_offset = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=token_offset.device), token_offset], dim=0
        )

        # 按样本分组：第 i 个样本的嵌入为 token_embeddings[token_offset[i]:token_offset[i+1]]
        token_embeddings_list = []
        for i in range(len(token_offset) - 1):
            start_i, end_i = token_offset[i], token_offset[i + 1]
            token_embeddings_list.append(token_embeddings[start_i:end_i])
        return token_embeddings_list

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # 兼容 Qwen 数据处理：pixel_values / image_grid_thw / position_ids 等通过 kwargs 透传
        pixel_values = kwargs.pop("pixel_values", None)
        images_for_mllm = pixel_values if pixel_values is not None else images
        mllm_out = self.mllm(
            input_ids=input_ids,
            labels=labels,
            images=images_for_mllm,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = mllm_out["hidden_states"]

        pred_masks = None
        pred_3d_masks = None
        if hidden_states is not None:
            image_hidden = self.image_decoder.project_hidden_states(hidden_states)
            point_hidden = self.point_decoder.project_hidden_states(hidden_states)

            image_pred_embeddings = self._extract_token_embeddings(
                input_ids, image_hidden, self.seg_token_idx
            )
            point_pred_embeddings = self._extract_token_embeddings(
                input_ids, point_hidden, self.seg_token_idx
            )

            images_for_sam = images if images is not None else pixel_values
            if images_for_sam is not None:
                image_embeddings = self.image_decoder.get_visual_embs(images_for_sam)
                original_size_list, resize_list = self._normalize_size_lists(
                    images_for_sam,
                    kwargs.get("original_size_list"),
                    kwargs.get("resize_list"),
                )
                pred_masks = self.image_decoder.generate_2d_masks(
                    image_pred_embeddings, image_embeddings, resize_list, original_size_list
                )
                if pred_masks is not None:
                    pred_masks = pred_masks.sigmoid_()

            if point_clouds is not None:
                pred_3d_masks = self.point_decoder.generate_3d_masks(
                    point_pred_embeddings, point_clouds
                )

        image_logits = pred_masks
        point_logits = pred_3d_masks
        if image_logits is None:
            image_logits = self.image_decoder(
                hidden_states, images=images, image_features=image_features
            )
        if point_logits is None:
            point_logits = self.point_decoder(
                hidden_states, point_clouds=point_clouds, point_features=point_features
            )
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
