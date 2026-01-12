# coding=utf-8
# Copyright 2025 2D-3D Joint Affordance Project
#
# 基于 Qwen3-VL-MOE 架构的 2D-3D 联合理解模型
# 参考: transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
#
# 核心思路:
#   - 继承 Qwen3VLMoeForConditionalGeneration，复用其 LLM backbone 和生成能力
#   - 保留原有的 visual (ViT) 模块以支持 2D 图片输入
#   - 植入 3D Point Encoder (如 PointBERT) 处理 3D 点云
#   - 通过 Projector 将 3D 特征投影到 LLM 隐藏空间
#   - 使用 mask scatter 方式将图片和点云特征融合到文本 embedding 中
#   - 支持 2D 图片、3D 点云、或两者同时输入的多模态理解

from dataclasses import dataclass
from typing import Optional, Union, List

import torch
import torch.nn as nn

from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache

# 导入本地的 Qwen3VLMoe 模块
from .qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeModel,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeCausalLMOutputWithPast,
)
from .qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig, 
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)


# ============================================================================
# 配置类
# ============================================================================

class Qwen3PointLLMConfig(PretrainedConfig):
    """
    Qwen3PointLLM 的配置类 (支持 2D 图片 + 3D 点云)
    
    继承自 PretrainedConfig，用于存储模型的所有配置参数
    
    Args:
        text_config (`dict`, *optional*):
            Qwen3VLMoeTextConfig 的配置字典，用于初始化 LLM backbone
        vision_config (`dict`, *optional*):
            Qwen3VLMoeVisionConfig 的配置字典，用于初始化 2D 视觉编码器
        point_encoder_type (`str`, *optional*, defaults to `"pointbert"`):
            点云编码器类型，支持 "pointbert", "pointnet2" 等
        point_feature_dim (`int`, *optional*, defaults to 768):
            点云编码器输出的特征维度
        num_point_tokens (`int`, *optional*, defaults to 256):
            点云编码后的 token 数量
        point_token_id (`int`, *optional*, defaults to 151663):
            点云占位符 token 的 ID
        point_start_token_id (`int`, *optional*, defaults to 151664):
            点云开始标记的 token ID
        point_end_token_id (`int`, *optional*, defaults to 151665):
            点云结束标记的 token ID
        image_token_id (`int`, *optional*, defaults to 151655):
            图片占位符 token 的 ID (与原始 Qwen3VLMoe 保持一致)
        video_token_id (`int`, *optional*, defaults to 151656):
            视频占位符 token 的 ID
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            视觉开始标记的 token ID
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            视觉结束标记的 token ID
    """
    model_type = "qwen3_point_llm"
    
    def __init__(
        self,
        text_config: dict = None,
        vision_config: dict = None,
        point_encoder_type: str = "pointbert",
        point_feature_dim: int = 768,
        num_point_tokens: int = 256,
        # 点云特殊 token ID (使用新的 ID 避免与图片/视频冲突)
        point_token_id: int = 151663,
        point_start_token_id: int = 151664,
        point_end_token_id: int = 151665,
        # 图片/视频特殊 token ID (与原始 Qwen3VLMoe 保持一致)
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 初始化文本配置
        if text_config is None:
            self.text_config = Qwen3VLMoeTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Qwen3VLMoeTextConfig(**text_config)
        else:
            self.text_config = text_config
        
        # 初始化视觉配置 (2D 图片)
        if vision_config is None:
            self.vision_config = Qwen3VLMoeVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
            
        # 点云编码器配置
        self.point_encoder_type = point_encoder_type
        self.point_feature_dim = point_feature_dim
        self.num_point_tokens = num_point_tokens
        
        # 点云特殊 token ID
        self.point_token_id = point_token_id
        self.point_start_token_id = point_start_token_id
        self.point_end_token_id = point_end_token_id
        
        # 图片/视频特殊 token ID (与原始 Qwen3VLMoe 保持一致)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        
        # 从 text_config 获取 hidden_size
        self.hidden_size = self.text_config.hidden_size


# ============================================================================
# 输出类
# ============================================================================

@dataclass
class Qwen3PointLLMCausalLMOutputWithPast(ModelOutput):
    """
    Qwen3PointLLM 的输出类 (支持 2D 图片 + 3D 点云)
    
    继承自 ModelOutput，包含语言模型的所有输出
    
    Args:
        loss (`torch.FloatTensor`, *optional*):
            语言建模损失
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            语言模型头的预测分数
        past_key_values (`Cache`, *optional*):
            用于加速解码的 KV 缓存
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            各层的隐藏状态
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            注意力权重
        point_features (`torch.FloatTensor`, *optional*):
            点云编码器输出的特征
        image_features (`torch.FloatTensor`, *optional*):cur
            图片编码器输出的特征
        rope_deltas (`torch.LongTensor`, *optional*):
            RoPE 位置编码的增量
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    point_features: Optional[torch.FloatTensor] = None
    image_features: Optional[torch.FloatTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None


# ============================================================================
# 点云编码器 (简化版 PointBERT)
# ============================================================================

class SimplePointEncoder(nn.Module):
    """
    简化版点云编码器
    
    将点云 (B, N, 6) 编码为 (B, num_tokens, feature_dim) 的特征
    其中 6 = XYZ (3) + RGB/Normal (3)
    
    实际使用时可替换为 PointBERT, Point-BERT, ULIP 等预训练模型
    
    Args:
        num_points (`int`): 输入点云的点数
        num_tokens (`int`): 输出的 token 数量
        feature_dim (`int`): 输出特征维度
        input_dim (`int`): 输入点的维度 (默认 6: XYZ + RGB)
    """
    def __init__(
        self,
        num_points: int = 8192,
        num_tokens: int = 256,
        feature_dim: int = 768,
        input_dim: int = 6,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        
        # 简单的点云编码: 使用 1D 卷积进行特征提取
        # 实际应用中应替换为 PointBERT 等预训练模型
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, feature_dim, 1),
        )
        
        # 将 N 个点聚合为 num_tokens 个 token
        # 使用自适应池化 + 线性层
        self.token_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool1d(num_tokens),
        )
        
        # Token-wise 的特征增强
        self.token_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        
    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_clouds: (B, N, 6) 点云数据，包含 XYZ 和 RGB/Normal
            
        Returns:
            point_features: (B, num_tokens, feature_dim) 点云特征
        """
        # (B, N, 6) -> (B, 6, N)
        x = point_clouds.transpose(1, 2)
        
        # 特征提取: (B, 6, N) -> (B, feature_dim, N)
        x = self.encoder(x)
        
        # Token 聚合: (B, feature_dim, N) -> (B, feature_dim, num_tokens)
        x = self.token_aggregator(x)
        
        # (B, feature_dim, num_tokens) -> (B, num_tokens, feature_dim)
        x = x.transpose(1, 2)
        
        # Token 编码增强
        x = self.token_encoder(x)
        
        return x


# ============================================================================
# 投影器 (Projector)
# ============================================================================

class PointProjector(nn.Module):
    """
    点云特征投影器
    
    将点云编码器输出的特征投影到 LLM 的隐藏空间
    参考 Qwen3VLMoe 中的 Qwen3VLMoeVisionPatchMerger
    
    Args:
        input_dim (`int`): 输入特征维度 (点云编码器输出)
        output_dim (`int`): 输出特征维度 (LLM hidden_size)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear_fc1 = nn.Linear(input_dim, output_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(output_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_tokens, input_dim) 点云特征
            
        Returns:
            projected: (B, num_tokens, output_dim) 投影后的特征
        """
        x = self.norm(x)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


# ============================================================================
# 主模型: Qwen3PointLLM
# ============================================================================

class Qwen3PointLLM(Qwen3VLMoePreTrainedModel):
    """
    基于 Qwen3-VL-MOE 的 2D-3D 联合理解模型
    
    架构说明:
        1. 继承 Qwen3VLMoePreTrainedModel，复用其权重初始化和预训练加载逻辑
        2. 保留原始的 Qwen3VLMoeVisionModel 处理 2D 图片
        3. 使用 Qwen3VLMoeTextModel 作为 LLM backbone
        4. 植入自定义的 Point Encoder 处理 3D 点云
        5. 通过 Projector 将点云特征对齐到 LLM 空间
        6. 使用 mask scatter 方式将图片和点云特征融合到文本 embedding
    
    支持的输入模式:
        - 仅文本
        - 文本 + 2D 图片
        - 文本 + 3D 点云
        - 文本 + 2D 图片 + 3D 点云 (联合理解)
    
    Args:
        config (`Qwen3PointLLMConfig`): 模型配置
    """
    config_class = Qwen3PointLLMConfig
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock"]
    
    def __init__(self, config: Qwen3PointLLMConfig):
        super().__init__(config)
        
        # 1. 保留原始的 2D 视觉编码器 (ViT)
        self.visual = Qwen3VLMoeVisionModel._from_config(config.vision_config)
        
        # 2. 初始化 LLM backbone (文本部分)
        self.language_model = Qwen3VLMoeTextModel(config.text_config)
        
        # 3. 初始化 LM Head
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, 
            config.text_config.vocab_size, 
            bias=False
        )
        
        # 4. 植入 3D Point Encoder
        # 输入: (Batch, N_points, 6) - XYZ + RGB/Normal
        # 输出: (Batch, num_tokens, point_feature_dim)
        self.point_encoder = SimplePointEncoder(
            num_points=8192,
            num_tokens=config.num_point_tokens,
            feature_dim=config.point_feature_dim,
        )
        
        # 5. 植入 Point Projector (维度对齐)
        # 将点云特征从 point_feature_dim 投影到 LLM 的 hidden_size
        self.point_projector = PointProjector(
            input_dim=config.point_feature_dim,
            output_dim=config.hidden_size,
        )
        
        # 6. 存储特殊 token ID
        self.point_token_id = config.point_token_id
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        
        # 用于缓存 rope_deltas
        self.rope_deltas = None
        
        # 初始化权重
        self.post_init()
        
    def get_input_embeddings(self):
        return self.language_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
        
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    # ========================================================================
    # 2D 图片处理方法 (复用原始 Qwen3VLMoe 的逻辑)
    # ========================================================================
    
    def get_image_features(
        self, 
        pixel_values: torch.FloatTensor, 
        image_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        编码 2D 图片为特征向量
        
        Args:
            pixel_values: 图片像素值
            image_grid_thw: 图片的时间、高度、宽度网格信息
            
        Returns:
            image_embeds: 图片特征列表
            deepstack_image_embeds: DeepStack 特征列表
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
    
    def get_video_features(
        self, 
        pixel_values_videos: torch.FloatTensor, 
        video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        编码视频为特征向量 (与图片处理相同)
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw)
    
    # ========================================================================
    # 3D 点云处理方法
    # ========================================================================
        
    def get_point_features(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        编码点云并投影到 LLM 空间
        
        Args:
            point_clouds: (B, N, 6) 点云数据
            
        Returns:
            point_embeds: (B, num_tokens, hidden_size) 投影后的点云特征
        """
        # A. 提取 3D 特征: (B, N, 6) -> (B, num_tokens, point_feature_dim)
        point_features = self.point_encoder(point_clouds)
        
        # B. 投影到 LLM 空间: (B, num_tokens, point_feature_dim) -> (B, num_tokens, hidden_size)
        point_embeds = self.point_projector(point_features)
        
        return point_embeds
    
    # ========================================================================
    # 多模态融合方法
    # ========================================================================
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        point_features: Optional[torch.FloatTensor] = None,
    ):
        """
        获取多模态占位符的 mask，并验证 token 数量是否匹配
        
        支持 2D 图片、视频和 3D 点云的占位符检测
        
        Args:
            input_ids: (B, seq_len) 输入 token ID
            inputs_embeds: (B, seq_len, hidden_size) 文本 embedding
            image_features: 图片特征 (可选)
            video_features: 视频特征 (可选)
            point_features: 点云特征 (可选)
            
        Returns:
            image_mask, video_mask, point_mask: 各模态的占位符 mask
        """
        # 图片 mask
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        
        # 视频 mask
        special_video_mask = input_ids == self.video_token_id
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: "
                f"tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )
        
        # 点云 mask
        special_point_mask = input_ids == self.point_token_id
        n_point_tokens = special_point_mask.sum()
        special_point_mask = special_point_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if point_features is not None and inputs_embeds[special_point_mask].numel() != point_features.numel():
            raise ValueError(
                f"Point features and point tokens do not match: "
                f"tokens: {n_point_tokens}, features {point_features.shape[0]}"
            )
        
        return special_image_mask, special_video_mask, special_point_mask
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        计算 RoPE 位置编码索引
        
        复用 Qwen3VLMoeModel 的逻辑，支持图片和视频的位置编码
        点云使用简单的线性位置编码
        """
        # 处理视频网格
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            for i, input_ids_i in enumerate(total_input_ids):
                input_ids_i = input_ids_i[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                        
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3PointLLMCausalLMOutputWithPast]:
        """
        前向传播
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入 token ID，其中 point_token_id 作为点云占位符
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                注意力 mask
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                位置 ID
            past_key_values (`Cache`, *optional*):
                KV 缓存，用于加速生成
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                直接提供 embedding，跳过 embedding 层
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算语言建模损失的标签
            point_clouds (`torch.FloatTensor` of shape `(batch_size, num_points, 6)`, *optional*):
                点云数据，包含 XYZ 坐标和 RGB/Normal
            cache_position (`torch.LongTensor`, *optional*):
                缓存位置
            use_cache (`bool`, *optional*):
                是否使用 KV 缓存
            output_attentions (`bool`, *optional*):
                是否输出注意力权重
            output_hidden_states (`bool`, *optional*):
                是否输出隐藏状态
            return_dict (`bool`, *optional*):
                是否返回 ModelOutput 对象
                
        Returns:
            `Qwen3PointLLMCausalLMOutputWithPast` 或 `tuple`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 1. 获取文本 embedding
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # 2. 处理点云输入
        point_features = None
        if point_clouds is not None:
            # A. 编码点云并投影
            point_embeds = self.get_point_features(point_clouds)
            point_features = point_embeds  # 保存用于输出
            
            # B. 获取点云占位符 mask
            point_mask = self._get_point_placeholder_mask(
                input_ids, inputs_embeds, point_embeds
            )
            
            # C. 融合点云 embedding 到文本 embedding
            inputs_embeds = self._merge_point_embeddings(
                inputs_embeds, point_embeds, point_mask
            )
        
        # 3. 通过 LLM backbone
        outputs = self.language_model(
            input_ids=None,  # 已经转换为 inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # 4. 计算 logits
        logits = self.lm_head(hidden_states)
        
        # 5. 计算损失
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return Qwen3PointLLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            point_features=point_features,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        point_clouds=None,
        **kwargs,
    ):
        """
        为生成准备输入
        
        在第一次前向传播后，点云已经被编码并融合到 KV 缓存中，
        后续生成步骤不需要再次处理点云
        """
        # 如果不是第一次生成（有 past_key_values），则不需要点云
        if past_key_values is not None:
            if cache_position is not None and cache_position[0] != 0:
                point_clouds = None
        
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            "use_cache": use_cache,
            "point_clouds": point_clouds,
        }
        
        return model_inputs


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "Qwen3PointLLMConfig",
    "Qwen3PointLLM",
    "Qwen3PointLLMCausalLMOutputWithPast",
    "SimplePointEncoder",
    "PointProjector",
]
