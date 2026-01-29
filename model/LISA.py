"""
LISA (Large Language and Vision Assistant) 模型实现
结合了 LLaVA（Large Language and Vision Assistant）和 SAM（Segment Anything Model）
用于语言引导的图像分割任务
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel, CLIPImageProcessor

from utils.common import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .pointnet2_utils import PointCloud3DSegmentor



class LisaMetaModel:
    """
    LISA 元模型类
    定义了 LISA 特定的模块：SAM 视觉模型和文本到分割的投影层
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # 如果配置中没有相关属性，从 kwargs 中获取
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        """
        初始化 LISA 特定模块
        
        Args:
            config: 模型配置对象
        """
        # 构建 SAM（Segment Anything Model）视觉编码器
        # SAM 用于生成图像嵌入和分割掩码
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        # 默认冻结 SAM 的所有参数
        for param in self.visual_model.parameters():
            param.requires_grad = False
        # 如果配置要求训练 mask decoder，则解冻其参数
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # 投影层：将语言模型的隐藏状态投影到 SAM 的 prompt 嵌入空间
        # 用于将文本特征转换为分割提示
        in_dim = config.hidden_size  # 语言模型的隐藏维度
        out_dim = config.out_dim  # SAM prompt encoder 的嵌入维度（通常为 256）
        text_fc = [
            nn.Linear(in_dim, in_dim),  # 第一层线性变换
            nn.ReLU(inplace=True),  # ReLU 激活
            nn.Linear(in_dim, out_dim),  # 第二层线性变换，输出到目标维度
            nn.Dropout(0.0),  # Dropout（当前为 0，即不使用）
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        # 投影层参数需要训练
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # 3D点云分割器
        self.point_cloud_segmentor = PointCloud3DSegmentor(
            embed_dim=out_dim,  # 与 SAM prompt encoder 的嵌入维度一致
            num_heads=8,
            num_decoder_layers=3,
            max_text_len=77
        )
        self.point_cloud_segmentor.train()
        for param in self.point_cloud_segmentor.parameters():
            param.requires_grad = True

class LisaModel(LisaMetaModel, LlavaLlamaModel):
    """
    LISA 模型主体
    继承自 LisaMetaModel（LISA 特定模块）和 LlavaLlamaModel（LLaVA + LLaMA 基础架构）
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        # 配置 LLaVA 相关参数
        self.config.use_cache = False  # 训练时不使用缓存
        self.config.vision_tower = self.config.mm_vision_tower  # 视觉编码器
        self.config.mm_vision_select_feature = "patch"  # 选择 patch 特征（而非 CLS token）
        self.config.image_aspect_ratio = "square"  # 图像宽高比设为正方形
        self.config.image_grid_pinpoints = None  # 不使用网格定位
        self.config.tune_mm_mlp_adapter = False  # 不微调多模态 MLP 适配器
        self.config.freeze_mm_mlp_adapter = True  # 冻结多模态 MLP 适配器
        self.config.pretrain_mm_mlp_adapter = None  # 不使用预训练的 MLP 适配器
        self.config.mm_use_im_patch_token = False  # 不使用图像 patch token


class LISAForCausalLM(LlavaLlamaForCausalLM):
    """
    LISA 因果语言模型
    继承自 LlavaLlamaForCausalLM，添加了分割功能
    
    架构组成：
    1. LLaVA 基础架构（多模态语言模型）
    2. SAM 视觉编码器（用于图像嵌入）
    3. SAM Mask Decoder（用于生成分割掩码）
    4. 文本到分割的投影层（连接语言特征和分割提示）
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 如果配置中没有相关属性，从 kwargs 中获取并设置
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            # 损失函数权重
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        # 分割标记的索引（[SEG] token）
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.aff_token_idx = kwargs.pop("aff_token_idx") # 2D aff token
        # self.pc_token_idx = kwargs.pop("pc_token_idx")  # 3D aff token

        super().__init__(config)

        # 创建 LISA 模型主体
        self.model = LisaModel(config, **kwargs)

        # 语言模型头：将隐藏状态映射到词汇表
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化 CLIP 图像预处理器
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            config.mm_vision_tower
        )

        # 初始化权重并应用最终处理
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        获取图像的视觉嵌入（使用 SAM 的图像编码器）- 批量推理版本
        
        Args:
            pixel_values: 输入图像张量，形状为 [batch_size, C, H, W]，已满足SAM图像编码器的输入要求
            
        Returns:
            image_embeddings: 图像嵌入，形状为 [batch_size, embed_dim, H', W']
        """
        # 无梯度推理，SAM图像编码器无需训练梯度
        with torch.no_grad():
            # 可选：推理前清理无用到的CUDA缓存（轻量操作，不影响批量推理）
            # torch.cuda.empty_cache()
            image_embeddings = self.model.visual_model.image_encoder(pixel_values)
        
        return image_embeddings
    
    def get_point_cloud_embs(self, point_clouds: torch.FloatTensor):
        """
        获取3D点云的特征嵌入
        Input:
            point_clouds: 点云数据, [B, 3, N] 或 [B, N, 3]
        Return:
            point_embeddings: 点云特征, [B, out_dim]
        """
        # 确保点云格式为 [B, 3, N]
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)
        point_embeddings = self.model.point_cloud_encoder(point_clouds)
        return point_embeddings
    
    def preprocess_clip_images(self, images_clip: torch.FloatTensor):
        """
        预处理 CLIP 图像 TODO: 优化掉clip，统一使用gpu计算
        
        Args:
            images_clip: 输入图像张量，形状为 [B, C, H, W]，值范围 [0, 1] 或 [0, 255]
            
        Returns:
            processed_images: 预处理后的图像张量，形状为 [B, C, H', W']
        """
        # 如果图像已经是归一化的 [0, 1] 范围，转换为 [0, 255]
        if images_clip.max() <= 1.0:
            images_clip = images_clip * 255.0
        
        # 转换为 numpy 格式以便使用 CLIPImageProcessor
        # CLIPImageProcessor 期望输入为 PIL Image 或 numpy array
        batch_size = images_clip.shape[0]
        device = images_clip.device
        dtype = images_clip.dtype
        
        # 将张量转换为 numpy (B, H, W, C) 格式
        images_np = images_clip.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')
        
        # 使用 CLIP 预处理器处理每张图像
        processed_images_list = []
        for i in range(batch_size):
            # 预处理单张图像
            processed = self.clip_image_processor(
                images=images_np[i],
                return_tensors="pt"
            )
            processed_images_list.append(processed['pixel_values'][0])
        
        # 堆叠并转移到原始设备
        processed_images = torch.stack(processed_images_list).to(device=device, dtype=dtype)
        
        return processed_images

    def forward(self, **kwargs):
        """
        前向传播入口
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            模型输出
        """
        # 如果包含 past_key_values（用于生成），使用父类方法
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        # 否则使用自定义的 model_forward
        return self.model_forward(**kwargs)
    
    def _extract_token_embeddings(self, output_ids, last_hidden_state, token_idx, has_image):
        """
        提取特定 token 的嵌入并按样本分组
        
        Args:
            output_ids: 输出的 token IDs [B, L]
            last_hidden_state: 投影后的隐藏状态 [B, L', C]
            token_idx: 要提取的 token 索引
            has_image: 是否有图像输入（用于调试信息）
            
        Returns:
            token_embeddings_list: 按样本分组的 token 嵌入列表
        """
        # 找到指定 token 的位置
        token_mask = output_ids[:, 1:] == token_idx
        
        # 关键修复：动态调整 token_mask 的长度以匹配 last_hidden_state
        # last_hidden_state.shape: [B, L', C]
        # token_mask.shape: [B, L-1]
        actual_seq_len = last_hidden_state.shape[1]
        current_mask_len = token_mask.shape[1]
        
        if actual_seq_len > current_mask_len:
            # 需要在前面填充（图像特征被插入到序列开头）
            padding_len = actual_seq_len - current_mask_len
            token_mask = torch.cat(
                [torch.zeros((token_mask.shape[0], padding_len), dtype=torch.bool, device=token_mask.device), token_mask],
                dim=1,
            )
        elif actual_seq_len < current_mask_len:
            # 截断到实际长度
            token_mask = token_mask[:, :actual_seq_len]
        
        # 提取 token 的嵌入
        token_embeddings = last_hidden_state[token_mask]
        
        # 按样本分组
        token_counts = token_mask.int().sum(-1)
        token_offset = token_counts.cumsum(-1)
        token_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=token_offset.device), token_offset], dim=0)
        
        token_embeddings_list = []
        for i in range(len(token_offset) - 1):
            start_i, end_i = token_offset[i], token_offset[i + 1]
            token_embeddings_list.append(token_embeddings[start_i:end_i])
        
        return token_embeddings_list
    
    def _generate_2d_masks(self, pred_embeddings_list, image_embeddings, resize_list, original_size_list):
        """
        使用 SAM 生成 2D 分割掩码
        
        Args:
            pred_embeddings_list: 按样本分组的预测嵌入列表
            image_embeddings: SAM 图像嵌入 [B, C, H, W]
            resize_list: 图像尺寸调整列表
            original_size_list: 原始图像尺寸列表
            
        Returns:
            pred_masks: 预测的 2D 掩码张量 [B, H, W] 或 None（如果所有样本都没有有效嵌入）
        """
        batch_size = len(pred_embeddings_list)
        multimask_output = False
        
        # 检查哪些样本有有效的嵌入
        valid_indices = [i for i in range(batch_size) if len(pred_embeddings_list[i]) > 0]
        
        if len(valid_indices) == 0:
            # 所有样本都没有特殊 token，返回 None
            return None
        
        # 收集有效样本的掩码
        pred_masks_list = []
        
        for i in valid_indices:
            # 将文本特征转换为 SAM 提示嵌入
            (sparse_embeddings, dense_embeddings) = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embeddings_list[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings_list[i].dtype)
            
            # 使用 SAM 的 mask decoder 生成掩码
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # 后处理掩码：调整到原始图像尺寸
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=original_size_list[i],
            )
            pred_masks_list.append(pred_mask[:, 0])  # [num_tokens, H, W]
        
        # 将列表转换为张量 [B_valid, num_tokens, H, W]
        # 注意：每个样本可能有不同数量的 token，这里假设每个样本只有一个 token
        # 如果有多个 token，取平均或最大值
        batch_pred_masks = []
        for pred_mask in pred_masks_list:
            if pred_mask.shape[0] > 1:
                # 多个 token，取平均
                pred_mask = pred_mask.mean(dim=0, keepdim=True)  # [1, H, W]
            batch_pred_masks.append(pred_mask[0])  # [H, W]
        
        # 堆叠成 [B_valid, H, W]
        batch_pred_masks = torch.stack(batch_pred_masks, dim=0)
        
        # 为无效样本填充零掩码
        if len(valid_indices) < batch_size:
            H, W = batch_pred_masks.shape[1], batch_pred_masks.shape[2]
            full_pred_masks = torch.zeros(
                batch_size, H, W,
                dtype=batch_pred_masks.dtype,
                device=batch_pred_masks.device
            )
            for idx, valid_idx in enumerate(valid_indices):
                full_pred_masks[valid_idx] = batch_pred_masks[idx]
            return full_pred_masks
        else:
            return batch_pred_masks
    
    def _generate_3d_masks(self, pred_embeddings_list, point_clouds):
        """
        使用 PointNet++ 生成 3D 点云掩码（支持批处理）
        
        Args:
            pred_embeddings_list: 按样本分组的预测嵌入列表，每个元素形状 [num_tokens_i, embed_dim]
            point_clouds: 点云数据 [B, 3, N]
            
        Returns:
            pred_3d_masks: 预测的 3D 掩码张量 [B, N] 或 None（如果所有样本都没有有效嵌入）
        """
        # 确保点云格式为 [B, 3, N]
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)
        
        batch_size = point_clouds.shape[0]
        num_points = point_clouds.shape[2]
        
        # 检查哪些样本有有效的嵌入
        valid_indices = [i for i in range(batch_size) if len(pred_embeddings_list[i]) > 0]
        
        if len(valid_indices) == 0:
            # 所有样本都没有特殊 token，返回 None
            return None
        
        # 准备批处理数据
        # 1. 收集有效样本的点云
        valid_point_clouds = point_clouds[valid_indices]  # [B_valid, 3, N]
        
        # 2. 填充文本特征到相同长度
        max_text_len = max(len(pred_embeddings_list[i]) for i in valid_indices)
        embed_dim = pred_embeddings_list[valid_indices[0]].shape[-1]
        
        # 创建批处理的文本特征和掩码
        batch_text_feat = torch.zeros(
            len(valid_indices), max_text_len, embed_dim,
            dtype=pred_embeddings_list[valid_indices[0]].dtype,
            device=pred_embeddings_list[valid_indices[0]].device
        )
        batch_text_mask = torch.zeros(
            len(valid_indices), max_text_len,
            dtype=torch.bool,
            device=pred_embeddings_list[valid_indices[0]].device
        )
        
        for batch_idx, orig_idx in enumerate(valid_indices):
            text_len = len(pred_embeddings_list[orig_idx])
            batch_text_feat[batch_idx, :text_len] = pred_embeddings_list[orig_idx]
            batch_text_mask[batch_idx, :text_len] = True
        
        # 3. 批量调用 3D 分割器
        batch_pred_3d_masks = self.model.point_cloud_segmentor(
            valid_point_clouds,  # [B_valid, 3, N]
            batch_text_feat,     # [B_valid, max_text_len, embed_dim]
            batch_text_mask      # [B_valid, max_text_len]
        )  # [B_valid, N]
        
        # 4. 重组输出，为无效样本填充零掩码
        if len(valid_indices) < batch_size:
            # 创建完整的批次张量
            full_pred_3d_masks = torch.zeros(
                batch_size, num_points,
                dtype=batch_pred_3d_masks.dtype,
                device=batch_pred_3d_masks.device
            )
            # 填充有效样本的预测
            for idx, valid_idx in enumerate(valid_indices):
                full_pred_3d_masks[valid_idx] = batch_pred_3d_masks[idx]
            return full_pred_3d_masks
        else:
            # 所有样本都有效，直接返回
            return batch_pred_3d_masks

    def model_forward(
        self,
        images: torch.FloatTensor = None,
        images_clip: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_masks: torch.LongTensor = None,
        offset: torch.LongTensor = None,
        # img_masks_tensor: torch.Tensor = None,
        original_size_list: List[List] = None,
        resize_list: List[List] = None,
        inference: bool = False,
        point_clouds: torch.FloatTensor = None,  # 3D点云输入 [B, 3, N] 或 [B, N, 3]
        # pc_masks_tensor: torch.FloatTensor = None,  # 3D点云真实掩码列表
        batch_size: int = None,  # 从外部传入的 batch_size（可选）
        img_valid_mask: torch.BoolTensor = None,  # 图像有效性标记 [B]
        pc_valid_mask: torch.BoolTensor = None,  # 点云有效性标记 [B]
        pc_valid_lengths: torch.LongTensor = None,  # 点云有效长度 [B]
        **kwargs,
    ):
        """
        模型前向传播主函数（支持动态模态输入和有效性屏蔽）
        
        Args:
            images: SAM 输入图像，形状 [batch_size, C, H, W]，可选
            images_clip: CLIP 输入图像，形状 [batch_size, C, H', W']，可选
            input_ids: 输入 token IDs
            labels: 文本标签（用于计算语言模型损失）
            attention_masks: 注意力掩码
            offset: 批次偏移量，用于处理不同长度的序列
            original_size_list: 原始图像尺寸列表 [(H, W), ...]
            resize_list: 调整后的图像尺寸列表 [(H, W), ...]
            inference: 是否为推理模式（不计算损失）
            point_clouds: 3D点云输入
            batch_size: 从外部传入的 batch_size（可选，如果未提供则自动计算）
            img_valid_mask: 图像有效性标记 [B]，True 表示该样本有真实图像
            pc_valid_mask: 点云有效性标记 [B]，True 表示该样本有真实点云
            pc_valid_lengths: 点云有效长度 [B]，表示每个样本的真实点数
            
        Returns:
            包含损失和预测结果的字典
        """
        # 获取 batch_size（优先使用传入的值，否则自动计算）
        if batch_size is None:
            batch_size = input_ids.shape[0]
        
        # debug: 必须有valid输入
        if img_valid_mask is None or pc_valid_mask is None or pc_valid_lengths is None:
            raise ValueError

        # 当所有样本都无效时（全0填充），has_valid_image/has_valid_point_cloud 应该为 False
        has_valid_image = img_valid_mask.any()
        has_valid_point_cloud = pc_valid_mask.any()
        
        # 获取 SAM 图像嵌入（仅在有有效图像输入时）
        image_embeddings = None
        if has_valid_image:
            image_embeddings = self.get_visual_embs(images)

        # # 找到所有 [SEG] token 和 [AFF] token 的位置
        # # 根据输入模态决定要查找的 token
        # if has_valid_image and has_valid_point_cloud:
        #     # 两种模态都有，查找 [SEG] 和 [AFF]
        #     seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx) | (input_ids[:, 1:] == self.aff_token_idx)
        # elif has_valid_image:
        #     # 只有图像，只查找 [SEG]
        #     seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)
        # else:
        #     # 只有点云，只查找 [AFF]
        #     seg_token_mask = (input_ids[:, 1:] == self.aff_token_idx)
        
        # 使用 [SEG] 同时作为点云和图片的token
        if has_valid_image or has_valid_point_cloud:
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        else:
            # 纯文本批次，没有任何有效模态
            seg_token_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
        
        seg_token_mask = torch.cat([
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1), dtype=torch.bool, device=seg_token_mask.device),
            ], dim=1)

        # 统一处理图像预处理和扩展（仅在有有效图像时）
        if has_valid_image:
            if inference:
                # 推理模式：扩展单张图像以匹配序列长度
                assert images_clip.shape[0] == 1
                length = input_ids.shape[0]
                images_clip = self.preprocess_clip_images(images_clip)
                images_clip = images_clip.expand(length, -1, -1, -1).contiguous()
            else:
                # 训练模式：根据 offset 为每个样本准备对应的图像
                images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        images_clip[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    images_clip_list.append(images_clip_i)
                images_clip = torch.cat(images_clip_list, dim=0)
                images_clip = self.preprocess_clip_images(images_clip)
        else:
            images_clip = None

        # 统一调用父类前向传播（LLaVA）
        # 推理和训练的唯一区别是是否传入labels
        output = super().forward(
            images=images_clip if has_valid_image else None,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels if not inference else None,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states

        # 将语言模型的隐藏状态投影到 prompt 嵌入空间
        assert len(self.model.text_hidden_fcs) == 1
        last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states[-1])
        
        # 确保 last_hidden_state 始终是 3D 张量 [B, L', C]
        if last_hidden_state.dim() == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)  # [1, L', C]
        
        # 动态调整 seg_token_mask 的长度以匹配 last_hidden_state
        # last_hidden_state.shape: [B, L', C]
        # seg_token_mask.shape: [B, L]
        # 需要将 seg_token_mask 扩展到 L' 的长度
        actual_seq_len = last_hidden_state.shape[1]
        current_mask_len = seg_token_mask.shape[1]
        
        if actual_seq_len > current_mask_len:
            # 需要在前面填充（图像特征被插入到序列开头）
            padding_len = actual_seq_len - current_mask_len
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], padding_len), dtype=torch.bool, device=seg_token_mask.device), seg_token_mask],
                dim=1,
            )
        elif actual_seq_len < current_mask_len:
            # 理论上不应该发生
            raise ValueError
            seg_token_mask = seg_token_mask[:, :actual_seq_len]
        
        # 提取 [SEG]/[AFF] token 位置的嵌入作为分割提示
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        # 计算特殊 token 的累积偏移量
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=seg_token_offset.device), seg_token_offset], dim=0
        )

        # 根据批次偏移量调整（只有在有 offset 时才调整）
        if offset is not None:
            seg_token_offset = seg_token_offset[offset]

        # 将预测嵌入按样本分组
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # ========== 2D 分割：使用 SAM 生成分割掩码（仅在有有效图像输入时）==========
        pred_masks = None
        if has_valid_image:
            pred_masks = self._generate_2d_masks(pred_embeddings, image_embeddings, resize_list, original_size_list)
            pred_masks = pred_masks * img_valid_mask[:, None, None]
            
        # ========== 3D 分割：使用 PointNet++ 处理点云（仅在有有效点云输入时）==========
        pred_3d_masks = None
        if has_valid_point_cloud:
            pred_3d_masks = self._generate_3d_masks(pred_embeddings, point_clouds)
            pred_3d_masks = pred_3d_masks * pc_valid_mask[:, None]

        # 返回预测结果和中间数据
        return {
            "output": output,  # 语言模型输出
            "pred_masks": pred_masks,  # [B, H, W] 或 None
            "pred_3d_masks": pred_3d_masks,  # [B, N] 或 None
            'has_valid_image': has_valid_image, 
            'has_valid_point_cloud': has_valid_point_cloud,
        }
