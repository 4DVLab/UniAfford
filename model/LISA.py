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

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, debug_gradient_graph)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class GuidedPointBlock(nn.Module):
    """
    引导点块 (GPB)：使用文本特征引导点云特征的交叉注意力模块
    类似于参考代码中的 gpb 模块
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(GuidedPointBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, text_feat, point_feat, text_mask=None):
        """
        Args:
            text_feat: 文本特征 [B, L, C]
            point_feat: 点云特征 [B, N, C]
            text_mask: 文本掩码 [B, L]，True 表示有效位置
        Returns:
            enhanced_point_feat: 增强后的点云特征 [B, N, C]
        """
        # 交叉注意力：点云特征作为 query，文本特征作为 key/value
        if text_mask is not None:
            # MultiheadAttention 的 key_padding_mask: True 表示忽略
            key_padding_mask = ~text_mask
        else:
            key_padding_mask = None
        
        attn_out, _ = self.cross_attn(point_feat, text_feat, text_feat, key_padding_mask=key_padding_mask)
        point_feat = self.norm1(point_feat + attn_out)
        point_feat = self.norm2(point_feat + self.ffn(point_feat))
        return point_feat


class PointCloud3DSegmentor(nn.Module):
    """
    3D点云分割器
    
    架构：
    1. PointNet++ Set Abstraction 编码器：逐层下采样点云
    2. Transformer Decoder：使用文本 token 作为 query 提取点云特征
    3. PointNet++ Feature Propagation 解码器：上采样回原始点数
    4. 输出每个点的分割掩码
    """
    def __init__(self, embed_dim=256, num_heads=8, num_decoder_layers=3, max_text_len=77):
        super(PointCloud3DSegmentor, self).__init__()
        self.embed_dim = embed_dim
        
        # ========== PointNet++ 编码器 (Set Abstraction) ==========
        # SA1: N -> 512 点
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=3,  # 只有xyz坐标
            mlp=[64, 64, 128], 
            group_all=False
        )
        # SA2: 512 -> 128 点
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3,  # 上一层特征 + xyz
            mlp=[128, 128, 256], 
            group_all=False
        )
        # SA3: 128 -> 32 点
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=32, 
            in_channel=256 + 3, 
            mlp=[256, 256, 512], 
            group_all=False
        )
        # SA4: 32 -> 8 点 (最深层)
        self.sa4 = PointNetSetAbstraction(
            npoint=8, radius=1.6, nsample=16, 
            in_channel=512 + 3, 
            mlp=[512, 512, embed_dim], 
            group_all=False
        )
        
        # ========== 特征投影层 ==========
        # 将各层特征投影到统一的 embed_dim
        self.proj_sa1 = nn.Conv1d(128, embed_dim, 1)
        self.proj_sa2 = nn.Conv1d(256, embed_dim, 1)
        self.proj_sa3 = nn.Conv1d(512, embed_dim, 1)
        # SA4 输出已经是 embed_dim
        
        # ========== 引导点块 (GPB) ==========
        # 在每个上采样阶段使用文本特征引导
        self.gpb4 = GuidedPointBlock(embed_dim, num_heads)
        self.gpb3 = GuidedPointBlock(embed_dim, num_heads)
        self.gpb2 = GuidedPointBlock(embed_dim, num_heads)
        self.gpb1 = GuidedPointBlock(embed_dim, num_heads)
        
        # ========== PointNet++ 解码器 (Feature Propagation) ==========
        # FP4: 8 -> 32 点
        self.fp4 = PointNetFeaturePropagation(
            in_channel=embed_dim + embed_dim,  # 上采样特征 + skip connection
            mlp=[embed_dim, embed_dim]
        )
        # FP3: 32 -> 128 点
        self.fp3 = PointNetFeaturePropagation(
            in_channel=embed_dim + embed_dim,
            mlp=[embed_dim, embed_dim]
        )
        # FP2: 128 -> 512 点
        self.fp2 = PointNetFeaturePropagation(
            in_channel=embed_dim + embed_dim,
            mlp=[embed_dim, embed_dim]
        )
        # FP1: 512 -> N 点 (原始点数)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=embed_dim + 3,  # 上采样特征 + 原始xyz
            mlp=[embed_dim, embed_dim]
        )
        
        # ========== Transformer Decoder ==========
        # 使用文本 token 作为 query，点云特征作为 memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 位置编码
        self.pos_1d = nn.Parameter(torch.randn(1, max_text_len, embed_dim) * 0.02)
        
        # ========== 输出头 ==========
        # 不需要额外的输出头，直接使用点积计算掩码
        
    def forward(self, xyz, text_feat, text_mask=None):
        """
        Args:
            xyz: 点云坐标 [B, 3, N] 或 [B, N, 3]
            text_feat: 文本特征 [B, L, C]，来自 LLM 的隐藏状态
            text_mask: 文本掩码 [B, L]，True 表示有效 token
        Returns:
            pred_mask: 每个点的分割掩码 [B, N]
        """
        # 确保点云格式为 [B, 3, N]
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)
        
        B, _, N = xyz.shape
        L = text_feat.shape[1]
        
        # ========== 编码阶段 ==========
        # 保存每层的坐标和特征用于 skip connection
        # p_i = [xyz_i, feat_i]
        
        # 原始点云
        l0_xyz = xyz  # [B, 3, N]
        l0_points = None
        
        # SA1: N -> 512
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 3, 512], [B, 128, 512]
        l1_points_proj = self.proj_sa1(l1_points)  # [B, embed_dim, 512]
        
        # SA2: 512 -> 128
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 128], [B, 256, 128]
        l2_points_proj = self.proj_sa2(l2_points)  # [B, embed_dim, 128]
        
        # SA3: 128 -> 32
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 32], [B, 512, 32]
        l3_points_proj = self.proj_sa3(l3_points)  # [B, embed_dim, 32]
        
        # SA4: 32 -> 8
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B, 3, 8], [B, embed_dim, 8]
        
        # ========== 解码阶段（带文本引导）==========
        # 在最深层使用 GPB 进行文本引导
        # l4_points: [B, embed_dim, 8] -> [B, 8, embed_dim]
        l4_points_t = l4_points.transpose(-2, -1)
        l4_points_t = self.gpb4(text_feat, l4_points_t, text_mask)
        l4_points = l4_points_t.transpose(-2, -1)  # [B, embed_dim, 8]
        
        # FP4: 8 -> 32
        up_feat = self.fp4(l3_xyz, l4_xyz, l3_points_proj, l4_points)  # [B, embed_dim, 32]
        up_feat_t = up_feat.transpose(-2, -1)
        up_feat_t = self.gpb3(text_feat, up_feat_t, text_mask)
        up_feat = up_feat_t.transpose(-2, -1)
        
        # FP3: 32 -> 128
        up_feat = self.fp3(l2_xyz, l3_xyz, l2_points_proj, up_feat)  # [B, embed_dim, 128]
        up_feat_t = up_feat.transpose(-2, -1)
        up_feat_t = self.gpb2(text_feat, up_feat_t, text_mask)
        up_feat = up_feat_t.transpose(-2, -1)
        
        # FP2: 128 -> 512
        up_feat = self.fp2(l1_xyz, l2_xyz, l1_points_proj, up_feat)  # [B, embed_dim, 512]
        up_feat_t = up_feat.transpose(-2, -1)
        up_feat_t = self.gpb1(text_feat, up_feat_t, text_mask)
        up_feat = up_feat_t.transpose(-2, -1)
        
        # FP1: 512 -> N (原始点数)
        up_feat = self.fp1(l0_xyz, l1_xyz, l0_xyz, up_feat)  # [B, embed_dim, N]
        
        # ========== Transformer Decoder ==========
        # 使用文本 token 作为 query，上采样后的点云特征作为 memory
        # text_feat: [B, L, C]
        # up_feat: [B, C, N] -> [B, N, C]
        memory = up_feat.transpose(-2, -1)  # [B, N, C]
        
        # 添加位置编码
        query_pos = self.pos_1d[:, :L, :]  # [1, L, C]
        tgt = text_feat + query_pos
        
        # Transformer Decoder
        if text_mask is not None:
            tgt_key_padding_mask = ~text_mask  # True 表示忽略
        else:
            tgt_key_padding_mask = None
        
        decoded_text = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [B, L, C]
        
        # 应用文本掩码
        if text_mask is not None:
            decoded_text = decoded_text * text_mask.unsqueeze(-1).float()
        
        # ========== 生成掩码 ==========
        # 使用点积计算每个点与文本的相关性
        # decoded_text: [B, L, C], up_feat: [B, C, N]
        # 输出: [B, L, N]
        point_text_sim = torch.einsum('blc,bcn->bln', decoded_text, up_feat)
        
        # 对文本维度求平均（考虑掩码）
        if text_mask is not None:
            # 只对有效 token 求平均
            mask_sum = text_mask.float().sum(1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            pred_mask = point_text_sim.sum(1) / (mask_sum.squeeze(-1) + 1e-8)  # [B, N]
        else:
            pred_mask = point_text_sim.mean(1)  # [B, N]
        
        # 应用 sigmoid 得到概率
        pred_mask = torch.sigmoid(pred_mask)
        
        return pred_mask

# future
class PointCloudEncoder(nn.Module):
    """3D点云编码器，基于PointNet++"""
    def __init__(self, out_dim=256):
        super(PointCloudEncoder, self).__init__()
        # PointNet++ Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # 投影层，将点云特征映射到与文本特征相同的维度
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
    
    def forward(self, xyz):
        """
        Input:
            xyz: 点云数据, [B, 3, N]
        Return:
            point_features: 点云全局特征, [B, out_dim]
        """
        B, _, N = xyz.shape
        
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # l3_points: [B, 1024, 1]
        x = l3_points.view(B, 1024)
        x = self.fc(x)
        
        return x

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    计算 Dice 损失，类似于掩码的广义 IoU
    
    Dice 系数衡量两个集合的重叠程度，Dice 损失 = 1 - Dice 系数
    
    Args:
        inputs: 任意形状的浮点张量，每个样本的预测值
        targets: 与 inputs 相同形状的浮点张量，存储二元分类标签
                (0 表示负类，1 表示正类)
        num_masks: 掩码数量，用于归一化
        scale: 缩放因子，用于数值稳定性
        eps: 防止除零的小常数
        
    Returns:
        Dice 损失值
    """
    inputs = inputs.sigmoid()  # 将预测值转换为概率
    inputs = inputs.flatten(1, 2)  # 展平空间维度
    targets = targets.flatten(1, 2)
    # 计算 Dice 系数：2 * |A ∩ B| / (|A| + |B|)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    计算 sigmoid 交叉熵损失（二元交叉熵）
    
    Args:
        inputs: 任意形状的浮点张量，每个样本的预测值（logits）
        targets: 与 inputs 相同形状的浮点张量，存储二元分类标签
                (0 表示负类，1 表示正类)
        num_masks: 掩码数量，用于归一化
        
    Returns:
        损失张量
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


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
        获取图像的视觉嵌入（使用 SAM 的图像编码器）
        
        Args:
            pixel_values: 输入图像张量，形状为 [batch_size, C, H, W]
            
        Returns:
            image_embeddings: 图像嵌入，形状为 [batch_size, embed_dim, H', W']
        """
        with torch.no_grad():  # SAM 图像编码器不需要梯度
            image_embeddings_list = []
            # 逐个处理图像（避免内存溢出）
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                # 使用 SAM 的图像编码器提取特征
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            # 拼接所有图像的嵌入
            image_embeddings = torch.cat(image_embeddings_list, 0)
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
        预处理 CLIP 图像
        
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
                [torch.zeros((token_mask.shape[0], padding_len)).bool().cuda(), token_mask],
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
        token_offset = torch.cat([torch.zeros(1).long().cuda(), token_offset], dim=0)
        
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
            image_embeddings: SAM 图像嵌入
            resize_list: 图像尺寸调整列表
            original_size_list: 原始图像尺寸列表
            
        Returns:
            pred_masks: 预测的 2D 掩码列表
        """
        pred_masks = []
        multimask_output = False
        
        for i in range(len(pred_embeddings_list)):
            if len(pred_embeddings_list[i]) > 0:
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
                pred_masks.append(pred_mask[:, 0])
        
        return pred_masks
    
    def _generate_3d_masks(self, pred_embeddings_list, point_clouds):
        """
        使用 PointNet++ 生成 3D 点云掩码
        
        Args:
            pred_embeddings_list: 按样本分组的预测嵌入列表
            point_clouds: 点云数据 [B, 3, N]
            
        Returns:
            pred_3d_masks: 预测的 3D 掩码列表
        """
        # 确保点云格式为 [B, 3, N]
        if point_clouds.shape[1] != 3:
            point_clouds = point_clouds.permute(0, 2, 1)
        
        pred_3d_masks = []
        batch_size = point_clouds.shape[0]
        
        for i in range(batch_size):
            if len(pred_embeddings_list[i]) > 0:
                # text_feat: [num_tokens, embed_dim] -> [1, num_tokens, embed_dim]
                text_feat = pred_embeddings_list[i].unsqueeze(0)
                text_mask = torch.ones(1, text_feat.shape[1], dtype=torch.bool, device=text_feat.device)
                pc_i = point_clouds[i:i+1]  # [1, 3, N]
                
                # 使用 3D 分割器
                pred_3d_mask = self.model.point_cloud_segmentor(pc_i, text_feat, text_mask)  # [1, N]
                pred_3d_masks.append(pred_3d_mask.squeeze(0))  # [N]
            else:
                # 如果没有特殊 token，返回全零掩码
                pred_3d_masks.append(torch.zeros(point_clouds.shape[2], device=point_clouds.device))
        
        return pred_3d_masks

    def model_forward(
        self,
        images: torch.FloatTensor = None,
        images_clip: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_masks: torch.LongTensor = None,
        offset: torch.LongTensor = None,
        masks_list: List[torch.FloatTensor] = None,
        original_size_list: List[torch.Tensor] = None,
        resize_list: List[tuple] = None,
        inference: bool = False,
        point_clouds: torch.FloatTensor = None,  # 3D点云输入 [B, 3, N] 或 [B, N, 3]
        point_masks_list: List[torch.FloatTensor] = None,  # 3D点云真实掩码列表
        **kwargs,
    ):
        """
        模型前向传播主函数（支持动态模态输入）
        
        Args:
            images: SAM 输入图像，形状 [batch_size, C, H, W]，可选
            images_clip: CLIP 输入图像，形状 [batch_size, C, H', W']，可选
            input_ids: 输入 token IDs
            labels: 文本标签（用于计算语言模型损失）
            attention_masks: 注意力掩码
            offset: 批次偏移量，用于处理不同长度的序列
            masks_list: 真实分割掩码列表（Ground Truth）
            original_size_list: 原始图像尺寸列表 [(H, W), ...]
            resize_list: 调整后的图像尺寸列表 [(H, W), ...]
            inference: 是否为推理模式（不计算损失）
            point_clouds: 3D点云输入
            point_masks_list: 3D点云真实掩码列表
            
        Returns:
            包含损失和预测结果的字典
        """
        # 判断输入模态
        has_image = images is not None and images_clip is not None
        has_point_cloud = point_clouds is not None
        
        # 至少需要一种模态
        if not has_image and not has_point_cloud:
            raise ValueError("至少需要提供图像或点云输入")
        
        # 获取 batch_size
        if has_image:
            batch_size = images.shape[0]
            if offset is not None:
                assert batch_size == len(offset) - 1
        else:
            batch_size = point_clouds.shape[0] if has_point_cloud else 1
        
        # 获取 SAM 图像嵌入（仅在有图像输入时）
        image_embeddings = None
        if has_image:
            image_embeddings = self.get_visual_embs(images)

        # 找到所有 [SEG] token 和 [AFF] token 的位置
        # 根据输入模态决定要查找的 token
        if has_image and has_point_cloud:
            # 两种模态都有，查找 [SEG] 和 [AFF]
            seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx) | (input_ids[:, 1:] == self.aff_token_idx)
        elif has_image:
            # 只有图像，只查找 [SEG]
            seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)
        else:
            # 只有点云，只查找 [AFF]
            seg_token_mask = (input_ids[:, 1:] == self.aff_token_idx)
        
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
                dim=1,
            )

        # 统一处理图像预处理和扩展
        if has_image:
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
            images=images_clip if has_image else None,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels if not inference else None,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states

        # 将语言模型的隐藏状态投影到 prompt 嵌入空间
        assert len(self.model.text_hidden_fcs) == 1
        last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states[-1])
        
        # 关键修复：动态调整 seg_token_mask 的长度以匹配 last_hidden_state
        # last_hidden_state.shape: [B, L', C]
        # seg_token_mask.shape: [B, L]
        # 需要将 seg_token_mask 扩展到 L' 的长度
        actual_seq_len = last_hidden_state.shape[1]
        current_mask_len = seg_token_mask.shape[1]
        
        if actual_seq_len > current_mask_len:
            # 需要在前面填充（图像特征被插入到序列开头）
            padding_len = actual_seq_len - current_mask_len
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], padding_len)).bool().cuda(), seg_token_mask],
                dim=1,
            )
        elif actual_seq_len < current_mask_len:
            # 理论上不应该发生，但为了安全起见进行截断
            seg_token_mask = seg_token_mask[:, :actual_seq_len]
        
        # 提取 [SEG]/[AFF] token 位置的嵌入作为分割提示
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        # 计算特殊 token 的累积偏移量
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
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

        # ========== 2D 分割：使用 SAM 生成分割掩码（仅在有图像输入时）==========
        pred_masks = []
        if has_image:
            pred_masks = self._generate_2d_masks(pred_embeddings, image_embeddings, resize_list, original_size_list)

        # ========== 3D 分割：使用 PointNet++ 处理点云（仅在有点云输入时）==========
        pred_3d_masks = []
        if has_point_cloud:
            pred_3d_masks = self._generate_3d_masks(pred_embeddings, point_clouds)

        # 推理模式：只返回预测结果
        if inference:
            return {
                "pred_masks": pred_masks if has_image else None,
                "gt_masks": masks_list if has_image else None,
                "pred_3d_masks": pred_3d_masks if has_point_cloud else None,
                "gt_3d_masks": point_masks_list if has_point_cloud else None,
            }

        # ========== 训练模式：计算损失 ==========
        # 计算语言模型损失（交叉熵）
        ce_loss = output.loss * self.ce_loss_weight
        
        # 计算 2D 分割掩码损失（仅在有图像输入时）
        # 使用 ce_loss * 0 保持计算图连接
        mask_bce_loss = ce_loss * 0.0
        mask_dice_loss = ce_loss * 0.0
        
        if has_image and len(pred_masks) > 0:
            mask_bce_loss_sum = 0
            mask_dice_loss_sum = 0
            num_masks = 0
            
            for batch_idx in range(len(pred_masks)):
                gt_mask = masks_list[batch_idx]
                pred_mask = pred_masks[batch_idx]
                assert gt_mask.shape[0] == pred_mask.shape[0], \
                    f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
                
                mask_bce_loss_sum += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
                mask_dice_loss_sum += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
                num_masks += gt_mask.shape[0]

            if num_masks > 0:
                mask_bce_loss = self.bce_loss_weight * mask_bce_loss_sum / num_masks
                mask_dice_loss = self.dice_loss_weight * mask_dice_loss_sum / num_masks
        
        mask_loss = mask_bce_loss + mask_dice_loss

        # 计算 3D 点云掩码损失（仅在有点云输入时）
        # 使用 ce_loss * 0 保持计算图连接
        mask_3d_bce_loss = ce_loss * 0.0
        mask_3d_dice_loss = ce_loss * 0.0
        
        if has_point_cloud and len(pred_3d_masks) > 0 and point_masks_list is not None:
            mask_3d_bce_loss_sum = 0
            mask_3d_dice_loss_sum = 0
            num_3d_masks = 0
            
            for batch_idx in range(len(pred_3d_masks)):
                if point_masks_list[batch_idx] is not None:
                    gt_3d_mask = point_masks_list[batch_idx]  # [N]
                    pred_3d_mask = pred_3d_masks[batch_idx]  # [N]
                    
                    if gt_3d_mask.shape == pred_3d_mask.shape:
                        # BCE 损失（pred_3d_mask 已经经过 sigmoid）
                        bce = F.binary_cross_entropy(
                            pred_3d_mask.clamp(1e-6, 1-1e-6), 
                            gt_3d_mask.float(), 
                            reduction='mean'
                        )
                        mask_3d_bce_loss_sum += bce
                        
                        # Dice 损失
                        intersection = (pred_3d_mask * gt_3d_mask.float()).sum()
                        union = pred_3d_mask.sum() + gt_3d_mask.float().sum()
                        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
                        mask_3d_dice_loss_sum += dice
                        num_3d_masks += 1
            
            if num_3d_masks > 0:
                mask_3d_bce_loss = self.bce_loss_weight * mask_3d_bce_loss_sum / num_3d_masks
                mask_3d_dice_loss = self.dice_loss_weight * mask_3d_dice_loss_sum / num_3d_masks
        
        mask_3d_loss = mask_3d_bce_loss + mask_3d_dice_loss

        # ========== 添加虚拟损失以保持所有参数连接到计算图 ==========
        # 这确保即使某些模块在当前批次中未使用，它们的参数仍然连接到 Loss
        dummy_loss = ce_loss * 0.0  # 虚拟损失的值始终为0
        
        # 1. 确保 point_cloud_segmentor 的所有参数连接到计算图
        for param in self.model.point_cloud_segmentor.parameters():
            if param.requires_grad:
                dummy_loss = dummy_loss + (param ** 2).sum() * 0.0
        
        # 2. 确保 SAM mask_decoder 的所有参数连接到计算图
        if hasattr(self.model, 'visual_model') and hasattr(self.model.visual_model, 'mask_decoder'):
            for param in self.model.visual_model.mask_decoder.parameters():
                if param.requires_grad:
                    dummy_loss = dummy_loss + (param ** 2).sum() * 0.0

        # 总损失（包含虚拟损失以保持计算图完整）
        loss = ce_loss + mask_loss + mask_3d_loss + dummy_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "mask_3d_bce_loss": mask_3d_bce_loss,
            "mask_3d_dice_loss": mask_3d_dice_loss,
            "mask_3d_loss": mask_3d_loss,
        }

    def evaluate(
        self,
        images_clip=None,
        images=None,
        input_ids=None,
        resize_list=None,
        original_size_list=None,
        max_new_tokens=32,
        tokenizer=None,
        point_clouds=None,
    ):
        """
        评估函数：生成文本并预测分割掩码（支持动态模态输入）
        
        Args:
            images_clip: CLIP 输入图像，可选
            images: SAM 输入图像，可选
            input_ids: 输入 token IDs
            resize_list: 图像尺寸调整列表，可选
            original_size_list: 原始图像尺寸列表，可选
            max_new_tokens: 最大生成 token 数
            tokenizer: 分词器（未使用）
            point_clouds: 3D点云数据，可选
            
        Returns:
            output_ids: 生成的 token IDs
            pred_2d_masks: 预测的 2D 分割掩码列表（如果有图像）或 None
            pred_3d_masks: 预测的 3D 点云掩码列表（如果有点云）或 None
        """
        # 判断输入模态
        has_image = images is not None and images_clip is not None
        has_point_cloud = point_clouds is not None
        
        # 至少需要一种模态
        if not has_image and not has_point_cloud:
            raise ValueError("至少需要提供图像或点云输入")
        
        with torch.no_grad():
            # ========== 1. 预处理图像（与训练/推理一致）==========
            if has_image:
                images_clip = self.preprocess_clip_images(images_clip)
            
            # ========== 2. 生成文本（包含 [SEG]/[AFF] token）==========
            outputs = self.generate(
                images=images_clip if has_image else None,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            
            # ========== 3. 投影隐藏状态到 prompt 空间（与训练/推理一致）==========
            assert len(self.model.text_hidden_fcs) == 1
            last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states)
            
            # ========== 4. 2D 分割：处理 [SEG] token（使用公共函数）==========
            pred_2d_masks = []
            if has_image:
                # 提取 [SEG] token 的嵌入
                seg_embeddings_list = self._extract_token_embeddings(
                    output_ids, last_hidden_state, self.seg_token_idx, has_image
                )
                
                # 获取 SAM 图像嵌入
                image_embeddings = self.get_visual_embs(images)
                
                # 生成 2D 掩码
                pred_2d_masks = self._generate_2d_masks(
                    seg_embeddings_list, image_embeddings, resize_list, original_size_list
                )
            
            # ========== 5. 3D 分割：处理 [AFF] token（使用公共函数）==========
            pred_3d_masks = []
            if has_point_cloud:
                # 提取 [AFF] token 的嵌入
                aff_embeddings_list = self._extract_token_embeddings(
                    output_ids, last_hidden_state, self.aff_token_idx, has_image
                )
                
                # 生成 3D 掩码
                pred_3d_masks = self._generate_3d_masks(aff_embeddings_list, point_clouds)
        
        return output_ids, pred_2d_masks if has_image else None, pred_3d_masks if has_point_cloud else None

