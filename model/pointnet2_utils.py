"""copied from pointnet++"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc /= m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, dtype=xyz.dtype, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, dtype=xyz.dtype, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)"""
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


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
        
        # 位置编码（初始化时使用float32，会在forward时自动转换为输入的dtype）
        self.pos_1d = nn.Parameter(torch.randn(1, max_text_len, embed_dim, dtype=torch.float32) * 0.02)
        
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
        
        # 添加位置编码（确保类型与 text_feat 一致）
        query_pos = self.pos_1d[:, :L, :].to(dtype=text_feat.dtype, device=text_feat.device)  # [1, L, C]
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
            decoded_text = decoded_text * text_mask.unsqueeze(-1).to(decoded_text.dtype)
        
        # ========== 生成掩码 ==========
        # 使用点积计算每个点与文本的相关性
        # decoded_text: [B, L, C], up_feat: [B, C, N]
        # 输出: [B, L, N]
        up_feat = up_feat.to(decoded_text.dtype)
        point_text_sim = torch.einsum('blc,bcn->bln', decoded_text, up_feat)
        
        # 对文本维度求平均（考虑掩码）
        if text_mask is not None:
            # 只对有效 token 求平均
            mask_sum = text_mask.to(decoded_text.dtype).sum(1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
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
