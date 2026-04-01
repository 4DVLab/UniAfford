from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from .models.point_prompt_training.point_transformer_v3m2_sonata import PointTransformerV3


class PointCloudPrefixEncoder(nn.Module):
    """
    将点云编码为可变长度 prefix embedding，供 MLLM 前缀拼接使用。

    Notes:
    - 不设置固定前缀 token 上限；长度由点云骨干输出决定。
    - 同一 batch 内按最大长度补齐，并返回 mask 区分有效/补齐位置。
    """

    def __init__(
        self,
        out_hidden_size: int,
        compute_dtype: torch.dtype,
        backbone_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = dict(
                in_channels=3,
                order=("z", "z-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(128, 128, 128, 128, 128),
                dec_depths=(2, 2, 2, 2),
                dec_channels=(64, 64, 128, 256),
                dec_num_head=(4, 4, 8, 16),
                dec_patch_size=(128, 128, 128, 128),
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.1,
                layer_scale=None,
                pre_norm=True,
                shuffle_orders=True,
                enable_rpe=False,
                enable_flash=False,
                upcast_attention=False,
                upcast_softmax=False,
                traceable=False,
                mask_token=False,
                enc_mode=True,
                freeze_encoder=False,
            )

        kwargs = dict(backbone_kwargs)
        kwargs["enc_mode"] = True
        self.point_backbone = PointTransformerV3(**kwargs)
        in_dim = int(kwargs.get("enc_channels", (32, 64, 128, 256, 512))[-1])
        self.proj = nn.Linear(in_dim, out_hidden_size)
        self.compute_dtype = compute_dtype
        self.to(dtype=compute_dtype)

    @staticmethod
    def _build_pointcept_batch(point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将 [B, N, 3] 点云改写为 Pointcept 所需字典。

        字段语义：
        - coord: [sum(N), 3]，所有样本坐标拼接后的展平坐标。
        - feat:  [sum(N), 3]，这里直接复用坐标作为输入特征。
        - batch: [sum(N)]，每个点属于哪个样本的 batch 索引。
        - offset:[B]，每个样本在展平序列中的结束下标（累积点数）。
        - grid_size: 体素网格大小，供 Pointcept 稀疏化/分块使用。
        """
        bsz, num_points, _ = point_clouds.shape
        coord = point_clouds.reshape(-1, 3).contiguous()
        feat = coord
        batch = (
            torch.arange(bsz, device=point_clouds.device)
            .unsqueeze(1)
            .repeat(1, num_points)
            .reshape(-1)
            .long()
        )
        offset = (
            torch.arange(1, bsz + 1, device=point_clouds.device, dtype=torch.long)
            * num_points
        )
        return dict(
            coord=coord,
            feat=feat,
            batch=batch,
            offset=offset,
            grid_size=0.02,
        )

    @staticmethod
    def _split_by_batch(feat: torch.Tensor, batch: torch.Tensor, bsz: int) -> List[torch.Tensor]:
        """
        将展平后的点特征按 batch id 还原为 list：
        - 第 i 个元素形状为 [Ki, C]，Ki 为该样本输出 token 数（可变长）。
        """
        tokens: List[torch.Tensor] = []
        for i in range(bsz):
            idx = batch == i
            tokens.append(feat[idx])
        return tokens

    def forward(
        self,
        point_clouds: Optional[torch.Tensor],
        pc_valid_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        详细步骤：
        1) 规范输入形状到 [B, N, 3]，并转为计算精度（bf16/fp16/fp32）。
        2) 构造 Pointcept 批输入字典（coord/feat/batch/offset）。
        3) 经 PointTransformerV3(enc_mode=True) 提取点级语义 token。
        4) 线性投影到 Qwen hidden 维度，得到可与 text embedding 拼接的特征。
        5) 按 batch id 拆回每样本 token 列表（变长）。
        6) 结合 pc_valid_lengths 过滤无效样本（长度置 0）。
        7) 按 batch 内最大长度补齐，输出 prefix_embeds 与 prefix_mask。

        Returns:
            prefix_embeds: [B, K_max, H]，补齐后的前缀 embedding。
            prefix_mask:   [B, K_max] (bool)，True 表示有效前缀 token。
        """
        if point_clouds is None:
            return None, None

        # Step 1. 输入合法性检查 + 形状规范化。
        if point_clouds.dim() != 3:
            raise ValueError(f"point_clouds should be [B, N, 3] or [B, 3, N], got {tuple(point_clouds.shape)}")

        if point_clouds.shape[-1] != 3 and point_clouds.shape[1] == 3:
            point_clouds = point_clouds.permute(0, 2, 1).contiguous()
        elif point_clouds.shape[-1] != 3:
            raise ValueError(f"point_clouds last dim must be 3, got {tuple(point_clouds.shape)}")

        # Step 2. 统一计算精度，避免后续 backbone/proj 类型不一致。
        point_clouds = point_clouds.to(self.compute_dtype)
        bsz = point_clouds.shape[0]

        # Step 3. Pointcept 编码：从点坐标获得点级语义特征。
        data_dict = self._build_pointcept_batch(point_clouds)
        point = self.point_backbone(data_dict)

        # Step 4. 通道投影到 MLLM hidden_size，便于与文本 embedding 在同空间拼接。
        point_feat = self.proj(point.feat.to(self.compute_dtype))

        # Step 5. 展平特征按样本拆分（变长 token 序列）。
        token_list = self._split_by_batch(point_feat, point.batch, bsz)

        # Step 6. 无效点云样本（pc_valid_lengths==0）显式置空，避免引入噪声前缀。
        if pc_valid_lengths is not None:
            valid_flags = (pc_valid_lengths > 0).tolist()
            token_list = [
                tok if valid_flags[i] else tok.new_zeros((0, tok.shape[-1]))
                for i, tok in enumerate(token_list)
            ]

        # Step 7. 计算 batch 内最大前缀长度；若全为空则返回 None。
        max_len = max((tok.shape[0] for tok in token_list), default=0)
        if max_len == 0:
            return None, None

        # Step 8. 补齐为 [B, K_max, H] 并构建 bool mask。
        hidden = token_list[0].shape[-1]
        prefix_embeds = token_list[0].new_zeros((bsz, max_len, hidden))
        prefix_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=prefix_embeds.device)
        for i, tok in enumerate(token_list):
            cur_len = tok.shape[0]
            if cur_len == 0:
                continue
            prefix_embeds[i, :cur_len] = tok
            prefix_mask[i, :cur_len] = True
        return prefix_embeds, prefix_mask
