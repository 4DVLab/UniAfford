import json
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn

from .models.point_prompt_training.point_transformer_v3m2_sonata import PointTransformerV3
from utils.checkpoint_utils import extract_state_dict


POINT_XYZ_CHANNELS = 3
POINT_BACKBONE_SHARED_CONFIG_KEYS = {
    "in_channels",
    "order",
    "stride",
    "enc_depths",
    "enc_channels",
    "enc_num_head",
    "enc_patch_size",
    "mlp_ratio",
    "qkv_bias",
    "qk_scale",
    "attn_drop",
    "proj_drop",
    "drop_path",
    "layer_scale",
    "pre_norm",
    "shuffle_orders",
    "enable_rpe",
    "enable_flash",
    "upcast_attention",
    "upcast_softmax",
    "traceable",
    "mask_token",
}
POINT_BACKBONE_DECODER_CONFIG_KEYS = {
    "dec_depths",
    "dec_channels",
    "dec_num_head",
    "dec_patch_size",
}
POINT_BACKBONE_STATE_PREFIXES = (
    "",
    "module.",
    "point_backbone.",
    "module.point_backbone.",
    "point_encoder.point_backbone.",
    "module.point_encoder.point_backbone.",
    "mllm.point_encoder.point_backbone.",
    "module.mllm.point_encoder.point_backbone.",
)


class PointCloudEncoder(nn.Module):
    """
    将点云编码为两种不同语义的特征：
    1) `mllm_point_tokens`: 编码器末端的较短 token，供 MLLM 注入。
    2) `per_point_features`: 解码器末端的逐点特征，供 3D decoder 逐点做相似度。

    Notes:
    - 两者共享同一次 PTV3 backbone 前向，但语义职责不同，不能混用。
    """

    def __init__(
        self,
        out_hidden_size: int,
        compute_dtype: torch.dtype,
        backbone_config: Optional[Dict],
    ):
        super().__init__()
        if backbone_config is None:
            kwargs = {}
        elif hasattr(backbone_config, "to_dict"):
            kwargs = dict(backbone_config.to_dict())
        else:
            kwargs = dict(backbone_config)

        kwargs.pop("enc_mode", None)
        # 这里启用解码器，并通过 PTV3 的 return_dual=True 同时拿到：
        # 1) 编码器末端 enc_point（较短 token，给 MLLM）
        # 2) 解码器末端 dec_point（逐点特征，给 3D decoder）
        kwargs["enc_mode"] = False
        self.point_backbone = PointTransformerV3(**kwargs)
        enc_dim = int(kwargs.get("enc_channels", (32, 64, 128, 256, 512))[-1])
        self.in_channels = int(kwargs.get("in_channels", POINT_XYZ_CHANNELS))
        self.proj = nn.Linear(enc_dim, out_hidden_size)
        self.compute_dtype = compute_dtype
        self.pretrained_info = None
        self.to(dtype=compute_dtype)

    @staticmethod
    def _to_plain_backbone_config(backbone_config: Optional[Dict]) -> Dict[str, Any]:
        if backbone_config is None:
            return {}
        if hasattr(backbone_config, "to_dict"):
            return dict(backbone_config.to_dict())
        return dict(backbone_config)

    @staticmethod
    def _extract_pretrained_backbone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        nested_candidates = [
            payload.get("point_encoder_backbone"),
            payload.get("point_prefix_backbone_kwargs"),
            payload.get("backbone"),
            ((payload.get("mllm") or {}).get("point_encoder_backbone") if isinstance(payload.get("mllm"), dict) else None),
            (
                ((payload.get("model_config") or {}).get("mllm") or {}).get("point_encoder_backbone")
                if isinstance(payload.get("model_config"), dict)
                else None
            ),
        ]
        for candidate in nested_candidates:
            if isinstance(candidate, dict):
                filtered = {
                    k: v
                    for k, v in candidate.items()
                    if k in POINT_BACKBONE_SHARED_CONFIG_KEYS or k in POINT_BACKBONE_DECODER_CONFIG_KEYS or k == "enc_mode"
                }
                if filtered:
                    return filtered

        filtered_root = {
            k: v
            for k, v in payload.items()
            if k in POINT_BACKBONE_SHARED_CONFIG_KEYS or k in POINT_BACKBONE_DECODER_CONFIG_KEYS or k == "enc_mode"
        }
        if filtered_root:
            return filtered_root
        raise ValueError("预训练配置中未找到 point backbone 相关字段。")

    @staticmethod
    def _resolve_pretrained_config_path(
        checkpoint_path: str,
        pretrained_config_path: Optional[str] = None,
    ) -> Optional[str]:
        if pretrained_config_path:
            return pretrained_config_path
        guessed_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.exists(guessed_path):
            return guessed_path
        return None

    @classmethod
    def _load_pretrained_bundle(
        cls,
        checkpoint_path: str,
        pretrained_config_path: Optional[str] = None,
        map_location: str | torch.device = "cpu",
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], str]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到 point backbone 预训练权重: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = extract_state_dict(checkpoint)

        resolved_config_path = cls._resolve_pretrained_config_path(
            checkpoint_path,
            pretrained_config_path,
        )
        if resolved_config_path is not None:
            with open(resolved_config_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return cls._extract_pretrained_backbone_payload(payload), state_dict, resolved_config_path

        if isinstance(checkpoint, dict):
            for key in ("config", "model_config"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, dict):
                    try:
                        return cls._extract_pretrained_backbone_payload(candidate), state_dict, f"{checkpoint_path}:{key}"
                    except ValueError:
                        pass

        raise ValueError(
            "未找到 point backbone 的预训练配置。"
            " 请显式传入 `--point_backbone_pretrained_config`，"
            "或将 `config.json` 放到权重同目录，"
            "或确保 ckpt 内部包含 `config/model_config`。"
        )

    @classmethod
    def _build_backbone_config_for_pretrained(
        cls,
        backbone_config: Optional[Dict],
        checkpoint_path: str,
        pretrained_config_path: Optional[str] = None,
        map_location: str | torch.device = "cpu",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pretrained_cfg, _, resolved_config_source = cls._load_pretrained_bundle(
            checkpoint_path,
            pretrained_config_path,
            map_location=map_location,
        )
        merged_cfg = cls._to_plain_backbone_config(backbone_config)
        pretrained_in_channels = int(pretrained_cfg.get("in_channels", merged_cfg.get("in_channels", POINT_XYZ_CHANNELS)))
        if pretrained_in_channels < POINT_XYZ_CHANNELS:
            raise ValueError(
                "point backbone 预训练配置的 in_channels 小于 3，无法承载当前 xyz 输入。"
                f" got in_channels={pretrained_in_channels}"
            )

        copied_keys: List[str] = []
        for key in POINT_BACKBONE_SHARED_CONFIG_KEYS:
            if key in pretrained_cfg:
                merged_cfg[key] = pretrained_cfg[key]
                copied_keys.append(key)

        if not bool(pretrained_cfg.get("enc_mode", False)):
            for key in POINT_BACKBONE_DECODER_CONFIG_KEYS:
                if key in pretrained_cfg:
                    merged_cfg[key] = pretrained_cfg[key]
                    copied_keys.append(key)

        # SONATA 常见输入为 [coord, color, normal] 共 9 维；当前项目仅提供 xyz，
        # 因此保留预训练通道数，并在 forward 时仅写前 3 个通道，剩余通道固定为 0。
        merged_cfg["in_channels"] = pretrained_in_channels
        merged_cfg["enc_mode"] = False
        return merged_cfg, {
            "config_path": resolved_config_source,
            "copied_keys": copied_keys,
            "pretrained_enc_mode": bool(pretrained_cfg.get("enc_mode", False)),
            "pretrained_in_channels": pretrained_in_channels,
        }

    @staticmethod
    def _strip_state_prefix(
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
    ) -> OrderedDict[str, torch.Tensor]:
        if prefix == "":
            return OrderedDict(state_dict.items())
        return OrderedDict(
            (k[len(prefix) :], v)
            for k, v in state_dict.items()
            if k.startswith(prefix)
        )

    def _match_pretrained_state_dict(
        self,
        raw_state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[OrderedDict[str, torch.Tensor], str]:
        module_state = self.point_backbone.state_dict()
        best_prefix = ""
        best_state = OrderedDict()
        best_match_count = -1

        for prefix in POINT_BACKBONE_STATE_PREFIXES:
            candidate = self._strip_state_prefix(raw_state_dict, prefix)
            matched = OrderedDict(
                (key, value)
                for key, value in candidate.items()
                if key in module_state and tuple(module_state[key].shape) == tuple(value.shape)
            )
            if len(matched) > best_match_count:
                best_match_count = len(matched)
                best_prefix = prefix
                best_state = matched

        if best_match_count <= 0:
            raise ValueError("未在预训练 checkpoint 中找到可匹配当前 point backbone 的参数。")
        return best_state, (best_prefix or "<raw>")

    def load_pretrained(
        self,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到 point backbone 预训练权重: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        raw_state_dict = extract_state_dict(checkpoint)
        loadable_state, matched_prefix = self._match_pretrained_state_dict(raw_state_dict)
        missing, unexpected = self.point_backbone.load_state_dict(loadable_state, strict=False)
        self.pretrained_info = {
            **(self.pretrained_info or {}),
            "weight_path": checkpoint_path,
            "matched_prefix": matched_prefix,
            "loaded_tensors": len(loadable_state),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        return self.pretrained_info

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        out_hidden_size: int,
        compute_dtype: torch.dtype,
        backbone_config: Optional[Dict] = None,
        pretrained_config_path: Optional[str] = None,
        map_location: str | torch.device = "cpu",
    ) -> "PointCloudEncoder":
        merged_cfg, info = cls._build_backbone_config_for_pretrained(
            backbone_config=backbone_config,
            checkpoint_path=checkpoint_path,
            pretrained_config_path=pretrained_config_path,
            map_location=map_location,
        )
        encoder = cls(
            out_hidden_size=out_hidden_size,
            compute_dtype=compute_dtype,
            backbone_config=merged_cfg,
        )
        encoder.pretrained_info = info
        encoder.load_pretrained(checkpoint_path, map_location=map_location)
        return encoder

    @staticmethod
    def _build_pointcept_batch(point_clouds: torch.Tensor, in_channels: int) -> Dict[str, torch.Tensor]:
        """
        将 [B, N, 3] 点云改写为 Pointcept 所需字典。

        字段语义：
        - coord: [sum(N), 3]，所有样本坐标拼接后的展平坐标。
        - feat:  [sum(N), C]，兼容 SONATA 的 [coord, color, normal] 风格输入。
                 当前项目只有 xyz，因此写入前 3 个通道，其余通道固定为 0。
        - batch: [sum(N)]，每个点属于哪个样本的 batch 索引。
        - offset:[B]，每个样本在展平序列中的结束下标（累积点数）。
        - grid_size: 体素网格大小，供 Pointcept 稀疏化/分块使用。
        """
        bsz, num_points, _ = point_clouds.shape
        coord = point_clouds.reshape(-1, 3).contiguous()
        if in_channels < POINT_XYZ_CHANNELS:
            raise ValueError(f"point backbone in_channels 必须 >= 3，当前为 {in_channels}")
        feat = coord.new_zeros((coord.shape[0], in_channels))
        feat[:, :POINT_XYZ_CHANNELS] = coord
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
        7) 按 batch 内最大长度补齐，输出给 MLLM 的短 token 及其 mask。

        Returns:
            mllm_point_tokens: [B, K_max, H]，给 MLLM 注入的短 token。
            mllm_point_token_mask: [B, K_max] (bool)，短 token 的有效位。
        """
        shared = self.encode_shared(point_clouds=point_clouds, pc_valid_lengths=pc_valid_lengths)
        if shared is None:
            return None, None
        return shared["mllm_point_tokens"], shared["mllm_point_token_mask"]

    def encode_shared(
        self,
        point_clouds: Optional[torch.Tensor],
        pc_valid_lengths: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        单次 backbone 前向，输出两种尺度的特征：
        - mllm_point_tokens/mllm_point_token_mask: enc_point 产生的短 token，注入 MLLM
        - per_point_features/per_point_mask: 与原始点数对齐的逐点特征，供 3D decoder 做相似度场
        """
        if point_clouds is None:
            return None

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

        # Step 3. Pointcept 编码：单次前向同时返回 enc_point 与 dec_point。
        data_dict = self._build_pointcept_batch(point_clouds, in_channels=self.in_channels)
        dual_out = self.point_backbone(data_dict, return_dual=True)
        enc_point = dual_out["enc_point"]
        dec_point = dual_out["dec_point"]

        # Step 4. enc_point -> 投影成 MLLM token；dec_point -> 保持逐点特征给 decoder。
        enc_feat = enc_point.feat.to(self.compute_dtype)     # [sum(K_i), C_enc]
        dec_feat = dec_point.feat.to(self.compute_dtype)     # [sum(N_i), C_dec]
        proj_feat = self.proj(enc_feat)                      # [sum(K_i), H_mllm]

        # Step 5. 分别按 batch 拆回：
        # - enc/proj 列表：较短 token 序列
        # - dec 列表：逐点特征序列（应与输入点数对齐）
        enc_list = self._split_by_batch(proj_feat, enc_point.batch, bsz)
        point_list = self._split_by_batch(dec_feat, dec_point.batch, bsz)

        # Step 6. 无效点云样本（pc_valid_lengths==0）显式置空，避免引入噪声前缀。
        if pc_valid_lengths is not None:
            valid_flags = (pc_valid_lengths > 0).tolist()
            point_list = [
                tok if valid_flags[i] else tok.new_zeros((0, tok.shape[-1]))
                for i, tok in enumerate(point_list)
            ]
            enc_list = [
                tok if valid_flags[i] else tok.new_zeros((0, tok.shape[-1]))
                for i, tok in enumerate(enc_list)
            ]

        valid_lengths = []
        for i in range(bsz):
            if pc_valid_lengths is not None:
                valid_lengths.append(int(pc_valid_lengths[i].item()))
            else:
                valid_lengths.append(int(point_list[i].shape[0]))

        # Step 6. enc_point 本身已经是较短的高层语义 token，直接作为 MLLM 注入序列。
        token_list = enc_list

        max_token_len = max((tok.shape[0] for tok in token_list), default=0)
        num_points = int(point_clouds.shape[1])
        if num_points == 0:
            return {
                "mllm_point_tokens": None,
                "mllm_point_token_mask": None,
                "per_point_features": None,
                "per_point_mask": None,
            }

        # Step 7. 分别补齐短 token 与逐点特征。
        # 其中 per_point_features 明确按原始点数 N 对齐，供 decoder 直接逐点做相似度。
        token_hidden = token_list[0].shape[-1] if max_token_len > 0 else enc_list[0].shape[-1]
        point_hidden = point_list[0].shape[-1]
        mllm_point_tokens = enc_list[0].new_zeros((bsz, max_token_len, token_hidden))
        mllm_point_token_mask = torch.zeros((bsz, max_token_len), dtype=torch.bool, device=enc_list[0].device)
        per_point_features = point_list[0].new_zeros((bsz, num_points, point_hidden))
        per_point_mask = torch.zeros((bsz, num_points), dtype=torch.bool, device=point_list[0].device)

        for i in range(bsz):
            tok = token_list[i]
            tok_len = tok.shape[0]
            if tok_len > 0:
                mllm_point_tokens[i, :tok_len] = tok
                mllm_point_token_mask[i, :tok_len] = True

            pts = point_list[i]
            pt_len = max(0, min(valid_lengths[i], int(pts.shape[0])))
            if pt_len > 0:
                if int(pts.shape[0]) < num_points:
                    raise ValueError(
                        f"decoder point features should align with original point count, "
                        f"got sample_points={int(pts.shape[0])}, expected_at_least={num_points}"
                    )
                per_point_features[i, :pt_len] = pts[:pt_len]
                per_point_mask[i, :pt_len] = True

        return {
            "mllm_point_tokens": mllm_point_tokens,
            "mllm_point_token_mask": mllm_point_token_mask,
            "per_point_features": per_point_features,
            "per_point_mask": per_point_mask,
        }
