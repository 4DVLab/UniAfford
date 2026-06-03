"""
联合可供性模型配置，集中管理模型所需的全部配置项。
"""
from copy import deepcopy
import json
import torch
from typing import Optional, Dict
from utils.common import resolve_dtype, FUNCTIONAL_TOKENS

class Configs:
    """基础配置类，提供快速属性更新能力。"""

    defaults: Dict = {}

    def _merge_defaults(self, config_dict: Optional[Dict] = None, overrides: Optional[Dict] = None):
        """先合并类级 defaults，再叠加显式配置。"""
        merged = deepcopy(self.defaults)
        if config_dict is not None:
            merged.update(config_dict)
        if overrides:
            merged.update(overrides)
        return merged

    def __init__(self, config_dict: Optional[Dict] = None, **overrides):
        merged = self._merge_defaults(config_dict, overrides)
        self.__dict__.update(merged)

    def update(self, config_dict: Optional[Dict] = None, **overrides):
        if config_dict is not None:
            self.__dict__.update(config_dict)
        self.__dict__.update(overrides)
        return self

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def _serialize_value(value):
        """递归转成 JSON 友好的结构。"""
        if isinstance(value, Configs):
            return value.to_json_dict()
        if isinstance(value, torch.dtype):
            if value == torch.bfloat16:
                return "bf16"
            if value == torch.float16:
                return "fp16"
            if value == torch.float32:
                return "fp32"
            return str(value).replace("torch.", "")
        if isinstance(value, dict):
            return {str(k): Configs._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Configs._serialize_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # 兜底：保持可序列化，不让导出失败（例如 tokenizer 句柄）
        return str(value)

    def to_json_dict(self):
        """用于配置落盘的字典（区别于训练时 to_dict 语义）。"""
        return {k: self._serialize_value(v) for k, v in self.__dict__.items()}

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)


class PointEncoderBackboneConfigs(Configs):
    """点云编码器 backbone（SONATA / PTV3）相关配置。"""

    defaults = {
        "in_channels": 3,
        "order": ("z", "z-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_num_head": (2, 4, 8, 16, 32),
        "enc_patch_size": (128, 128, 128, 128, 128),
        "dec_depths": (2, 2, 2, 2),
        "dec_channels": (64, 64, 128, 256),
        "dec_num_head": (4, 4, 8, 16),
        "dec_patch_size": (128, 128, 128, 128),
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.1,
        "layer_scale": None,
        "pre_norm": True,
        "shuffle_orders": True,
        "enable_rpe": False,
        "enable_flash": False,
        "upcast_attention": False,
        "upcast_softmax": False,
        "traceable": False,
        "mask_token": False,
        "enc_mode": False,
        "freeze_encoder": False,
    }


class MLLMConfigs(Configs):
    """多模态大模型相关配置。"""

    defaults = {
        "qwen_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen_attn_implementation": "flash_attention_2",
        "compute_dtype": "bf16",
        "model_max_length": 512,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_heads": 8,
        "tokenizer": None,
        "train_mask_decoder": True,
        "out_dim": 256,
        "functional_tokens": FUNCTIONAL_TOKENS,
        "enable_point_encoder": True,
        "point_encoder_backbone": None,
        "point_encoder_pretrained": None,
        "point_encoder_pretrained_config": None,
        "restore_from_checkpoint": False,
        "serialized_processor_files": None,
        "serialized_model_config_files": None,
        "serialized_model_class_name": None,
        # 统一的任务/placeholder 注册表；新增任务时在这里追加 "task": "<token>"。
        "task_placeholders": {
            "text": "<text>",
            "img": "<img_aff>",
            "pc": "<pc_aff>",
            # "latent": "<latent>",
        },
        # 推理阶段 router 选中非 text task token 时，是否用 hidden state 作为下一步输入。
        # False 时所有 token（包括 <img_aff>/<pc_aff>）都使用 MLLM 原生 embedding lookup。
        "use_router_latent_feedback": True,
    }

    def __init__(self, config_dict: Optional[Dict] = None, **overrides):
        raw = {}
        if config_dict is not None:
            raw.update(config_dict)
        raw.update(overrides)

        # 兼容旧命名，避免已有配置/权重元数据失效。
        if "enable_pc_prefix" in raw and "enable_point_encoder" not in raw:
            raw["enable_point_encoder"] = raw.pop("enable_pc_prefix")
        if "point_prefix_backbone_kwargs" in raw and "point_encoder_backbone" not in raw:
            raw["point_encoder_backbone"] = raw.pop("point_prefix_backbone_kwargs")

        raw = self._merge_defaults(raw)

        # 在 fp32 的时候禁用 flash_attention_2，因为 flash_attention_2 只支持 bf16。
        resolved_dtype = resolve_dtype(raw["compute_dtype"])
        raw["compute_dtype"] = resolved_dtype
        if resolved_dtype != torch.bfloat16:
            raw["qwen_attn_implementation"] = None

        backbone_cfg = raw.get("point_encoder_backbone", None)
        if backbone_cfg is None:
            raw["point_encoder_backbone"] = PointEncoderBackboneConfigs()
        elif isinstance(backbone_cfg, PointEncoderBackboneConfigs):
            raw["point_encoder_backbone"] = backbone_cfg
        else:
            raw["point_encoder_backbone"] = PointEncoderBackboneConfigs(backbone_cfg)

        super().__init__(raw)


class ImageDecoderConfigs(Configs):
    """视觉编码器及对齐相关配置。"""

    defaults = {
        "compute_dtype": "fp32",
        "hidden_size": 256,
        "num_heads": 8,
        "mm_vision_select_feature": "patch",
        "image_aspect_ratio": "square",
        "image_grid_pinpoints": None,
        "image_out_dim": 1,
        "tune_mm_mlp_adapter": False,
        "freeze_mm_mlp_adapter": True,
        "pretrain_mm_mlp_adapter": None,
        "mm_use_im_patch_token": False,
        "use_cache": False,
    }

    def __init__(self, config_dict: Optional[Dict] = None, **overrides):
        raw = self._merge_defaults(config_dict, overrides)
        raw["compute_dtype"] = resolve_dtype(raw["compute_dtype"])
        super().__init__(raw)


class PointDecoderConfigs(Configs):
    """图像/点云解码器相关配置。"""

    defaults = {
        "compute_dtype": "fp32",
        "hidden_size": 512,
        "num_heads": 8,
        "backbone_mode": "independent",     # {"shared", "independent"}
        "decode_mode": "similarity",        # {"similarity", "prompt"}
        "grid_size": 0.02,
        "backbone_kwargs": None,
        "backbone_out_channels": 64,
    }

    def __init__(self, config_dict: Optional[Dict] = None, **overrides):
        raw = self._merge_defaults(config_dict, overrides)
        raw["backbone_mode"] = str(raw["backbone_mode"]).lower()
        assert raw["backbone_mode"] in {"shared", "independent"}, f"Unsupported point decoder backbone_mode: {raw['backbone_mode']}"

        raw["decode_mode"] = str(raw["decode_mode"]).lower()
        assert raw["decode_mode"] in {"similarity", "prompt"}, f"Unsupported point decoder decode_mode: {raw['decode_mode']}"

        raw["compute_dtype"] = resolve_dtype(raw["compute_dtype"])
        backbone_kwargs = raw["backbone_kwargs"]
        if backbone_kwargs is None:
            backbone_kwargs = PointEncoderBackboneConfigs().to_dict()
        elif hasattr(backbone_kwargs, "to_dict"):
            backbone_kwargs = backbone_kwargs.to_dict()
        else:
            backbone_kwargs = dict(backbone_kwargs)
        backbone_kwargs["enc_mode"] = False
        raw["backbone_kwargs"] = backbone_kwargs
        if raw.get("backbone_out_channels", None) is None:
            dec_channels = backbone_kwargs.get("dec_channels", PointEncoderBackboneConfigs.defaults["dec_channels"])
            raw["backbone_out_channels"] = int(dec_channels[0])
        super().__init__(raw)


class JointAffordanceConfig(Configs):
    """联合可供性模型的配置类。"""

    def __init__(
        self,
        config_dict: Optional[Dict] = None,
        mllm_config: Optional[MLLMConfigs | Dict] = None,
        image_decoder: Optional[ImageDecoderConfigs | Dict] = None,
        point_decoder: Optional[PointDecoderConfigs | Dict] = None,
        **kwargs,
    ):
        raw = {}
        if config_dict is not None:
            raw.update(config_dict)
        raw.update(kwargs)

        # 兼容不同命名风格：mllm / mllm_config
        if mllm_config is None and "mllm" in raw:
            mllm_config = raw.pop("mllm")
        if mllm_config is None and "mllm_config" in raw:
            mllm_config = raw.pop("mllm_config")
        if image_decoder is None and "image_decoder" in raw:
            image_decoder = raw.pop("image_decoder")
        if point_decoder is None and "point_decoder" in raw:
            point_decoder = raw.pop("point_decoder")

        super().__init__(raw)

        self.mllm = mllm_config if isinstance(mllm_config, MLLMConfigs) else MLLMConfigs(mllm_config)
        self.image_decoder = (
            image_decoder if isinstance(image_decoder, ImageDecoderConfigs) else ImageDecoderConfigs(image_decoder)
        )
        self.point_decoder = (
            point_decoder if isinstance(point_decoder, PointDecoderConfigs) else PointDecoderConfigs(point_decoder)
        )

        if self.mllm is not None:
            self.tokenizer = self.mllm.tokenizer

    def __getattr__(self, name):
        for cfg in (self.mllm, self.image_decoder, self.point_decoder):
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


__all__ = [
    "Configs",
    "PointEncoderBackboneConfigs",
    "MLLMConfigs",
    "ImageDecoderConfigs",
    "PointDecoderConfigs",
    "JointAffordanceConfig",
]
