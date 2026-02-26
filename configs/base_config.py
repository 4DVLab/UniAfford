"""
联合可供性模型配置，集中管理模型所需的全部配置项。
"""
import json
import torch
from typing import Optional
from utils.common import resolve_dtype, SEG_TOKEN, AFF_TOKEN

class Configs:
    """基础配置类，提供快速属性更新能力。"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
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


class MLLMConfigs(Configs):
    """多模态大模型相关配置。"""

    def __init__(
        self,
        qwen_model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        qwen_attn_implementation: str = "flash_attention_2",
        compute_dtype: str = "bf16",
        model_max_length: int = 512,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_heads: int = 8,
        tokenizer: Optional[object] = None,
        train_mask_decoder: bool = True,
        out_dim: int = 256,
        seg_token: str = SEG_TOKEN,
        aff_token: str = AFF_TOKEN,
        **kwargs,
    ):

        # 在fp32的时候禁用 flash_attention_2，因为flash_attention_2只支持bf16，在fp32的时候使用默认的attn_implementation
        qwen_attn_implementation = qwen_attn_implementation if compute_dtype == 'bf16' else None
        super().__init__(
            qwen_model_name_or_path=qwen_model_name_or_path,
            qwen_attn_implementation=qwen_attn_implementation,
            compute_dtype=resolve_dtype(compute_dtype),
            model_max_length=model_max_length,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            tokenizer=tokenizer,
            train_mask_decoder=train_mask_decoder,
            out_dim=out_dim,
            seg_token=seg_token,
            aff_token=aff_token,
            **kwargs,
        )


class ImageDecoderConfigs(Configs):
    """视觉编码器及对齐相关配置。"""

    def __init__(
        self,
        compute_dtype: Optional[torch.dtype] = 'fp32',
        hidden_size: int = 256,
        num_heads: int = 8,
        mm_vision_select_feature: str = "patch",
        image_aspect_ratio: str = "square",
        image_grid_pinpoints: Optional[object] = None,
        image_out_dim: int = 1,
        tune_mm_mlp_adapter: bool = False,
        freeze_mm_mlp_adapter: bool = True,
        pretrain_mm_mlp_adapter: Optional[str] = None,
        mm_use_im_patch_token: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):  
        super().__init__(
            compute_dtype=resolve_dtype(compute_dtype),
            hidden_size=hidden_size,
            num_heads=num_heads,
            mm_vision_select_feature=mm_vision_select_feature,
            image_aspect_ratio=image_aspect_ratio,
            image_grid_pinpoints=image_grid_pinpoints,
            image_out_dim=image_out_dim,
            tune_mm_mlp_adapter=tune_mm_mlp_adapter,
            freeze_mm_mlp_adapter=freeze_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            mm_use_im_patch_token=mm_use_im_patch_token,
            use_cache=use_cache,
            **kwargs,
        )


class PointDecoderConfigs(Configs):
    """图像/点云解码器相关配置。"""

    def __init__(
        self,
        compute_dtype: Optional[torch.dtype] = 'fp32',
        hidden_size: int = 128,
        num_heads: int = 8,
        # point_out_dim: int = None, # 为空则可以适配任意点数的输入输出
        **kwargs,
    ):
        super().__init__(
            compute_dtype=resolve_dtype(compute_dtype),
            hidden_size=hidden_size,
            num_heads=num_heads,
            # point_out_dim=point_out_dim,
            **kwargs,
        )
        

class JointAffordanceConfig(Configs):
    """联合可供性模型的配置类。"""
    def __init__(
        self,
        mllm_config:Optional[MLLMConfigs] = None,
        image_decoder: Optional[ImageDecoderConfigs] = None,
        point_decoder: Optional[PointDecoderConfigs] = None,
        **kwargs,
    ):  
        super().__init__(**kwargs)
        self.mllm = mllm_config or MLLMConfigs()
        self.image_decoder = image_decoder or ImageDecoderConfigs()
        self.point_decoder = point_decoder or PointDecoderConfigs()

        if self.mllm is not None:
            self.tokenizer = self.mllm.tokenizer

    def __getattr__(self, name):
        for cfg in (self.mllm, self.image_decoder, self.point_decoder):
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")



__all__ = [
    "Configs",
    "MLLMConfigs",
    "ImageDecoderConfigs",
    "PointDecoderConfigs",
    "JointAffordanceConfig",
]
