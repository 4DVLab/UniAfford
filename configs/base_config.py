"""
联合可供性模型配置，集中管理模型所需的全部配置项。
"""
import torch
from typing import Optional
from utils.common import resolve_dtype

class Configs:
    """基础配置类，提供快速属性更新能力。"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def to_dict(self):
        return dict(self.__dict__)


class MLLMConfigs(Configs):
    """多模态大模型相关配置。"""

    def __init__(
        self,
        # version: str = "../pretrained/llava-llama-2-13b-chat-lightning-preview",
        qwen_model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        qwen_attn_implementation: str = "flash_attention_2",
        qwen_dtype: str = "bf16",
        model_max_length: int = 512,
        # conv_type: str = "llava_llama_2",
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_heads: int = 8,
        tokenizer: Optional[object] = None,
        # use_mm_start_end: bool = True,
        # vision_tower: str = "../pretrained/clip-vit-large-patch14",
        # vision_pretrained: Optional[str] = "../pretrained/sam_vit_h_4b8939.pth",
        train_mask_decoder: bool = True,
        out_dim: int = 256,
        ce_loss_weight: float = 1.0,
        dice_loss_weight: float = 0.5,
        bce_loss_weight: float = 2.0,
        seg_token_idx: Optional[int] = None,
        aff_token_idx: Optional[int] = None,
    ):
        super().__init__(
            # version=version,
            qwen_model_name_or_path=qwen_model_name_or_path,
            qwen_attn_implementation=qwen_attn_implementation,
            qwen_dtype=resolve_dtype(qwen_dtype),
            model_max_length=model_max_length,
            # conv_type=conv_type,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            tokenizer=tokenizer,
            # use_mm_start_end=use_mm_start_end,
            # vision_tower=vision_tower,
            # vision_pretrained=vision_pretrained,
            train_mask_decoder=train_mask_decoder,
            out_dim=out_dim,
            ce_loss_weight=ce_loss_weight,
            dice_loss_weight=dice_loss_weight,
            bce_loss_weight=bce_loss_weight,
            seg_token_idx=seg_token_idx,
            aff_token_idx=aff_token_idx,
        )


class ImageDecoderConfigs(Configs):
    """视觉编码器及对齐相关配置。"""

    def __init__(
        self,
        compute_dtype: Optional[torch.dtype] = None,
        hidden_size: int = 768,
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
            compute_dtype=compute_dtype,
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
        compute_dtype: Optional[torch.dtype] = None,
        hidden_size: int = 768,
        num_heads: int = 8,
        # point_out_dim: int = None, # 为空则可以适配任意点数的输入输出
        **kwargs,
    ):
        super().__init__(
            compute_dtype=compute_dtype,
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
        compute_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):  
        super().__init__(compute_dtype=compute_dtype, **kwargs)
        self.mllm = mllm_config or MLLMConfigs()
        self.image_decoder = image_decoder or ImageDecoderConfigs(compute_dtype=compute_dtype)
        self.point_decoder = point_decoder or PointDecoderConfigs(compute_dtype=compute_dtype)

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
