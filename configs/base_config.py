"""
联合可供性模型配置。
参考 training_config.py 的组织方式，集中管理模型所需的全部配置项。
"""
from __future__ import annotations

from typing import Optional

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
        version: str = "../pretrained/llava-llama-2-13b-chat-lightning-preview",
        model_max_length: int = 512,
        conv_type: str = "llava_llama_2",
        vocab_size: int = 32000,
        hidden_size: int = 768,
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
            version=version,
            model_max_length=model_max_length,
            conv_type=conv_type,
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
        hidden_size: int = 768,
        num_heads: int = 8,
        point_out_dim: int = None, # 为空则可以适配任意点数的输入输出
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            point_out_dim=point_out_dim,
            **kwargs,
        )

class JointAffModelConfigs(Configs):
    """联合可供性模型的配置类。"""
    def __init__(
        self,
        mllm_config:Optional[MLLMConfigs] = None,
        visual_decoder: Optional[ImageDecoderConfigs] = None,
        point_decoder: Optional[PointDecoderConfigs] = None,
        **kwargs,
    ):  
        super().__init__(**kwargs)
        self.mllm = mllm_config or MLLMConfigs()
        self.visual_decoder = visual_decoder or ImageDecoderConfigs()
        self.point_decoder = point_decoder or PointDecoderConfigs()

        if self.mllm is not None:
            self.tokenizer = self.mllm.tokenizer

    def __getattr__(self, name):
        for cfg in (self.mllm, self.visual_decoder, self.point_decoder):
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


JointAffordanceConfig = JointAffModelConfigs



__all__ = [
    "Configs",
    "MLLMConfigs",
    "ImageDecoderConfigs",
    "PointDecoderConfigs",
    "JointAffordanceConfig",
]
