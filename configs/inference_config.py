"""
推理配置：控制推理/验证阶段的运行方式与输出行为。
"""
from typing import Optional, Dict

from configs import base_config


class InferenceConfig(base_config.Configs):
    defaults = {
        # 覆盖优先级：命令行参数 > InferenceConfig.defaults > training_config.json。
        # 默认为 None 表示不覆盖 TrainingConfig；若在此处填具体值，则会覆盖训练配置。
        "precision": None,
        "mask_threshold_2d": None,
        "mask_threshold_3d": None,
    }

    def __init__(
        self,
        config_dict: Optional[Dict] = None,
        **overrides,
    ):
        raw = {}
        if config_dict is not None:
            raw.update({k: v for k, v in config_dict.items() if v is not None})
        raw.update({k: v for k, v in overrides.items() if v is not None})
        super().__init__(raw)
