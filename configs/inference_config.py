"""
推理配置：控制推理/验证阶段的运行方式与输出行为。
"""
from typing import Optional, Dict

from configs import base_config


class InferenceConfig(base_config.Configs):
    defaults = {
        "device": "cuda",
        "precision": "bf16",
        "image_size": 1024,
        "num_points": 2048,
        "batch_size": 1,
        "num_workers": 4,
        "split": "test",
        "save_predictions": False,
        "output_dir": "./validation_output",
        "mask_threshold_2d": 0.0,
        "mask_threshold_3d": 0.5,
    }

    def __init__(
        self,
        config_dict: Optional[Dict] = None,
        **overrides,
    ):
        raw = {}
        if config_dict is not None:
            raw.update(config_dict)
        raw.update(overrides)
        super().__init__(raw)
