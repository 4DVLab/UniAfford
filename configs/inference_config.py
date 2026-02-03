"""
推理配置：控制推理/验证阶段的运行方式与输出行为。
"""
from __future__ import annotations
from configs import base_config


class InferenceConfig(base_config.Configs):
    def __init__(
        self,
        device: str = "cuda",
        precision: str = "bf16",
        image_size: int = 1024,
        num_points: int = 2048,
        batch_size: int = 1,
        num_workers: int = 4,
        split: str = "test",
        save_predictions: bool = False,
        output_dir: str = "./validation_output",
        mask_threshold_2d: float = 0.0,
        mask_threshold_3d: float = 0.5,
    ):
        super().__init__(
            device=device,
            precision=precision,
            image_size=image_size,
            num_points=num_points,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
            save_predictions=save_predictions,
            output_dir=output_dir,
            mask_threshold_2d=mask_threshold_2d,
            mask_threshold_3d=mask_threshold_3d,
        )
