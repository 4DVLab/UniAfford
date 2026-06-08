"""
NOTE: 所有的相对目录都是基于项目（仓库）根目录而言
"""
import os
import json
from copy import deepcopy
from typing import List, Optional, Union

import torch
from utils.common import resolve_dtype

from .base_config import Configs, ImageDecoderConfigs, JointAffordanceConfig, MLLMConfigs, PointDecoderConfigs


class DeepSpeedConfigs(Configs):
    """DeepSpeed 相关配置，所有项以属性存储，通过 to_dict() 转为 DeepSpeed 所需字典。"""

    defaults = {
        # 与 DeepSpeed 顶层键对应的属性
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "precision": "fp32",
        "gradient_clipping": 1.0,
        # zero_optimization 相关（默认 ZeRO-3：模型+优化器+梯度全分片）
        "zero_stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        # 优化器 Offload（显存不足时把优化器状态放 CPU）
        "offload_optimizer_device": "cpu",
        "offload_optimizer_pin_memory": True,
        # 是否使用分层学习率（为 True 时不写 optimizer/scheduler）
        "use_layerwise_lr": True,
        # optimizer / scheduler 用到的训练参数
        "lr": 0.003,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "epochs": 250,
        "steps_per_epoch": None,
        "warmup_min_lr": 0.0,
        "warmup_num_steps": 100,
        "warmup_type": "linear",
    }

    def __init__(self, config_dict: Optional[dict] = None, **overrides):
        raw = self._merge_defaults(config_dict, overrides)
        raw["precision"] = resolve_dtype(raw["precision"])
        super().__init__(raw)

    def to_dict(self) -> dict:
        """将当前属性转换为 DeepSpeed 配置字典（嵌套结构）。"""
        ds_config = {
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": {"enabled": self.precision == torch.half},
            "bf16": {"enabled": self.precision == torch.bfloat16},
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_stage,
                "allgather_partitions": self.allgather_partitions,
                "allgather_bucket_size": int(self.allgather_bucket_size),
                "overlap_comm": self.overlap_comm,
                "reduce_scatter": self.reduce_scatter,
                "reduce_bucket_size": int(self.reduce_bucket_size),
                "contiguous_gradients": self.contiguous_gradients,
                "offload_optimizer": {
                    "device": self.offload_optimizer_device,
                    "pin_memory": self.offload_optimizer_pin_memory,
                },
                # "offload_param": {
                #     "device": self.offload_param_device,
                #     "pin_memory": self.offload_param_pin_memory,
                # },
            },
        }
        if not self.use_layerwise_lr:
            total_steps = (self.epochs * self.steps_per_epoch) if self.steps_per_epoch is not None else 0
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "betas": (self.beta1, self.beta2),
                },
            }
            ds_config["scheduler"] = {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": total_steps,
                    "warmup_min_lr": self.warmup_min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                    "warmup_type": self.warmup_type,
                },
            }
        return ds_config

class LoRAConfigs(Configs):
    """
    LoRA 相关配置，与 DeepSpeedConfigs 一致：全部以属性存储。
    - to_dict()：序列化/日志用。
    - to_peft_config()：返回 peft.LoraConfig，供 get_peft_model 等直接使用。
    """

    defaults = {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    def __init__(self, config_dict: Optional[dict] = None, **overrides):
        raw = self._merge_defaults(config_dict, overrides)
        target_modules = raw["lora_target_modules"]
        raw["lora_target_modules"] = (
            target_modules
            if isinstance(target_modules, list)
            else [m.strip() for m in str(target_modules).split(",") if m.strip()]
        )
        super().__init__(raw)

    def to_dict(self) -> dict:
        """属性转成普通字典，便于序列化或日志。"""
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def to_peft_config(self):
        """返回 peft.LoraConfig，供 get_peft_model(model, config.lora.to_peft_config()) 使用。"""
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )

class TrainingConfig(Configs):
    """
    训练配置类，用于存储模型训练的所有超参数。详细说明见同目录下README.md
    """
    defaults = {
        # 基础配置
        "local_rank": 0,
        "vis_save_path": "./vis_output",
        # 数据配置
        "dataset_dir": "../datasets/merged1-2-3/",
        "log_base_dir": "../runs",
        "exp_name": "joint-aff-exp",
        "image_size": (1024, 1024),
        "num_points": 2048,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "use_sample_cache": False,  # 是否启用样本缓存以提高数据加载速度，适用于小批量数据且显存充裕的情况
        # 训练配置
        "epochs": 250,
        "samples_per_epoch": None,  # 数据大时设置为训练数据集的 70% 以上，比较小时设置为 90%以上。默认全量训练数据集
        "steps_per_epoch": None,
        "batch_size": 1,  # 每卡 batch 大小
        "grad_accumulation_steps": 1,
        "val_batch_size": 10,  # 验证时每卡 batch 大小
        "workers": 4,
        "print_freq": 1,
        # 默认使用短回答模板，降低 generate 阶段自回归格式漂移；关闭后恢复原来的丰富自然语言回答。
        "use_simple_answer_template": False,
        # 训练策略：默认全量训练，需要冻结的模块统一写到 name_of_params_to_freeze
        # 默认冻结qwen的视觉编码器， point_encoder (pretrained SONATA)的点云编码器， image_decoder (SAM)的图像编码器
        "name_of_params_to_freeze": "mllm.model.visual, point_encoder.point_backbone, image_decoder.visual_model.image_encoder",
        # 优化器配置
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.0,
        # 分层学习率（可选）
        "use_layerwise_lr": True,
        "llm_lr": None,        # 默认为 lr * 0.01 (5e-5)
        "vision_2d_lr": 5e-6,  
        "vision_3d_lr": 5e-4,  
        # 学习率调度器配置
        "warmup_num_steps": 100,
        "warmup_min_lr": 0,
        "warmup_type": "linear",
        # 其他配置
        "num_classes_per_sample": 3,
        # 预测概率图二值化阈值（可由验证集自动搜索后写回 training_config.json）
        "mask_threshold_2d": 0.5,
        "mask_threshold_3d": 0.5,
        # GT 二值化阈值：用于对齐不同 benchmark/zero-shot setting，不应由验证集自动修改
        "gt_threshold_2d": 0.5,
        "gt_threshold_3d": 0.5,
        # 验证集搜索预测阈值的候选范围
        "auto_select_mask_threshold": True,
        "write_selected_mask_threshold_to_config": True,
        "gradient_checkpointing": True,
        # 损失配置
        "focal_loss_weight": 2.0,
        "dice_loss_weight": 0.5,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "bce_loss_weight": 2.0,
        "pc_dice_loss_weight": 0.5,
        "ce_loss_weight": 1.0,
        # 高级配置
        "exclude_val": False,
        "no_eval": False,
        "eval_only": False,
        "auto_resume": True,
        "resume": "",
        "start_epoch": 0,
        # 运行时属性
        "distributed": False,
    }

    def __init__(
        self,
        config_dict: Optional[dict] = None,
        model_config: Optional[JointAffordanceConfig | dict] = None,
        deepspeed_config: Optional[DeepSpeedConfigs | dict] = None,
        lora_config: Optional[LoRAConfigs | dict] = None,
        **kwargs,
    ):
        raw = self._merge_defaults(config_dict, kwargs)

        if model_config is None and "model_config" in raw:
            model_config = raw.pop("model_config")
        if deepspeed_config is None and "deepspeed_config" in raw:
            deepspeed_config = raw.pop("deepspeed_config")
        if deepspeed_config is None and "deepspeed" in raw:
            deepspeed_config = raw.pop("deepspeed")
        if lora_config is None and "lora_config" in raw:
            lora_config = raw.pop("lora_config")
        if lora_config is None and "lora" in raw:
            lora_config = raw.pop("lora")

        if model_config is None:
            model_config = JointAffordanceConfig(
                mllm_config=MLLMConfigs(compute_dtype="fp32"),  # Qwen必须使用bf16以使用flash-attn
                image_decoder=ImageDecoderConfigs(compute_dtype="fp32"),  # 分布式训练要求保持参数精度一致
                point_decoder=PointDecoderConfigs(compute_dtype="fp32"),
            )
        elif not isinstance(model_config, JointAffordanceConfig):
            model_config = JointAffordanceConfig(model_config)

        deepspeed_config = (
            deepspeed_config
            if isinstance(deepspeed_config, DeepSpeedConfigs)
            else DeepSpeedConfigs(deepspeed_config)
        )
        lora_config = (
            lora_config
            if isinstance(lora_config, LoRAConfigs)
            else LoRAConfigs(lora_config)
        )

        batch_size = raw["batch_size"]
        samples_per_epoch = raw["samples_per_epoch"]
        steps_per_epoch = raw["steps_per_epoch"]
        if samples_per_epoch is not None and steps_per_epoch is None:
            # 仅作为初始化兜底；train_fsdp.py 会在构建 DataLoader 后用真实值覆盖。
            raw["steps_per_epoch"] = max(1, (samples_per_epoch + batch_size - 1) // max(1, batch_size))

        params_to_freeze = raw["name_of_params_to_freeze"]
        raw["name_of_params_to_freeze"] = (
            params_to_freeze
            if isinstance(params_to_freeze, list)
            else [m.strip() for m in str(params_to_freeze).split(",") if m.strip()]
        )

        lr = raw["lr"]
        if raw.get("llm_lr", None) is None:
            raw["llm_lr"] = lr * 0.01
        if raw.get("vision_2d_lr", None) is None:
            raw["vision_2d_lr"] = lr
        if raw.get("vision_3d_lr", None) is None:
            raw["vision_3d_lr"] = lr

        log_base_dir = raw["log_base_dir"]
        exp_name = raw["exp_name"]
        raw["log_dir"] = os.path.join(log_base_dir, exp_name)

        super().__init__(raw, model_config=model_config, deepspeed=deepspeed_config, lora=lora_config)

    def to_json_dict(self, include_deepspeed: bool = True):
        data = super().to_json_dict()
        if not include_deepspeed:
            data.pop("deepspeed", None)
        return data

    def save_json(self, path: str, include_deepspeed: bool = True):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(include_deepspeed=include_deepspeed), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json_dict(cls, data: dict) -> "TrainingConfig":
        """从 JSON 字典恢复 TrainingConfig。"""
        payload = dict(data)

        model_data = payload.pop("model_config", {}) or {}
        # 兼容不同命名（mllm / mllm_config）
        mllm_data = model_data.pop("mllm", None)
        if mllm_data is None:
            mllm_data = model_data.pop("mllm_config", {}) or {}
        image_data = model_data.pop("image_decoder", {}) or {}
        point_data = model_data.pop("point_decoder", {}) or {}

        # tokenizer 句柄不可从 JSON 恢复，统一置空并在运行时重建
        mllm_data["tokenizer"] = None

        model_cfg = JointAffordanceConfig(
            mllm_config=MLLMConfigs(**mllm_data),
            image_decoder=ImageDecoderConfigs(**image_data),
            point_decoder=PointDecoderConfigs(**point_data),
            **model_data,
        )

        ds_data = payload.pop("deepspeed", {}) or {}
        lora_data = payload.pop("lora", {}) or {}
        ds_cfg = DeepSpeedConfigs(**ds_data)
        lora_cfg = LoRAConfigs(**lora_data)

        # 这两个字段由构造逻辑推导得到，不参与反序列化输入
        payload.pop("log_dir", None)
        payload.pop("distributed", None)
        # JSON 中 tuple 会变 list，这里恢复常用结构
        if isinstance(payload.get("image_size"), list):
            payload["image_size"] = tuple(payload["image_size"])

        return cls(
            config_dict=payload,
            model_config=model_cfg,
            deepspeed_config=ds_cfg,
            lora_config=lora_cfg,
        )

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """从 JSON 文件恢复 TrainingConfig。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)
    
