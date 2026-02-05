"""
NOTE: 所有的相对目录都是基于项目（仓库）根目录而言
"""
import os
from re import S
from typing import Optional

import torch
from utils.common import resolve_dtype

from .base_config import Configs, JointAffordanceConfig


class DeepSpeedConfigs(Configs):
    """DeepSpeed 相关配置，所有项以属性存储，通过 to_dict() 转为 DeepSpeed 所需字典。"""

    def __init__(
        self,
        # 与 DeepSpeed 顶层键对应的属性
        train_micro_batch_size_per_gpu: int = 1,
        gradient_accumulation_steps: int = 1,
        precision: Optional[torch.dtype] = None,
        gradient_clipping: float = 1.0,

        # zero_optimization 相关
        zero_stage: int = 2,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        reduce_scatter: bool = True,
        reduce_bucket_size: float = 5e8,
        allgather_bucket_size: float = 5e8,
        # 是否使用分层学习率（为 True 时不写 optimizer/scheduler）
        use_layerwise_lr: bool = True,
        # optimizer / scheduler 用到的训练参数
        lr: float = 0.003,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epochs: int = 250,
        steps_per_epoch: Optional[int] = None,
        warmup_min_lr: float = 0.0,
        warmup_num_steps: int = 100,
        warmup_type: str = "linear",
        **kwargs,
    ):
        super().__init__(
            train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            precision=precision,
            gradient_clipping=gradient_clipping,
            zero_stage=zero_stage,
            contiguous_gradients=contiguous_gradients,
            overlap_comm=overlap_comm,
            reduce_scatter=reduce_scatter,
            reduce_bucket_size=reduce_bucket_size,
            allgather_bucket_size=allgather_bucket_size,
            use_layerwise_lr=use_layerwise_lr,
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_min_lr=warmup_min_lr,
            warmup_num_steps=warmup_num_steps,
            warmup_type=warmup_type,
            **kwargs,
        )

    def to_dict(self) -> dict:
        """将当前属性转换为 DeepSpeed 配置字典（嵌套结构）。"""
        precision = self.precision if self.precision is not None else torch.float32
        ds_config = {
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": {"enabled": precision == torch.half},
            "bf16": {"enabled": precision == torch.bfloat16},
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_stage,
                "contiguous_gradients": self.contiguous_gradients,
                "overlap_comm": self.overlap_comm,
                "reduce_scatter": self.reduce_scatter,
                "reduce_bucket_size": int(self.reduce_bucket_size),
                "allgather_bucket_size": int(self.allgather_bucket_size),
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

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "q_proj, k_proj, v_proj, o_proj",
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        **kwargs,
    ):
        super().__init__(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=[m.strip() for m in lora_target_modules.split(",") if m.strip()],
            bias=bias,
            task_type=task_type,
            **kwargs,
        )

    def to_dict(self) -> dict:
        """属性转成普通字典，便于序列化或日志。"""
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
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
            target_modules=list(self.lora_target_modules),
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )

class TrainingConfig(Configs):
    """
    训练配置类，用于存储模型训练的所有超参数。详细说明见同目录下README.md
    """
    
    def __init__(
        self,
        # 基础配置
        local_rank=0,
        precision="fp32",  # fp32。 bf16会报错，fp16会导致nan
        model_config: Optional[JointAffordanceConfig] = None,
        
        # 模型配置
        image_size=(1024, 1024), # h,w
        
        # 数据配置
        dataset_dir="../datasets/merged1-2-3/",
        log_base_dir="../runs",
        exp_name="joint-aff-exp",
        num_points=2048,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        use_sample_cache=False,  # 是否启用样本缓存以提高数据加载速度，适用于小批量数据且显存充裕的情况
        
        # 训练配置
        epochs=250,
        samples_per_epoch=None,  # 数据大时设置为训练数据集的 70% 以上，比较小时设置为 90%以上。默认全量训练数据集
        steps_per_epoch=None,
        batch_size=1,
        grad_accumulation_steps=1,
        val_batch_size=10,
        workers=4,
        print_freq=1,
        name_of_params_to_train="visual_model, vision_tower, mm_projector, text_hidden_fcs, point_cloud_segmentor",

        # 优化器配置
        lr=0.003,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.0,
        
        # 分层学习率（可选）
        use_layerwise_lr=True,
        llm_lr=None,        # 默认为 lr * 0.1
        vision_2d_lr=None,  # 默认为 lr
        vision_3d_lr=None,  # 默认为 lr
        llm_param_keywords="lora_, base_layer",
        vision_2d_param_keywords="mask_decoder, text_hidden_fcs",
        vision_3d_param_keywords="point_cloud_segmentor",
        
        # 学习率调度器配置
        warmup_num_steps=100,
        warmup_min_lr=0,
        warmup_type="linear",
        
        # 其他配置
        num_classes_per_sample=3,
        mask_threshold_2d=0.0,
        mask_threshold_3d=0.5,
        gradient_checkpointing=True,
        
        # 高级配置
        exclude_val=False,
        no_eval=False,
        eval_only=False,
        auto_resume=True,
        resume="",
        start_epoch=0,
        vis_save_path="./vis_output",
        **kwargs,
    ):  
        
        
        if samples_per_epoch is not None:
            steps_per_epoch = samples_per_epoch // batch_size

        resolved_precision = resolve_dtype(precision)

        super().__init__(
            # 基础配置
            local_rank = local_rank,
            
            precision = resolved_precision,

            model_config = model_config or JointAffordanceConfig(),
            vis_save_path = vis_save_path,
            
            # 模型配置
            image_size = image_size,
            
            # 数据配置
            dataset_dir = dataset_dir,
            log_base_dir = log_base_dir,
            exp_name = exp_name,
            num_points = num_points,
            train_ratio = train_ratio,
            val_ratio = val_ratio,
            test_ratio = test_ratio,
            use_sample_cache = use_sample_cache,
            
            # 训练配置
            epochs = epochs,
            steps_per_epoch = steps_per_epoch ,
            samples_per_epoch = samples_per_epoch,
            
            batch_size = batch_size,
            grad_accumulation_steps = grad_accumulation_steps,
            val_batch_size = val_batch_size,
            workers = workers,
            print_freq = print_freq,
            name_of_params_to_train = [m.strip() for m in name_of_params_to_train.split(",") if m.strip()],
            
            # 优化器配置
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            weight_decay = weight_decay,
            
            # 分层学习率
            use_layerwise_lr = use_layerwise_lr,
            llm_lr = llm_lr if llm_lr is not None else lr * 0.1,
            vision_2d_lr = vision_2d_lr if vision_2d_lr is not None else lr,
            vision_3d_lr = vision_3d_lr if vision_3d_lr is not None else lr,
            llm_param_keywords = [m.strip() for m in llm_param_keywords.split(",") if m.strip()],
            vision_2d_param_keywords = [m.strip() for m in vision_2d_param_keywords.split(",") if m.strip()],
            vision_3d_param_keywords = [m.strip() for m in vision_3d_param_keywords.split(",") if m.strip()],
            
            # 学习率调度器配置
            warmup_num_steps = warmup_num_steps,
            warmup_min_lr = warmup_min_lr,
            warmup_type = warmup_type,

            # DeepSpeed / LoRA 配置对象
            deepspeed = DeepSpeedConfigs(),
            lora = LoRAConfigs(),
            
            # 其他配置
            num_classes_per_sample = num_classes_per_sample,
            mask_threshold_2d = mask_threshold_2d,
            mask_threshold_3d = mask_threshold_3d,
            gradient_checkpointing = gradient_checkpointing,
            
            # 高级配置
            exclude_val = exclude_val,
            no_eval = no_eval,
            eval_only = eval_only,
            auto_resume = auto_resume,
            resume = resume,
            start_epoch = start_epoch,
            
            # 运行时属性
            log_dir = os.path.join(log_base_dir, exp_name),
            distributed = False,
            **kwargs,
        )
    
