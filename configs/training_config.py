"""
NOTE: 所有的相对目录都是基于项目（仓库）根目录而言
"""
import os
import torch

class TrainingConfig:
    """
    训练配置类，用于存储模型训练的所有超参数。详细说明见同目录下README.md
    """
    
    def __init__(
        self,
        # 基础配置
        local_rank=0,
        version="../pretrained/llava-llama-2-13b-chat-lightning-preview",
        precision="fp16",  # fp32, bf16, fp16
        
        # 模型配置
        image_size=(1024, 1024), # h,w
        model_max_length=512,
        vision_tower="../pretrained/clip-vit-large-patch14",
        vision_pretrained="../pretrained/sam_vit_h_4b8939.pth",
        out_dim=256,
        
        # 数据配置
        dataset_dir="../datasets/merged1-2-3/",
        log_base_dir="../runs",
        exp_name="joint-aff-exp",
        num_points=2048,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        
        # 训练配置
        epochs=250,
        steps_per_epoch=100,
        batch_size=1,
        grad_accumulation_steps=10,
        val_batch_size=1,
        workers=4,
        print_freq=1,
        name_of_params_to_train="visual_model,vision_tower,mm_projector,text_hidden_fcs,point_cloud_segmentor",

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
        
        # 学习率调度器配置
        warmup_num_steps=100,
        warmup_min_lr=0,
        warmup_type="linear",
        
        # DeepSpeed 配置
        gradient_clipping=1.0,
        zero_stage=2,
        reduce_bucket_size=5e8,
        allgather_bucket_size=5e8,
        
        # 损失权重
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        
        # LoRA 配置
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        
        # 其他配置
        num_classes_per_sample=3,
        mask_threshold_2d=0.0,
        mask_threshold_3d=0.5,
        gradient_checkpointing=True,
        train_mask_decoder=True,
        use_mm_start_end=True,
        conv_type="llava_llama_2",  # llava_v1, llava_llama_2
        
        # 高级配置
        exclude_val=False,
        no_eval=False,
        eval_only=False,
        auto_resume=True,
        resume="",
        start_epoch=0,
        vis_save_path="./vis_output",
    ):
        # 基础配置
        self.local_rank = local_rank
        self.version = version
        
        self.precision = torch.float32
        match precision:
            case 'fp16':
                self.precision = torch.half
            case "bf16":
                self.precision = torch.bfloat16

        self.vis_save_path = vis_save_path
        
        # 模型配置
        self.image_size = image_size
        self.model_max_length = model_max_length
        self.vision_tower = vision_tower
        self.vision_pretrained = vision_pretrained
        self.out_dim = out_dim
        
        # 数据配置
        self.dataset_dir = dataset_dir
        self.log_base_dir = log_base_dir
        self.exp_name = exp_name
        self.num_points = num_points
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 训练配置
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.print_freq = print_freq
        self.name_of_params_to_train = name_of_params_to_train.split(",")
        
        # 优化器配置
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
        # 分层学习率
        self.use_layerwise_lr = use_layerwise_lr
        self.llm_lr = llm_lr if llm_lr is not None else lr * 0.1
        self.vision_2d_lr = vision_2d_lr if vision_2d_lr is not None else lr
        self.vision_3d_lr = vision_3d_lr if vision_3d_lr is not None else lr
        
        # 学习率调度器配置
        self.warmup_num_steps = warmup_num_steps
        self.warmup_min_lr = warmup_min_lr
        self.warmup_type = warmup_type
        
        # DeepSpeed 配置
        self.gradient_clipping = gradient_clipping
        self.zero_stage = zero_stage
        self.reduce_bucket_size = reduce_bucket_size
        self.allgather_bucket_size = allgather_bucket_size
        
        # 损失权重
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        
        # LoRA 配置
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules.split(',')
        
        # 其他配置
        self.num_classes_per_sample = num_classes_per_sample
        self.mask_threshold_2d = mask_threshold_2d
        self.mask_threshold_3d = mask_threshold_3d
        self.gradient_checkpointing = gradient_checkpointing
        self.train_mask_decoder = train_mask_decoder
        self.use_mm_start_end = use_mm_start_end
        self.conv_type = conv_type
        
        # 高级配置
        self.exclude_val = exclude_val
        self.no_eval = no_eval
        self.eval_only = eval_only
        self.auto_resume = auto_resume
        self.resume = resume
        self.start_epoch = start_epoch
        
        # 运行时属性
        self.log_dir = os.path.join(self.log_base_dir, self.exp_name)
        self.seg_token_idx = None
        self.aff_token_idx = None
        self.distributed = False
        
        # 验证配置
        self._validate()
    
    def _validate(self):
        """验证配置参数"""
        if self.precision not in ["fp32", "bf16", "fp16"]:
            raise ValueError(f"precision must be one of ['fp32', 'bf16', 'fp16'], got {self.precision}")
        
        if self.conv_type not in ["llava_v1", "llava_llama_2"]:
            raise ValueError(f"conv_type must be one of ['llava_v1', 'llava_llama_2'], got {self.conv_type}")
    
    def get_deepspeed_config(self):
        """
        生成 DeepSpeed 配置字典
        
        Returns:
            dict: DeepSpeed 配置
        """
        ds_config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.grad_accumulation_steps,
            "fp16": {
                "enabled": self.precision == "fp16",
            },
            "bf16": {
                "enabled": self.precision == "bf16",
            },
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": int(self.reduce_bucket_size),
                "allgather_bucket_size": int(self.allgather_bucket_size),
            },
        }
        
        # 如果不使用分层学习率，添加优化器和调度器配置
        if not self.use_layerwise_lr:
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
                    "total_num_steps": self.epochs * self.steps_per_epoch,
                    "warmup_min_lr": self.warmup_min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                    "warmup_type": self.warmup_type,
                },
            }
        # 如果使用分层学习率，不在配置中指定优化器和调度器
        # DeepSpeed 会使用我们传入的参数组和手动创建的优化器/调度器
        
        return ds_config
    
    def get_lora_config(self):
        """
        生成 LoRA 配置对象
        
        Returns:
            LoraConfig 或 None: 如果 lora_r > 0 返回 LoraConfig，否则返回 None
        """
        if self.lora_r <= 0:
            return None
        
        from peft import LoraConfig
        
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
