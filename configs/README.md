# 训练配置说明

## 快速开始

### 基本使用

```python
from configs import TrainingConfig

# 使用默认配置
config = TrainingConfig()

# 自定义配置
config = TrainingConfig(
    exp_name="my_experiment",
    epochs=100,
    lr=0.001,
    batch_size=2,
)
```

## 主要配置项

### 基础配置
- `local_rank`: 分布式训练节点排名 (默认: 0)
- `version`: 预训练模型路径 (默认: "../pretrained/llava-llama-2-13b-chat-lightning-preview")
- `precision`: 训练精度，可选 "fp32", "bf16", "fp16" (默认: "fp16")

### 模型配置
- `image_size`: 输入图像尺寸 (默认: 1024)
- `model_max_length`: 最大序列长度 (默认: 512)
- `vision_tower`: 视觉编码器路径 (默认: "../pretrained/clip-vit-large-patch14")
- `vision_pretrained`: SAM 预训练权重路径 (默认: "../pretrained/sam_vit_h_4b8939.pth")

### 数据配置
- `dataset_dir`: 数据集目录 (默认: "../datasets/merged1-2-3/")
- `log_base_dir`: 日志基础目录 (默认: "./runs")
- `exp_name`: 实验名称 (默认: "lisa")
- `num_points`: 点云点数 (默认: 2048)
- `train_ratio`: 训练集比例 (默认: 0.7)
- `val_ratio`: 验证集比例 (默认: 0.15)
- `test_ratio`: 测试集比例 (默认: 0.15)

### 训练配置
- `epochs`: 训练轮数 (默认: 250)
- `steps_per_epoch`: 每轮步数 (默认: 100)
- `batch_size`: 批次大小 (默认: 1)
- `grad_accumulation_steps`: 梯度累积步数 (默认: 10)
- `workers`: 数据加载进程数 (默认: 4)

### 优化器配置
- `lr`: 基础学习率 (默认: 0.0003)
- `beta1`: Adam beta1 (默认: 0.9)
- `beta2`: Adam beta2 (默认: 0.95)

### 分层学习率（可选）
- `use_layerwise_lr`: 是否启用分层学习率 (默认: False)
- `llm_lr`: LLM 学习率 (默认: lr * 0.1)
- `vision_2d_lr`: 2D 视觉学习率 (默认: lr)
- `vision_3d_lr`: 3D 点云学习率 (默认: lr)

### 损失权重
- `ce_loss_weight`: 交叉熵损失权重 (默认: 1.0)
- `dice_loss_weight`: Dice 损失权重 (默认: 0.5)
- `bce_loss_weight`: BCE 损失权重 (默认: 2.0)

### LoRA 配置
- `lora_r`: LoRA 秩 (默认: 8)
- `lora_alpha`: LoRA alpha (默认: 16)
- `lora_dropout`: LoRA dropout (默认: 0.05)
- `lora_target_modules`: 目标模块 (默认: "q_proj,v_proj")

## 使用示例

### 示例 1: 快速实验
```python
config = TrainingConfig(
    exp_name="quick_test",
    epochs=10,
    steps_per_epoch=50,
)
```

### 示例 2: 高精度训练
```python
config = TrainingConfig(
    exp_name="high_precision",
    precision="bf16",
    epochs=500,
    lr=0.0001,
    use_layerwise_lr=True,
    llm_lr=0.00001,
    vision_2d_lr=0.0001,
    vision_3d_lr=0.0001,
)
```

### 示例 3: 大批次训练
```python
config = TrainingConfig(
    exp_name="large_batch",
    batch_size=1,
    grad_accumulation_steps=32,  # 有效批次大小 = 32
    lr=0.001,
)
```

### 示例 4: 自定义数据集
```python
config = TrainingConfig(
    exp_name="custom_data",
    dataset_dir="/path/to/your/dataset",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    num_points=4096,
)
```

## 注意事项

1. **批次大小限制**: 当前 `batch_size` 建议设置为 1，通过 `grad_accumulation_steps` 增加有效批次大小
2. **精度选择**: 
   - `fp16`: 速度快，显存占用少，但可能不稳定
   - `bf16`: 平衡速度和稳定性（推荐）
   - `fp32`: 最稳定，但速度慢，显存占用大
3. **分层学习率**: 启用后可以为不同模块设置不同学习率，通常 LLM 使用较小学习率
4. **自动恢复**: `auto_resume=True` 时会自动从最新 checkpoint 恢复训练
