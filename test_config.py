from configs import TrainingConfig

# 测试配置类
config = TrainingConfig()
print('✓ 配置导入成功')
print(f'  实验名称: {config.exp_name}')
print(f'  学习率: {config.lr}')
print(f'  训练轮数: {config.epochs}')

# 测试 DeepSpeed 配置生成
ds_config = config.get_deepspeed_config()
print('\n✓ DeepSpeed 配置生成成功')
print(f'  - batch_size: {ds_config["train_micro_batch_size_per_gpu"]}')
print(f'  - grad_accumulation: {ds_config["gradient_accumulation_steps"]}')
print(f'  - zero_stage: {ds_config["zero_optimization"]["stage"]}')
print(f'  - fp16: {ds_config["fp16"]["enabled"]}')
print(f'  - gradient_clipping: {ds_config["gradient_clipping"]}')

# 测试 LoRA 配置生成
lora_config = config.get_lora_config()
if lora_config:
    print('\n✓ LoRA 配置生成成功')
    print(f'  - lora_r: {lora_config.r}')
    print(f'  - lora_alpha: {lora_config.lora_alpha}')
    print(f'  - lora_dropout: {lora_config.lora_dropout}')
    print(f'  - target_modules: {lora_config.target_modules}')
else:
    print('\n✓ LoRA 已禁用 (lora_r=0)')

# 测试自定义配置
print('\n测试自定义配置:')
custom_config = TrainingConfig(
    exp_name="test_experiment",
    lr=0.001,
    use_layerwise_lr=True,
    lora_r=16,
    zero_stage=3,
)
print(f'  实验名称: {custom_config.exp_name}')
print(f'  学习率: {custom_config.lr}')
print(f'  LLM学习率: {custom_config.llm_lr}')
print(f'  2D视觉学习率: {custom_config.vision_2d_lr}')
print(f'  3D视觉学习率: {custom_config.vision_3d_lr}')
print(f'  LoRA秩: {custom_config.lora_r}')
print(f'  ZeRO阶段: {custom_config.zero_stage}')

print('\n✓ 所有测试通过！')
