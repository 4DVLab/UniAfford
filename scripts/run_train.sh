#!/bin/bash
# 在仓库根目录运行
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TORCH_CUDA_ARCH_LIST="8.0"

# 离线模式（避免下载模型/数据集）
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 禁用Inductor多进程编译，提升训练稳定性（非调试专属，训练也建议保留）
export TORCHINDUCTOR_NO_PARALLEL_COMPILE=1

deepspeed \
--master_addr=127.0.0.1 \
--master_port=29501 \
train_new.py \
--vision_pretrained ../pretrained/sam_vit_h_4b8939.pth \
--log_dir ../runs/joint-aff-exp-01/ \
--dataset_dir ../datasets/merged1-2-3/ \
--qwen_model ../pretrained/Qwen/Qwen3-VL-8B-Instruct