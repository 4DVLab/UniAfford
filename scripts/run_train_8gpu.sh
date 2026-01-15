#!/bin/bash
# 8卡 V100 分布式训练启动脚本 (不使用 DeepSpeed)

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

# 训练参数
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NUM_GPUS=8

# 模型和数据路径 (请根据实际情况修改)
VERSION="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
VISION_PRETRAINED="PATH_TO_SAM_ViT-H"
DATASET_DIR="./dataset"
LOG_DIR="./runs"
EXP_NAME="lisa_8gpu_ddp"

# 训练超参数
BATCH_SIZE=2
GRAD_ACCUM_STEPS=10
EPOCHS=10
STEPS_PER_EPOCH=500
LR=0.0003

# V100 使用 fp16 (不支持 bf16)
PRECISION="fp16"

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    try_train.py \
    --version ${VERSION} \
    --vision_pretrained ${VISION_PRETRAINED} \
    --dataset_dir ${DATASET_DIR} \
    --log_base_dir ${LOG_DIR} \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --epochs ${EPOCHS} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --lr ${LR} \
    --precision ${PRECISION} \
    --gradient_checkpointing \
    --train_mask_decoder \
    --use_mm_start_end \
    --auto_resume \
    --use_point_cloud
