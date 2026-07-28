#!/bin/bash
# LISA 2D-3D UniAfford Demo 启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径（请根据实际情况修改）
MODEL_PATH="./runs/lisa/ckpt_model"
SAM_PATH="./pretrained/sam_vit_h_4b8939.pth"

# 启动参数
DEVICE="cuda:0"
PRECISION="fp16"
IMAGE_SIZE=1024
NUM_POINTS=2048
PORT=7860

# 启动应用
python app.py \
    --model_path $MODEL_PATH \
    --vision_pretrained $SAM_PATH \
    --device $DEVICE \
    --precision $PRECISION \
    --image_size $IMAGE_SIZE \
    --num_points $NUM_POINTS \
    --port $PORT \
    --share  # 如果不需要公共链接，删除此行
