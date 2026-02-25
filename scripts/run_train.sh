#!/bin/bash
# 在仓库根目录运行
# 用法: ./scripts/run_train.sh fsdp
#    或 bash ./scripts/run_train.sh ds
# 不指定则默认ds

# ================================ 环境配置 ================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="8.0"

# 离线模式（避免下载模型/数据集）
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 禁用Inductor多进程编译，提升训练稳定性（非调试专属，训练也建议保留）
export TORCHINDUCTOR_NO_PARALLEL_COMPILE=1

# ================================ 训练参数 ================================

COMMON_ARGS=(
  --vision_pretrained ../pretrained/sam_vit_h_4b8939.pth
  --log_dir ../runs/joint-aff-debug/
  --dataset_dir ../datasets/merged1-2-3/
  --qwen_model ../pretrained/Qwen/Qwen3-VL-2B-Instruct
)

# ================================ 选择 fsdp?ds ================================

BACKEND="${1:-fsdp}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ "$BACKEND" == "fsdp" ]]; then
  echo "Launching FSDP training (torchrun, nproc_per_node=$NGPU)"
  torchrun \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NGPU" \
    train_fsdp.py \
    "${COMMON_ARGS[@]}"
elif [[ "$BACKEND" == "ds" ]]; then
  echo "Launching DeepSpeed training"
  deepspeed \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train_ds.py \
    "${COMMON_ARGS[@]}"
else
  echo "Usage: $0 [fsdp|ds]" >&2
  echo "  fsdp - PyTorch FSDP (train_fsdp.py)" >&2
  echo "  ds   - DeepSpeed (train_ds.py)" >&2
  exit 1
fi