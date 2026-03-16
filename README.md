
# QuickStart
## Environments
## Run

# Train

# Dataset

## 数据集目录结构

按 `obj_type` 组织，典型结构如下：

```bash
dataset_root/
├── Spoon/
│   ├── Instruction.csv  # instruction, obj, aff, img_id
│   ├── Image/
│   │   ├── rgb/  # 假设 Spoon_1 同时包含grasp、contain的标注，Spoon_2只有grasp，and Spoon_3只有contain
│   │   │   ├── Spoon_1.png
│   │   │   ├── Spoon_2.png
│   │   │   └── Spoon_3.png
│   │   └── mask/
│   │       ├── grasp/
│   │       │   ├── Spoon_1_grasp.png  # 命名： obj_id_aff.png
│   │       │   └── Spoon_2_grasp.png
│   │       └── contain/
│   │           ├── Spoon_1_contain.png
│   │           └── Spoon_3_contain.png
│   └── PointCloud/
│       ├── Spoon_1.csv  # x, y, z, aff1的分布概率, aff2(如有), ...
│       ├── Spoon_2.csv
│       └── Spoon_3.csv
├── Mug/...
├── metadata.json  # 分割统计信息（比例、样本数、随机种子等）
├── train.json      # 训练集分割
├── val.json        # 验证集分割
└── test.json       # 测试集分割
```

说明：

- `Ins` 和 `Image` 共用 id 编码，其是从 `RAGNet` 的数据集中获取； `Point` 则单独编码，其是从 `PIADv2` 和 `AGPIL` 数据集中获取的。
- `Image` 的 mask 按 `aff_type` 分子目录保存，`Ins` 和 `Point` 的则将 `aff_type` 写入  `obj_type` 对应的行内。

## 分割配置文件

分割由 `SplitManager` 执行，生成四个文件：`metadata.json`、`train.json`、`val.json`、`test.json`。`JointDataset` 通过 `split_file='train.json'` 等形式加载对应分割。

`train.json` / `val.json` / `test.json` 各自为独立结构，示例（以 train.json 为例）：

```json
{
  "Instruction": {
    "Spoon": {
      "grasp": [1, 2, 3, 4, 5],
      "contain": [1, 2, 3],
      "lift": [1, 2]
    }
  },
  "Image": {
    "Spoon": {
      "grasp": [101, 102, 103],
      "contain": [101, 104]
    }
  },
  "PointCloud": {
    "Spoon": {
      "grasp": [301, 302, 303],
      "contain": [301, 304]
    }
  }
}
```



- `Instruction`: `{obj_type: {aff_type: [ins_id, ...]}}`
- `Image`: `{obj_type: {aff_type: [img_id, ...]}}`（或旧格式 `[[img_id, img_mask_idx], ...]`）
- `PointCloud`: `{obj_type: {aff_type: [pc_id, ...]}}`（与 Image 一致，仅保存 id）

说明：

- 统一使用 `aff_type` 语义，不再依赖索引定位 mask
- `Image` 通过 `img_id` 取样本后，用 `image.get_mask_by_aff(aff_type)` 获取 GT
- `PointCloud` 通过 `pc_id` 取样本后，用 `pc.get_mask_by_aff(aff_type)` 获取 GT

## 数据分割策略

`SplitManager.split()` 先构建分组，再按比例切分：

- 图文对：按 `Instruction.aff_type` 与同 id 的 `Image` 精确配对
- 点云对：按点云 `aff_mask_dict` 的 aff_type 拆成 `(pc_id, mask_idx)` 列表（mask_idx 仅用于兼容旧格式，配对时以 aff_type 为准）
- 每个 `(obj_type, aff_type)` 分组内去重、打乱、再切分

小样本组保护：

- 当分组样本数较小且 `val/test` 比例大于 0 时，至少为 `val/test` 分配 5 条（可行时）
- 防止 `val/test` 长期为 0 或极少样本导致评估失真

# Pipeline

## 训练输入数据管道

###  JointDataSample.get_data()→ _build_sample() 单样本输出

`JointDataSample.get_data()` 返回原始数据，`utils/dataset.py` 中 `JointAffordanceTorchDataset._build_sample()` 将其转为模型可用的单样本字典：

```python
# 单样本输出（CPU 张量，可选字段依模态存在性而定）
{
    # 元信息
    "sample_id": int,           # 样本全局 id
    "obj_type": str,            # 物体类型，如 "Chair"
    "aff_type": str,            # affordance 类型，如 "sit"

    # MLLM 文本与 Qwen3-VL 视觉
    "input_ids": Tensor[1, L],           # 文本 token ids（含图像占位 token），int64
    "labels": Tensor[1, L],              # 语言建模标签，非监督位为 IGNORE_INDEX，int64
    "pixel_values": Tensor[N, C, H, W],  # Qwen3-VL 视觉输入（可选），mllm_precision
    "image_grid_thw": Tensor[N, 3],      # Qwen3-VL 图像网格 (T,H,W)（可选），int64

    # 2D 分割（有图像时存在）
    "images": Tensor[3, H, W],           # 2D 分支输入图像，image_precision
    "img_gt": Tensor[H, W],              # 2D 监督掩码，image_precision

    # 3D 分割（有点云时存在）
    "point_clouds": Tensor[N, 3],        # 3D 分支输入点云，point_precision
    "pc_gt": Tensor[N],                  # 3D 监督掩码，point_precision
}
```

### joint_affordance_collate_fn 批处理输出

`utils/dataset.py` 中 `joint_affordance_collate_fn()` 将单样本列表批处理为：

```python
# 批处理输出（B = batch_size）
{
    # MLLM 输入
    "input_ids": Tensor[B, L],           # 文本 token ids，pad 至 batch 内最大长度
    "labels": Tensor[B, L],              # 语言建模标签，pad 位为 IGNORE_INDEX
    "attention_mask": Tensor[B, L],      # 有效 token 掩码（非 pad 为 True）
    "pixel_values": Tensor[B, 3, H, W],  # Qwen3-VL 视觉输入（缺失样本用占位图回填）
    "image_grid_thw": Tensor[B, 3],       # Qwen3-VL 图像网格

    # 2D 分割输入
    "images": Tensor[B, 3, H, W],        # 2D 输入图像（统一 padding 至 output_image_size）
    "img_gt_tensor": Tensor[B, H, W],     # 2D 监督掩码（无效样本为 0）
    "original_size_list": [(h, w), ...],  # 各样本原始 GT 尺寸
    "img_valid_mask": Tensor[B],         # bool，是否有有效 2D 监督

    # 3D 分割输入
    "point_clouds": Tensor[B, N, 3],     # 点云（无效样本用 0 填充）
    "pc_gt_tensor": Tensor[B, N],        # 3D 监督掩码（无效样本为 0）
    "pc_valid_lengths": Tensor[B],      # 各样本有效点数，0 表示无效

    # 元信息（列表，与 batch 一一对应）
    "sample_id": [int, ...],
    "obj_type": [str, ...],
    "aff_type": [str, ...],
}
```

## 模型输入输出管道

###  模型输入（`JointAffordanceModel.forward`）

```python
# 输入字典（与 collate_fn 输出一致）
{
    "input_ids": Tensor[B, L],
    "labels": Tensor[B, L] | None,
    "attention_mask": Tensor[B, L] | None,
    "pixel_values": Tensor[B, 3, H, W] | None,
    "image_grid_thw": Tensor[B, 3] | None,

    "images": Tensor[B, 3, H, W],
    "original_size_list": [(h, w), ...],
    "img_valid_mask": Tensor[B],
    "img_gt_tensor": Tensor[B, H, W] | None,

    "point_clouds": Tensor[B, N, 3],
    "pc_valid_lengths": Tensor[B],
    "pc_gt_tensor": Tensor[B, N] | None,

    "obj_type": [str, ...],   # 用于解析 obj-aff token
    "aff_type": [str, ...],
}
```

### 模型输出（`JointAffordanceModel.forward`）

```python
# 输出字典
{
    "image_logits": Tensor[B, H, W] | None,   # 2D 分割 logits
    "point_logits": Tensor[B, N] | None,      # 3D 分割 logits
    "token_ids": Tensor[B, L] | None,         # 语言模型 argmax(logits)
    "ce_loss": Tensor | None,                  # 语言建模交叉熵（传入 labels 时）
    "hidden_states": Tensor | None,           # 仅 return_hidden_states=True 时
    "output": MLLMOutput | None,               # 仅 return_mllm_output=True 时
}
```

# Cite
