
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

分割由 `utils/data_process/create_split.py` 中的 `SplitManager` 执行，生成四个文件：`metadata.json`、`train.json`、`val.json`、`test.json`。`JointDataset` 通过 `split_file='train.json'` 等形式加载对应分割。

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

数据分割由 `utils/data_process/create_split.py` 中的 `SplitManager.split()` 执行，核心流程如下（默认从磁盘扫描 IDs，也可切换到内存 IDs 来源）：

1. **构建分组（按 `obj_type + aff_type`）**
   - **图文分组**：遍历 `Instruction`，使用同 id 的 `Image` 做配对，得到 `(ins_id, img_id)`。
   - **点云分组**：遍历 `PointCloud.aff_mask_dict`，按 aff 拆分为 `((pc_id, mask_idx),)` 的中间结构（`mask_idx` 仅兼容旧数据格式，后续以 aff 语义为准）。

2. **组内处理**
   - 每个 `(obj_type, aff_type)` 组先去重、再打乱。
   - 可选采样：
     - `sample_rate`：按比例采样；
     - `min_sample_per_group`：每组最低保留；
     - `max_sample_per_group`：每组上限裁剪。

3. **按比例切分 train/val/test**
   - 对每个分组独立按 `train_ratio / val_ratio / test_ratio` 切分。
   - 当 `val/test` 启用时，代码会尽量保证每组 `val/test` 至少有基本样本数（当前实现中最小目标为 5），并在样本不足时做回退调整。
   - 过小分组（当前实现中 `n_total <= 20`）会被跳过，避免极小样本噪声影响评估。

4. **写出分割文件**
   - 输出 `train.json / val.json / test.json`，结构为：
     - `Instruction: {obj: {aff: [ins_id, ...]}}`
     - `Image: {obj: {aff: [img_id, ...]}}`
     - `PointCloud: {obj: {aff: [pc_id, ...]}}`
   - 同时写出 `metadata.json`，包含：
     - `train_sample / val_sample / test_sample / total_sample`
     - 分割比例、随机种子
     - 各 split 下按模态与 `obj-aff` 的计数统计（`obj_aff_count_by_split`）。

> 说明：`JointDataset(split_file='train.json'|'val.json'|'test.json')` 会按对应 split 独立加载数据，并在 `pair_samples()` 中统一按 aff 语义对齐三模态样本。

### Train-only 分割（整库作为训练集）

当前 `SplitManager.split()` 支持将整个数据集只切分为训练集：

- `train_ratio=1.0, val_ratio=0.0, test_ratio=0.0`
- 或者等价地让 `val/test` 为 `0`

在该模式下：

- 不再强制 `val/test` 的最小样本数
- 不再触发 holdout 场景的小样本跳过逻辑
- 仅写出 `train.json`（以及 `metadata.json`），不会额外输出 `val.json/test.json`

### data_process 与分割文件

- `utils/data_process/external_datasets_processing.py`
  - 在 `load_and_save` 处理完成后，会默认调用 `create_split.save_split_from_disk(..., train=1,val=0,test=0)` 生成 train-only 分割文件。
- `utils/data_process/merge_datasets.py`
  - 合并后会重新加载并重写数据（重新编号、排序）。
  - 可选 `--save_split` 生成 train-only 分割文件。

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

    "obj_type": [str, ...],   # 样本元信息（记录/分析）
    "aff_type": [str, ...],
}
```

### 软路由（Router）分支策略

当前模型在 `JointAffordanceModel` 中采用 **text/img/pc 三路软路由**，用于让 `MLLM last_hidden_state` 自适应选择下游分支：

1. **路由预测**
   - 对每个 token 的隐藏向量 `h[b, l, :]` 通过 `route_head` 得到 `route_logits[b, l, 3]`；
   - `softmax` 得到 `route_probs`，三类分别对应 `text / img / pc`。

2. **分支专用头 + 软聚合**
   - 所有 token 同时经过 `img_branch_head` 与 `pc_branch_head`；
   - 用 `route_probs[..., img]`、`route_probs[..., pc]` 对 token 级特征做加权平均，得到样本级 `img_emb` 与 `pc_emb`；
   - `img_emb` 输入 `image_decoder`，`pc_emb` 输入 `point_decoder`。

3. **占位 token 回写（保持顺序）**
   - 仅用于输出记录：按 `hard_route=argmax(route_logits)` 保持原 token 顺序回写 `token_ids`；
   - 路由到 img 的位置替换为 `<img_aff>`，路由到 pc 的位置替换为 `<pc_aff>`；
   - 不再依赖 `<img_obj_aff>/<pc_obj_aff>` 这类按类别展开 token。

4. **路由监督与稳定性**
   - 训练文本中显式保留 `<img_aff>/<pc_aff>` 作为 token-level 路由标签；
   - 损失由 `L_route`（路由 CE）与 `L_bal`（负载均衡）提供，避免路由塌缩到单一路径。
   - 语言 CE 会忽略 `<img_aff>/<pc_aff>` 标签位，这些位置主要由 router + 下游分割损失学习。

> 说明：Router 负责“分流与聚合”，具体 2D/3D 解码仍在各自 decoder 中执行；这种解耦更利于维护与调试。

### 与旧方案对比（按 token_id 提取 seg token）

旧方案（token_id 提取）：

- **做法**：先在词表注册 `<img_obj_aff>/<pc_obj_aff>` 等功能 token，再在输出 token_ids 中查找这些 token 的位置提取 hidden state。
- **优点**：
  - 规则直观、可解释性强（“看到某 token 就走某分支”）；
  - 早期调试方便（可直接对齐 token 字符串）。
- **缺点**：
  - 强依赖固定 token 设计，扩展到新对象/新语义需要持续注入 token；
  - 对开放词汇泛化较弱（未注册 token 无法路由）；
  - 训练容易变成“背 token 模板”而非学习跨模态语义分流。

当前方案（软路由）：

- **做法**：不再依赖 `<img_obj_aff>/<pc_obj_aff>` 的字符串匹配，直接由 `last_hidden_state -> route_head` 学习 token 级分流。
- **优点**：
  - 不绑定固定功能 token，任意语义 token 都可被路由到 2D/3D 分支；
  - 更利于跨对象、跨表达方式的泛化；
  - 路由决策可结合上下文（自注意力后的 hidden state）而非单 token 规则。
- **缺点**：
  - 训练稳定性更依赖损失设计（`L_route`/`L_bal`）与日志监控；
  - 可解释性相对下降，需要额外记录路由统计与 token 分析。

### Router 如何知道 `<Aff>` 位置

`<Aff>` 位置信号来自数据构造与标签对齐，而不是推理时字符串匹配：

1. `utils/dataset.py::_build_text()` 在 assistant 答案中写入 `<img_aff>/<pc_aff>` 占位；
2. `_build_qwen_inputs()` 仅对 assistant 片段赋监督标签（其余位置为 `IGNORE_INDEX`）；
3. 模型里 `_build_route_mask_from_labels()` 使用与验证一致的 next-token 对齐规则（标签位置 `p` 对应预测位置 `p-1`）构造路由监督位置；
4. `L_route` 在这些位置监督 text/img/pc 路由类别，`L_bal` 约束分布稳定；同时 CE 忽略 `<img_aff>/<pc_aff>` 标签位，避免语言头硬性学习占位 token。

### 模型输出（`JointAffordanceModel.forward`）

```python
# 输出字典
{
    "image_logits": Tensor[B, H, W] | None,   # 2D 分割 logits
    "point_logits": Tensor[B, N] | None,      # 3D 分割 logits
    "token_ids": Tensor[B, L] | None,         # 路由后 token ids（img/pc 位置回写为 <img_aff>/<pc_aff>）
    "ce_loss": Tensor | None,                  # 语言建模交叉熵（传入 labels 时）
    "route_logits": Tensor[B, L, 3] | None,   # text/img/pc 路由 logits
    "route_probs": Tensor[B, L, 3] | None,    # 路由概率
    "hidden_states": Tensor | None,           # 仅 return_hidden_states=True 时
    "output": MLLMOutput | None,               # 仅 return_mllm_output=True 时
}
```

# Cite
