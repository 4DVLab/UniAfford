# 数据集格式与使用说明

本文档说明 2D–3D UniAfford 使用的数据集组织方式。该文件是自包含的，可以在发布数据集时直接作为数据集包内的 `README.md` 使用。

## 总览

数据集按物体类别 `obj_type` 组织。每个物体类别目录下可以同时包含文本指令、2D 图像标注和 3D 点云标注。跨模态关联不默认依赖相同 ID，而是通过 `Instruction.csv` 中的 `img_id` / `pc_id` 显式绑定。

```text
dataset_root/
├── spoon/
│   ├── Instruction.csv
│   ├── Image/
│   │   ├── rgb/
│   │   │   ├── spoon_1.png
│   │   │   └── spoon_2.png
│   │   └── mask/
│   │       ├── grasp/
│   │       │   ├── spoon_1_grasp.png
│   │       │   └── spoon_2_grasp.png
│   │       └── contain/
│   │           └── spoon_1_contain.png
│   └── PointCloud/
│       ├── spoon_1.csv
│       └── spoon_2.csv
├── metadata.json
├── train.json
├── val.json
└── test.json
```

## 命名规范

- `obj_type`：物体语义类别名，可以包含空格，例如 `spoon`、`mug`。
- `aff_type`：affordance 语义类别名，可以包含空格，例如 `grasp`、`contain`。
- `Image/rgb/`：RGB 文件名为 `{obj_type}_{img_id}.png`。
- `Image/mask/<aff_type>/`：mask 文件名为 `{obj_type}_{img_id}_{aff_type}.png`。
- `PointCloud/`：点云文件名为 `{obj_type}_{pc_id}.csv`。

项目工具在转换和保存数据时会倾向于将 `obj_type` 与 `aff_type` 规范化为小写。加载器会尽量兼容历史数据中的大写或驼峰前缀，但新发布的数据集建议统一使用小写语义名。

## Instruction.csv

每个物体类别目录下可包含一个 `Instruction.csv`，固定六列：

```csv
ins,obj_type,aff_type,id,img_id,pc_id
```

字段含义：

| 字段 | 含义 |
|---|---|
| `ins` | 自然语言指令或查询。 |
| `obj_type` | 物体类别，例如 `spoon`。 |
| `aff_type` | affordance 类别，例如 `grasp`。 |
| `id` | instruction 自身 ID。 |
| `img_id` | 可选，关联的 RGB/mask 样本 ID。 |
| `pc_id` | 可选，关联的点云样本 ID。 |

`img_id` 与 `pc_id` 可以同时存在、只存在其中一个，或都为空。因此同一数据集中可以混合 image-only、point-cloud-only 与多模态配对样本。

## 2D 图像标注

RGB 图像位于：

```text
<obj_type>/Image/rgb/
```

Affordance mask 按 `aff_type` 分目录存放：

```text
<obj_type>/Image/mask/<aff_type>/
```

示例：

```text
spoon/Image/rgb/spoon_1.png
spoon/Image/mask/grasp/spoon_1_grasp.png
spoon/Image/mask/contain/spoon_1_contain.png
```

mask 按单通道图像读取，可以是二值 `0/255`，也可以是连续热力图。训练与验证代码会在指标计算前做必要的归一化和阈值处理。

## 3D 点云标注

点云 CSV 位于：

```text
<obj_type>/PointCloud/
```

前三列固定为坐标，后续每一列对应一个 affordance mask：

```csv
x,y,z,grasp,contain
0.012,0.031,0.552,0.88,0.01
0.018,0.029,0.548,0.11,0.65
```

规则：

- 前三列必须为 `x,y,z`。
- 第四列起，每个列名都是一个 `aff_type`。
- 每个 affordance 列中的值是逐点概率（0~1）。
- 同一个点云可以通过多个 affordance 列保存多个标注。

## Split 文件

`train.json`、`val.json`、`test.json` 使用相同结构：

```json
{
  "Instruction": {
    "spoon": {
      "grasp": [1, 2, 3],
      "contain": [4, 5]
    }
  },
  "Image": {
    "spoon": {
      "grasp": [101, 102],
      "contain": [104]
    }
  },
  "PointCloud": {
    "spoon": {
      "grasp": [301, 302],
      "contain": [304]
    }
  }
}
```

结构为：

```text
<modality> -> <obj_type> -> <aff_type> -> [id, ...]
```

说明：

- `Instruction` 中的 ID 对应 `Instruction.csv` 的 `id` 字段。
- `Image` 中的 ID 对应 `img_id`，会在 `Image/rgb/` 和 `Image/mask/<aff_type>/` 下解析。
- `PointCloud` 中的 ID 对应 `pc_id`，会在点云 CSV 中读取对应的 affordance 列。
- `metadata.json` 由项目工具生成时，会记录 split 比例、随机种子、样本数和各模态统计信息。

## 在代码中加载

在仓库根目录下可通过：

```python
from utils.base_dataset import JointDataset

dataset = JointDataset(
    dataset_root="/path/to/dataset_root",
    split_file="train.json",
)
samples = dataset.load_all_data().samples
```

训练时，原始样本会进一步由 `utils.dataset.JointAffordanceTorchDataset` 转换为模型输入，包括文本 token、Qwen-VL 图像输入、2D mask、点云与 3D mask。

## 生成 Split

已有完整目录后，可以使用项目工具生成 split：

```bash
python utils/data_process/create_split.py --help
```

常见模式：

- 标准训练评估：生成 `train.json` / `val.json` / `test.json`。
- 全库训练：设置 `train_ratio=1.0, val_ratio=0.0, test_ratio=0.0`，只生成训练 split 与 `metadata.json`。

## 可视化

如果只想快速检查单个样本，可使用代码仓库中的 `utils/base_dataset.py -s` 基础可视化能力。若需要批量导出 2D 叠图、3D 点云渲染图，请参考代码仓库中的 `docs/rendering.md`。
