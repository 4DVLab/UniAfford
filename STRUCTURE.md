# 项目结构与功能说明

面向后续重构整理当前仓库的文件结构与功能定位，基于 `e:\快速导航\实习\研究\2D-3D Joint Affordance\main` 现有代码。

## 目录结构一览

```
.
├─ train_ds.py                 # 主训练入口（DeepSpeed + LISA + LoRA + 分层学习率）
├─ validate.py                 # 推理/评估脚本（加载 checkpoint 评估 2D/3D 指标，可保存预测）
├─ app.py                      # Gradio Demo（2D/3D 分割可视化）
├─ try.py                      # 实验/替代实现片段（优化版 mask 生成/forward，未接入主流程）
├─ configs/
│  ├─ __init__.py               # 导出 TrainingConfig
│  ├─ training_config.py        # 训练参数/DeepSpeed/LoRA 配置定义
│  └─ README.md                 # 配置说明（当前文件中文显示疑似编码问题）
├─ model/
│  ├─ LISA.py                   # LISA 主模型：LLaVA + SAM + PointNet++ 3D 分割
│  ├─ pointnet2_utils.py        # PointNet++ / 3D 分割组件
│  ├─ joint_affordance.py        # 早期/未完成的联合 affordance 模型（标注为未使用）
│  ├─ en_decoder.py             # 文本/图像/点云 encoder/decoder（部分占位/未完成）
│  ├─ qwen_aff.py               # 基于 Qwen3-VL-MOE 的 2D/3D 联合模型实验
│  ├─ llava/                    # LLaVA 相关代码（语言-视觉基础模型）
│  ├─ segment_anything/         # SAM 相关代码（2D 分割）
│  └─ qwen3_vl_moe/              # Qwen3-VL-MOE 模型代码
├─ utils/
│  ├─ base_dataset.py           # 数据集结构/加载/保存/划分（三模态样本构建）
│  ├─ dataset.py                # PyTorch Dataset + collate_fn + DatasetManager
│  ├─ metrics.py                # 2D/3D 指标与 TensorBoard 记录
│  ├─ calculator.py             # loss 与指标的数学计算
│  ├─ common.py                 # 常量、日志、通用工具
│  └─ data_process/
│     ├─ external_datasets_processing.py  # 外部数据集格式转换/整合
│     ├─ merge_datasets.py                # 合并多数据集（按 obj/模态）
│     ├─ resort_none_aff.py               # 根据 Instruction.csv 重新归类 aff
│     └─ rename.py                        # 重命名 obj/aff（含数据迁移）
├─ scripts/
│  ├─ run_train_8gpu.sh         # 8 卡训练脚本（torchrun）
│  ├─ run_app.sh                # 启动 Gradio Demo 脚本
│  └─ vscode_debug_launch.json  # VSCode 调试配置
├─ requirements.txt             # 训练依赖
├─ requirements_app.txt         # Demo 依赖
└─ README.md                    # TODO 列表（当前中文显示疑似编码问题）
```

## 主要入口脚本

- `train_ds.py`
  - 初始化 tokenizer 与 LISA 模型；添加 `[SEG]` / `[AFF]` token。
  - 支持 LoRA；支持分层学习率（LLM / 2D 视觉 / 3D 点云 / 其他）。
  - 使用 `DatasetManager` 构建训练/验证集，`collate_fn` 处理多模态 batch。
  - 训练中计算：语言模型 CE、2D mask loss、3D mask loss、dummy_loss。
  - 支持 DeepSpeed 训练与断点续训；保存完整 DS checkpoint + 轻量权重。

- `validate.py`
  - 加载训练好的权重（适配 DeepSpeed inference）。
  - 基于 `DatasetManager` 构建测试集；计算 2D/3D 指标。
  - 可选保存预测结果：2D mask png、3D 点云 csv。

- `app.py`
  - Gradio 在线 Demo：图像/点云 + 文本提示 -> 分割可视化。
  - 内置 `LISAInferenceEngine` 完成推理与可视化。

- `try.py`
  - 试验性的优化版 `_generate_2d_masks/_generate_3d_masks/model_forward`。
  - 当前未被主训练/推理脚本引用，可视为草稿/备选实现。

## 核心模块说明

### model/
- `LISA.py`
  - 组合 LLaVA（语言+视觉）与 SAM（2D mask）和 PointNet++（3D mask）。
  - 通过 `[SEG]` token 抽取文本隐层特征，分别生成 2D/3D mask。
  - 2D：SAM prompt encoder + mask decoder；3D：PointCloud3DSegmentor。

- `pointnet2_utils.py`
  - PointNet++ 基础层与 3D 分割模块 `PointCloud3DSegmentor`。
  - 包含多尺度采样、特征传播、引导注意力等。

- `joint_affordance.py`
  - 标注为“未完成/未使用”的旧版联合模型。

- `en_decoder.py`
  - Text/Image/Point 编码器与解码器的原型实现。
  - 含 Qwen 文本编码器包装，部分 decoder 为占位或未完成。

- `qwen_aff.py`
  - 基于 Qwen3-VL-MOE 的 2D/3D 联合模型实验。

- `llava/`, `segment_anything/`, `qwen3_vl_moe/`
  - 外部模型代码拷贝/改造，作为底层依赖。

### utils/
- `base_dataset.py`
  - 定义三模态基础类：`Instruction` / `Image` / `PointCloud`。
  - `JointDataset` 负责读取数据集、划分 train/val/test、样本组装。
  - 典型数据结构：
    - `Instruction.csv`：`ins,obj_type,aff_type,id`
    - `Image/rgb/<obj>_<id>.png`
    - `Image/mask/<aff>/<obj>_<id>_<aff>.png`
    - `PointCloud/<obj>_<id>.csv`（x,y,z + 各 aff mask 列）
    - `info.json`：统计信息/计数

- `dataset.py`
  - `DatasetManager`：封装 `JointDataset` 并提供 PyTorch Dataset。
  - `BaseDataset`/`TrainDataset`：样本预处理、缓存、随机采样。
  - `collate_fn`：统一文本/图像/点云的 batch 对齐与 padding。

- `metrics.py`
  - `MetricsTracker` 统一管理 loss 与 2D/3D 指标。
  - 2D：gIoU/cIoU；3D：MAE/AUC/aIoU/SIM。
  - `TensorBoardLogger` 记录训练/验证指标。

- `calculator.py`
  - 2D/3D loss（BCE/Dice）与 3D 指标计算工具。

- `common.py`
  - token 常量、日志、辅助函数（如 `dict_to_cuda`）。

### utils/data_process/
- `external_datasets_processing.py`：将多种外部数据集转换为统一格式。
- `merge_datasets.py`：合并多个数据集到统一根目录。
- `resort_none_aff.py`：依据 `Instruction.csv` 重排 `None` aff 数据。
- `rename.py`：重命名 obj/aff，并迁移数据。

## 训练/评估流程概览

1. `train_ds.py` -> `TrainingConfig` -> 初始化 tokenizer/model。
2. `DatasetManager` 构建 PyTorch Dataset + `collate_fn`。
3. DeepSpeed 初始化训练（支持 LoRA、分层学习率）。
4. 每步计算 CE/2D/3D loss + 评估指标。
5. 保存完整 DS checkpoint + 轻量权重。
6. `validate.py` 加载权重，计算 2D/3D 指标，可保存预测结果。

## 配置与脚本

- `configs/training_config.py`
  - 训练超参、DeepSpeed、LoRA、分层学习率等集中管理。

- `scripts/run_train_8gpu.sh`
  - torchrun 8 卡训练示例（当前脚本内引用 `try_train.py`）。

- `scripts/run_app.sh`
  - 启动 Gradio Demo 示例。

## 依赖

- `requirements.txt`：训练依赖（deepspeed/torch/transformers 等）。
- `requirements_app.txt`：Demo 依赖（gradio/trimesh/open3d 等）。

## 重构准备：已观察到的不一致/待确认点

- `app.py` 中引用 `utils.utils`，当前仓库不存在该模块（可能应为 `utils.common`）。
- `scripts/run_train_8gpu.sh` 调用 `try_train.py`，当前仓库未发现该文件。
- `joint_affordance.py` / `en_decoder.py` 多处标注未完成，可能属于历史/实验代码。
- `configs/README.md` 与根目录 `README.md` 中文显示疑似编码问题。

如需进一步细化（例如数据集字段、训练/推理参数约束、模型依赖关系图），可以在此文档基础上继续扩展。
