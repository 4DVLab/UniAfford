# UniAfford: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception

<p align="center">
  <a href="README_zh.md">中文</a> |
  <a href="README.md">English</a>
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv-coming%20soon-ee4c2c.svg" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/🤗%20Weights-coming%20soon-yellow.svg" alt="Model Weights"></a>
  <a href=""><img src="https://img.shields.io/badge/🤗%20Dataset-coming%20soon-yellow.svg" alt="Dataset"></a>
</p>

**UniAfford** 是一个基于多模态大语言模型（MLLM）的统一二维和三维可供性感知框架，支持图像、点云或者两者同时输入的二维可供性感知，旨在实现具备泛化能力的跨模态 affordance perception。

我们提出了 **Token Router for Tasks** 多任务训练范式，通过将共享 MLLM 的上下文隐状态动态路由至不同任务分支，使密集可供性监督能够直接塑造共享语义表示，而无需依赖预定义的文本任务 token。

为支持统一跨模态学习，我们进一步构建了 **UniAfford-Data**，一个按照“物体-可供性“分类的大规模可供性数据集，包含二维像素级标注、三维点级标注以及语言指令。

## 资源

| 类型 | 链接 |
|---|---|
| 项目主页 | [Project Page](https://4dvlab.github.io/UniAfford) |
| 论文 | **UniAfford: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception** |
| 模型权重 | [Hugging Face Model](https://huggingface.co/yiqian7a/UniAfford-4B) |
| 数据集 | [Hugging Face Dataset](https://huggingface.co/datasets/yiqian7a/UniAfford-Data) |
| 数据集格式 | [docs/DATASET.md](docs/DATASET.md) |
| 进阶文档 | [docs/README_ADVANCED.md](docs/README_ADVANCED.md) |

## 仓库结构

- `configs/`：训练、模型与推理配置。
- `model/`：UniAfford 模型，以及 Segment Anything、PointCept、Qwen-VL 相关子模块。
- `utils/`：数据集、指标、checkpoint、数据处理与渲染工具。
- `scripts/`：训练、demo、checkpoint 与可视化相关脚本入口。
- `train_ds.py` / `train_fsdp.py` / `validate.py` / `app.py`：训练、验证与 demo 顶层入口。

## 快速开始

### 环境与安装

推荐 Python 3.10+，CUDA 12.4+，pytorch 2.6+；`requirements.txt` 中提供测验过的最低版本，根据需要安装其中注释掉的部分依赖库。

```bash
# 安装前先确保文件中的torch\transformer\peft等依赖库的版本适配您的机器，
pip install -r requirements.txt --no-build-isolation
```

### 运行

可交互的网页版demo，只需要下载checkpoint即可运行：

```bash
python app.py --checkpoint_path /path/to/checkpoint --device cuda --port 7860
```

## 数据与权重准备

仓库不内置训练数据、Qwen 权重、SAM 权重或点云 backbone 权重。请在运行前准备：

- 数据集根目录，格式见 [docs/DATASET.md](docs/DATASET.md)。公开数据集链接占位：[Hugging Face Dataset](https://huggingface.co/datasets/your-org/2d-3d-uniafford)。
- 训练好的 UniAfford 模型权重。模型权重链接占位：[Hugging Face Model](https://huggingface.co/your-org/2d-3d-uniafford)。
- Qwen / Qwen3-VL 模型路径。
- SAM 视觉 backbone 权重，例如 `sam_vit_h_4b8939.pth`。
- 点云 backbone 权重，例如 SONATA / PointTransformer 相关 checkpoint。

默认路径可在 `scripts/run_train.sh` 或 `configs/` 中调整。

### 数据集格式

数据按 `obj_type` 组织，每个物体类别下包含 `Instruction.csv`、`Image/` 与 `PointCloud/`：

```text
dataset_root/
├── spoon/
│   ├── Instruction.csv
│   ├── Image/
│   │   ├── rgb/
│   │   └── mask/<aff_type>/
│   └── PointCloud/
├── train.json
├── val.json
└── test.json
```

完整数据集格式与使用说明见 [docs/DATASET.md](docs/DATASET.md)。训练 batch 字段、模型输入输出、调优指南与指标说明见 [docs/README_ADVANCED.md](docs/README_ADVANCED.md)。

## 可视化与渲染

基础单样本查看由 `utils/base_dataset.py` 提供；批量导出论文图、grid 图或 IAGNet/Mitsuba 风格点云图时使用 `utils/rendering/`：

```bash
python -m utils.rendering.generate_render_manifest --dataset-root /path/to/dataset_root --output docs/auto_render_manifest.json
python -m utils.rendering.batch_render --render-json docs/auto_render_manifest.json
python -m utils.rendering.render_point --render-json docs/auto_render_manifest.json
```

详细参数与渲染分层见 [docs/rendering.md](docs/rendering.md)。

## 致谢

特别感谢[LISA](https://github.com/JIA-Lab-research/LISA)、[Sonata](https://xywu.me/sonata/)、Qwen3-VL、IAGNet、GREAT、[Affordance-R1](https://github.com/hq-King/Affordance-R1)、DAG 等开源项目作者及其代码对本项目的启发和帮助。

## Citation

如果该项目对您有帮助，请引用我们的文章并给项目一个 star：

```bibtex
正式发表后在此处添加 BibTeX。
```
