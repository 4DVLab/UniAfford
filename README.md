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

**UniAfford** is a unified 2D–3D affordance perception framework built upon Multimodal Large Language Models (MLLMs). It supports affordance perception from image-only, point-cloud-only, or joint multimodal inputs, aiming to achieve generalizable cross-modal affordance perception. 

We propose **Token Router for Tasks**, a multitask training paradigm that dynamically routes contextual hidden states from a shared MLLM into task-specific branches, enabling dense affordance supervision to directly shape shared semantic representations without relying on predefined textual task tokens.

To support unified cross-modal learning, we further construct **UniAfford-Data**, a large-scale affordance dataset organized under a unified object–affordance taxonomy, containing pixel-level 2D annotations, point-level 3D annotations, and language instructions.

## Resources

| Type | Link |
|---|---|
| Project page | [Project Page](https://4dvlab.github.io/UniAfford) |
| Paper | **UniAfford: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception** |
| Model weights | [Hugging Face Model](https://huggingface.co/yiqian7a/UniAfford-4B) |
| Dataset | [Hugging Face Dataset](https://huggingface.co/datasets/yiqian7a/UniAfford-Data) |
| Dataset format | [docs/DATASET.md](docs/DATASET.md) |
| Advanced guide | [docs/README_ADVANCED.md](docs/README_ADVANCED.md) |

## Quick Start

### Environment and Installation

Recommended environment: Python 3.10+, CUDA 12.4+, and PyTorch 2.6+. The versions in `requirements.txt` are intended for reproducing the full training stack.

```bash
# Training / validation stack, including DeepSpeed, transformers, Open3D, etc.
pip install -r requirements.txt

# Gradio demo only
pip install -r requirements_app.txt
```

### Run

```bash
# Training. Edit dataset, pretrained weight, and log paths in scripts/run_train.sh first.
bash scripts/run_train.sh fsdp   # or ds. DeepSpeed is more prone to OOM from peak memory usage in this project, while FSDP is relatively stable.

# Evaluation
python validate.py \
  --checkpoint_path /path/to/checkpoint.pth \
  --dataset_dir /path/to/dataset_root \
  --split test

# Gradio demo
python app.py --checkpoint_path /path/to/checkpoint --device cuda --port 7860
```

## Data and Weights

This repository does not include training data, Qwen weights, SAM weights, or point-cloud backbone weights. Please prepare the following before running:

- Dataset root. See [docs/DATASET.md](docs/DATASET.md). Placeholder: [Hugging Face Dataset](https://huggingface.co/datasets/your-org/2d-3d-uniafford).
- Trained UniAfford model weights. Placeholder: [Hugging Face Model](https://huggingface.co/your-org/2d-3d-uniafford).
- Qwen / Qwen3-VL model path.
- SAM visual backbone weights, for example `sam_vit_h_4b8939.pth`.
- Point-cloud backbone weights, for example SONATA / PointTransformer checkpoints.

Default paths can be changed in `scripts/run_train.sh` or `configs/`.

## Dataset Format

The dataset is organized by `obj_type`. Each object category contains `Instruction.csv`, `Image/`, and `PointCloud/`:

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

See [docs/DATASET.md](docs/DATASET.md) for the complete dataset format and usage. See [docs/README_ADVANCED.md](docs/README_ADVANCED.md) for batch fields, model inputs/outputs, tuning notes, and metric definitions.

## Visualization and Rendering

`utils/base_dataset.py` provides basic single-sample visualization. For batch figure export, grid images, or IAGNet/Mitsuba-style point-cloud rendering, use `utils/rendering/`:

```bash
python -m utils.rendering.generate_render_manifest --dataset-root /path/to/dataset_root --output docs/auto_render_manifest.json
python -m utils.rendering.batch_render --render-json docs/auto_render_manifest.json
python -m utils.rendering.render_point --render-json docs/auto_render_manifest.json
```

See [docs/rendering.md](docs/rendering.md) for rendering details.

## Repository Structure

- `configs/`: training, model, and inference configurations.
- `model/`: the joint affordance model and related Segment Anything, PointCept, and Qwen-VL modules.
- `utils/`: dataset loading, metrics, checkpoints, data processing, and rendering utilities.
- `scripts/`: training, demo, checkpoint, and visualization entry points.
- `train_ds.py` / `train_fsdp.py` / `validate.py` / `app.py`: top-level training, validation, and demo scripts.

## Acknowledgements

We sincerely thank the authors of [LISA](https://github.com/JIA-Lab-research/LISA), [Sonata](https://xywu.me/sonata/), Qwen3-VL, IAGNet, GREAT, [Affordance-R1](https://github.com/hq-King/Affordance-R1), DAG, and other open-source projects for their inspiration and help through their code.

## Citation

If this project is useful to your research, please cite our paper and consider starring the repository:

```bibtex
BibTeX will be added after publication.
```
