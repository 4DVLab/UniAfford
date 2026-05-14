# 渲染与可视化分层说明

本仓库将「看数据」与「按清单出图」分成三条线，避免把 `base_dataset` 的交互查看与 manifest 批量管线混为一谈。实现已集中在 [`utils/rendering/`](../utils/rendering/)（清单批量 + IAGNet/Mitsuba），根目录 [`utils/batch_render.py`](../utils/batch_render.py) 为兼容层 re-export。

## 三层职责

| 层级 | 用途 | 入口 / 模块 |
|------|------|----------------|
| **A. 数据集内省** | 对磁盘上的单个 RGB / 点云 CSV 调用 `show()`，快速目测 | [`utils/base_dataset.py`](../utils/base_dataset.py)：`-s` / `--show` |
| **B. 清单批量导出** | 读 JSON manifest，2D 叠图、`3d/` 静态图（Open3D ↔ Matplotlib）、可选 grid；`render.point_cloud.backend` 为 `iagnet` / `mitsuba*` 时委托给 **C** | [`utils/batch_render.py`](../utils/batch_render.py) 或 [`utils/rendering/batch_render.py`](../utils/rendering/batch_render.py)，`--render-json` |
| **C. IAGNet / Mitsuba** | 写 Mitsuba XML，可选 Python API 或 CLI 渲染 EXR、转 JPG | 实现：[`utils/rendering/render_point.py`](../utils/rendering/render_point.py)；CLI 包装：[`scripts/render_point.py`](../scripts/render_point.py) |

Manifest 格式见 [`docs/render_manifest_example.json`](render_manifest_example.json)。从数据集自动生成清单见 [`scripts/generate_render_manifest.py`](../scripts/generate_render_manifest.py)。

## 推荐命令（仓库根目录）

**单文件交互查看（无 manifest）**

```bash
python utils/base_dataset.py -s path/to/sample.png path/to/pointcloud.csv
```

**清单批量导出（2D + 3D 静态 / grid / 或 manifest 内联 IAGNet 配置）**

```bash
python utils/batch_render.py --render-json docs/render_manifest_example.json
python utils/batch_render.py --render-json docs/render_manifest_example.json --dataset-root /path/to/dataset_root
```

**仅 IAGNet 风格 XML / Mitsuba（消费同一 manifest 的 `point_clouds`）**

```bash
python scripts/render_point.py --render-json docs/render_manifest_example.json
python scripts/render_point.py --render-json docs/render_manifest_example.json --run-mitsuba
```

## 依赖提示

- **B** 通常需要 `opencv-python`、`numpy`；Open3D 静态后端需安装 `open3d`。
- **C** 需要可选的 Mitsuba 与（若 `--convert-jpg`）`OpenEXR` / `Imath`。

## 代码引用关系

- `utils.rendering.batch_render.render_targets_from_json` 在 `backend ∈ {iagnet, mitsuba, mitsuba_iagnet}` 时 **延迟** `import utils.rendering.render_point.export_iagnet_style`，避免未安装 Mitsuba 时无法导入批量模块。
