# 批量渲染与高质量出图说明

`utils/rendering/` 中的代码用于**清单驱动的批量出图**与**高质量点云渲染**，主要服务于论文图、报告图、数据集批量格式化和结果检查等需要稳定输出文件的场景。该流程不是日常快速可视化的必要入口；如果只是查看单个 RGB、mask 或点云样本，优先使用 `utils/base_dataset.py` 中已有的基础可视化支持。

其中，点云高质量渲染流程参考 IAGNet 的 Mitsuba 渲染方式，包括点云归一化、小球化、相机视角、地面、面积光和阴影设置。本仓库中的实现并不是重新设计一套渲染风格，而是将 IAGNet 风格的点云出图流程封装、迁移到 `utils/rendering/` 下，并适配本数据集的 manifest 批量处理方式。

本文档仅说明以下批量出图相关文件的使用方式：

- `utils/rendering/generate_render_manifest.py`：从 metadata/split JSON 自动生成渲染清单。
- `utils/rendering/batch_render.py`：读取 manifest，批量导出 2D 叠图、普通 3D 静态图、grid，或按配置转交给 IAGNet/Mitsuba 管线。
- `utils/rendering/render_point.py`：仅消费 manifest 中的 `point_clouds`，生成 IAGNet 风格 Mitsuba XML，并可继续渲染 EXR/JPG。
- `docs/render_manifest_example.json`：manifest 示例与默认渲染参数。

## 适用范围

建议在以下场景中使用 `utils/rendering`：

- 批量导出整个或部分数据集的 2D affordance 叠图。
- 为论文、PPT 或 README 生成统一尺寸、统一颜色、统一布局的图。
- 对点云生成更接近 IAGNet 风格的高质量小球、地面、光照和阴影图。
- 将多个样本合并为 grid 图片，用于统一展示。

如果目的只是确认数据能否读取、mask 是否对齐、点云形状是否大致正确，则无需使用 manifest 流程。此类快速查看由 `utils/base_dataset.py` 的基础可视化功能承担。

## 1. 生成 Manifest

manifest 是批量渲染流程的输入文件，用于描述数据集根目录、输出目录、渲染参数，以及待渲染的 `images` 和 `point_clouds` 列表。完整格式参考 `docs/render_manifest_example.json`。

从数据集的 metadata/split 自动生成 2D+3D 渲染清单：

```bash
python -m utils.rendering.generate_render_manifest ^
  --dataset-root "E:\path\to\dataset_root" ^
  --output docs\auto_render_manifest.json ^
  --render-output-dir "../outputs/rendered_targets" ^
  --max-per-aff 10 ^
  --backend iagnet ^
  --output-mode both
```

仅生成整个 2D 数据集的渲染清单：

```bash
python -m utils.rendering.generate_render_manifest ^
  --dataset-root "E:\path\to\dataset_root" ^
  --output docs\full_2d_render_manifest.json ^
  --render-output-dir "../outputs/rendered_2d" ^
  --images-only ^
  --max-per-aff 0 ^
  --output-mode single
```

常用参数：

- `--metadata-file`：指定 split/metadata JSON。不指定时，脚本会在数据集根目录下依次尝试 `render_ids.json`、`split.json`、`dataset_split.json`、`all.json`、`train.json`、`val.json`、`test.json`、`info.json`、`metadata.json`。
- `--images-only`：只生成 `images`，`point_clouds` 为空，适合批量渲染 2D 数据集。
- `--max-per-aff 0`：每个物体、每个 affordance 保留全部可用 ID；大于 0 时表示抽样上限。
- `--obj-types` / `--aff-types`：按物体或 affordance 过滤，例如 `--obj-types Spoon,Mug`。
- `--copy-selected-to` / `--use-copy-root`：把选中的 RGB、mask、点云复制成一个小子集，并让 manifest 指向该子集。

## 2. 批量渲染 2D / 3D

使用 `batch_render.py` 读取 manifest 并执行批量导出：

```bash
python -m utils.rendering.batch_render --render-json docs\auto_render_manifest.json
```

如需临时覆盖 manifest 中的 `dataset_root`，可使用：

```bash
python -m utils.rendering.batch_render ^
  --render-json docs\auto_render_manifest.json ^
  --dataset-root "E:\path\to\dataset_root"
```

输出形式由 manifest 中的 `output.mode` 控制：

- `single`：每个样本单独保存。2D 输出到 `output_dir/2d/`，普通 3D 输出到 `output_dir/3d/`。
- `grid`：只保存 `grid_2d.jpg` / `grid_3d.jpg`。
- `both`：同时保存单图和 grid。

2D 叠图参数位于 `render.image`：

- `alpha`：高亮透明度。
- `color_rgb`：高亮颜色，默认红色 `[255, 0, 0]`。
- `threshold`：mask 二值化阈值，代码使用 `mask > threshold`。对 0/255 mask 可用 `0.0`；如果输入是 sigmoid 概率图，通常用 `0.5`。

3D 渲染后端由 `render.point_cloud.backend` 控制：

- `iagnet` / `mitsuba` / `mitsuba_iagnet`：转交给 `utils.rendering.render_point.export_iagnet_style`，使用参考 IAGNet 的 Mitsuba 流程生成 XML，并根据 `render.point_cloud.iagnet` 决定是否继续渲染 EXR/JPG。
- `realistic` / `open3d` / `sphere`：使用 Open3D 离屏渲染，小球、材质、光照和软阴影为近似效果。
- `matplotlib`：轻量回退后端，无真实光影。

## 3. 仅运行 IAGNet / Mitsuba 点云渲染

该入口只处理 manifest 中的 `point_clouds`，适用于仅需要生成高质量点云图的情况。渲染参数和姿态设置参考 IAGNet，并在本仓库中封装为可批量调用的脚本。

仅生成 Mitsuba XML：

```bash
python -m utils.rendering.render_point --render-json docs\auto_render_manifest.json
```

上述命令默认将 XML 写入 `output_dir/iagnet_xml/`。如需继续使用 Mitsuba Python API 渲染 EXR，可运行：

```bash
python -m utils.rendering.render_point ^
  --render-json docs\auto_render_manifest.json ^
  --run-mitsuba
```

如需同时将 EXR 转换为 JPG，可运行：

```bash
python -m utils.rendering.render_point ^
  --render-json docs\auto_render_manifest.json ^
  --run-mitsuba ^
  --convert-jpg
```

如果当前环境没有可用 CUDA，或遇到 LLVM / Dr.Jit 相关问题，可切换到 CPU fallback：

```bash
python -m utils.rendering.render_point ^
  --render-json docs\auto_render_manifest.json ^
  --run-mitsuba ^
  --mitsuba-variant scalar_rgb ^
  --convert-jpg
```

也可以使用 Mitsuba CLI：

```bash
python -m utils.rendering.render_point ^
  --render-json docs\auto_render_manifest.json ^
  --run-mitsuba ^
  --mitsuba-renderer cli ^
  --mitsuba-bin mitsuba ^
  --convert-jpg
```

IAGNet 风格渲染的关键参数位于 `render.point_cloud.iagnet`：

- `sphere_radius`：点云小球半径，对应 IAGNet 风格的小球化点云。
- `sample_count`：Mitsuba 采样数，越大越慢但噪声越低。
- `fov` / `camera_origin`：相机视角。
- `radiance`：面积光强度。
- `background_radiance`：均匀环境光。过大会洗白颜色并冲淡阴影，默认建议保持 `0.0`。
- `ground_scale`：地面大小。
- `max_points`：最多渲染多少个点。
- `mitsuba_variant`：默认 `cuda_ad_rgb`；CPU fallback 用 `scalar_rgb`。
- `exr_wait_timeout` / `exr_wait_interval`：等待 EXR/JPG 文件稳定，避免输出文件尚未完全写入就开始转换，导致 0 字节或损坏文件。

## 4. 依赖

基础批量 2D/普通 3D：

```bash
pip install opencv-python numpy pillow matplotlib
pip install open3d
```

IAGNet/Mitsuba 高质量点云渲染：

```bash
pip install mitsuba OpenEXR Imath
```

`OpenEXR` / `Imath` 只在 `--convert-jpg` 时需要。只生成 XML 不需要安装 Mitsuba；使用 Python API 直接渲染时才需要安装 `mitsuba`。

## 5. 文件关系

- `generate_render_manifest.py` 只负责生成 manifest，不执行渲染。
- `batch_render.py` 是清单总入口；它会先渲染 `images`，再根据 `render.point_cloud.backend` 选择普通 3D 或 IAGNet/Mitsuba。
- `render_point.py` 是参考 IAGNet 打包迁移后的高质量点云入口；它只处理 `point_clouds`，不会渲染 2D。
- `base_dataset.py` 保留基础可视化能力。日常查看数据时优先使用该入口，不需要走上述批量出图流程。
