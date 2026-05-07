
"""Batch visualization export utilities for 2D/3D affordance targets."""

import os
import warnings
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_dataset import Modality
from utils.common import resolve_path

RENDER_MANIFEST_SCHEMA_NOTE = """
Render manifest JSON:
{
  "dataset_root": "path/to/dataset_root",
  "output_dir": "outputs/rendered_targets",
  "output": {
    "mode": "single | grid | both",
    "grid": {
      "columns": 4,
      "rows": null,
      "cell_width": 800,
      "cell_height": 800,
      "padding": 12,
      "background": [255, 255, 255]
    }
  },
  "render": {
    "image": {"alpha": 0.5, "color_rgb": [255, 0, 0], "threshold": 0.0, "extension": ".jpg"},
    "point_cloud": {
      "backend": "realistic",
      "size": 800,
      "elev": 30,
      "azim": 45,
      "point_size": 8,
      "sphere_radius": 0.024,
      "sphere_resolution": 8,
      "max_points": 6000,
      "extension": ".jpg"
    }
  },
  "images": [{"name": "optional_name", "obj_type": "spoon", "img_id": 120, "aff": "grasp"}],
  "point_clouds": [{"name": "optional_name", "obj_type": "spoon", "pc_id": 186, "aff": "wrapgrasp"}]
}

output.mode:
- single: 每个目标分别输出到 2d/ 和 3d/
- grid: 只输出 grid_2d.jpg 和 grid_3d.jpg
- both: 同时输出单图和合并图
"""

def _resolve_relative_path(path_value: str, base_dir: str) -> str:
    """解析 manifest 中的路径；相对路径优先相对 manifest 所在目录。"""
    if not path_value:
        return path_value
    expanded = os.path.expanduser(os.path.expandvars(str(path_value)))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _load_render_manifest(manifest_path: str, dataset_root_override: Optional[str] = None) -> Dict[str, Any]:
    manifest_path = resolve_path(manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("渲染 JSON 必须是对象，格式参照 docs/render_manifest_example.json")

    dataset_root = dataset_root_override or manifest.get("dataset_root")
    if not dataset_root:
        raise ValueError("渲染 JSON 需要提供 dataset_root，或命令行传入 --dataset-root")
    manifest["dataset_root"] = _resolve_relative_path(dataset_root, manifest_dir)
    manifest["output_dir"] = _resolve_relative_path(
        manifest.get("output_dir", "outputs/rendered_targets"),
        manifest_dir,
    )
    return manifest


def _render_obj_dir(dataset_root: str, obj_type: str) -> Tuple[str, str]:
    normalized = Modality._normalize_label(obj_type)
    obj_dir_map = {
        Modality._normalize_label(d): d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    }
    return normalized, obj_dir_map.get(normalized, obj_type)


def _find_image_rgb_path(dataset_root: str, obj_type: str, real_obj: str, img_id: int) -> Optional[str]:
    rgb_dir = os.path.join(dataset_root, real_obj, "Image", "rgb")
    for prefix in (obj_type, real_obj):
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(rgb_dir, f"{prefix}_{img_id}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _load_render_image_target(
    dataset_root: str,
    obj_type: str,
    img_id: int,
    aff_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    obj_type, real_obj = _render_obj_dir(dataset_root, obj_type)
    aff_type = Modality._normalize_label(aff_type)
    rgb_path = _find_image_rgb_path(dataset_root, obj_type, real_obj, img_id)
    if rgb_path is None:
        raise FileNotFoundError(f"找不到 RGB 图像: obj={obj_type}, img_id={img_id}")

    img = cv2.imread(rgb_path)
    if img is None:
        raise ValueError(f"读取 RGB 图像失败: {rgb_path}")

    mask_dir = os.path.join(dataset_root, real_obj, "Image", "mask", aff_type)
    mask_path = None
    for prefix in (obj_type, real_obj):
        candidate = os.path.join(mask_dir, f"{prefix}_{img_id}_{aff_type}.png")
        if os.path.exists(candidate):
            mask_path = candidate
            break
    if mask_path is None:
        raise FileNotFoundError(f"找不到 2D mask: obj={obj_type}, img_id={img_id}, aff={aff_type}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"读取 2D mask 失败: {mask_path}")
    return img, mask


def _load_render_point_target(
    dataset_root: str,
    obj_type: str,
    pc_id: int,
    aff_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    obj_type, real_obj = _render_obj_dir(dataset_root, obj_type)
    aff_type = Modality._normalize_label(aff_type)
    pc_dir = os.path.join(dataset_root, real_obj, "PointCloud")
    pc_path = None
    for prefix in (obj_type, real_obj):
        candidate = os.path.join(pc_dir, f"{prefix}_{pc_id}.csv")
        if os.path.exists(candidate):
            pc_path = candidate
            break
    if pc_path is None:
        raise FileNotFoundError(f"找不到点云 CSV: obj={obj_type}, pc_id={pc_id}")

    with open(pc_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        header = first_line.split(',') if first_line else []
        data = np.loadtxt(f, delimiter=',')
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    if data.shape[1] < 3:
        raise ValueError(f"点云 CSV 至少需要 xyz 三列: {pc_path}")

    labels = [Modality._normalize_label(label) for label in header[3:]]
    if aff_type not in labels:
        raise ValueError(f"点云 CSV 中没有 aff 列: aff={aff_type}, file={pc_path}")
    mask_idx = labels.index(aff_type)
    return data[:, :3], data[:, 3 + mask_idx]


def _render_image_overlay_affordance_r1(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_rgb: Tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> np.ndarray:
    """按 Affordance-R1 风格渲染：原图上叠加纯红半透明 mask。"""
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0
    binary = mask_float > float(threshold)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    color = np.asarray(color_rgb, dtype=np.float32)
    result_rgb = img_rgb.copy()
    result_rgb[binary] = img_rgb[binary] * (1.0 - alpha) + color * alpha
    result_bgr = cv2.cvtColor(np.clip(result_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def _prepare_point_cloud_render_data(
    points: np.ndarray,
    mask: np.ndarray,
    max_points: Optional[int] = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """归一化点云并生成红/灰 affordance 颜色。"""
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    mask = mask[valid]
    if points.size == 0:
        raise ValueError("点云为空或全部为非有限值")

    if max_points and points.shape[0] > int(max_points):
        sample_idx = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[sample_idx]
        mask = mask[sample_idx]

    mins = np.amin(points, axis=0)
    maxs = np.amax(points, axis=0)
    center = (mins + maxs) / 2.0
    scale = float(np.amax(maxs - mins))
    if scale <= 0:
        scale = 1.0
    points = (points - center) / scale

    mask = np.clip(mask, 0.0, None)
    if mask.max() > 1.0:
        mask = mask / mask.max()
    mask = np.sqrt(np.clip(mask, 0.0, 1.0))
    base_color = np.array([190, 190, 190], dtype=np.float32) / 255.0
    affordance_color = np.array([255, 0, 0], dtype=np.float32) / 255.0
    colors = base_color + (affordance_color - base_color) * mask[:, None]
    return points, colors


def _render_point_cloud_matplotlib(
    points: np.ndarray,
    colors: np.ndarray,
    size: int = 800,
    elev: float = 30,
    azim: float = 45,
    point_size: float = 12,
) -> np.ndarray:
    """轻量 3D 点云渲染后端；无真实光照，但依赖少。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _render_point_cloud_realistic(
    points: np.ndarray,
    colors: np.ndarray,
    size: int = 800,
    elev: float = 30,
    azim: float = 45,
    sphere_radius: float = 0.024,
    sphere_resolution: int = 8,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    ground_plane: bool = True,
) -> np.ndarray:
    """Open3D 离屏渲染：把点转成小球，使用 lit material 和阴影。"""
    import open3d as o3d
    from open3d.visualization import rendering

    mesh = o3d.geometry.TriangleMesh()
    sphere_resolution = max(3, int(sphere_resolution))
    sphere_radius = float(sphere_radius)
    for point, color in zip(points, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius,
            resolution=sphere_resolution,
        )
        sphere.translate(point.astype(np.float64), relative=True)
        sphere.paint_uniform_color(color.astype(np.float64))
        mesh += sphere
    mesh.compute_vertex_normals()

    renderer = rendering.OffscreenRenderer(int(size), int(size))
    bg = np.array(
        [background_rgb[0] / 255.0, background_rgb[1] / 255.0, background_rgb[2] / 255.0, 1.0],
        dtype=np.float32,
    )
    renderer.scene.set_background(bg)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_roughness = 0.62
    material.base_reflectance = 0.35
    renderer.scene.add_geometry("affordance_spheres", mesh, material)

    if ground_plane:
        plane_size = 20.0
        plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.01)
        plane.translate((-plane_size / 2.0, -plane_size / 2.0, -0.58), relative=True)
        plane.paint_uniform_color((bg[0], bg[1], bg[2]))
        plane.compute_vertex_normals()
        plane_material = rendering.MaterialRecord()
        plane_material.shader = "defaultLit"
        plane_material.base_roughness = 0.75
        renderer.scene.add_geometry("ground", plane, plane_material)

    try:
        renderer.scene.scene.set_sun_light(
            [-0.45, -0.35, -0.82],
            [1.0, 1.0, 1.0],
            85000,
        )
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            [-0.45, -0.35, -0.82],
        )
    except Exception:
        pass

    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    radius = 2.05
    eye = [
        radius * np.cos(elev_rad) * np.cos(azim_rad),
        radius * np.cos(elev_rad) * np.sin(azim_rad),
        radius * np.sin(elev_rad),
    ]
    renderer.setup_camera(35.0, [0.0, 0.0, 0.0], eye, [0.0, 0.0, 1.0])
    rendered = renderer.render_to_image()
    rgb = np.asarray(rendered)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _render_point_cloud_static(
    points: np.ndarray,
    mask: np.ndarray,
    size: int = 800,
    elev: float = 30,
    azim: float = 45,
    point_size: float = 12,
    max_points: Optional[int] = 6000,
    backend: str = "realistic",
    sphere_radius: float = 0.024,
    sphere_resolution: int = 8,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    ground_plane: bool = True,
) -> np.ndarray:
    """用固定视角导出点云图；realistic 后端提供小球、材质与光照。"""
    points, colors = _prepare_point_cloud_render_data(points, mask, max_points=max_points)
    backend = str(backend or "realistic").lower()
    if backend in {"realistic", "open3d", "sphere"}:
        try:
            return _render_point_cloud_realistic(
                points,
                colors,
                size=size,
                elev=elev,
                azim=azim,
                sphere_radius=sphere_radius,
                sphere_resolution=sphere_resolution,
                background_rgb=background_rgb,
                ground_plane=ground_plane,
            )
        except Exception as exc:
            warnings.warn(f"Open3D realistic 渲染失败，回退到 matplotlib: {exc}")
    elif backend != "matplotlib":
        warnings.warn(f"未知 3D 渲染 backend={backend}，回退到 matplotlib")

    return _render_point_cloud_matplotlib(
        points,
        colors,
        size=size,
        elev=elev,
        azim=azim,
        point_size=point_size,
    )


def _safe_render_name(item: Dict[str, Any], index: int, modality: str, aff_type: str) -> str:
    name = item.get("name")
    if not name:
        obj = Modality._normalize_label(item.get("obj_type", "obj"))
        item_id = item.get(f"{modality}_id", index)
        name = f"{obj}_{modality}{item_id}_{aff_type}"
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(name))
    return safe.strip('_') or f"item_{index}"


def _fit_image_to_cell(img: np.ndarray, width: int, height: int, background: Tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _save_image_grid(images: List[np.ndarray], output_path: str, grid_cfg: Dict[str, Any]) -> None:
    if not images:
        return
    columns = int(grid_cfg.get("columns", 4))
    if columns <= 0:
        raise ValueError("grid.columns 必须大于 0")
    rows = grid_cfg.get("rows")
    rows = int(rows) if rows is not None else int(np.ceil(len(images) / columns))
    if rows <= 0:
        raise ValueError("grid.rows 必须大于 0")
    if rows * columns < len(images):
        raise ValueError(f"grid rows*columns 不足以容纳 {len(images)} 张图")

    cell_width = int(grid_cfg.get("cell_width", 800))
    cell_height = int(grid_cfg.get("cell_height", 800))
    padding = int(grid_cfg.get("padding", 12))
    if cell_width <= 0 or cell_height <= 0 or padding < 0:
        raise ValueError("grid.cell_width/cell_height 必须大于 0，padding 不能为负数")
    bg_rgb = tuple(int(v) for v in grid_cfg.get("background", [255, 255, 255]))
    background = (bg_rgb[2], bg_rgb[1], bg_rgb[0])

    grid_h = rows * cell_height + (rows + 1) * padding
    grid_w = columns * cell_width + (columns + 1) * padding
    canvas = np.full((grid_h, grid_w, 3), background, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        y0 = padding + row * (cell_height + padding)
        x0 = padding + col * (cell_width + padding)
        cell = _fit_image_to_cell(image, cell_width, cell_height, background)
        canvas[y0:y0 + cell_height, x0:x0 + cell_width] = cell

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def render_targets_from_json(manifest_path: str, dataset_root_override: Optional[str] = None) -> Dict[str, List[str]]:
    """根据 JSON manifest 批量导出 2D/3D affordance target 可视化。"""
    manifest = _load_render_manifest(manifest_path, dataset_root_override=dataset_root_override)
    dataset_root = manifest["dataset_root"]
    output_dir = manifest["output_dir"]
    image_items = manifest.get("images", [])
    point_items = manifest.get("point_clouds", [])
    if not isinstance(image_items, list) or not isinstance(point_items, list):
        raise ValueError("渲染 JSON 的 images 与 point_clouds 必须是数组")
    if not image_items and not point_items:
        raise ValueError("渲染 JSON 需要提供非空 images 或 point_clouds 数组")

    output_cfg = manifest.get("output", {})
    mode = str(output_cfg.get("mode", "single")).lower()
    if mode not in {"single", "grid", "both"}:
        raise ValueError("output.mode 只能是 single、grid 或 both")
    save_single = mode in {"single", "both"}
    save_grid = mode in {"grid", "both"}

    render_cfg = manifest.get("render", {})
    image_cfg = render_cfg.get("image", {})
    point_cfg = render_cfg.get("point_cloud", {})
    image_ext = image_cfg.get("extension", ".jpg")
    point_ext = point_cfg.get("extension", ".jpg")
    os.makedirs(output_dir, exist_ok=True)

    image_results: List[np.ndarray] = []
    point_results: List[np.ndarray] = []
    saved_paths = {"2d": [], "3d": [], "grid": [], "iagnet_xml": [], "iagnet_exr": [], "iagnet_jpg": []}

    for idx, item in enumerate(image_items):
        if not isinstance(item, dict):
            warnings.warn(f"跳过非法 images[{idx}]：必须是对象")
            continue
        obj_type = item.get("obj_type")
        if not obj_type:
            warnings.warn(f"跳过 images[{idx}]：缺少 obj_type")
            continue
        if item.get("img_id") is None or item.get("aff") is None:
            warnings.warn(f"跳过 images[{idx}]：缺少 img_id 或 aff")
            continue

        img_id = int(item["img_id"])
        img_aff = Modality._normalize_label(item["aff"])
        img_bgr, img_mask = _load_render_image_target(dataset_root, obj_type, img_id, img_aff)
        overlay = _render_image_overlay_affordance_r1(
            img_bgr,
            img_mask,
            alpha=float(image_cfg.get("alpha", 0.5)),
            color_rgb=tuple(image_cfg.get("color_rgb", [255, 0, 0])),
            threshold=float(image_cfg.get("threshold", 0.0)),
        )
        image_results.append(overlay)
        if save_single:
            name = _safe_render_name(item, idx, "img", img_aff)
            out_path = os.path.join(output_dir, "2d", f"{name}{image_ext}")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, overlay)
            saved_paths["2d"].append(out_path)

    point_backend = str(point_cfg.get("backend", "realistic")).lower()
    if point_items and point_backend in {"iagnet", "mitsuba", "mitsuba_iagnet"}:
        from scripts.render_points import export_iagnet_style

        iagnet_cfg = point_cfg.get("iagnet", {})
        iagnet_outputs = export_iagnet_style(
            manifest_path,
            dataset_root_override=dataset_root,
            output_dir_override=output_dir,
            run_mitsuba=bool(iagnet_cfg.get("run_mitsuba", False)),
            convert_jpg=bool(iagnet_cfg.get("convert_jpg", False)),
            mitsuba_bin=str(iagnet_cfg.get("mitsuba_bin", "mitsuba")),
            mitsuba_renderer=str(iagnet_cfg.get("mitsuba_renderer", "python")),
            mitsuba_variant=str(iagnet_cfg.get("mitsuba_variant", "cuda_ad_rgb")),
            exr_wait_timeout=float(iagnet_cfg.get("exr_wait_timeout", 60.0)),
            exr_wait_interval=float(iagnet_cfg.get("exr_wait_interval", 0.25)),
        )
        saved_paths["iagnet_xml"].extend(iagnet_outputs.get("xml", []))
        saved_paths["iagnet_exr"].extend(iagnet_outputs.get("exr", []))
        saved_paths["iagnet_jpg"].extend(iagnet_outputs.get("jpg", []))
    else:
        for idx, item in enumerate(point_items):
            if not isinstance(item, dict):
                warnings.warn(f"跳过非法 point_clouds[{idx}]：必须是对象")
                continue
            obj_type = item.get("obj_type")
            if not obj_type:
                warnings.warn(f"跳过 point_clouds[{idx}]：缺少 obj_type")
                continue
            if item.get("pc_id") is None or item.get("aff") is None:
                warnings.warn(f"跳过 point_clouds[{idx}]：缺少 pc_id 或 aff")
                continue

            pc_id = int(item["pc_id"])
            pc_aff = Modality._normalize_label(item["aff"])
            points, pc_mask = _load_render_point_target(dataset_root, obj_type, pc_id, pc_aff)
            point_img = _render_point_cloud_static(
                points,
                pc_mask,
                size=int(point_cfg.get("size", 800)),
                elev=float(point_cfg.get("elev", 30)),
                azim=float(point_cfg.get("azim", 45)),
                point_size=float(point_cfg.get("point_size", 10)),
                max_points=point_cfg.get("max_points", 6000),
                backend=point_backend,
                sphere_radius=float(point_cfg.get("sphere_radius", 0.024)),
                sphere_resolution=int(point_cfg.get("sphere_resolution", 8)),
                background_rgb=tuple(point_cfg.get("background", [255, 255, 255])),
                ground_plane=bool(point_cfg.get("ground_plane", True)),
            )
            point_results.append(point_img)
            if save_single:
                name = _safe_render_name(item, idx, "pc", pc_aff)
                out_path = os.path.join(output_dir, "3d", f"{name}{point_ext}")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, point_img)
                saved_paths["3d"].append(out_path)

    if save_grid:
        grid_cfg = output_cfg.get("grid", {})
        grid_2d = os.path.join(output_dir, "grid_2d.jpg")
        grid_3d = os.path.join(output_dir, "grid_3d.jpg")
        if image_results:
            _save_image_grid(image_results, grid_2d, grid_cfg)
            saved_paths["grid"].append(grid_2d)
        if point_results:
            _save_image_grid(point_results, grid_3d, grid_cfg)
            saved_paths["grid"].append(grid_3d)

    return saved_paths



    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量渲染 2D/3D 数据")
    parser.add_argument('--render-json', type=str, required=True,
                        help='批量导出模式：读取 JSON manifest，按 images 与 point_clouds 独立列表渲染保存，JSON格式参照 docs/render_manifest_example.json')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='可选：覆盖 JSON manifest 中的 dataset_root')
    args = parser.parse_args()

    saved_paths = render_targets_from_json(args.render_json, dataset_root_override=args.dataset_root)
    total = sum(len(paths) for paths in saved_paths.values())
    print(f"批量渲染完成，共保存 {total} 个文件")
    for group, paths in saved_paths.items():
        for path in paths:
            print(f"{group}: {path}")
