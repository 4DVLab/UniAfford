"""
生成渲染 manifest：扫描数据集目录，自动构建渲染清单。

对于同时包含 Image（图像）和 PointCloud（点云）数据的每个物体，
本脚本会查找既存在于 2D 掩码、又出现在 3D 点云 CSV 表头的 affordance。
对于每个共同的 affordance，分别随机导出不超过 N 个图像 ID 与 N 个点云 ID，组合为 manifest，
供 utils/batch_render.py 和 scripts/render_points.py 等脚本批量渲染使用。
"""
import argparse
import json
import os
import random
import sys
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.base_dataset import Modality  # noqa: E402


# 默认渲染配置改为从 JSON 文件加载
def load_default_render_config(json_path: str = None) -> dict:
    """
    加载默认渲染 JSON 配置。默认路径为 docs/render_manifest_example.json。
    """
    if json_path is None:
        json_path = os.path.join(REPO_ROOT, "docs", "render_manifest_example.json")
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # 移除非配置参数，仅保留 output, render，用作 DEFAULT_RENDER_CONFIG
    config_slim = {k: v for k, v in config.items() if k in ("output", "render")}
    return config_slim

DEFAULT_RENDER_CONFIG = load_default_render_config()




def _normalize(label: str) -> str:
    return Modality._normalize_label(label)


def _safe_name(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
    return safe.strip('_') or "target"


def _path_for_json(path: str, base_dir: str, absolute: bool = False) -> str:
    path = os.path.abspath(path)
    if absolute:
        return path
    rel = os.path.relpath(path, base_dir)
    return rel.replace(os.sep, "/")


def _list_object_dirs(dataset_root: str, obj_filter: Optional[Set[str]] = None) -> List[str]:
    object_dirs = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if not os.path.isdir(path):
            continue
        if obj_filter is not None and _normalize(name) not in obj_filter:
            continue
        image_dir = os.path.join(path, "Image")
        pc_dir = os.path.join(path, "PointCloud")
        if os.path.isdir(image_dir) and os.path.isdir(pc_dir):
            object_dirs.append(name)
    return object_dirs


def _find_rgb_path(dataset_root: str, obj_type: str, img_id: int) -> Optional[str]:
    rgb_dir = os.path.join(dataset_root, obj_type, "Image", "rgb")
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(rgb_dir, f"{obj_type}_{img_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    normalized = _normalize(obj_type)
    if normalized != obj_type:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(rgb_dir, f"{normalized}_{img_id}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _extract_id_from_mask_name(filename: str, obj_type: str, aff_type: str) -> Optional[int]:
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in {".png", ".jpg", ".jpeg"}:
        return None

    prefixes = [f"{obj_type}_"]
    normalized = _normalize(obj_type)
    if normalized != obj_type:
        prefixes.append(f"{normalized}_")
    suffix = f"_{aff_type}"
    for prefix in prefixes:
        if stem.startswith(prefix) and stem.endswith(suffix):
            id_part = stem[len(prefix):-len(suffix)]
            try:
                return int(id_part)
            except ValueError:
                return None
    return None


def _scan_image_affordances(dataset_root: str, obj_type: str) -> Dict[str, List[int]]:
    mask_root = os.path.join(dataset_root, obj_type, "Image", "mask")
    aff_to_ids: Dict[str, List[int]] = {}
    if not os.path.isdir(mask_root):
        return aff_to_ids

    for aff_dir in sorted(os.listdir(mask_root)):
        aff_path = os.path.join(mask_root, aff_dir)
        if not os.path.isdir(aff_path):
            continue
        aff = _normalize(aff_dir)
        ids: Set[int] = set()
        for filename in os.listdir(aff_path):
            img_id = _extract_id_from_mask_name(filename, obj_type, aff_dir)
            if img_id is None:
                img_id = _extract_id_from_mask_name(filename, _normalize(obj_type), aff)
            if img_id is not None and _find_rgb_path(dataset_root, obj_type, img_id):
                ids.add(img_id)
        if ids:
            aff_to_ids[aff] = sorted(ids)
    return aff_to_ids


def _extract_pc_id(filename: str, obj_type: str) -> Optional[int]:
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        return None
    prefixes = [f"{obj_type}_"]
    normalized = _normalize(obj_type)
    if normalized != obj_type:
        prefixes.append(f"{normalized}_")
    for prefix in prefixes:
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):])
            except ValueError:
                return None
    return None


def _read_pc_affordances(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line.startswith("#"):
        first_line = first_line[1:].strip()
    header = [col.strip() for col in first_line.split(",") if col.strip()]
    return [_normalize(col) for col in header[3:]]


def _scan_point_affordances(dataset_root: str, obj_type: str) -> Dict[str, List[int]]:
    pc_root = os.path.join(dataset_root, obj_type, "PointCloud")
    aff_to_ids: Dict[str, Set[int]] = {}
    if not os.path.isdir(pc_root):
        return {}

    for filename in sorted(os.listdir(pc_root)):
        pc_id = _extract_pc_id(filename, obj_type)
        if pc_id is None:
            continue
        csv_path = os.path.join(pc_root, filename)
        try:
            affs = _read_pc_affordances(csv_path)
        except OSError:
            continue
        for aff in affs:
            aff_to_ids.setdefault(aff, set()).add(pc_id)
    return {aff: sorted(ids) for aff, ids in aff_to_ids.items() if ids}


def _sample_ids(ids: Sequence[int], limit: int, rng: Optional[random.Random]) -> List[int]:
    ids = list(ids)
    if limit <= 0 or len(ids) <= limit:
        return sorted(ids)
    if rng is None:
        return sorted(ids[:limit])
    return sorted(rng.sample(ids, limit))


def _iter_selected_affs(
    image_affs: Dict[str, List[int]],
    point_affs: Dict[str, List[int]],
    aff_filter: Optional[Set[str]] = None,
) -> Iterable[str]:
    shared = set(image_affs.keys()) & set(point_affs.keys())
    if aff_filter is not None:
        shared &= aff_filter
    return sorted(shared)


def build_render_manifest(
    dataset_root: str,
    output_dir: str,
    max_per_aff: int = 10,
    seed: Optional[int] = None,
    obj_types: Optional[Sequence[str]] = None,
    aff_types: Optional[Sequence[str]] = None,
    render_backend: str = "realistic",
    output_mode: str = "both",
    absolute_paths: bool = False,
    manifest_path: Optional[str] = None,
) -> Dict:
    dataset_root = os.path.abspath(dataset_root)
    manifest_base = os.path.dirname(os.path.abspath(manifest_path)) if manifest_path else os.getcwd()
    rng = random.Random(seed) if seed is not None else None
    obj_filter = {_normalize(obj) for obj in obj_types} if obj_types else None
    aff_filter = {_normalize(aff) for aff in aff_types} if aff_types else None

    manifest = deepcopy(DEFAULT_RENDER_CONFIG)
    manifest["dataset_root"] = _path_for_json(dataset_root, manifest_base, absolute=absolute_paths)
    manifest["output_dir"] = output_dir
    manifest["output"]["mode"] = output_mode
    manifest["render"]["point_cloud"]["backend"] = render_backend
    manifest["images"] = []
    manifest["point_clouds"] = []
    manifest["metadata"] = {
        "source_dataset_root": dataset_root,
        "max_per_aff": max_per_aff,
        "seed": seed,
        "selection": "shared_affordances_per_object",
    }

    for obj_type in _list_object_dirs(dataset_root, obj_filter=obj_filter):
        image_affs = _scan_image_affordances(dataset_root, obj_type)
        point_affs = _scan_point_affordances(dataset_root, obj_type)
        for aff in _iter_selected_affs(image_affs, point_affs, aff_filter=aff_filter):
            image_ids = _sample_ids(image_affs[aff], max_per_aff, rng)
            point_ids = _sample_ids(point_affs[aff], max_per_aff, rng)
            obj_norm = _normalize(obj_type)
            aff_norm = _normalize(aff)
            for img_id in image_ids:
                manifest["images"].append(
                    {
                        "name": _safe_name(f"{obj_norm}_{aff_norm}_img{img_id}"),
                        "obj_type": obj_type,
                        "img_id": img_id,
                        "aff": aff_norm,
                    }
                )
            for pc_id in point_ids:
                manifest["point_clouds"].append(
                    {
                        "name": _safe_name(f"{obj_norm}_{aff_norm}_pc{pc_id}"),
                        "obj_type": obj_type,
                        "pc_id": pc_id,
                        "aff": aff_norm,
                    }
                )

    manifest["metadata"]["num_images"] = len(manifest["images"])
    manifest["metadata"]["num_point_clouds"] = len(manifest["point_clouds"])
    return manifest


def _split_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a dataset and generate a render manifest.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--render-output-dir", default="../outputs/rendered_targets",
                        help="output_dir written into the generated manifest.")
    parser.add_argument("--max-per-aff", type=int, default=10,
                        help="Max image IDs and max point-cloud IDs sampled per shared obj-aff.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. If omitted, the first sorted IDs are used.")
    parser.add_argument("--obj-types", default=None,
                        help="Optional comma-separated object filter, e.g. Spoon,Mug.")
    parser.add_argument("--aff-types", default=None,
                        help="Optional comma-separated affordance filter.")
    parser.add_argument("--backend", default="realistic",
                        choices=("realistic", "open3d", "sphere", "matplotlib", "iagnet", "mitsuba", "mitsuba_iagnet"),
                        help="Point-cloud backend written into render.point_cloud.backend.")
    parser.add_argument("--output-mode", default="both", choices=("single", "grid", "both"),
                        help="Manifest output.mode.")
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Write absolute dataset_root instead of a path relative to the manifest.")
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    manifest = build_render_manifest(
        dataset_root=args.dataset_root,
        output_dir=args.render_output_dir,
        max_per_aff=args.max_per_aff,
        seed=args.seed,
        obj_types=_split_csv_arg(args.obj_types),
        aff_types=_split_csv_arg(args.aff_types),
        render_backend=args.backend,
        output_mode=args.output_mode,
        absolute_paths=args.absolute_paths,
        manifest_path=output_path,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Render manifest saved: {output_path}")
    print(f"Images: {len(manifest['images'])}, point clouds: {len(manifest['point_clouds'])}")


if __name__ == "__main__":
    main()
