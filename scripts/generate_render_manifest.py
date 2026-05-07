"""
生成渲染 manifest：基于数据集 metadata/split 自动构建渲染清单。

对于 metadata/split 中同时包含 Image（图像）和 PointCloud（点云）数据的每个物体，
本脚本会查找两个模态共有的 affordance。
对于每个共同的 affordance，分别随机导出不超过 N 个图像 ID 与 N 个点云 ID，组合为 manifest，
供 utils/batch_render.py 和 scripts/render_points.py 等脚本批量渲染使用。
"""
import argparse
import json
import os
import random
import shutil
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def _entry_primary_id(entry: Any) -> Optional[int]:
    """Extract the primary id from split entries: int, [id, ...], or {id: ...}."""
    try:
        if isinstance(entry, dict):
            value = entry.get("id", entry.get("img_id", entry.get("pc_id")))
            return int(value) if value not in (None, "", "None", "none") else None
        if isinstance(entry, (list, tuple)):
            return int(entry[0]) if entry else None
        return int(entry)
    except (TypeError, ValueError):
        return None


def _collect_ids_from_modality_map(modality_map: Any) -> Dict[str, Dict[str, List[int]]]:
    """Normalize {obj: {aff: [ids]}} metadata into lower-case obj/aff maps."""
    collected: Dict[str, Dict[str, Set[int]]] = {}
    if not isinstance(modality_map, dict):
        return {}

    for obj_type, aff_map in modality_map.items():
        if not isinstance(aff_map, dict):
            continue
        obj_norm = _normalize(obj_type)
        for aff_type, entries in aff_map.items():
            aff_norm = _normalize(aff_type)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                entry_id = _entry_primary_id(entry)
                if entry_id is not None:
                    collected.setdefault(obj_norm, {}).setdefault(aff_norm, set()).add(entry_id)
    return {
        obj: {aff: sorted(ids) for aff, ids in aff_map.items() if ids}
        for obj, aff_map in collected.items()
    }


def _merge_modality_ids(
    target: Dict[str, Dict[str, Set[int]]],
    source: Dict[str, Dict[str, List[int]]],
) -> None:
    for obj_type, aff_map in source.items():
        for aff_type, ids in aff_map.items():
            target.setdefault(obj_type, {}).setdefault(aff_type, set()).update(ids)


def _finalize_modality_ids(data: Dict[str, Dict[str, Set[int]]]) -> Dict[str, Dict[str, List[int]]]:
    return {
        obj: {aff: sorted(ids) for aff, ids in aff_map.items() if ids}
        for obj, aff_map in data.items()
    }


def _collect_render_ids_from_metadata(metadata: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
    """Read object/aff/id lists from split-style metadata files."""
    image_ids: Dict[str, Dict[str, Set[int]]] = {}
    point_ids: Dict[str, Dict[str, Set[int]]] = {}

    # Supports either top-level Image/PointCloud or split sections: train/val/test.
    _merge_modality_ids(image_ids, _collect_ids_from_modality_map(metadata.get("Image", metadata.get("img"))))
    _merge_modality_ids(point_ids, _collect_ids_from_modality_map(metadata.get("PointCloud", metadata.get("pc"))))
    for split_name in ("train", "val", "test"):
        split_data = metadata.get(split_name)
        if not isinstance(split_data, dict):
            continue
        _merge_modality_ids(image_ids, _collect_ids_from_modality_map(split_data.get("Image", split_data.get("img"))))
        _merge_modality_ids(point_ids, _collect_ids_from_modality_map(split_data.get("PointCloud", split_data.get("pc"))))

    return _finalize_modality_ids(image_ids), _finalize_modality_ids(point_ids)


def _candidate_metadata_files(dataset_root: str, explicit_path: Optional[str] = None) -> List[str]:
    if explicit_path:
        return [os.path.abspath(explicit_path)]
    names = (
        "render_ids.json",
        "split.json",
        "dataset_split.json",
        "all.json",
        "train.json",
        "val.json",
        "test.json",
        "info.json",
        "metadata.json",
    )
    return [os.path.join(dataset_root, name) for name in names]


def _load_ids_from_metadata(dataset_root: str, metadata_file: Optional[str] = None) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]], Optional[str]]:
    """Load Image/PointCloud ids from the first useful metadata/split file."""
    merged_images: Dict[str, Dict[str, Set[int]]] = {}
    merged_points: Dict[str, Dict[str, Set[int]]] = {}
    used_files: List[str] = []

    for path in _candidate_metadata_files(dataset_root, metadata_file):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        image_ids, point_ids = _collect_render_ids_from_metadata(data)
        if image_ids or point_ids:
            _merge_modality_ids(merged_images, image_ids)
            _merge_modality_ids(merged_points, point_ids)
            used_files.append(path)
        if metadata_file and used_files:
            break

    return _finalize_modality_ids(merged_images), _finalize_modality_ids(merged_points), ";".join(used_files) if used_files else None


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


def _find_mask_path(dataset_root: str, obj_type: str, aff_type: str, img_id: int) -> Optional[str]:
    mask_dir = os.path.join(dataset_root, obj_type, "Image", "mask", aff_type)
    normalized_obj = _normalize(obj_type)
    prefixes = [obj_type]
    if normalized_obj != obj_type:
        prefixes.append(normalized_obj)
    for prefix in prefixes:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(mask_dir, f"{prefix}_{img_id}_{aff_type}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _find_point_path(dataset_root: str, obj_type: str, pc_id: int) -> Optional[str]:
    pc_dir = os.path.join(dataset_root, obj_type, "PointCloud")
    normalized_obj = _normalize(obj_type)
    prefixes = [obj_type]
    if normalized_obj != obj_type:
        prefixes.append(normalized_obj)
    for prefix in prefixes:
        candidate = os.path.join(pc_dir, f"{prefix}_{pc_id}.csv")
        if os.path.exists(candidate):
            return candidate
    return None


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


def _filter_image_ids_by_files(dataset_root: str, obj_type: str, aff_type: str, ids: Sequence[int]) -> List[int]:
    """Keep ids whose RGB and mask files are present."""
    kept = []
    for img_id in ids:
        if not _find_rgb_path(dataset_root, obj_type, img_id):
            continue
        if _find_mask_path(dataset_root, obj_type, aff_type, img_id):
            kept.append(int(img_id))
    return sorted(set(kept))


def _filter_point_ids_by_files(dataset_root: str, obj_type: str, ids: Sequence[int]) -> List[int]:
    kept = []
    for pc_id in ids:
        if _find_point_path(dataset_root, obj_type, pc_id):
            kept.append(int(pc_id))
    return sorted(set(kept))


def _copy_file_preserve_relative(src: str, dataset_root: str, output_root: str) -> Optional[str]:
    if not src or not os.path.exists(src):
        return None
    rel_path = os.path.relpath(src, dataset_root)
    dst = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_selected_files(manifest: Dict, source_dataset_root: str, output_root: str) -> Dict[str, int]:
    """Copy selected RGB/mask/point-cloud files while preserving dataset layout."""
    source_dataset_root = os.path.abspath(source_dataset_root)
    output_root = os.path.abspath(output_root)
    copied = {"rgb": 0, "mask": 0, "point_cloud": 0}
    seen: Set[str] = set()

    for item in manifest.get("images", []):
        obj_type = item.get("obj_type")
        img_id = item.get("img_id")
        aff_type = item.get("aff")
        if obj_type is None or img_id is None or aff_type is None:
            continue
        for src, key in (
            (_find_rgb_path(source_dataset_root, obj_type, int(img_id)), "rgb"),
            (_find_mask_path(source_dataset_root, obj_type, aff_type, int(img_id)), "mask"),
        ):
            if src and src not in seen:
                _copy_file_preserve_relative(src, source_dataset_root, output_root)
                seen.add(src)
                copied[key] += 1

    for item in manifest.get("point_clouds", []):
        obj_type = item.get("obj_type")
        pc_id = item.get("pc_id")
        if obj_type is None or pc_id is None:
            continue
        src = _find_point_path(source_dataset_root, obj_type, int(pc_id))
        if src and src not in seen:
            _copy_file_preserve_relative(src, source_dataset_root, output_root)
            seen.add(src)
            copied["point_cloud"] += 1

    return copied


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
    metadata_file: Optional[str] = None,
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
        "source_metadata_file": None,
        "max_per_aff": max_per_aff,
        "seed": seed,
        "selection": "shared_affordances_per_object",
    }

    metadata_image_ids, metadata_point_ids, used_metadata = _load_ids_from_metadata(dataset_root, metadata_file=metadata_file)
    manifest["metadata"]["source_metadata_file"] = used_metadata
    if not metadata_image_ids or not metadata_point_ids:
        raise ValueError("未从 metadata/split 文件中找到可用的 Image/PointCloud ID")

    obj_candidates = sorted(set(metadata_image_ids.keys()) & set(metadata_point_ids.keys()))
    if obj_filter is not None:
        obj_candidates = [obj for obj in obj_candidates if obj in obj_filter]

    for obj_norm in obj_candidates:
        obj_type = obj_norm
        image_affs = metadata_image_ids.get(obj_norm)
        point_affs = metadata_point_ids.get(obj_norm)
        if image_affs is None or point_affs is None:
            continue

        for aff in _iter_selected_affs(image_affs, point_affs, aff_filter=aff_filter):
            image_ids = _filter_image_ids_by_files(dataset_root, obj_type, aff, image_affs[aff])
            point_ids = _filter_point_ids_by_files(dataset_root, obj_type, point_affs[aff])
            image_ids = _sample_ids(image_ids, max_per_aff, rng)
            point_ids = _sample_ids(point_ids, max_per_aff, rng)
            if not image_ids or not point_ids:
                continue
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
    parser.add_argument("--metadata-file", default=None,
                        help="Optional split/metadata JSON. If omitted, train/val/test/info/metadata files under dataset_root are tried.")
    parser.add_argument("--output-mode", default="both", choices=("single", "grid", "both"),
                        help="Manifest output.mode.")
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Write absolute dataset_root instead of a path relative to the manifest.")
    parser.add_argument("--copy-selected-to", default=None,
                        help="Optional directory to copy selected RGB/mask/point-cloud files into, preserving dataset layout.")
    parser.add_argument("--use-copy-root", action="store_true",
                        help="When --copy-selected-to is set, write manifest dataset_root to the copied subset directory.")
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
        metadata_file=args.metadata_file,
    )

    if args.copy_selected_to:
        copy_root = os.path.abspath(args.copy_selected_to)
        copied = copy_selected_files(manifest, os.path.abspath(args.dataset_root), copy_root)
        manifest["metadata"]["copied_selected_to"] = copy_root
        manifest["metadata"]["copied_selected_counts"] = copied
        if args.use_copy_root:
            manifest_base = os.path.dirname(output_path)
            manifest["dataset_root"] = _path_for_json(copy_root, manifest_base, absolute=args.absolute_paths)
        print(
            "Selected files copied: "
            f"rgb={copied['rgb']}, mask={copied['mask']}, point_cloud={copied['point_cloud']} -> {copy_root}"
        )

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Render manifest saved: {output_path}")
    print(f"Images: {len(manifest['images'])}, point clouds: {len(manifest['point_clouds'])}")


if __name__ == "__main__":
    main()
