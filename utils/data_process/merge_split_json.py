import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

try:
    from create_split import _dump_split_part, _entry_sort_key, _normalize_ids, SplitManager
except ImportError:
    from utils.data_process.create_split import _dump_split_part, _entry_sort_key, _normalize_ids, SplitManager


MODALITIES = ("Instruction", "Image", "PointCloud")


def _entry_key(entry) -> Tuple:
    if isinstance(entry, dict):
        return ("dict", entry.get("id"), entry.get("img_id"), entry.get("pc_id"))
    return ("id", entry)


def _merge_entry_lists(existing: List, incoming: Iterable) -> List:
    merged = list(existing)
    seen = {_entry_key(entry) for entry in merged}
    for entry in incoming:
        key = _entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    return sorted(merged, key=_entry_sort_key)


def load_split_json(path: str) -> Dict[str, Dict[str, Dict[str, list]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_ids(data)


def merge_split_json_files(input_paths: Iterable[str]) -> Dict[str, Dict[str, Dict[str, list]]]:
    merged = {modality: {} for modality in MODALITIES}
    for input_path in input_paths:
        split_ids = load_split_json(input_path)
        for modality in MODALITIES:
            for obj_name, aff_map in split_ids.get(modality, {}).items():
                obj_bucket = merged[modality].setdefault(obj_name, {})
                for aff_name, ids in aff_map.items():
                    obj_bucket[aff_name] = _merge_entry_lists(obj_bucket.get(aff_name, []), ids)
    return merged


def save_split_json(split_ids: Dict[str, Dict[str, Dict[str, list]]], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        _dump_split_part(split_ids, f)


def build_metadata(split_ids: Dict[str, Dict[str, Dict[str, list]]], input_paths: Iterable[str]) -> Dict:
    return {
        "source_files": [os.path.abspath(path) for path in input_paths],
        "total_sample": SplitManager._count_paired_samples(split_ids),
        "obj_aff_count": SplitManager._counts(split_ids),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并多个 split JSON 文件，按 modality/object/affordance 对 id 去重合并。"
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="输入 split JSON 文件列表，例如 train.json val.json other_val.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出的新 split JSON 文件路径",
    )
    parser.add_argument(
        "--metadata_output",
        default=None,
        help="可选：保存合并统计 metadata JSON 的路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for input_path in args.inputs:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入 split JSON 不存在: {input_path}")

    merged = merge_split_json_files(args.inputs)
    save_split_json(merged, args.output)

    metadata = build_metadata(merged, args.inputs)
    if args.metadata_output:
        metadata_dir = os.path.dirname(os.path.abspath(args.metadata_output))
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(args.metadata_output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"已合并 {len(args.inputs)} 个 split JSON: {args.output}")
    print(f"合并后 paired sample 估计数: {metadata['total_sample']}")


if __name__ == "__main__":
    main()
