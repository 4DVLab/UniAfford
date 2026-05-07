import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODALITIES = ("Instruction", "Image", "PointCloud")
ALIASES = {
    "Instruction": ("Instruction", "ins"),
    "Image": ("Image", "img"),
    "PointCloud": ("PointCloud", "pc"),
}


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None", "none"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry_primary_id(entry: Any) -> Optional[int]:
    if isinstance(entry, dict):
        return _optional_int(entry.get("id", entry.get("ins_id")))
    if isinstance(entry, (list, tuple)):
        return _optional_int(entry[0]) if entry else None
    return _optional_int(entry)


def _entry_linked_id(entry: Any, key: str) -> Optional[int]:
    if not isinstance(entry, dict):
        return None
    return _optional_int(entry.get(key))


def _normalize_instruction_entry(entry: Any) -> Optional[Dict[str, Optional[int]]]:
    if isinstance(entry, str):
        stripped = entry.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                pass

    if isinstance(entry, dict):
        ins_id = _entry_primary_id(entry)
        img_id = _entry_linked_id(entry, "img_id")
        pc_id = _entry_linked_id(entry, "pc_id")
        if ins_id is None and img_id is None and pc_id is None:
            return None
        return {"id": ins_id, "img_id": img_id, "pc_id": pc_id}

    if isinstance(entry, (list, tuple)):
        if not entry:
            return None
        return {
            "id": _optional_int(entry[0]),
            "img_id": _optional_int(entry[1]) if len(entry) > 1 else None,
            "pc_id": _optional_int(entry[2]) if len(entry) > 2 else None,
        }

    ins_id = _optional_int(entry)
    if ins_id is None:
        return None
    return {"id": ins_id, "img_id": None, "pc_id": None}


def _entry_key(entry: Any) -> Any:
    if isinstance(entry, dict):
        return (_entry_primary_id(entry), _entry_linked_id(entry, "img_id"), _entry_linked_id(entry, "pc_id"))
    if isinstance(entry, (list, tuple)):
        return tuple(entry)
    return entry


def _dedup_entries(entries: Iterable[Any]) -> List[Any]:
    out = []
    seen = set()
    for entry in entries:
        key = _entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _normalize_split(raw: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    split = {mod: {} for mod in MODALITIES}
    for mod, keys in ALIASES.items():
        src = {}
        for key in keys:
            if key in raw:
                src = raw.get(key) or {}
                break
        for obj, aff_map in src.items():
            split[mod].setdefault(str(obj), {})
            for aff, entries in (aff_map or {}).items():
                if mod == "Instruction":
                    cleaned = [
                        normalized
                        for entry in (entries or [])
                        for normalized in [_normalize_instruction_entry(entry)]
                        if normalized is not None
                    ]
                else:
                    cleaned = [
                        entry
                        for entry in (entries or [])
                        if _entry_primary_id(entry) is not None
                    ]
                if cleaned:
                    split[mod][str(obj)][str(aff)] = _dedup_entries(cleaned)
    return split


def _entry_by_id(entries: Iterable[Any]) -> Dict[int, Any]:
    out = {}
    for entry in entries or []:
        entry_id = _entry_primary_id(entry)
        if entry_id is not None:
            out[entry_id] = entry
    return out


def _next_unused(entries: List[Any], used_ids: set[int]) -> Optional[Any]:
    for entry in entries:
        entry_id = _entry_primary_id(entry)
        if entry_id is None or entry_id not in used_ids:
            if entry_id is not None:
                used_ids.add(entry_id)
            return entry
    return None


def _compose_group_entries(
    ins_entries: List[Any],
    img_entries: List[Any],
    pc_entries: List[Any],
) -> List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
    """Mirror JointDataset's grouping of one object-affordance split group."""
    remaining_img = list(img_entries or [])
    remaining_pc = list(pc_entries or [])
    img_entry_by_id = _entry_by_id(remaining_img)
    pc_entry_by_id = _entry_by_id(remaining_pc)
    used_img_ids: set[int] = set()
    used_pc_ids: set[int] = set()
    grouped = []

    for ins_entry in ins_entries or []:
        normalized_ins = _normalize_instruction_entry(ins_entry)
        if normalized_ins is None:
            continue

        bound_img_id = normalized_ins.get("img_id")
        bound_pc_id = normalized_ins.get("pc_id")

        if bound_img_id is not None:
            img_entry = img_entry_by_id.get(bound_img_id, bound_img_id)
            used_img_ids.add(bound_img_id)
        else:
            img_entry = _next_unused(remaining_img, used_img_ids)

        if bound_pc_id is not None:
            pc_entry = pc_entry_by_id.get(bound_pc_id, bound_pc_id)
            used_pc_ids.add(bound_pc_id)
        else:
            pc_entry = _next_unused(remaining_pc, used_pc_ids)

        grouped.append((normalized_ins, img_entry, pc_entry))

    leftover_img = [entry for entry in remaining_img if _entry_primary_id(entry) not in used_img_ids]
    leftover_pc = [entry for entry in remaining_pc if _entry_primary_id(entry) not in used_pc_ids]
    for i in range(max(len(leftover_img), len(leftover_pc))):
        grouped.append((
            None,
            leftover_img[i] if i < len(leftover_img) else None,
            leftover_pc[i] if i < len(leftover_pc) else None,
        ))

    return grouped


def compute_split_stats(split: Dict[str, Any]) -> Dict[str, int]:
    split = _normalize_split(split)
    objects = set()
    affordances = set()
    object_affordance_pairs = set()
    image_only = 0
    point_cloud_only = 0
    paired = 0

    all_objects = set().union(*(set(split[mod].keys()) for mod in MODALITIES))
    for obj in all_objects:
        all_affs = set()
        for mod in MODALITIES:
            all_affs.update(split[mod].get(obj, {}).keys())
        for aff in all_affs:
            objects.add(obj)
            affordances.add(aff)
            object_affordance_pairs.add((obj, aff))

            groups = _compose_group_entries(
                split["Instruction"].get(obj, {}).get(aff, []),
                split["Image"].get(obj, {}).get(aff, []),
                split["PointCloud"].get(obj, {}).get(aff, []),
            )
            for _, img_entry, pc_entry in groups:
                has_image = img_entry is not None
                has_pc = pc_entry is not None
                if has_image and has_pc:
                    paired += 1
                elif has_image:
                    image_only += 1
                elif has_pc:
                    point_cloud_only += 1

    return {
        "image_only": image_only,
        "point_cloud_only": point_cloud_only,
        "paired": paired,
        "objects": len(objects),
        "affordances": len(affordances),
        "object_affordance_pairs": len(object_affordance_pairs),
    }


def load_split(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_split_paths(dataset_root: str, split_names: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for split_name in split_names:
        filename = split_name if split_name.endswith(".json") else f"{split_name}.json"
        path = os.path.join(dataset_root, filename)
        if os.path.exists(path):
            yield split_name.removesuffix(".json"), path


def format_latex_rows(rows: List[Tuple[str, Dict[str, int]]]) -> str:
    lines = []
    for split_name, stats in rows:
        label = split_name[:1].upper() + split_name[1:]
        lines.append(
            f"{label} & {stats['image_only']} & {stats['point_cloud_only']} & "
            f"{stats['paired']} & {stats['objects']} & {stats['affordances']} & "
            f"{stats['object_affordance_pairs']} \\\\"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract UniAfford-Data split statistics for the appendix table."
    )
    parser.add_argument("-d", "--dataset-root", required=True, help="Dataset root containing train/val/test JSON files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Split names or JSON filenames.")
    parser.add_argument("--format", choices=["latex", "json"], default="latex", help="Output format.")
    args = parser.parse_args()

    rows = []
    for split_name, path in iter_split_paths(os.path.abspath(args.dataset_root), args.splits):
        rows.append((split_name, compute_split_stats(load_split(path))))

    if not rows:
        raise FileNotFoundError(f"No split files found under {args.dataset_root}: {args.splits}")

    if args.format == "json":
        print(json.dumps({name: stats for name, stats in rows}, ensure_ascii=False, indent=2))
    else:
        print(format_latex_rows(rows))


if __name__ == "__main__":
    main()
