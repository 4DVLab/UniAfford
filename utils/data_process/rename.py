"""
按 README 规范重命名数据集目录、文件名和元数据。

命名规范：
- obj_type / aff_type 使用小写字母和空格
- 下划线自动转换为空格

可直接在下面两个字典里预定义重命名规则，一键转换；若字典为空且命令行未指定
--obj/--aff/--rename-map，则自动遍历输入目录下的所有物体和 affordance 并转换为标准形式。
"""
import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from tqdm import tqdm


# 需要人工合并同义名时，直接在这里填，例如 {"wrapgrasp": "wrap-grasp"}。
OBJ_RENAME_MAP = {
    "Top door of refrigerator": "refrigerator",
    "Bottom door of refrigerator": "refrigerator",
    "Right door of refrigerator": "refrigerator",
    "Left door of refrigerator": "refrigerator",
    "Top door of refrigertor": "refrigerator",
    "Bottom door of refrigertor": "refrigerator",
    "Right door of refrigertor": "refrigerator",
    "Left door of refrigertor": "refrigerator",
    "Top door of referigertor": "refrigerator",
    "Bottom door of referigertor": "refrigerator",
    "Right door of referigertor": "refrigerator",
    "Left door of referigertor": "refrigerator",
    "Left door of wardrobe": "wardrobe",
    "Power-drill": "power drill",

    # "scroll wheel": "computer mouse",
}
AFF_RENAME_MAP = {
    "wrap-grasp": "wrapgrasp",
    "secure grip": "grip",
    "secure-grip": "grip",
    "pinch grip": "pinch",
    "pick-up": "pick up",
    # n. -> v.
    "grasping": "grasp",
    "stiring": "stir",
    "turning": "turn",
    "cutting": "cut",
    "peeling": "peel",
    "punching": "punch",
    "screwing": "screw",
    "hammering": "hammer",
    "drilling": "drill",
    "sawing": "saw",
    "chopping": "chop",
    "opening": "open",
    "closing": "close",
    "lifting": "lift",
    "placing": "place",
    "putting": "put",
    "taking": "take",
    "putting": "put",
    "unclasping": "unclasp",
    
}


def standardize_name(name):
    if name is None:
        return ""
    name = str(name).strip().replace("_", " ").lower()
    return re.sub(r"\s+", " ", name)


def normalize_mapping(mapping):
    """将重命名字典的 key/value 都先标准化，保证不同大小写/下划线写法可匹配。"""
    return {
        standardize_name(old): standardize_name(new)
        for old, new in (mapping or {}).items()
        if standardize_name(old) and standardize_name(new)
    }


def load_rename_map(path):
    if not path:
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "objects" in data or "affordances" in data:
        return data.get("objects", {}), data.get("affordances", {})
    return data.get("obj", {}), data.get("aff", {})


def mapped_name(name, rename_map):
    """先标准化待匹配名称，再查标准化后的重命名字典；未命中时返回标准化名称。"""
    std = standardize_name(name)
    return rename_map.get(std, std)


def collect_names_from_json_value(value, objects, affordances):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "obj_type" and isinstance(item, str):
                objects.add(item)
            elif key == "aff_type" and isinstance(item, str):
                affordances.add(item)
            collect_names_from_json_value(item, objects, affordances)
    elif isinstance(value, list):
        for item in value:
            collect_names_from_json_value(item, objects, affordances)


def collect_names_from_metadata(dataset_root):
    """只从轻量 JSON 元数据收集名称，避免逐个扫描图片/点云文件。"""
    objects = set()
    affordances = set()
    for filename in ("metadata.json", "info.json", "train.json", "val.json", "test.json"):
        path = os.path.join(dataset_root, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if filename == "info.json" and isinstance(data, dict):
            for obj_dict in data.values():
                if not isinstance(obj_dict, dict):
                    continue
                for obj, aff_dict in obj_dict.items():
                    objects.add(obj)
                    if isinstance(aff_dict, dict):
                        affordances.update(k for k in aff_dict.keys() if k != "ID")
        collect_names_from_json_value(data, objects, affordances)
    return objects, affordances


def build_full_maps(dataset_root, obj_rename=None, aff_rename=None, map_file=None):
    file_obj_map, file_aff_map = load_rename_map(map_file)
    obj_map = normalize_mapping({**OBJ_RENAME_MAP, **file_obj_map})
    aff_map = normalize_mapping({**AFF_RENAME_MAP, **file_aff_map})

    if obj_rename:
        old, new = obj_rename
        obj_map[standardize_name(old)] = standardize_name(new)
    if aff_rename:
        old, new = aff_rename
        aff_map[standardize_name(old)] = standardize_name(new)

    objects, affordances = collect_names_from_metadata(dataset_root)
    for obj in objects:
        std = standardize_name(obj)
        obj_map.setdefault(std, std)
    for aff in affordances:
        std = standardize_name(aff)
        aff_map.setdefault(std, std)
    return obj_map, aff_map


def extract_sample_id(filename, old_obj_name):
    stem, _ = os.path.splitext(filename)
    if stem.startswith(old_obj_name + "_"):
        rest = stem[len(old_obj_name) + 1:]
        return rest.split("_", 1)[0]
    match = re.search(r"_(\d+)(?:_|$)", stem)
    if match:
        return match.group(1)
    return None


def extract_sample_suffix(filename, old_obj_name):
    stem, _ = os.path.splitext(filename)
    if stem.startswith(old_obj_name + "_"):
        rest = stem[len(old_obj_name) + 1:]
        parts = rest.split("_", 1)
        return parts[1] if len(parts) > 1 else ""
    match = re.search(r"_(\d+)(?:_(.*))?$", stem)
    return match.group(2) or "" if match else ""


def format_dataset_filename(new_obj_name, new_id, ext, aff_name=None, suffix=None):
    ext = ext.lower()
    if aff_name:
        return f"{new_obj_name}_{new_id}_{aff_name}{ext}"
    if suffix:
        return f"{new_obj_name}_{new_id}_{suffix}{ext}"
    return f"{new_obj_name}_{new_id}{ext}"


def _modality_state(rename_state, obj_name, modality):
    obj_state = rename_state[obj_name]
    if modality not in obj_state:
        obj_state[modality] = {"next_id": 1, "id_map": {}}
    return obj_state[modality]


def allocate_modality_id(rename_state, obj_name, modality, old_obj_name, old_id):
    state = _modality_state(rename_state, obj_name, modality)
    key = (old_obj_name, str(old_id))
    if key not in state["id_map"]:
        state["id_map"][key] = state["next_id"]
        state["next_id"] += 1
    return state["id_map"][key]


def lookup_modality_id(rename_state, obj_name, modality, old_obj_name, old_id):
    if not old_id:
        return ""
    state = _modality_state(rename_state, obj_name, modality)
    return state["id_map"].get((old_obj_name, str(old_id)), old_id)


def write_instruction(src_path, dst_path, old_obj_name, obj_name, obj_map, aff_map, rename_state):
    fieldnames = ["ins", "obj_type", "aff_type", "id", "img_id", "pc_id"]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    write_header = not os.path.exists(dst_path)
    with open(src_path, "r", newline="", encoding="utf-8-sig") as input_f, open(
        dst_path, "a", newline="", encoding="utf-8"
    ) as output_f:
        reader = csv.DictReader(input_f)
        writer = csv.DictWriter(output_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in reader:
            old_id = row.get("id", "") or f"row-{reader.line_num}"
            new_id = allocate_modality_id(rename_state, obj_name, "Instruction", old_obj_name, old_id)
            writer.writerow({
                "ins": row.get("ins", ""),
                "obj_type": mapped_name(row.get("obj_type") or obj_name, obj_map),
                "aff_type": mapped_name(row.get("aff_type", ""), aff_map),
                "id": new_id,
                "img_id": lookup_modality_id(rename_state, obj_name, "Image", old_obj_name, row.get("img_id", "")),
                "pc_id": lookup_modality_id(rename_state, obj_name, "PointCloud", old_obj_name, row.get("pc_id", "")),
            })


def write_pointcloud(src_path, dst_path, aff_map):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # 只重写首行 header 中的 aff 列名；首行之后的点云数值数据用二进制原样透传。
    with open(src_path, "rb") as input_f, open(dst_path, "wb") as output_f:
        first = input_f.readline()
        if first:
            newline = b"\r\n" if first.endswith(b"\r\n") else b"\n"
            line = first.rstrip(b"\r\n").decode("utf-8")
            prefix = "# " if line.startswith("# ") else ""
            clean_header = line[2:] if line.startswith("# ") else line
            cols = [col.strip() for col in clean_header.split(",")]
            if len(cols) > 3:
                cols = cols[:3] + [mapped_name(col, aff_map) for col in cols[3:]]
            output_f.write((prefix + ",".join(cols)).encode("utf-8") + newline)
        shutil.copyfileobj(input_f, output_f)


def copy_image_dir(src_dir, dst_dir, old_obj_name, new_obj_name, aff_map, rename_state):
    rgb_dir = os.path.join(src_dir, "rgb")
    if os.path.isdir(rgb_dir):
        dst_rgb_dir = os.path.join(dst_dir, "rgb")
        os.makedirs(dst_rgb_dir, exist_ok=True)
        for filename in tqdm(os.listdir(rgb_dir), desc=f"Image-rgb-{old_obj_name}", leave=False):
            src_file = os.path.join(rgb_dir, filename)
            if not os.path.isfile(src_file):
                continue
            old_id = extract_sample_id(filename, old_obj_name) or filename
            new_id = allocate_modality_id(rename_state, new_obj_name, "Image", old_obj_name, old_id)
            _, ext = os.path.splitext(filename)
            dst_name = format_dataset_filename(new_obj_name, new_id, ext)
            shutil.copy2(src_file, os.path.join(dst_rgb_dir, dst_name))

    mask_dir = os.path.join(src_dir, "mask")
    if os.path.isdir(mask_dir):
        for aff_dir in tqdm(os.listdir(mask_dir), desc=f"Mask-{old_obj_name}", leave=False):
            src_aff_dir = os.path.join(mask_dir, aff_dir)
            if not os.path.isdir(src_aff_dir):
                continue
            new_aff = mapped_name(aff_dir, aff_map)
            dst_aff_dir = os.path.join(dst_dir, "mask", new_aff)
            os.makedirs(dst_aff_dir, exist_ok=True)
            for filename in tqdm(os.listdir(src_aff_dir), desc=f"{new_aff}", leave=False):
                src_file = os.path.join(src_aff_dir, filename)
                if not os.path.isfile(src_file):
                    continue
                old_id = extract_sample_id(filename, old_obj_name) or filename
                new_id = allocate_modality_id(rename_state, new_obj_name, "Image", old_obj_name, old_id)
                _, ext = os.path.splitext(filename)
                dst_name = format_dataset_filename(new_obj_name, new_id, ext, aff_name=new_aff)
                shutil.copy2(src_file, os.path.join(dst_aff_dir, dst_name))

    for subdir in tqdm(os.listdir(src_dir), desc=f"Image-extra-{old_obj_name}", leave=False):
        if subdir in {"rgb", "mask"}:
            continue
        src_subdir = os.path.join(src_dir, subdir)
        if not os.path.isdir(src_subdir):
            continue
        dst_subdir = os.path.join(dst_dir, subdir)
        os.makedirs(dst_subdir, exist_ok=True)
        for filename in tqdm(os.listdir(src_subdir), desc=f"{old_obj_name}/{subdir}", leave=False):
            src_file = os.path.join(src_subdir, filename)
            if not os.path.isfile(src_file):
                continue
            old_id = extract_sample_id(filename, old_obj_name) or filename
            new_id = allocate_modality_id(rename_state, new_obj_name, "Image", old_obj_name, old_id)
            suffix = extract_sample_suffix(filename, old_obj_name)
            _, ext = os.path.splitext(filename)
            dst_name = format_dataset_filename(new_obj_name, new_id, ext, suffix=suffix)
            shutil.copy2(src_file, os.path.join(dst_subdir, dst_name))


def merge_json_dict(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge_json_dict(dst[key], value)
        else:
            dst[key] = value


def rewrite_info(src_path, dst_path, obj_map, aff_map):
    with open(src_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    output = {}
    for modality, obj_dict in info.items():
        output[modality] = defaultdict(dict)
        for obj, aff_dict in (obj_dict or {}).items():
            new_obj = mapped_name(obj, obj_map)
            for aff, value in (aff_dict or {}).items():
                new_aff = "ID" if aff == "ID" else mapped_name(aff, aff_map)
                cur = output[modality].setdefault(new_obj, {}).get(new_aff, 0)
                output[modality][new_obj][new_aff] = cur + value
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def remap_split_entry(entry, modality, old_obj, new_obj, rename_state):
    if isinstance(entry, dict):
        remapped = dict(entry)
        if entry.get("id") not in (None, "", "None", "none"):
            remapped["id"] = lookup_modality_id(rename_state, new_obj, "Instruction", old_obj, entry.get("id"))
        if entry.get("img_id") not in (None, "", "None", "none"):
            remapped["img_id"] = lookup_modality_id(rename_state, new_obj, "Image", old_obj, entry.get("img_id"))
        if entry.get("pc_id") not in (None, "", "None", "none"):
            remapped["pc_id"] = lookup_modality_id(rename_state, new_obj, "PointCloud", old_obj, entry.get("pc_id"))
        return remapped
    return lookup_modality_id(rename_state, new_obj, modality, old_obj, entry)


def rewrite_split_ids(data, obj_map, aff_map, rename_state):
    """同步重写 split 文件中的 obj/aff key 和因合并产生的新 ID。"""
    if not isinstance(data, dict):
        return rewrite_split_value(data, obj_map, aff_map)

    out = {}
    for modality, obj_data in data.items():
        if modality not in {"Instruction", "Image", "PointCloud"} or not isinstance(obj_data, dict):
            out[modality] = rewrite_split_value(obj_data, obj_map, aff_map)
            continue
        out.setdefault(modality, {})
        for old_obj, aff_data in obj_data.items():
            if not isinstance(aff_data, dict):
                continue
            new_obj = mapped_name(old_obj, obj_map)
            out[modality].setdefault(new_obj, {})
            for old_aff, entries in aff_data.items():
                new_aff = mapped_name(old_aff, aff_map)
                if not isinstance(entries, list):
                    entries = [entries]
                remapped_entries = [
                    remap_split_entry(entry, modality, old_obj, new_obj, rename_state)
                    for entry in entries
                ]
                out[modality][new_obj].setdefault(new_aff, []).extend(remapped_entries)
    return out


def rewrite_split_value(value, obj_map, aff_map, parent_key=None):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            new_key = key
            if standardize_name(key) in obj_map:
                new_key = mapped_name(key, obj_map)
            elif standardize_name(key) in aff_map:
                new_key = mapped_name(key, aff_map)
            out[new_key] = rewrite_split_value(item, obj_map, aff_map, parent_key=key)
        return out
    if isinstance(value, list):
        return [rewrite_split_value(item, obj_map, aff_map, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        if parent_key == "obj_type":
            return mapped_name(value, obj_map)
        if parent_key == "aff_type":
            return mapped_name(value, aff_map)
    return value


def rewrite_metadata_counts(counts, obj_map, aff_map):
    """重写 metadata.obj_aff_count_by_split 中的 obj/aff 统计 key。"""
    if not isinstance(counts, dict):
        return counts
    out = {}
    for split_name, split_data in counts.items():
        if not isinstance(split_data, dict):
            out[split_name] = split_data
            continue
        out[split_name] = {}
        for modality, modality_data in split_data.items():
            if not isinstance(modality_data, dict):
                out[split_name][modality] = modality_data
                continue
            out[split_name][modality] = {}
            for obj_name, obj_counts in modality_data.items():
                if obj_name == "total":
                    out[split_name][modality][obj_name] = obj_counts
                    continue
                new_obj = mapped_name(obj_name, obj_map)
                dst_obj_counts = out[split_name][modality].setdefault(new_obj, {})
                if not isinstance(obj_counts, dict):
                    out[split_name][modality][new_obj] = obj_counts
                    continue
                for aff_name, count in obj_counts.items():
                    if aff_name == "total":
                        dst_obj_counts[aff_name] = dst_obj_counts.get(aff_name, 0) + count
                    else:
                        new_aff = mapped_name(aff_name, aff_map)
                        dst_obj_counts[new_aff] = dst_obj_counts.get(new_aff, 0) + count
    return out


def rewrite_metadata_value(value, obj_map, aff_map, parent_key=None):
    """彻底重写 metadata 中可能出现的 obj/aff 名称，同时保留系统字段名。"""
    protected_keys = {
        "ratios", "train", "val", "test", "random_seed", "total_samples",
        "paired_samples", "id_source", "balance_modalities", "sample_rate",
        "max_sample_per_group", "min_sample_per_group", "obj_aff_count_by_split",
        "Instruction", "Image", "PointCloud", "total",
    }
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "obj_aff_count_by_split":
                out[key] = rewrite_metadata_counts(item, obj_map, aff_map)
                continue
            if key in protected_keys:
                new_key = key
            elif parent_key in {"Instruction", "Image", "PointCloud"}:
                new_key = mapped_name(key, obj_map)
            elif isinstance(item, (int, float)) and key not in protected_keys:
                new_key = mapped_name(key, aff_map)
            else:
                key_as_obj = mapped_name(key, obj_map)
                key_as_aff = mapped_name(key, aff_map)
                new_key = key_as_obj if key_as_obj != standardize_name(key) else key_as_aff
            out[new_key] = rewrite_metadata_value(item, obj_map, aff_map, parent_key=new_key)
        return out
    if isinstance(value, list):
        return [rewrite_metadata_value(item, obj_map, aff_map, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        if parent_key in {"obj", "obj_type"}:
            return mapped_name(value, obj_map)
        if parent_key in {"aff", "aff_type"}:
            return mapped_name(value, aff_map)
    return value


def copy_object_dir(src_dir, dst_dir, old_obj_name, new_obj_name, obj_map, aff_map, rename_state):
    os.makedirs(dst_dir, exist_ok=True)
    entries = os.listdir(src_dir)

    pc_src_dir = os.path.join(src_dir, "PointCloud")
    if os.path.isdir(pc_src_dir):
        pc_dst = os.path.join(dst_dir, "PointCloud")
        os.makedirs(pc_dst, exist_ok=True)
        for filename in tqdm(os.listdir(pc_src_dir), desc=f"PointCloud-{old_obj_name}", leave=False):
            pc_src = os.path.join(pc_src_dir, filename)
            if os.path.isfile(pc_src) and filename.endswith(".csv"):
                old_id = extract_sample_id(filename, old_obj_name) or filename
                new_id = allocate_modality_id(rename_state, new_obj_name, "PointCloud", old_obj_name, old_id)
                _, ext = os.path.splitext(filename)
                pc_name = format_dataset_filename(new_obj_name, new_id, ext)
                write_pointcloud(pc_src, os.path.join(pc_dst, pc_name), aff_map)

    image_src_dir = os.path.join(src_dir, "Image")
    if os.path.isdir(image_src_dir):
        copy_image_dir(image_src_dir, os.path.join(dst_dir, "Image"), old_obj_name, new_obj_name, aff_map, rename_state)

    ins_src = os.path.join(src_dir, "Instruction.csv")
    if os.path.isfile(ins_src):
        write_instruction(
            ins_src,
            os.path.join(dst_dir, "Instruction.csv"),
            old_obj_name,
            new_obj_name,
            obj_map,
            aff_map,
            rename_state,
        )

    handled = {"PointCloud", "Image", "Instruction.csv"}
    for entry in tqdm(entries, desc=f"Object-extra-{old_obj_name}", leave=False):
        if entry in handled:
            continue
        src_path = os.path.join(src_dir, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(dst_dir, entry), dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, os.path.join(dst_dir, entry))


def rewrite_dataset(dataset_root, obj_map, aff_map):
    tmp_root = dataset_root.rstrip(os.sep) + ".rename_tmp"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)
    rename_state = defaultdict(dict)

    entries = os.listdir(dataset_root)
    dir_entries = [entry for entry in entries if os.path.isdir(os.path.join(dataset_root, entry))]
    file_entries = [entry for entry in entries if not os.path.isdir(os.path.join(dataset_root, entry))]

    for entry in tqdm(dir_entries, desc="重写物体目录"):
        src_path = os.path.join(dataset_root, entry)
        new_obj = mapped_name(entry, obj_map)
        copy_object_dir(src_path, os.path.join(tmp_root, new_obj), entry, new_obj, obj_map, aff_map, rename_state)

    for entry in tqdm(file_entries, desc="重写元数据"):
        src_path = os.path.join(dataset_root, entry)
        dst_path = os.path.join(tmp_root, entry)
        if entry == "info.json":
            rewrite_info(src_path, dst_path, obj_map, aff_map)
        elif entry in {"train.json", "val.json", "test.json"}:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(rewrite_split_ids(data, obj_map, aff_map, rename_state), f, ensure_ascii=False, indent=2)
        elif entry == "metadata.json":
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(rewrite_metadata_value(data, obj_map, aff_map), f, ensure_ascii=False, indent=2)
        else:
            shutil.copy2(src_path, dst_path)

    for entry in tqdm(os.listdir(dataset_root), desc="清理旧数据", leave=False):
        path = os.path.join(dataset_root, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for entry in tqdm(os.listdir(tmp_root), desc="写回新数据", leave=False):
        shutil.move(os.path.join(tmp_root, entry), os.path.join(dataset_root, entry))
    shutil.rmtree(tmp_root)


def main():
    parser = argparse.ArgumentParser(description="按 README 规范重命名数据集中的物体和 affordance 名称")
    parser.add_argument("-d", "--dataset-root", type=str, required=True, help="数据集根目录")
    parser.add_argument("-o", "--obj", type=str, nargs="+", metavar=("OLD", "NEW"), help="重命名物体类型")
    parser.add_argument("-a", "--aff", type=str, nargs=2, metavar=("OLD", "NEW"), help="重命名 affordance 类型")
    parser.add_argument("--rename-map", type=str, default=None, help="JSON 重命名字典，格式 {'objects':{}, 'affordances':{}}")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise ValueError(f"数据集根目录不存在: {dataset_root}")

    obj_rename = None
    if args.obj:
        if len(args.obj) == 1:
            obj_rename = (args.obj[0], args.obj[0])
        elif len(args.obj) == 2:
            obj_rename = tuple(args.obj)
        else:
            raise ValueError("参数 --obj 只允许 1 个或 2 个值")
    aff_rename = tuple(args.aff) if args.aff else None

    obj_map, aff_map = build_full_maps(dataset_root, obj_rename=obj_rename, aff_rename=aff_rename, map_file=args.rename_map)
    print("物体重命名映射:")
    for old, new in sorted(obj_map.items()):
        if old != new:
            print(f"  {old} -> {new}")
    print("Affordance 重命名映射:")
    for old, new in sorted(aff_map.items()):
        if old != new:
            print(f"  {old} -> {new}")

    rewrite_dataset(dataset_root, obj_map, aff_map)
    print(f"完成重命名并规范化数据集: {dataset_root}")


if __name__ == "__main__":
    main()
