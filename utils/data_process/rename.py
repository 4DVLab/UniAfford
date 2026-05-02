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
    "Top door of refrigertor": "refrigerator",
    "Bottom door of refrigertor": "refrigerator",
    "Right door of refrigertor": "refrigerator",
    "Left door of refrigertor": "refrigerator",
    "Power-drill": "power drill",

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


def rewrite_dataset_filename(filename, old_obj_name, new_obj_name, aff_name=None):
    _, ext = os.path.splitext(filename)
    sample_id = extract_sample_id(filename, old_obj_name)
    if sample_id:
        if aff_name:
            return f"{new_obj_name}_{sample_id}_{aff_name}{ext.lower()}"
        return f"{new_obj_name}_{sample_id}{ext.lower()}"
    stem, ext = os.path.splitext(filename)
    return f"{standardize_name(stem)}{ext.lower()}"


def write_instruction(src_path, dst_path, obj_name, obj_map, aff_map):
    fieldnames = ["ins", "obj_type", "aff_type", "id", "img_id", "pc_id"]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", newline="", encoding="utf-8-sig") as input_f, open(
        dst_path, "w", newline="", encoding="utf-8"
    ) as output_f:
        reader = csv.DictReader(input_f)
        writer = csv.DictWriter(output_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "ins": row.get("ins", ""),
                "obj_type": mapped_name(row.get("obj_type") or obj_name, obj_map),
                "aff_type": mapped_name(row.get("aff_type", ""), aff_map),
                "id": row.get("id", ""),
                "img_id": row.get("img_id", ""),
                "pc_id": row.get("pc_id", ""),
            })


def write_pointcloud(src_path, dst_path, aff_map):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as input_f, open(dst_path, "w", encoding="utf-8") as output_f:
        first = input_f.readline()
        if first:
            prefix = "# " if first.startswith("# ") else ""
            clean_header = first[2:] if first.startswith("# ") else first
            cols = [col.strip() for col in clean_header.strip().split(",")]
            if len(cols) > 3:
                cols = cols[:3] + [mapped_name(col, aff_map) for col in cols[3:]]
            output_f.write(prefix + ",".join(cols) + "\n")
        shutil.copyfileobj(input_f, output_f)


def copy_image_dir(src_dir, dst_dir, old_obj_name, new_obj_name, aff_map):
    subdirs = os.listdir(src_dir)
    for subdir in tqdm(subdirs, desc=f"Image-{old_obj_name}", leave=False):
        src_subdir = os.path.join(src_dir, subdir)
        if not os.path.isdir(src_subdir):
            continue
        if subdir == "mask":
            for aff_dir in tqdm(os.listdir(src_subdir), desc=f"Mask-{old_obj_name}", leave=False):
                src_aff_dir = os.path.join(src_subdir, aff_dir)
                if not os.path.isdir(src_aff_dir):
                    continue
                new_aff = mapped_name(aff_dir, aff_map)
                dst_aff_dir = os.path.join(dst_dir, "mask", new_aff)
                os.makedirs(dst_aff_dir, exist_ok=True)
                for filename in tqdm(os.listdir(src_aff_dir), desc=f"{new_aff}", leave=False):
                    src_file = os.path.join(src_aff_dir, filename)
                    if os.path.isfile(src_file):
                        dst_name = rewrite_dataset_filename(filename, old_obj_name, new_obj_name, new_aff)
                        shutil.copy2(src_file, os.path.join(dst_aff_dir, dst_name))
        else:
            dst_subdir = os.path.join(dst_dir, subdir)
            os.makedirs(dst_subdir, exist_ok=True)
            for filename in tqdm(os.listdir(src_subdir), desc=f"{old_obj_name}/{subdir}", leave=False):
                src_file = os.path.join(src_subdir, filename)
                if os.path.isfile(src_file):
                    dst_name = rewrite_dataset_filename(filename, old_obj_name, new_obj_name)
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
                output[modality].setdefault(new_obj, {})[new_aff] = value
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


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


def copy_object_dir(src_dir, dst_dir, old_obj_name, new_obj_name, obj_map, aff_map):
    os.makedirs(dst_dir, exist_ok=True)
    for entry in tqdm(os.listdir(src_dir), desc=f"Object-{old_obj_name}", leave=False):
        src_path = os.path.join(src_dir, entry)
        if entry == "Instruction.csv" and os.path.isfile(src_path):
            write_instruction(src_path, os.path.join(dst_dir, entry), new_obj_name, obj_map, aff_map)
        elif entry == "PointCloud" and os.path.isdir(src_path):
            pc_dst = os.path.join(dst_dir, entry)
            os.makedirs(pc_dst, exist_ok=True)
            for filename in tqdm(os.listdir(src_path), desc=f"PointCloud-{old_obj_name}", leave=False):
                pc_src = os.path.join(src_path, filename)
                if os.path.isfile(pc_src) and filename.endswith(".csv"):
                    pc_name = rewrite_dataset_filename(filename, old_obj_name, new_obj_name)
                    write_pointcloud(pc_src, os.path.join(pc_dst, pc_name), aff_map)
        elif entry == "Image" and os.path.isdir(src_path):
            copy_image_dir(src_path, os.path.join(dst_dir, entry), old_obj_name, new_obj_name, aff_map)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(dst_dir, entry), dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, os.path.join(dst_dir, entry))


def rewrite_dataset(dataset_root, obj_map, aff_map):
    tmp_root = dataset_root.rstrip(os.sep) + ".rename_tmp"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    entries = os.listdir(dataset_root)
    for entry in tqdm(entries, desc="重写数据集"):
        src_path = os.path.join(dataset_root, entry)
        dst_path = os.path.join(tmp_root, entry)
        if os.path.isdir(src_path):
            new_obj = mapped_name(entry, obj_map)
            copy_object_dir(src_path, os.path.join(tmp_root, new_obj), entry, new_obj, obj_map, aff_map)
        elif entry == "info.json":
            rewrite_info(src_path, dst_path, obj_map, aff_map)
        elif entry in {"train.json", "val.json", "test.json", "metadata.json"}:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(rewrite_split_value(data, obj_map, aff_map), f, ensure_ascii=False, indent=2)
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
