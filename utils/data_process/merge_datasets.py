"""
将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）但不处理info.json
"""
import csv
import os
import sys
import argparse
from collections import defaultdict
import shutil
import json
import re
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import clean_quotes
from utils.base_dataset import (
    create_info_dict,
    save_info,
)
from utils.data_process.create_split import SplitManager


MODALITY_DIRS = {"Image", "PointCloud"}


def standardize_name(name):
    """README 规范：小写，去首尾空白，下划线转换为空格，连续空白压缩为一个空格。"""
    if name is None:
        return ""
    name = str(name).strip().replace("_", " ").lower()
    return re.sub(r"\s+", " ", name)


def standardize_info_dict(info):
    """合并 info.json 时同步规范化 obj/aff key。"""
    normalized = create_info_dict()
    for modality, obj_dict in (info or {}).items():
        if modality not in normalized:
            normalized[modality] = defaultdict(lambda: defaultdict(int))
        for obj_type, vals in (obj_dict or {}).items():
            obj_name = standardize_name(obj_type)
            if not obj_name:
                continue
            for aff_type, value in (vals or {}).items():
                aff_name = "ID" if aff_type == "ID" else standardize_name(aff_type)
                normalized[modality][obj_name][aff_name] = max(
                    normalized[modality][obj_name][aff_name],
                    value,
                )
    return normalized


def merge_info_dict(target, source):
    for modality, obj_dict in source.items():
        for obj_type, vals in obj_dict.items():
            for key, value in vals.items():
                target[modality][obj_type][key] = max(target[modality][obj_type][key], value)


def standardize_dataset_filename(filename, old_obj_name, new_obj_name, aff_name=None):
    """按统一格式重写 Image/PointCloud 文件名；无法识别 id 时保留原 basename 的标准化形式。"""
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    old_prefix = old_obj_name
    sample_id = None
    if stem.startswith(old_prefix + "_"):
        rest = stem[len(old_prefix) + 1:]
        sample_id = rest.split("_", 1)[0]
    else:
        id_match = re.search(r"_(\d+)(?:_|$)", stem)
        if id_match:
            sample_id = id_match.group(1)

    if sample_id:
        if aff_name:
            return f"{new_obj_name}_{sample_id}_{aff_name}{ext}"
        return f"{new_obj_name}_{sample_id}{ext}"
    return f"{standardize_name(stem)}{ext}"


def copy_csv_with_standard_names(src_path, dst_path, obj_name=None):
    """复制 csv，同时规范化 Instruction 字段或 PointCloud aff header。"""
    with open(src_path, "r", newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        header = next(csv.reader([first_line.strip()])) if first_line.strip() else []

    if header[:6] == ["ins", "obj_type", "aff_type", "id", "img_id", "pc_id"]:
        fieldnames = ["ins", "obj_type", "aff_type", "id", "img_id", "pc_id"]
        mode = "a" if os.path.exists(dst_path) else "w"
        with open(dst_path, mode, newline="", encoding="utf-8") as output_f:
            writer = csv.DictWriter(output_f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            with open(src_path, "r", newline="", encoding="utf-8") as input_f:
                reader = csv.DictReader(input_f)
                for row in reader:
                    writer.writerow({
                        "ins": clean_quotes(row.get("ins", "")),
                        "obj_type": obj_name or standardize_name(row.get("obj_type", "")),
                        "aff_type": standardize_name(row.get("aff_type", "")),
                        "id": row.get("id", ""),
                        "img_id": row.get("img_id", ""),
                        "pc_id": row.get("pc_id", ""),
                    })
        return

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as input_f, open(dst_path, "w", encoding="utf-8") as output_f:
        first = input_f.readline()
        if first:
            prefix = "# " if first.startswith("# ") else ""
            clean_header = first[2:] if first.startswith("# ") else first
            cols = [col.strip() for col in clean_header.strip().split(",")]
            if len(cols) > 3:
                cols = cols[:3] + [standardize_name(col) for col in cols[3:]]
            output_f.write(prefix + ",".join(cols) + "\n")
        shutil.copyfileobj(input_f, output_f)


def copy_image_dir(src_dir, dst_dir, old_obj_name, new_obj_name):
    for subdir in os.listdir(src_dir):
        src_subdir = os.path.join(src_dir, subdir)
        if not os.path.isdir(src_subdir):
            continue
        if subdir == "mask":
            for aff_dir in os.listdir(src_subdir):
                src_aff_dir = os.path.join(src_subdir, aff_dir)
                if not os.path.isdir(src_aff_dir):
                    continue
                new_aff = standardize_name(aff_dir)
                dst_aff_dir = os.path.join(dst_dir, "mask", new_aff)
                os.makedirs(dst_aff_dir, exist_ok=True)
                for filename in os.listdir(src_aff_dir):
                    src_file = os.path.join(src_aff_dir, filename)
                    if not os.path.isfile(src_file):
                        continue
                    dst_file = os.path.join(
                        dst_aff_dir,
                        standardize_dataset_filename(filename, old_obj_name, new_obj_name, new_aff),
                    )
                    shutil.copy2(src_file, dst_file)
        else:
            dst_subdir = os.path.join(dst_dir, subdir)
            os.makedirs(dst_subdir, exist_ok=True)
            for filename in os.listdir(src_subdir):
                src_file = os.path.join(src_subdir, filename)
                if not os.path.isfile(src_file):
                    continue
                suffix = "obj mask" if subdir == "obj_mask" and "obj" in filename else None
                dst_name = standardize_dataset_filename(filename, old_obj_name, new_obj_name)
                if suffix is not None and not dst_name.endswith("_obj mask.png"):
                    stem, ext = os.path.splitext(dst_name)
                    dst_name = f"{stem}_{suffix}{ext}"
                shutil.copy2(src_file, os.path.join(dst_subdir, dst_name))


def copy_obj(obj_path, obj_name):
    """复制单个物体的数据到输出目录"""
    old_obj_name = os.path.basename(obj_path)
    obj_output_path = os.path.join(args.output, obj_name)
    os.makedirs(obj_output_path, exist_ok=True)

    for m in os.listdir(obj_path):
        m_path = os.path.join(obj_path, m)

        if os.path.isdir(m_path):
            dest_path = os.path.join(obj_output_path, m)
            if m == "Image":
                copy_image_dir(m_path, dest_path, old_obj_name, obj_name)
            elif m == "PointCloud":
                os.makedirs(dest_path, exist_ok=True)
                for filename in os.listdir(m_path):
                    src_file = os.path.join(m_path, filename)
                    if not os.path.isfile(src_file):
                        continue
                    dst_name = standardize_dataset_filename(filename, old_obj_name, obj_name)
                    copy_csv_with_standard_names(src_file, os.path.join(dest_path, dst_name))
            else:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(m_path, dest_path)

        elif m.endswith('.csv'):
            # 处理CSV文件
            output_csv = os.path.join(obj_output_path, m)
            copy_csv_with_standard_names(m_path, output_csv, obj_name=obj_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）")
    parser.add_argument('-i', '--input', type=str, nargs="+", help='输入数据集的根目录，按照物体-模态分类')
    parser.add_argument('-o', '--output', type=str, help='输出位置', default='/mnt/data/datasets/2D-3D-JointAffordance/merged')
    parser.add_argument('-f', '--filter', action='store_true', help='仅保留包含三元组的数据', default=False)
    parser.add_argument('--save_split', action='store_true', default=True, help='合并后额外分割数据集并保存分割文件 json')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 收集物体模态信息 & 合并 info
    obj_modalities = defaultdict(set)
    info_dict = create_info_dict()

    # 1) 先合并所有输入数据集的 info.json（数值取更大的）
    print("正在合并各数据集的 info.json ...")
    for dataset_dir in tqdm(args.input, desc="合并 info"):
        info_file = os.path.join(dataset_dir, 'info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                loaded_dict = json.load(f)
                merge_info_dict(info_dict, standardize_info_dict(loaded_dict))

    # 2) 根据是否启用 filter 决定过滤逻辑，并在启用 filter 时裁剪 info_dict
    if args.filter:
        print("正在收集物体模态信息并进行过滤...")

        # 遍历所有输入数据集目录，统计每个物体有哪些模态
        for dataset_dir in tqdm(args.input, desc="扫描数据集"):
            if not os.path.exists(dataset_dir):
                print(f"警告: 数据集目录不存在: {dataset_dir}")
                continue

            obj_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            for obj_name in tqdm(obj_list, desc=f"扫描物体 ({os.path.basename(dataset_dir)})", leave=False):
                obj_path = os.path.join(dataset_dir, obj_name)
                std_obj_name = standardize_name(obj_name)

                # 获取该物体的所有模态
                for modality in os.listdir(obj_path):
                    modality_path = os.path.join(obj_path, modality)
                    if os.path.isdir(modality_path) or modality.endswith('.csv'):
                        obj_modalities[std_obj_name].add(modality)

        print(f"找到 {len(obj_modalities)} 个物体")

        # 筛选出至少包含三种模态的物体
        filtered_objects = []
        for obj_name, modalities in obj_modalities.items():
            if len(modalities) >= 3:  # 包含三元组
                filtered_objects.append(obj_name)
                print(f"包含三元组: {obj_name} - 模态: {modalities}")

        print(f"筛选后保留 {len(filtered_objects)} 个物体")

        # 根据过滤结果裁剪 info_dict，只保留被筛选物体的信息
        if filtered_objects:
            keep_set = set(filtered_objects)
            for modality, obj_dict in info_dict.items():
                for obj_name in list(obj_dict.keys()):
                    if obj_name not in keep_set:
                        del obj_dict[obj_name]

        # 复制筛选后的数据
        for dataset_dir in tqdm(args.input, desc="复制数据集"):
            if not os.path.exists(dataset_dir):
                print(f"警告: 数据集目录不存在: {dataset_dir}")
                continue

            obj_list = [
                d for d in os.listdir(dataset_dir)
                if standardize_name(d) in filtered_objects and os.path.isdir(os.path.join(dataset_dir, d))
            ]
            for obj_name in tqdm(obj_list, desc=f"复制物体 ({os.path.basename(dataset_dir)})", leave=False):
                obj_path = os.path.join(dataset_dir, obj_name)
                copy_obj(obj_path, standardize_name(obj_name))

    else:
        # 不过滤，复制所有数据；info_dict 中已是所有输入数据集 info 的并集（数值取最大）
        print("复制所有数据（不过滤）...")
        for dataset_dir in tqdm(args.input, desc="复制数据集"):
            if not os.path.exists(dataset_dir):
                print(f"警告: 数据集目录不存在: {dataset_dir}")
                continue

            obj_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            for obj_name in tqdm(obj_list, desc=f"复制物体 ({os.path.basename(dataset_dir)})", leave=False):
                obj_path = os.path.join(dataset_dir, obj_name)
                copy_obj(obj_path, standardize_name(obj_name))

    # 3) 最终保存合并后的 info.json
    save_info(args.output, info_dict)

    # 可选保存分割文件
    if args.save_split:
        SplitManager(args.output).split(train_ratio=0.95, val_ratio=0.05, test_ratio=0.0, id_source="disk")

    print("数据处理完成!")
    print(f"输出目录: {args.output}")