"""
将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）但不处理info.json
"""
import csv
import os
import argparse
from collections import defaultdict
import shutil
import json
from common import clean_quotes
from tqdm import tqdm
from base_dataset import create_info_dict, load_info, save_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）")
    parser.add_argument('-i', '--input', type=str, nargs="+", help='输入数据集的根目录，按照物体-模态分类')
    parser.add_argument('-o', '--output', type=str, help='输出位置', default='/mnt/data/datasets/2D-3D-JointAffordance/merged')
    parser.add_argument('-f', '--filter', action='store_true', help='仅保留包含三元组的数据', default=False)

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)


    def copy_obj(obj_path, obj_name):
        """复制单个物体的数据到输出目录"""
        obj_output_path = os.path.join(args.output, obj_name)
        os.makedirs(obj_output_path, exist_ok=True)

        for m in os.listdir(obj_path):
            m_path = os.path.join(obj_path, m)

            if os.path.isdir(m_path):  # Point和Image目录可以直接复制
                dest_path = os.path.join(obj_output_path, m)
                if os.path.exists(dest_path):
                    # 如果目标已存在，合并内容
                    for item in os.listdir(m_path):
                        src_item = os.path.join(m_path, item)
                        dst_item = os.path.join(dest_path, item)
                        if os.path.isdir(src_item):
                            if os.path.exists(dst_item):
                                shutil.rmtree(dst_item)
                            shutil.copytree(src_item, dst_item)
                        else:
                            shutil.copy2(src_item, dst_item)
                else:
                    shutil.copytree(m_path, dest_path)

            elif m.endswith('.csv'):
                # 处理CSV文件
                output_csv = os.path.join(obj_output_path, m)
                fieldnames = ['ins', 'obj_type', 'aff_type', 'id']

                # 如果输出CSV已存在，追加模式；否则创建新文件
                mode = 'a' if os.path.exists(output_csv) else 'w'
                with open(output_csv, mode, newline='', encoding='utf-8') as output_f:
                    writer = csv.DictWriter(output_f, fieldnames=fieldnames)
                    if mode == 'w':
                        writer.writeheader()

                    with open(m_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            processed_row = {
                                'ins': clean_quotes(row.get("ins", "")),  # 兼容之前的问题
                                'obj_type': row.get('obj_type', ''),
                                'aff_type': row.get('aff_type', ''),
                                'id': row.get('id', '')
                            }
                            writer.writerow(processed_row)


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
                for m, _ in info_dict.items():
                    for obj_type, vals in loaded_dict.get(m, {}).items():
                        for key, v in vals.items():
                            info_dict[m][obj_type][key] = max(info_dict[m][obj_type][key], v)

    # 2) 根据是否启用 filter 决定复制逻辑，并在启用 filter 时裁剪 info_dict
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

                # 获取该物体的所有模态
                for modality in os.listdir(obj_path):
                    modality_path = os.path.join(obj_path, modality)
                    if os.path.isdir(modality_path) or modality.endswith('.csv'):
                        obj_modalities[obj_name].add(modality)

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

            obj_list = [d for d in os.listdir(dataset_dir)
                        if d in filtered_objects and os.path.isdir(os.path.join(dataset_dir, d))]
            for obj_name in tqdm(obj_list, desc=f"复制物体 ({os.path.basename(dataset_dir)})", leave=False):
                obj_path = os.path.join(dataset_dir, obj_name)
                copy_obj(obj_path, obj_name)

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
                copy_obj(obj_path, obj_name)

    # 3) 最终保存合并后的 info.json
    save_info(args.output, info_dict)

    print("数据处理完成!")
    print(f"输出目录: {args.output}")