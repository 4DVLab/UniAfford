"""
将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）
"""
import csv
import os
import argparse
from collections import defaultdict
import shutil
import json
from common import clean_quotes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将dataset.py处理的不同位置数据整合为同一个数据集（合并所有的Ins.csv）")
    parser.add_argument('-i', '--input', type=str, nargs="+", help='输入数据集的根目录，按照物体-模态分类')
    parser.add_argument('-o', '--output', type=str, help='输出位置', default='/mnt/data/datasets/sorted_23d')
    parser.add_argument('-f', '--filter', action='store_true', help='仅保留包含三元组的数据', default=True)

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


    # 收集物体模态信息
    obj_modalities = defaultdict(set)
    info_dict = {
            'PointCloud': defaultdict(lambda: defaultdict(int)),
            'Image': defaultdict(lambda: defaultdict(int)),
            'Instruction': defaultdict(lambda: defaultdict(int)),
        }

    if args.filter:
        print("正在收集物体模态信息...")

        # 遍历所有输入数据集目录
        for dataset_dir in args.input:
            # 加载info数据
            info_file = os.path.join(args.input, 'info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    loaded_dict = json.load(f)
                    for m, _ in info_dict.items():
                        for obj_type, vals in loaded_dict.get(k, {}).items():
                            for k, v in vals.items():
                                info_dict[m][obj_type][k] = max(info_dict[m][obj_type][k], v)

            if not os.path.exists(dataset_dir):
                print(f"警告: 数据集目录不存在: {dataset_dir}")
                continue

            for obj_name in os.listdir(dataset_dir):
                obj_path = os.path.join(dataset_dir, obj_name)
                if not os.path.isdir(obj_path):
                    continue

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

        # 复制筛选后的数据
        for dataset_dir in args.input:
            for obj_name in os.listdir(dataset_dir):
                if obj_name in filtered_objects:
                    obj_path = os.path.join(dataset_dir, obj_name)
                    if os.path.isdir(obj_path):
                        print(f"复制物体: {obj_name}")
                        copy_obj(obj_path, obj_name)

        with open(os.path.join(args.output, 'info.json'), 'w', encoding='utf-8') as f:
            save_info = {
                'PointCloud': defaultdict(lambda: defaultdict(int)),
                'Image': defaultdict(lambda: defaultdict(int)),
                'Instruction': defaultdict(lambda: defaultdict(int)),
            }

            for m,_ in info_dict:
                for o in info_dict[m]:
                    if o in filtered_objects:
                        save_info[m][o] = info_dict[m][o]

            json.dump(save_info, f, ensure_ascii=False, indent=2)

    else:
        # 不过滤，复制所有数据
        print("复制所有数据（不过滤）...")
        for dataset_dir in args.input:
            for obj_name in os.listdir(dataset_dir):
                obj_path = os.path.join(dataset_dir, obj_name)
                if os.path.isdir(obj_path):
                    print(f"复制物体: {obj_name}")
                    copy_obj(obj_path, obj_name)

        with open(os.path.join(args.output, 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=2)

    print("数据处理完成!")
    print(f"输出目录: {args.output}")