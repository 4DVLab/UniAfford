"""
根据Instruction.csv重新为图片及mask分类aff
"""

import csv
import os
import shutil
from tqdm import tqdm
import argparse
from base_dataset import create_info_dict, load_info, save_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据Instruction.csv重新为图片及mask分类aff")
    parser.add_argument('-c', '--csv_root', type=str, default='/mnt/data/datasets/output_csv_2',
                        help='Instruction.csv 所在的根目录')
    parser.add_argument('-i', '--input_root', type=str, default='/mnt/data/datasets/2D-3D-RAGNet',
                        help='输入图片的根目录')
    parser.add_argument('-o', '--output_root', type=str, default='/mnt/data/datasets/output_csv_2',
                        help='输出图片的根目录')
    parser.add_argument('-info', '--info_file', type=str, default=None,
                        help='指定info.json的文件位置并继承编号ID，否则初始化ID')
    
    args = parser.parse_args()

    csv_root = args.csv_root
    input_root = args.input_root
    output_root = args.output_root

    # 加载或创建 info_dict
    if args.info_file is not None:
        info_dict = load_info(args.info_file)
        print(f"已从 {args.info_file} 加载编号信息，将继续编号")
    else:
        info_dict = create_info_dict()
        print("未指定 info 文件，从头开始编号")

    # 获取目录下所有文件夹  
    obj_list = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    print(f"找到 {len(obj_list)} 个对象文件夹，开始处理...")

    for obj in tqdm(obj_list):
        csv_path = os.path.join(csv_root, obj, 'Instruction.csv')
        
        if not os.path.exists(csv_path):
            print(f"跳过: {csv_path} 不存在")
            continue

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            header = next(reader, None)

            input_base_dir = os.path.join(input_root, obj, 'Image')
            output_base_dir = os.path.join(output_root, obj, 'Image')

            os.makedirs(output_base_dir, exist_ok=True)
            os.makedirs(os.path.join(output_base_dir, 'rgb'), exist_ok=True)

            for row in reader:
                if not row or len(row) < 4: continue

                new_aff = row[2]  # aff_type
                old_idx = row[3]  # 原始 id
                
                # 获取新的编号（继续编号）
                info_dict['Image'][obj]['ID'] += 1
                new_idx = info_dict['Image'][obj]['ID']
                info_dict['Image'][obj][new_aff] += 1
                
                src_rgb_file = os.path.join(input_base_dir, 'rgb', f"{obj}_{old_idx}.png")
                src_mask_file = os.path.join(input_base_dir, 'mask', 'None', f"{obj}_{old_idx}_None.png")

                dst_rgb_file = os.path.join(output_base_dir, 'rgb', f"{obj}_{new_idx}.png")
                dst_mask_dir = os.path.join(output_base_dir, 'mask', new_aff)
                os.makedirs(dst_mask_dir, exist_ok=True)
                dst_mask_file = os.path.join(dst_mask_dir, f"{obj}_{new_idx}_{new_aff}.png")

                if os.path.exists(src_rgb_file):
                    shutil.copy2(src_rgb_file, dst_rgb_file)
                else:
                    print(f"警告: 源文件不存在 {src_rgb_file}")
                
                if os.path.exists(src_mask_file):
                    shutil.copy2(src_mask_file, dst_mask_file)
                else:
                    print(f"警告: 源文件不存在 {src_mask_file}")

    # 保存更新后的 info 文件
    save_info(output_root, info_dict)

    print("所有文件整理完成。")