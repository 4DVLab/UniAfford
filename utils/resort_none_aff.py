"""
根据Instruction.csv重新为图片及mask分类aff
"""

import csv
import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    csv_root = '/mnt/data/datasets/output_csv_2'
    input_root = '/mnt/data/datasets/2D-3D-RAGNet'
    output_root = '/mnt/data/datasets/output_csv_2'

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
            for row in reader:
                if not row or len(row) < 4: continue

                new_aff = row[2]  # aff_type
                idx = row[3]      # id
                
                src_rgb_file = os.path.join(input_base_dir, 'rgb', f"{obj}_{idx}.png")
                src_mask_file = os.path.join(input_base_dir, 'mask', 'None', f"{obj}_{idx}_None.png")


                dst_rgb_file = os.path.join(output_base_dir, 'rgb', f"{obj}_{idx}.png")
                dst_mask_file = os.path.join(output_base_dir, 'mask', new_aff, f"{obj}_{idx}_{new_aff}.png")

                shutil.copy2(src_rgb_file, dst_rgb_file)
                shutil.copy2(src_mask_file, dst_mask_file)

    print("所有文件整理完成。")