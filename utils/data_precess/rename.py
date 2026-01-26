"""
重命名某一个物体的名称或aff的名称（未测试）
"""
import os
import shutil
import argparse
from ..base_dataset import Instruction, Image, PointCloud


def rename_obj_type(dataset_root, obj_rename, aff_rename=None):
    """
    重命名物体类型，可选同时重命名 affordance，复用 datasets.py 中的加载方法
    
    Args:
        dataset_root: 数据集根目录
        obj_rename: 物体重命名元组 (old_obj_type, new_obj_type)
        aff_rename: affordance重命名元组 (old_aff_type, new_aff_type)，可选
            - 当 old_obj_type == new_obj_type 且仅提供 aff_rename 时，相当于只改同一物体下的 aff 名称
    """
    old_obj_type, new_obj_type = obj_rename
    old_obj_dir = os.path.join(dataset_root, old_obj_type)
    if not os.path.exists(old_obj_dir):
        print(f"警告: 物体目录不存在: {old_obj_dir}")
        return
    
    same_name = (old_obj_type == new_obj_type)

    # 1. 先加载新名称的物体（如果存在），以继承计数（仅在改名时需要）
    new_obj_dir = os.path.join(dataset_root, new_obj_type)
    if not same_name and os.path.exists(new_obj_dir):
        print(f"加载已存在的新名称物体: {new_obj_type}")
        # 加载 PointCloud
        for pc in PointCloud.load_all(dataset_root, obj_type=new_obj_type, keep_id=True):
            pc.free_memory()
        # 加载 Image
        for img in Image.load_all(dataset_root, obj_type=new_obj_type, keep_id=True):
            img.free_memory()
        # 加载 Instruction（只加载新名称的）
        Instruction.load_all(dataset_root, keep_id=True)
        # 过滤出新名称的 Instruction 以继承计数
        if new_obj_type in Instruction.all:
            for inst in Instruction.all[new_obj_type]:
                pass  # 只是加载以继承计数
        print(f"已加载新名称物体的计数信息")
    
    # 确保新物体目录下的 PointCloud / Image 目录存在
    new_pc_dir = os.path.join(new_obj_dir, 'PointCloud')
    new_img_dir = os.path.join(new_obj_dir, 'Image')
    os.makedirs(new_pc_dir, exist_ok=True)
    os.makedirs(new_img_dir, exist_ok=True)

    # 2. 加载旧名称的物体，强制指定 obj_type 为新名称（并可选重命名 aff）
    print(f"加载旧名称物体并重命名为: {new_obj_type}")
    
    # 加载 PointCloud
    old_pc_dir = os.path.join(old_obj_dir, 'PointCloud')
    if os.path.exists(old_pc_dir):
        for filename in os.listdir(old_pc_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(old_pc_dir, filename)
                print(f'loading PC: {file_path}')
                pc = PointCloud.load_file(
                    file_path,
                    obj_type=new_obj_type,  # 强制指定为新名称
                    keep_id=True
                )
                # 如果需要同时重命名 affordance，则替换 labels 中的名称
                if aff_rename is not None and pc.labels is not None:
                    old_aff, new_aff = aff_rename
                    pc.labels = [new_aff if label == old_aff else label
                                 for label in pc.labels]
                # 只保留有 labels 和 mask 的对象
                if pc.labels is not None and len(pc.labels) > 0 and pc.mask is not None:
                    pc.save_to(os.path.join(new_pc_dir, f'{pc.obj_type}_{pc.id}.csv'))
                pc.free_memory()
    
    # 加载 Image
    old_img_dir = os.path.join(old_obj_dir, 'Image')
    if os.path.exists(old_img_dir):
        rgb_dir = os.path.join(old_img_dir, 'rgb')
        if os.path.exists(rgb_dir):
            for rgb_file in os.listdir(rgb_dir):
                if rgb_file.lower().endswith(('.png', '.jpg')):
                    rgb_path = os.path.join(rgb_dir, rgb_file)
                    try:
                        print(f'loading IMG: {rgb_path}')
                        img = Image.load_file(
                            rgb_path,
                            obj_type=new_obj_type,  # 强制指定为新名称
                            keep_id=True
                        )
                        # 如果需要同时重命名 affordance，则替换 labels 中的名称
                        if aff_rename is not None and img.labels is not None:
                            old_aff, new_aff = aff_rename
                            img.labels = [new_aff if label == old_aff else label
                                          for label in img.labels]
                        # 只保留有 labels 和 mask 的对象
                        if img.labels is not None and len(img.labels) > 0 and len(img.mask) > 0:
                            img.save_to(new_img_dir)
                        img.free_memory()
                    except Exception as e:
                        print(f"Failed to load {rgb_path}: {e}")
                        continue
    
    # 加载 Instruction
    old_ins_file = os.path.join(old_obj_dir, 'Instruction.csv')
    if os.path.exists(old_ins_file):
        instructions = Instruction.load_file(old_ins_file, keep_id=True)
        # 更新 obj_type 为新名称，创建新的 Instruction 对象
        for inst in instructions:
            if inst.obj_type == old_obj_type:
                # 可选地同时重命名 aff_type
                new_aff_type = inst.aff_type
                if aff_rename is not None and inst.aff_type is not None:
                    old_aff, new_aff = aff_rename
                    if inst.aff_type == old_aff:
                        new_aff_type = new_aff

                # 创建新的 Instruction 对象，指定 obj_type 为新名称
                # 这样会在新名称的计数上继续增加
                Instruction(
                    inst.ins,
                    obj_type=new_obj_type,
                    aff_type=new_aff_type,
                    given_id=inst.id
                )
    
    # 3. 保存所有数据（只保存有 label/mask 的对象）
    print(f"保存重命名后的数据到: {new_obj_type}")
    
    # 保存 Instruction
    Instruction.save_all(dataset_root, obj_type=[new_obj_type], keep_id=True)
    
    # 4. 删除旧目录（仅当物体名发生变化时）
    if not same_name and os.path.exists(old_obj_dir):
        shutil.rmtree(old_obj_dir)
        print(f"已删除旧目录: {old_obj_dir}")



def main():
    parser = argparse.ArgumentParser(description="重命名数据集中的物体名称或affordance名称")
    parser.add_argument('-d', '--dataset-root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument(
        '-o', '--obj',
        type=str,
        nargs='+',
        metavar=('OLD', 'NEW'),
        help='重命名物体类型: --obj <旧名称> <新名称>；'
             '若只提供一个名称，则表示在该物体上仅重命名aff（需要配合 --aff）'
    )
    parser.add_argument('-a', '--aff', type=str, nargs=2, metavar=('OLD', 'NEW'),
                       help='重命名affordance类型: --aff <旧名称> <新名称> (默认处理所有物体)')
    
    args = parser.parse_args()
    
    dataset_root = os.path.abspath(args.dataset_root)
    
    if not os.path.isdir(dataset_root):
        raise ValueError(f"数据集根目录不存在: {dataset_root}")
    
    # 重命名物体类型（可选同时重命名 affordance）
    if args.obj:
        if len(args.obj) == 1:
            # 只提供一个物体名：表示在该物体上仅重命名 aff
            old_obj_type = new_obj_type = args.obj[0]
        elif len(args.obj) == 2:
            old_obj_type, new_obj_type = args.obj
        else:
            raise ValueError("参数 --obj 只允许 1 个或 2 个值")

        obj_rename = (old_obj_type, new_obj_type)
        aff_rename = tuple(args.aff) if args.aff else None

        # 无实际修改的情况：旧名=新名，且没有提供 aff 重命名
        if old_obj_type == new_obj_type and aff_rename is None:
            print(f"物体名称未变化，且未指定aff重命名，跳过：{old_obj_type}")
        else:
            print(f"\n开始重命名物体类型: {old_obj_type} -> {new_obj_type}")
            if aff_rename:
                old_aff, new_aff = aff_rename
                print(f"同时重命名 affordance: {old_aff} -> {new_aff}")
            rename_obj_type(dataset_root, obj_rename=obj_rename, aff_rename=aff_rename)
            print(f"完成重命名物体类型: {old_obj_type} -> {new_obj_type}\n")
    
    if not args.obj and not args.aff:
        parser.print_help()


if __name__ == "__main__":
    main()
