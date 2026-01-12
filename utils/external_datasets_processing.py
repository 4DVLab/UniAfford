"""
!!WARNING: 由于数据构建过程是一个一个数据集处理的，故其余数据集的处理代码仅供参考，可能存在部分问题。
    dataset.py文件只保证加载本数据集时不出现问题。
    注意所有的'Note'注释。
"""


import os
import json
import warnings
import numpy as np
import cv2
from collections import defaultdict
from base_dataset import Instruction, Image, PointCloud
from common import resolve_path

# 全局参数
DEFAULT_OUTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 输出的数据集位置（用于数据转换，训练推理时可忽略）
DEFAULT_INPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 加载的数据集位置（输入数据位置，或转换数据集的输入位置）


"""  ----------------------------------------------- PointCloud classes ----------------------------------------------  """

class AGPIL_PC(PointCloud):
    aff_type = [
        'grasp', 'contain', 'lift', 'open', 'lay',
        'sit', 'support', 'wrapgrasp', 'pour', 'move',
        'display', 'push', 'listen', 'wear', 'press',
        'cut', 'stab',     
    ] # 17
    all = {
        k: list() for k in [
            'Bag', 'Bed', 'Bottle', 'Bowl', 'Chair',
            'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',
            'Faucet', 'Hat', 'Keyboard', 'Knife', 'Laptop',
            'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture',
            'Table', 'TrashCan', 'Vase',
        ] # 23
    }

    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type=obj_type, mask=mask, labels=labels)
        AGPIL_PC.all[obj_type].append(self)
        AGPIL_PC.count[obj_type]['ID'] += 1


    @staticmethod
    def load_file(filepath, obj_type=None, keep_id=False) -> 'PointCloud':
        """ Plain Text like:
        cefccd231c34f213eec1a3147175f806068 Bed x y z 0.233132 0.0 0.0 ...
        """

        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                data.append(list(map(float, parts[2:])))
            
            

        data = np.asarray(data, dtype=float)

        # 筛选出 mask 中全 0 的列索引（按列判断，忽略前三列 xyz）
        zero_col_idx = np.where(np.all(data[:, 3:] == 0, axis=0))[0]
        # 根据列索引过滤掉对应的标签；此处先用 header 作为占位标签
        labels = [label for idx, label in enumerate(AGPIL_PC.aff_type) if idx not in zero_col_idx]
        pc_obj = AGPIL_PC(points = data[:, :3], obj_type=obj_type, labels=labels)

        if zero_col_idx.size > 0:
            data = np.delete(data, zero_col_idx+3, axis=1) 
        if data.shape[1] > 3:
            pc_obj.mask = data[:, 3:]
        
        return pc_obj

    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        def iterator():
            for obj_type in list(cls.all):
                for view in os.listdir(dataset_root_path):
                    for s in ['Seen', 'Unseen']:
                        for t in ['Test', 'Train']:
                            files_dir = os.path.join(dataset_root_path, view, s, 'Point', t, obj_type)
                            if not os.path.isdir(files_dir):
                                continue

                            for file in os.listdir(files_dir):
                                file_path = os.path.join(files_dir, file)
                                if os.path.isfile(file_path) and not file.startswith('.'):  # 忽略 .开头的文件
                                    print(f'loading PC{file_path}')
                                    yield cls.load_file(file_path, obj_type=obj_type)
        return iterator()

class PIADv2_PC(PointCloud):
    aff_type = [
        'grasp', 'contain', 'lift', 'open', 'lay',
        'sit', 'support', 'wrapgrasp', 'pour', 'move',
        'display', 'push', 'listen', 'wear', 'press',
        'cut', 'stab', 'carry', 'ride', 'clean',
        'play', 'beat', 'speak', 'pull'
    ]  # 24
    all = {
        k: list() for k in [
            'Backpack', 'Bag', 'Baseballbat', 'Bed', 'Bicycle',
            'Bottle', 'Bowl', 'Broom', 'Bucket', 'Chair',
            'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',
            'Faucet', 'Fork', 'Glasses', 'Guitar', 'Hammer',
            'Hat', 'Kettle', 'Keyboard', 'Knife', 'Laptop',
            'Microphone', 'Microwave', 'Mop', 'Motorcycle', 'Mug',
            'Refrigerator', 'Scissors', 'Skateboard', 'Spoon', 'StorageFurniture',
            'Suitcase', 'Surfboard', 'Table', 'Tennisracket', 'Toothbrush',
            'TrashCan', 'Umbrella', 'Vase'
        ] # 43
    }

    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type, mask, labels)
        PIADv2_PC.all[obj_type].append(self)
        PIADv2_PC.count[obj_type]['ID'] += 1

    @staticmethod
    def load_file(filepath, obj_type=None, aff_type=None) -> 'PointCloud':
        data = np.load(filepath)
        pc_obj = PIADv2_PC(points=data[:, :3], obj_type=obj_type, mask=data[:, 3:],labels=[aff_type])

        return pc_obj

    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        """
        Args:
            dataset_root_path: PIADv2数据集的位置，下层目录为 Seen,Unseen_aff,Unseen_obj(任一）
        """
        def iterator():
            for s in ['Seen', 'Unseen_aff', 'Unseen_obj']:
                if not os.path.isdir(os.path.join(dataset_root_path, s)): continue
                for t in ['test', 'train', 'val']:
                    dirpath = os.path.join(dataset_root_path, s, 'Point', t)
                    if not os.path.isdir(dirpath): continue

                    for obj_type in os.listdir(dirpath):
                        obj_dir = os.path.join(dirpath, obj_type)
                        for sub_dataset in os.listdir(obj_dir):
                            for aff in os.listdir(os.path.join(obj_dir, sub_dataset)):
                                file_dir = os.path.join(obj_dir, sub_dataset, aff)
                                for file in os.listdir(file_dir):
                                    file_path = os.path.join(file_dir, file)
                                    if os.path.isfile(file_path) and file.endswith('.npy'):
                                        print(f'loading PC{file_path}')
                                        yield cls.load_file(file_path, obj_type=obj_type, aff_type=aff)

                break # PIADv2的Seen,Unseen_aff,Unseen_obj三个数据集只是同一个数据集的不同划分，任意处理一个就行
        return iterator()

# discard
class LASO_PC(PointCloud):
    aff_type = [
        'lay', 'sit', 'support', 'grasp', 'lift',
        'contain', 'open', 'wrap_grasp', 'pour', 'move',
        'display', 'push', 'pull', 'listen', 'wear',
        'press', 'cut', 'stab'
    ]  # 18
    all = {
        k: list() for k in [
            "Bag", "Bed", "Bowl", "Clock", "Dishwasher",
            "Display", "Door", "Earphone", "Faucet", "Hat",
            "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave",
            "Mug", "Refrigerator", "Chair", "Scissors", "Table",
            "TrashCan", "Vase", "Bottle"
        ]# 23
    }

    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type, mask, labels)
        LASO_PC.all[obj_type].append(self)
        LASO_PC.count[obj_type]['ID'] += 1

    def load_file(self, obj_type=None, aff_type=None):
        raise NotImplementedError('懒得写，直接使用 LASO_PC.load_all')

    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        raise NotImplementedError('暂未实现')
        import pickle # load only needed
        def iterator():
            for t in ['test', 'train', 'val']:
                with open(os.path.join(dataset_root_path, f'objects_{t}.pkl'), 'rb') as f:
                    obj_points = pickle.load(f)
                with open(os.path.join(dataset_root_path, f'anno_{t}.pkl'), 'rb') as f:
                    obj_aff = pickle.load(f)

                for i, e in enumerate(zip(obj_points, obj_aff)):
                    print(f'loading PC: {file_path}')
                    yield cls(points=obj_points[i], obj_type=e[1]['class'], mask=e[1]['mask'], labels=[e[1]['affordance'],])

        return iterator()

class AffNet_PC(PointCloud):...


"""  ----------------------------------------------- Image classes ----------------------------------------------  """

class BoxedImage(Image):
    def __init__(self, img, obj_type, box:np.ndarray=None, labels=None, **kwargs):
        """
        Args:
            img: 图片数组
            obj_type: 物体类型
            box: 标注文本框的左上角(x1, y1)和右下角(x2, y2)组合成的数组(x1, y1, x2, y2)
            labels: 标签列表
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(img, obj_type=obj_type, labels=labels, **kwargs)

        # 直接将box区域划作mask
        if box is not None:
            box_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            box_mask[box[1]:box[3], box[0]:box[2]] = 1
            if len(self.mask) == 0:
                self.mask = [box_mask]
            else:
                self.mask.append(box_mask)
        self.dtype = 'Boxed'

class HeatImage(Image):
    def __init__(self, img:np.ndarray, obj_type, aff_mask:np.ndarray=None, labels=None, obj_mask:np.ndarray=None):
        super().__init__(img, obj_type=obj_type, aff_mask=aff_mask, labels=labels, obj_mask=obj_mask)
        self.dtype = 'HeatMap'
    # TODO: 热力图的标注转换

class HANDAL_IMG(Image):
    def __init__(self, img, obj_type, labels=None, aff_mask=None, obj_mask=None, visible_mask=None, **kwargs):
        """
        HANDAL数据集的Image子类
        
        Args:
            img: 图片数组
            obj_type: 物体类型
            labels: 标签列表
            aff_mask: affordance mask列表
            obj_mask: 物体mask
            visible_mask: 可见部分mask
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            img=img,
            obj_type=obj_type,
            labels=labels,
            aff_mask=aff_mask,
            obj_mask=obj_mask,
            visible_mask=visible_mask,
            **kwargs
        )
    
    @classmethod
    def load_file(cls, filepath, obj_type=None):
        raise NotImplementedError('懒得写，直接使用 HANDAL_IMG.load_all')

    @classmethod
    def load_all(cls, dir_path, obj_type, aff_type='grasp', **kwargs):
        """
        Args:
            dir_path: 指HANDAL数据集中一个压缩包解压后的位置
            obj_type: 手动指定这个压缩包下的物体种类
            aff_type: 默认只有抓取这一个动作
        """
        def iterator():
            """需要手动指定种类和文件目录"""
            for t in ['test', 'train']:
                path = os.path.join(dir_path, t)
                if not os.path.isdir(path):
                    continue

                for video_id in sorted(os.listdir(path)):
                    base_path = os.path.join(path, video_id)
                    if not os.path.isdir(base_path):
                        continue

                    # 获取该目录下所有 jpg 和 png 文件并排序
                    img_files = sorted(
                        f for f in os.listdir(os.path.join(path, video_id, 'rgb'))
                        if f.lower().endswith('.jpg') or f.lower().endswith('.png')
                    )

                    # 每 15 张取一张（约 8~9 张），按排序顺序抽取
                    for idx in range(0, len(img_files), 15):
                        fname = os.path.basename(img_files[idx])
                        o_id = os.path.splitext(fname)[0]

                        # 原始图片
                        img_path = os.path.join(base_path, 'rgb', fname)  # jpg
                        img = cv2.imread(img_path)

                        # 物体部分（含被遮挡）
                        obj_path = os.path.join(base_path, 'mask', f'{o_id}_000000.png')
                        obj_mask = cv2.imread(obj_path)

                        # handle部分的mask
                        aff_path = os.path.join(base_path, 'mask_parts', f'{o_id}_000000_handle.png')
                        aff_mask = cv2.imread(aff_path)

                        # 可见部分
                        visib_path = os.path.join(base_path, 'mask_visib', f'{o_id}_000000.png')
                        visib_mask = cv2.imread(visib_path)

                        obj = HANDAL_IMG(
                            img,
                            obj_type=obj_type,
                            labels=[aff_type],
                            aff_mask=[aff_mask],
                            obj_mask=obj_mask,
                            visible_mask=visib_mask,
                        )
                        print(f'loading IMG: {img_path}')
                        yield obj

        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root, obj_type, aff_type='grasp', **kwargs):
        for img in cls.load_all(input_root, obj_type=obj_type, aff_type=aff_type, **kwargs):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            img.save_to(dir_path)

class AGD20k_IMG(HeatImage):...

class RAGNet(Image):
    sub_dataset = [
        '3doi_easy_reasoning_val.pkl',
        # '3doi_val.pkl',
        'egoobjects_easy_reasoning_train.pkl',
        'egoobjects_hard_reasoning_train.pkl',
        'egoobjects_train.pkl',
        # 'graspnet_test_novel_val.pkl',
        # 'graspnet_test_seen_val.pkl',
        'graspnet_train.pkl',
        # 'handal_easy_reasoning_val.pkl',
        # 'handal_hard_reasoning_train.pkl',
        # 'handal_hard_reasoning_val.pkl',
        # 'handal_mini_val.pkl',
        # 'openx_train.pkl',
        # 'rlbench_train.pkl',
    ]
    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        """
        同时加载图片和文本数据集（绑定id）
        """
        import pickle

        def iterator():
            for sub_dataset in RAGNet.sub_dataset:
                with open(os.path.join(dataset_root_path, sub_dataset), 'rb') as f:
                    pickled_data = pickle.load(f)
                    pickled_data.sort(key=lambda x: x['frame_path'])

                for obj in pickled_data:
                    obj['frame_path'] = os.path.join(dataset_root_path, obj['frame_path'][7:])
                    obj['mask_path'] = os.path.join(dataset_root_path, obj['mask_path'][7:])

                    img = cv2.imread(obj['frame_path'])
                    if img is None: continue
                    print(f"loading IMG: {obj['frame_path']}")

                    img_obj = RAGNet(
                        img=img,
                        labels=['None'],  # HACK: 数据集里没有明确指定aff类型，无法分类保存
                        obj_mask=None,
                        aff_mask=[cv2.imread(obj['mask_path'], cv2.IMREAD_GRAYSCALE)],
                        obj_type = obj['task_object_class'].capitalize()
                    )
                    if 'answer' in obj:
                        Instruction(
                            obj['answer'],
                            obj_type=obj['task_object_class'].capitalize(),
                            aff_type='None',  # HACK: 数据集里没有明确指定aff类型，需要再做处理
                            given_id=img_obj.id
                        )

                    yield img_obj
        return iterator()



"""  -------------------------------------------- 单独运行时作为数据处理工具 -----------------------------------------------  """
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="根据不同的数据集选择不同的处理方式，整合为同一个数据集")
    parser.add_argument("-i", "--input_dir", type=str, help="输入数据集位置", default=DEFAULT_INPUT_DIR)
    parser.add_argument('-o', "--output_dir", type=str, help="输出数据集的绝对位置", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('-info', '--info_file', default=None, help='指定info.json的文件位置并继承编号ID，否则初始化ID')

    parser.add_argument("-m", "--modality", type=str, nargs="+", help="手动添加数据的模态，可选一个或多个",
                         default=['all'], choices=['pc', 'img', 'img_mask', 'ins', 'all'])
    parser.add_argument("-d", "--dataset", type=str, help="按照预设定数据集整理",
                         default=None, choices=['AGPIL', 'PIADv2', 'PIAD', 'RAGNet', 'HANDAL', 'AGD20K', 'LASO'])
    parser.add_argument("-a", "--aff_type", type=str, help="affordance种类", default=None)
    parser.add_argument("-t", "--obj_type", type=str, help="物体类型", default=None)
    parser.add_argument('-s', '--show', type=str, nargs="+", help='直接渲染点云文件的路径，选择时只执行渲染操作', default=[])

    args = parser.parse_args()


    # 兼容相对/绝对路径
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not os.path.isdir(input_dir):
        raise ValueError(fr'{input_dir} is not a valid directory')
    os.makedirs(output_dir, exist_ok=True)


    # 数据集的统计信息，只在构建数据集的时候使用
    info_file = os.path.join(DEFAULT_INPUT_DIR, 'info.json')
    info_dict = {
        'PointCloud': defaultdict(lambda: defaultdict(int)),  # 对应PointCloud.count
        'Image': defaultdict(lambda: defaultdict(int)),
        'Instruction': defaultdict(lambda: defaultdict(int)),
    }

    # 如果要增加某个数据集同时继续编号，则需要指定--info_file加载输出数据集位置下的info.json
    keep_id = False
    if args.info_file is not None:
        keep_id =True
        info_file = resolve_path(args.info_file)

        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                loaded_dict = json.load(f)
                for k, v in info_dict.items():
                    for obj_type, vals in loaded_dict.get(k, {}).items():
                        info_dict[k][obj_type] = defaultdict(int, vals)

                # 恢复 cls.count计数
                PointCloud.count = info_dict['PointCloud']
                Image.count = info_dict['Image']
                Instruction.count = info_dict['Instruction']
        else:
            warnings.warn(f"没有找到info.json: {info_file}, 使用初始info_dict")


    # 整理模态输入
    selected_modalities = set(args.modality)
    if 'all' in selected_modalities:
        selected_modalities = {'pc', 'img', 'img_mask', 'ins'}

    err = None
    try:
        if args.show:
            for f in args.show:
                file_path = resolve_path(f)
                match args.dataset:
                    case None:
                        pc = PointCloud.load_file(file_path)
                    case 'AGPIL':
                        pc = AGPIL_PC.load_file(file_path)
                    case 'PIADv2':
                        pc = PIADv2_PC.load_file(file_path)
                    case e:
                        raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')
                pc.show()

        else:
            if 'pc' in selected_modalities:
                match args.dataset:
                    case None:
                        PointCloud.load_and_save(input_dir, output_dir, keep_id=keep_id)
                    case 'AGPIL':
                        AGPIL_PC.load_and_save(input_dir, output_dir)
                    case 'PIADv2':
                        tmp = list(PIADv2_PC.load_all(input_dir))
                        PointCloud.deduplicate()
                        PointCloud.save_all(output_dir)
                    case 'PIAD':
                        ...
                    case 'LASO':
                        LASO_PC.load_and_save(input_dir, output_dir)
                    case e:
                        raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

            if 'img' in selected_modalities:
                match args.dataset:
                    case None:
                        Image.load_and_save(input_dir, output_dir, keep_id=keep_id)
                    case 'HANDAL':
                        assert args.obj_type is not None and args.aff_type is not None
                        HANDAL_IMG.load_and_save(input_dir, output_dir, obj_type=args.obj_type, aff_type=args.aff_type)
                    case 'RAGNet':
                        RAGNet.load_and_save(input_dir, output_dir)
                    case 'AGD20K' | 'AGD20k': ...
                    case e:
                        raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

            if 'ins' in selected_modalities:
                if args.dataset == 'RAGNet':
                    Instruction.save_all(output_dir)  # 直接保存之前加载的数据
    except Exception as e:
        err = e

    """  ----------------------------------- 保存信息文件 -------------------------------------  """

    # 更新 PointCloud.count（保留最大的计数）
    for obj, v in PointCloud.count.items():
        for aff, count in v.items():
            current_max = info_dict['PointCloud'][obj][aff]
            info_dict['PointCloud'][obj][aff] = max(count, current_max)

    # 更新 Image.id（保留最大的计数）
    for obj, v in Image.count.items():
        for aff, count in v.items():
            current_max = info_dict['Image'][obj][aff]
            info_dict['Image'][obj][aff] = max(count, current_max)

    # 更新 Instruction.id （直接覆盖）
    info_dict['Instruction'] = Instruction.count

    info_file = os.path.join(output_dir, 'info.json')
    with open(info_file, 'w') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)


    if err: raise err