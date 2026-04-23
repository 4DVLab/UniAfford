"""
!!WARNING: 由于数据构建过程是一个一个数据集处理的，故其余数据集的处理代码可能在更新迭代中存在部分问题，仅供参考。
    注意所有的'Note'注释。
"""

import sys
import os
import warnings
import numpy as np
import cv2
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.base_dataset import Instruction, Image, PointCloud, load_info, save_info, create_info_dict
from utils.common import resolve_path

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

    def __init__(self, points, obj_type, aff_mask_dict: dict = None):
        super().__init__(points, obj_type=obj_type, aff_mask_dict=aff_mask_dict)
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
        if zero_col_idx.size > 0:
            data = np.delete(data, zero_col_idx+3, axis=1) 
        aff_mask_dict = {}
        if data.shape[1] > 3:
            kept_labels = [label for idx, label in enumerate(AGPIL_PC.aff_type) if idx not in zero_col_idx]
            for idx, label in enumerate(kept_labels):
                aff_mask_dict[label] = data[:, 3 + idx]
        pc_obj = AGPIL_PC(points=data[:, :3], obj_type=obj_type, aff_mask_dict=aff_mask_dict)
        
        return pc_obj

    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        def iterator():
            for obj_type in list(cls.all):
                for view in os.listdir(dataset_root_path):
                    for s in ['Seen']: #, 'Unseen']:
                        for t in ['Train']: #, 'Test']:
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

    def __init__(self, points, obj_type, aff_mask_dict: dict = None):
        super().__init__(points, obj_type, aff_mask_dict=aff_mask_dict)
        PIADv2_PC.all[obj_type].append(self)
        PIADv2_PC.count[obj_type]['ID'] += 1

    @staticmethod
    def load_file(filepath, obj_type=None, aff_type=None) -> 'PointCloud':
        data = np.load(filepath)
        aff_mask_dict = {}
        if data.shape[1] > 3 and aff_type is not None:
            point_mask = data[:, 3]
            aff_mask_dict[str(aff_type)] = point_mask
        pc_obj = PIADv2_PC(points=data[:, :3], obj_type=obj_type, aff_mask_dict=aff_mask_dict)

        return pc_obj

    @classmethod
    def load_all(cls, dataset_root_path, **kwargs):
        """
        Args:
            dataset_root_path: PIADv2数据集的位置，下层目录为 Seen,Unseen_aff,Unseen_obj(任一）
        """
        def iterator():
            # PIADv2的Seen,Unseen_aff,Unseen_obj三个数据集只是同一个数据集的不同划分，根据需要处理一个就行
            for s in ['Seen']: #, 'Unseen_aff', 'Unseen_obj']:
                if not os.path.isdir(os.path.join(dataset_root_path, s)): continue
                for t in ['train', 'val']:#, 'test']:
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

    def __init__(self, points, obj_type, aff_mask_dict: dict = None):
        super().__init__(points, obj_type, aff_mask_dict=aff_mask_dict)
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
                    yield cls(
                        points=obj_points[i],
                        obj_type=e[1]['class'],
                        aff_mask_dict={str(e[1]['affordance']): e[1]['mask']},
                    )

        return iterator()

class AffordanceNet3D_PC(PointCloud):
    """
    适配 3DAffordanceNet/dataset/AffordanceNet.py 的 full-shape pkl 格式：
    - full_shape_{split}_data.pkl
    - 每条样本包含 shape_id / semantic class / affordance / full_shape
    - full_shape 内包含 coordinate [N,3] 与 label(dict: aff -> [N])
    """

    @staticmethod
    def _normalize_obj_type(name):
        return str(name).strip() if name is not None else "Unknown"

    @classmethod
    def _iter_split_data(cls, dataset_root_path, split):
        import pickle as pkl
        pkl_path = os.path.join(dataset_root_path, f"full_shape_{split}_data.pkl")
        if not os.path.isfile(pkl_path):
            return []
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
        return data if isinstance(data, list) else []

    @classmethod
    def load_all(cls, dataset_root_path, aff_type=None, **kwargs):
        """
        Args:
            dataset_root_path: 3DAffordanceNet 数据目录
            split: 'all'/'train'/'val'/'test'，默认 all
            aff_type: 可选，只保留指定 affordance（str 或 list）
        """
        aff_set = set([str(aff_type)]) if isinstance(aff_type, str) else (set(map(str, aff_type)) if aff_type is not None else None)

        def iterator():
            for sp in ['train', 'val']:#, 'test']:
                samples = cls._iter_split_data(dataset_root_path, sp)
                if not samples:
                    continue
                for info in samples:
                    full_shape = info.get("full_shape", {})
                    coords = full_shape.get("coordinate", None)
                    label_dict = full_shape.get("label", {})
                    if coords is None:
                        continue
                    points = np.asarray(coords, dtype=np.float32)
                    if points.ndim != 2 or points.shape[1] != 3:
                        continue

                    aff_names = info.get("affordance", [])
                    if not isinstance(aff_names, (list, tuple)):
                        aff_names = list(label_dict.keys())
                    if aff_set is not None:
                        aff_names = [a for a in aff_names if str(a) in aff_set]

                    aff_mask_dict = {}
                    for aff in aff_names:
                        mask = label_dict.get(aff)
                        if mask is None:
                            continue
                        arr = np.asarray(mask, dtype=np.float32).reshape(-1)
                        if arr.shape[0] != points.shape[0]:
                            continue
                        aff_mask_dict[str(aff)] = arr

                    obj_type = cls._normalize_obj_type(info.get("semantic class", "Unknown"))
                    print(f'loading PC {sp}: shape_id={info.get("shape_id", "N/A")}, obj={obj_type}')
                    yield cls(points=points, obj_type=obj_type, aff_mask_dict=aff_mask_dict)

        return iterator()


class GEAL_PC(PointCloud):
    """
    适配 GEAL 论文中的 Corrupted 3D Affordance 数据（LASO-C / PIAD-C，与 Hugging Face 发布结构一致），
    将 `dataset/corrupt.py` 中 CorruptDataset 所使用的 pickle 标注转为 `PointCloud`（进而可配合
    `PointCloud.save_all` / JointDataset 目录格式）。

    期望 **父目录** 下并列两个子数据集（名称可通过 ``subset_names`` 修改）::

        parent_root/
          LASO-C/
            point/
              {corrupt_type}_{level}.pkl
              ...
            text/
              Affordance-Question.csv
          PIAD-C/
            point/
              ...
            text/
              ...

    ``load_all`` 会依次扫描各子集 ``point/`` 下所有 ``.pkl``，将物体类别写为
    ``{子集名}_{class}``（如 ``LASO-C_Bed``），避免两子集类别名冲突、并保留来源。

    每条 pickle 记录需包含：class, affordance, mask, point。
    点云坐标按 pickle 内存储的（已损坏）坐标原样写入，不做 ``normalize_point_cloud``。
    """

    _DEFAULT_SUBSETS = ("LASO-C", "PIAD-C")

    @classmethod
    def load_all(
        cls,
        dataset_root_path,
        aff_type=None,
        obj_type=None,
        **kwargs,
    ):
        import pickle

        root = resolve_path(dataset_root_path)

        aff_set = None
        if aff_type is not None:
            aff_set = {aff_type} if isinstance(aff_type, str) else set(map(str, aff_type))
        obj_set = None
        if obj_type is not None:
            obj_set = {obj_type} if isinstance(obj_type, str) else set(obj_type)

        def _yield_from_pkl(pkl_path: str, subset_label: str):
            with open(pkl_path, "rb") as f:
                annotations = pickle.load(f)
            if not isinstance(annotations, (list, tuple)):
                warnings.warn(f"GEAL_PC: 跳过非列表 pkl: {pkl_path}")
                return
            base_msg = os.path.relpath(pkl_path, root)
            for data in annotations:
                obj_c = str(data.get("class", ""))
                aff = str(data.get("affordance", ""))
                if obj_set is not None and obj_c not in obj_set:
                    continue
                if aff_set is not None and aff not in aff_set:
                    continue
                pts = np.asarray(data.get("point"), dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    continue
                mask = np.asarray(data.get("mask"), dtype=np.float32).reshape(-1)
                if mask.shape[0] != pts.shape[0]:
                    continue
                aff_mask_dict = {aff: mask}
                obj_key = f"{subset_label}_{obj_c}"
                print(f"loading GEAL_PC [{base_msg}] -> {obj_key} / {aff}")
                yield cls(points=pts, obj_type=obj_key, aff_mask_dict=aff_mask_dict)

        def iterator():
            found_any_subset = False
            for subset in cls._DEFAULT_SUBSETS:
                sub_root = os.path.join(root, subset)
                if not os.path.isdir(sub_root):
                    warnings.warn(f"GEAL_PC: 子数据集目录不存在，已跳过: {sub_root}")
                    continue
                found_any_subset = True
                point_dir = os.path.join(sub_root, "point")
                if not os.path.isdir(point_dir):
                    warnings.warn(f"GEAL_PC: 无 point 目录，已跳过: {point_dir}")
                    continue
                pkl_files = sorted(
                    f for f in os.listdir(point_dir) if f.lower().endswith(".pkl")
                )
                if not pkl_files:
                    warnings.warn(f"GEAL_PC: point 目录内无 pkl: {point_dir}")
                    continue
                for fname in pkl_files:
                    pkl_path = os.path.join(point_dir, fname)
                    if not os.path.isfile(pkl_path):
                        continue
                    yield from _yield_from_pkl(pkl_path, subset)
            if not found_any_subset:
                raise FileNotFoundError(
                    f"GEAL_PC: 在 {root} 下未找到任何子数据集目录 {cls._DEFAULT_SUBSETS}，请检查路径与子文件夹命名。"
                )
        
        return iterator()


"""  ----------------------------------------------- Image classes ----------------------------------------------  """

class BoxedImage(Image):
    def __init__(self, img, obj_type, box:np.ndarray=None, aff_type: str = "box", **kwargs):
        """
        Args:
            img: 图片数组
            obj_type: 物体类型
            box: 标注文本框的左上角(x1, y1)和右下角(x2, y2)组合成的数组(x1, y1, x2, y2)
            **kwargs: 其他传递给父类的参数
        """
        aff_mask_dict = kwargs.pop("aff_mask_dict", None) or {}
        super().__init__(img, obj_type=obj_type, aff_mask_dict=aff_mask_dict, **kwargs)

        # 直接将box区域划作mask
        if box is not None:
            box_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            box_mask[box[1]:box[3], box[0]:box[2]] = 1
            self.aff_mask_dict[str(aff_type)] = box_mask
        self.dtype = 'Boxed'


class HANDAL_IMG(Image):
    directory_to_category = {
        "handal_dataset_adjustable_wrenches": "Wrench",
        "handal_dataset_measuring_cups": "Cup",
        "handal_dataset_mugs": "Mug",
        "handal_dataset_pots_pans": "Pot or Pan",
        "handal_dataset_power_drills": "Power-drill",
        "handal_dataset_ratchets": "Ratchet",
        "handal_dataset_screwdrivers": "Screwdriver",
        "handal_dataset_spatulashi": "Spatulashi",
        "handal_dataset_strainers": "Strainer",
        "handal_dataset_utensils": "Utensil"
    }
    def __init__(self, img, obj_type, aff_mask_dict=None, obj_mask=None, visible_mask=None, **kwargs):
        """
        HANDAL数据集的Image子类
        
        Args:
            img: 图片数组
            obj_type: 物体类型
            aff_mask_dict: affordance 掩码字典
            obj_mask: 物体mask
            visible_mask: 可见部分mask
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            img=img,
            obj_type=obj_type,
            aff_mask_dict=aff_mask_dict,
            obj_mask=obj_mask,
            visible_mask=visible_mask,
            **kwargs
        )
    
    @classmethod
    def load_file(cls, filepath, obj_type=None):
        raise NotImplementedError('懒得写，直接使用 HANDAL_IMG.load_all')

    @classmethod
    def load_all(cls, dir_path, obj_type=None, aff_type='grasp', **kwargs):
        """
        Args:
            dir_path: HANDAL 数据集根目录，或某一个子目录（如 handal_dataset_mugs）
            obj_type: 可选，手动指定物体类别。未指定时根据 directory_to_category 自动映射
            aff_type: 默认只有抓取这一个动作
        """
        def iterator():
            def _iter_one_dir(one_dir: str, mapped_obj: str):
                for t in ['train']:  # 'test'
                    path = os.path.join(one_dir, t)
                    if not os.path.isdir(path):
                        continue

                    for video_id in sorted(os.listdir(path)):
                        base_path = os.path.join(path, video_id)
                        if not os.path.isdir(base_path):
                            continue
                        rgb_dir = os.path.join(base_path, 'rgb')
                        if not os.path.isdir(rgb_dir):
                            continue

                        # 获取该目录下所有 jpg 和 png 文件并排序
                        img_files = sorted(
                            f for f in os.listdir(rgb_dir)
                            if f.lower().endswith('.jpg') or f.lower().endswith('.png')
                        )

                        # 每 20 张取一张，最多取 8 张
                        count = 0
                        for idx in range(0, len(img_files), 20):
                            fname = os.path.basename(img_files[idx])
                            o_id = os.path.splitext(fname)[0]

                            img_path = os.path.join(rgb_dir, fname)
                            img = cv2.imread(img_path)
                            if img is None:
                                continue

                            aff_path = os.path.join(base_path, 'mask_parts', f'{o_id}_000000_handle.png')
                            aff_mask = cv2.imread(aff_path, cv2.IMREAD_GRAYSCALE)
                            if aff_mask is None:
                                continue

                            obj = HANDAL_IMG(
                                img,
                                obj_type=mapped_obj,
                                aff_mask_dict={str(aff_type): aff_mask},
                            )
                            print(f'loading IMG: {img_path}')
                            yield obj
                            count += 1
                            if count > 7:
                                break

            # 自动模式：dir_path 为根目录，遍历映射表中所有子目录
            root_name = os.path.basename(os.path.abspath(dir_path))
            if root_name in cls.directory_to_category:
                mapped_obj = obj_type or cls.directory_to_category[root_name]
                yield from _iter_one_dir(dir_path, mapped_obj)
            else:
                for folder, category in cls.directory_to_category.items():
                    one_dir = os.path.join(dir_path, folder)
                    if not os.path.isdir(one_dir):
                        continue
                    mapped_obj = obj_type or category
                    yield from _iter_one_dir(one_dir, mapped_obj)

        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root, obj_type=None, aff_type='grasp', **kwargs):
        for img in cls.load_all(input_root, obj_type=obj_type, aff_type=aff_type, **kwargs):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            img.save_to(dir_path)

class AGD20k_IMG(Image):
    """
    AGD20K（Cross-View-AG）第一人称测试集：目录为
    ``{root}/{Seen|Unseen}/testset/egocentric/{affordance_name}/{object_name}/{rgb}``，
    像素级 GT 为同结构的 ``.../testset/GT/.../*.png``（灰度图，与 rgb 同名改扩展名）。

    导出时 ``obj_type`` 为 ``{affordance_name}_{object_name}``（文件夹名中的下划线已改为空格，二者之间仍用下划线连接），
    避免不同动作下相同物体文件夹名冲突；``aff_mask_dict`` 的 key 为规范化后的动作名（下划线→空格）。
    """

    _VALID_RGB_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, img, obj_type, aff_mask_dict=None, obj_mask=None, visible_mask=None, **kwargs):
        super().__init__(
            img=img,
            obj_type=obj_type,
            aff_mask_dict=aff_mask_dict,
            obj_mask=obj_mask,
            visible_mask=visible_mask,
            **kwargs,
        )

    @classmethod
    def load_all(cls, dataset_root_path, splits=None, subset='testset', **kwargs):
        """
        Args:
            dataset_root_path: AGD20K 根目录（含 Seen/ Unseen），或某一划分目录
            splits: 遍历的划分，默认 ``('Seen', 'Unseen')``；若 *dataset_root_path* 已为划分子目录则忽略
            subset: 含 GT 的子集名，默认为 ``testset``（与官方训练代码一致）
        """
        kwargs.pop('keep_id', None)

        def resolve_agd_root_and_splits(dataset_root_path, splits):
            """支持 ``AGD20K`` 根目录，或直接传入 ``.../Seen``、``.../Unseen`` 子目录。"""
            root = os.path.normpath(dataset_root_path)
            base = os.path.basename(root)
            if base in ('Seen', 'Unseen'):
                return os.path.dirname(root), (base,)
            if splits is None:
                splits = ('Seen', )#'Unseen')
            return root, tuple(splits)

        root, split_list = resolve_agd_root_and_splits(dataset_root_path, splits)

        def iterator():
            for sp in split_list:
                ego_root = os.path.join(root, sp, subset, 'egocentric')
                gt_root = os.path.join(root, sp, subset, 'GT')
                if not os.path.isdir(ego_root) or not os.path.isdir(gt_root):
                    continue

                for aff_name in sorted(os.listdir(ego_root)):
                    aff_ego = os.path.join(ego_root, aff_name)
                    if not os.path.isdir(aff_ego):
                        continue
                    for obj_id in sorted(os.listdir(aff_ego)):
                        obj_dir = os.path.join(aff_ego, obj_id)
                        if not os.path.isdir(obj_dir):
                            continue
                        for fname in sorted(os.listdir(obj_dir)):
                            lower = fname.lower()
                            if not lower.endswith(cls._VALID_RGB_EXT):
                                continue
                            stem, _ = os.path.splitext(fname)
                            mask_path = os.path.join(gt_root, aff_name, obj_id, stem + '.png')
                            if not os.path.isfile(mask_path):
                                continue

                            img_path = os.path.join(obj_dir, fname)
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            aff_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if aff_mask is None:
                                continue

                            aff_label = aff_name.replace('_', ' ')
                            obj_label = obj_id.replace('_', ' ')
                            print(f'loading IMG: {img_path}')
                            yield cls(
                                img,
                                obj_type=obj_label,
                                aff_mask_dict={aff_label: aff_mask},
                            )

        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root, splits=None, subset='testset', **kwargs):
        id_counter = {}
        for img in cls.load_all(input_root, splits=splits, subset=subset, **kwargs):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            else:
                id_counter[img.obj_type] = id_counter.get(img.obj_type, 0) + 1
                file_id = id_counter[img.obj_type]
            img.save_to(dir_path)
            img.free_memory()


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
                        obj_mask=None,
                        aff_mask_dict={'None': cv2.imread(obj['mask_path'], cv2.IMREAD_GRAYSCALE)},
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
                         default=None, choices=['AGPIL', 'PIADv2', 'PIAD', 'RAGNet', 'HANDAL', 'AGD20K', 'LASO', 'AffordanceNet3D', 'GEAL'])
    parser.add_argument("-a", "--aff_type", type=str, help="affordance种类", default=None)
    parser.add_argument("-t", "--obj_type", type=str, help="物体类型", default=None)
    parser.add_argument('-s', '--show', type=str, nargs="+", help='直接渲染点云文件的路径，选择时只执行渲染操作', default=[])
    parser.add_argument('--save_split', action='store_true', default=True,
                        help='处理完成后自动生成分割文件（默认开启）')

    args = parser.parse_args()


    # 兼容相对/绝对路径
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not os.path.isdir(input_dir):
        raise ValueError(fr'{input_dir} is not a valid directory')
    os.makedirs(output_dir, exist_ok=True)


    # 如果要增加某个数据集同时继续编号，则需要指定--info_file加载输出数据集位置下的info.json
    if args.info_file is not None:
        keep_id = True
        info_file_path = resolve_path(args.info_file)
        info_dict = load_info(info_file_path)
    else:
        keep_id = False
        info_dict = create_info_dict()


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
                    case 'GEAL':
                        raise TypeError(
                            'GEAL（-d GEAL）为 pickle 批量导出，不支持 -s/--show；'
                            '请对导出的 CSV 使用 -d None -s <path>'
                        )
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
                    case 'AffordanceNet3D':
                        AffordanceNet3D_PC.load_and_save(input_dir, output_dir)
                    case 'GEAL':
                        GEAL_PC.load_and_save(input_dir, output_dir)
                    case e:
                        raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

            if 'img' in selected_modalities:
                match args.dataset:
                    case None:
                        Image.load_and_save(input_dir, output_dir, keep_id=keep_id)
                    case 'HANDAL':
                        HANDAL_IMG.load_and_save(
                            input_dir,
                            output_dir,
                            obj_type=args.obj_type,
                            aff_type=args.aff_type or 'grasp',
                        )
                    case 'RAGNet':
                        RAGNet.load_and_save(input_dir, output_dir)
                    case 'AGD20K' | 'AGD20k':
                        AGD20k_IMG.load_and_save(input_dir, output_dir)
                    case e:
                        raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

            if 'ins' in selected_modalities:
                if args.dataset == 'RAGNet':
                    Instruction.save_all(output_dir)  # 直接保存之前加载的数据
    except Exception as e:
        err = e

    """  ----------------------------------- 保存信息文件 -------------------------------------  """
    save_info(output_dir, info_dict)

    # 处理完成后生成分割文件：train=1, val=0, test=0（整库作为训练集）
    # 采用磁盘直读方式，避免 load_and_save 流式处理导致的内存对象不完整问题。
    if not args.show and args.save_split and err is None:
        try:
            from .create_split import save_split_from_disk
        except ImportError:
            from create_split import save_split_from_disk
        save_split_from_disk(output_dir, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    if err: raise err