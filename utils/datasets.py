"""
!!WARNING: 由于数据构建过程是一个一个数据集处理的，故处理其余数据集的代码仅供参考，可能存在部分问题。
    dataset.py文件只保证加载本数据集时不出现问题。
    单独运行该程序时注意所有的'Note'注释。
"""


import os
import json
import warnings
import numpy as np
import open3d as o3d
import cv2
import csv
from collections import defaultdict


# 全局参数
DEFAULT_OUTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 输出的数据集位置（用于数据转换，训练推理时可忽略）
DEFAULT_INPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 加载的数据集位置（训练的输入数据位置，或转换数据集的输入位置）


"""  ----------------------------------------------- utils functions ----------------------------------------------  """

def resolve_path(path_str: str):
    """兼容相对/绝对路径，返回绝对路径。"""
    if path_str is None: return None
    return path_str if os.path.isabs(path_str) else os.path.abspath(os.path.join(os.getcwd(), path_str))


"""  ----------------------------------------------- PointCloud classes ----------------------------------------------  """

class PointCloud:
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    """
    count like: dict{
        'Bed': {
            'ID': 514,
            'sit': 114,
            'lay': '19',
            ...
        },
        Chair: {
            'ID': 81,
            'sit': 9,
        }
        ...
    }
    """

    def __init__(self, points, obj_type, mask:np.ndarray=None, labels:list=None, given_id:int=None):
        self.points = points
        self.obj_type = obj_type
        
        PointCloud.all[obj_type].append(self)
        PointCloud.count[obj_type]['ID'] += 1
        self.id = PointCloud.count[obj_type]['ID'] if given_id is None else given_id

        if labels is not None:
            for l in labels:
                PointCloud.count[obj_type][l] += 1

        self.mask = mask       # 对应点的aff的值
        self.labels = labels   # aff_mask对应列的标签

        self.is_sorted = False
        self.sort()            # 排序点云
        self._hash = None
        hash(self)

    """  ---------------------------------------- 读写相关 ---------------------------------------------  """
    def save_to(self, filepath):
        """统一保存为csv格式，第一行是标签，数据前三列是xyz，后面所有的列分别表示不同aff标注"""
        header = ['x', 'y', 'z']

        # 拼接xyz和mask
        if self.mask is not None:
            data = np.concatenate([self.points, self.mask], axis=1)
            header += self.labels
        else:
            data = self.points

        with open(filepath, 'w') as f:
            np.savetxt(f, data, delimiter=',', header=','.join(header))
    
    @staticmethod
    def load_file(filepath, obj_type=None, keep_id: bool=False) -> 'PointCloud':
        """
        Args:
            keep_id: 是否保持文件的id，默认False加载时重新分配id
        """
        obj_type = os.path.basename(os.path.dirname(filepath)) if obj_type is None else obj_type

        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            header = first_line.split(',') if first_line else []
            data = np.loadtxt(f, delimiter=',', skiprows=1)

        if keep_id:
            file_name = os.path.basename(filepath).strip('.csv')
            given_id = int(file_name.split('_')[1])
        else:
            given_id = None

        if len(header) > 3:
            pc_obj = PointCloud(points=data[:, :3],
                                mask=data[:, 3:],
                                obj_type=obj_type,
                                labels=header[3:],
                                given_id=given_id)
        else:
            pc_obj = PointCloud(points=data[:, :3],
                                obj_type=obj_type,
                                given_id=given_id)

        return pc_obj

    @classmethod
    def save_all(cls, dataset_root_path):
        for k, v in cls.all.items():
            dir_path = os.path.join(dataset_root_path, k, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            id = 0
            for e in v:
                if e is not None:
                    id += 1 # 重新按顺序分配id
                    e.save_to(os.path.join(dir_path, f'{e.obj_type}_{id}.csv')) # 保存时命名为 {obj_type}_{id}.csv

    @classmethod
    def load_all(cls, dataset_root_path, keep_id: bool=False):
        """
        从统一格式的数据集中批量加载 PointCloud

        Args:
            dataset_root_path: 根目录，结构为 {obj_type}/PointCloud/{obj_type}_{id}.csv
            keep_id: 是否保持文件名中的 id，而不是重新分配
        """
        def iterator():
            for obj_type in os.listdir(dataset_root_path):
                dir_path = os.path.join(dataset_root_path, obj_type, 'PointCloud')
                if not os.path.isdir(dir_path):
                    continue
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        print(f'loading PC: {file_path}')
                        yield cls.load_file(file_path, obj_type=obj_type, keep_id=keep_id)

        return iterator()
    
    @classmethod
    def load_and_save(cls, input_root, output_root, keep_id=False):
        for pc in cls.load_all(input_root, keep_id=keep_id):
            dir_path = os.path.join(output_root, pc.obj_type, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            pc.save_to(os.path.join(dir_path, f'{pc.obj_type}_{pc.id}.csv'))
            pc.free_memory()

    """  ---------------------------------------- 工具 ---------------------------------------------  """
    def __eq__(self, other):
        if not isinstance(other, PointCloud):
            raise TypeError(f"{type(other)}不是点云，不可比较")

        if self.points.ndim != other.points.ndim or self.points.shape != other.points.shape:
            return False

        return hash(self) == hash(other)

    def __del__(self):
        # 更新count
        PointCloud.count[self.obj_type]["ID"] -= 1
        for l in self.labels:
            PointCloud.count[self.obj_type][l] -= 1

        self.free_memory()

    def __hash__(self):
        if self._hash is None:
            self.sort(force=False)
            data = (
                self.points.shape,
                self.points.dtype.str,  # e.g., '<f8', '|i4'
                self.points.tobytes()
            )
            self._hash = hash(data)

        return self._hash


    def show(self, selected_labels:list=None):
        """
        Args:
            selected_labels: 只选择部分标签，否则都显示
        """
        if self.mask is not None and self.labels is not None and len(self.labels) > 0:
            for idx, label in enumerate(self.labels):
                if selected_labels is not None and label not in selected_labels: continue
                if self.mask.shape[1] <= idx: raise ValueError(f'Error in {self.obj_type}-{self.id}: mask的列数{self.mask.shape}和label的维度 {label} 不同')

                mask_col = self.mask[:, idx]
                
                # 白色背景
                colors = np.full((self.points.shape[0], 3), [0.8, 0.8, 0.8])
                
                # 有mask值的点：红色渐变，mask值越大颜色越浓（越亮）
                mask_mask = mask_col > 0  # 找到所有非零mask的点
                if mask_mask.any():
                    # 红色通道：从深红(0.3)到亮红(1.0)，根据归一化的mask值
                    colors[mask_mask, 0] = 0.3 + 0.7 * mask_col[mask_mask]  # 红色通道
                    colors[mask_mask, 1] = 0.0  # 绿色通道保持为0
                    colors[mask_mask, 2] = 0.0  # 蓝色通道保持为0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([pcd], window_name=f"Rendering label: {self.obj_type}-{self.id}-{label} (red)")
        else:
            # 无 mask/labels 时直接渲染
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            o3d.visualization.draw_geometries([pcd], window_name=f"Rendering label: {self.obj_type}-{self.id}")

    def sort(self, force=True):
        if self.is_sorted and not force: return False
        else:
            sort_idx = np.lexsort((self.points[:, 2], self.points[:, 1], self.points[:, 0]))
            self.points = self.points[sort_idx]
            self.mask = self.mask[sort_idx]

            self.is_sorted = True
            return True

    @classmethod
    def sort_by_id(cls):
        for obj_type in cls.all.keys():
            cls.all[obj_type].sort(key=lambda x: x.id)

    def free_memory(self):
        """释放自身占用的内存（不更改计数，用于不重复加载的情况）"""
        # 删除内部数组
        self.points = None
        self.mask = None
        self.labels = None

        # 删除self的记录
        PointCloud.all[self.obj_type][self.id - 1] = None

    def _merge(self, other):
        """合并两个点云标注并更新label、计数（默认点云hash相等）"""
        for i, l in enumerate(other.labels):
            if l not in self.labels:
                self.labels.append(l)
                self.mask = np.hstack((self.mask, other.mask[:, [i]]))
        del other
        return self

    @classmethod
    def deduplicate(cls):
        """根据hash值去重合并数据"""
        for obj_type, ls in cls.all.items():
            loaded = dict()
            for pc in ls:
                loaded[pc] = pc._merge(loaded[pc])

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

class Image:
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    
    def __init__(self, img:np.ndarray,
            obj_type,
            labels:list=None,
            aff_mask:list[np.ndarray]=None,
            obj_mask:np.ndarray=None,
            visible_mask:np.ndarray=None,
            given_id:int=None,
        ):
        """ 
        Args:
            aff_mask: affordance区域的标注信息
            obj_mask: 整个物体的区域信息（含被遮挡部分）
        """
        self.img = img
        # 确保img是uint8格式和三通道
        if img.dtype != np.uint8:
            self.img = np.clip(img, 0, 255).astype(np.uint8)
        if self.img.ndim == 2:  # 灰度图转三通道
            self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self.labels = []
        if labels:
            for l in labels:
                Image.count[obj_type][l] += 1
            self.labels = labels

        self.dtype = 'No-mask' if aff_mask is None else 'Segmented'
        self.obj_type = obj_type
        Image.all[obj_type].append(self)

        Image.count[self.obj_type]['ID'] += 1
        self.id = Image.count[self.obj_type]['ID'] if given_id is None else given_id

        self.mask = aff_mask if aff_mask is not None else []
        self.obj_mask = obj_mask
        self.visible_mask = visible_mask

    @classmethod
    def sort_by_id(cls):
        for obj_type in cls.all.keys():
            cls.all[obj_type].sort(key=lambda x: x.id)

    def save_to(self, dir_path):
        # dir_path 应该是目录，生成2~4个文件：原图 和 aff_mask，并在obj_mask目录下并排保存图片的物体mask和可见部分mask（如有）
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 保存img原图
        rgb_dir = os.path.join(dir_path, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        img_path = os.path.join(rgb_dir, f'{self.obj_type}_{self.id}.png')
        cv2.imwrite(img_path, self.img)

        # 保存aff_mask
        if len(self.mask) != 0:
            # 每个mask单独保存
            for idx, mask in enumerate(self.mask):
                mask_label_dir = os.path.join(dir_path, 'mask', self.labels[idx])
                os.makedirs(mask_label_dir, exist_ok=True)
                single_mask_path = os.path.join(mask_label_dir, f'{self.obj_type}_{self.id}_{self.labels[idx]}.png')
                cv2.imwrite(single_mask_path, mask)

        # 保存obj_mask和visib_mask（如有）
        obj_mask_dir = os.path.join(dir_path, 'obj_mask')
        if self.obj_mask is not None and self.obj_mask.size != 0:
            os.makedirs(obj_mask_dir, exist_ok=True)
            obj_mask_path = os.path.join(obj_mask_dir, f'{self.obj_type}_{self.id}_obj_mask.png')
            cv2.imwrite(obj_mask_path, self.obj_mask)
        if self.visible_mask is not None and self.visible_mask.size != 0:
            os.makedirs(obj_mask_dir, exist_ok=True)
            vis_mask_path = os.path.join(obj_mask_dir, f'{self.obj_type}_{self.id}_visible_mask.png')
            cv2.imwrite(vis_mask_path, self.visible_mask)

    def show(self, selected_labels:list=None, overlay=True, wait_key=True):
        """
        使用OpenCV显示图片和mask信息
        
        Args:
            selected_labels: 只显示指定的标签，如果为None则显示所有标签
            overlay: 是否在原图上叠加显示mask（彩色叠加），否则单独显示mask
            wait_key: 是否等待按键（True则按任意键关闭，False则显示1秒后自动关闭）
        """
        # 确保图片是uint8格式
        img_display = self.img.copy()
        if img_display.dtype != np.uint8:
            img_display = np.clip(img_display, 0, 255).astype(np.uint8)
        
        # 显示原图
        cv2.imshow(f'Original - {self.obj_type}_{self.id}', img_display)
        
        # 显示affordance masks
        if len(self.mask) > 0:
            for idx, mask in enumerate(self.mask):
                label = self.labels[idx] if idx < len(self.labels) else f'mask_{idx}'
                
                # 如果指定了selected_labels，跳过不在列表中的
                if selected_labels is not None and label not in selected_labels:
                    continue
                
                # 确保mask是uint8格式
                mask_display = mask.copy()
                if mask_display.dtype != np.uint8:
                    mask_display = (mask_display * 255).clip(0, 255).astype(np.uint8)
                
                if overlay:
                    # 在原图上叠加显示mask（使用红色半透明叠加）
                    mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
                    overlay_img = cv2.addWeighted(img_display, 0.7, mask_colored, 0.3, 0)
                    cv2.imshow(f'Overlay - {self.obj_type}_{self.id}_{label}', overlay_img)
                else:
                    # 单独显示mask
                    cv2.imshow(f'Mask - {self.obj_type}_{self.id}_{label}', mask_display)
        
        # 显示obj_mask
        if self.obj_mask is not None and self.obj_mask.size != 0:
            obj_mask_display = self.obj_mask.copy()
            if obj_mask_display.dtype != np.uint8:
                obj_mask_display = (obj_mask_display * 255).clip(0, 255).astype(np.uint8)
            
            if overlay:
                # 在原图上叠加显示（使用绿色）
                mask_colored = np.zeros_like(img_display)
                mask_colored[:, :, 1] = obj_mask_display  # 绿色通道
                overlay_img = cv2.addWeighted(img_display, 0.7, mask_colored, 0.3, 0)
                cv2.imshow(f'Overlay - {self.obj_type}_{self.id}_obj_mask', overlay_img)
            else:
                cv2.imshow(f'Obj Mask - {self.obj_type}_{self.id}', obj_mask_display)
        
        # 显示visible_mask
        if self.visible_mask is not None and self.visible_mask.size != 0:
            vis_mask_display = self.visible_mask.copy()
            if vis_mask_display.dtype != np.uint8:
                vis_mask_display = (vis_mask_display * 255).clip(0, 255).astype(np.uint8)
            
            if overlay:
                # 在原图上叠加显示（使用蓝色）
                mask_colored = np.zeros_like(img_display)
                mask_colored[:, :, 0] = vis_mask_display  # 蓝色通道
                overlay_img = cv2.addWeighted(img_display, 0.7, mask_colored, 0.3, 0)
                cv2.imshow(f'Overlay - {self.obj_type}_{self.id}_visible_mask', overlay_img)
            else:
                cv2.imshow(f'Visible Mask - {self.obj_type}_{self.id}', vis_mask_display)
        
        # 等待按键或自动关闭
        if wait_key:
            print(f"显示图片: {self.obj_type}_{self.id} - 按任意键关闭所有窗口")
            cv2.waitKey(0)
        else:
            cv2.waitKey(1000)  # 显示1秒
        
        cv2.destroyAllWindows()

    def resize(self, size, interpolation=cv2.INTER_LINEAR):
        """
        将图片及所有mask缩放到指定大小，返回新的Image对象
        
        Args:
            size: 目标尺寸，可以是：
                - (width, height) 元组
                - 单个整数（等比例缩放，保持宽高比）
                - (width, height) 字符串，如 "224x224"
            interpolation: 插值方法，默认 cv2.INTER_LINEAR
                - cv2.INTER_NEAREST: 最近邻插值（适合mask）
                - cv2.INTER_LINEAR: 双线性插值（适合图片）
                - cv2.INTER_CUBIC: 双三次插值
                - cv2.INTER_AREA: 区域插值（适合缩小）
        
        Returns:
            Image: 新的缩放后的Image对象（原对象不变）
        """
        # 解析size参数
        if isinstance(size, str):
            # 处理 "224x224" 格式
            parts = size.lower().split('x')
            if len(parts) == 2:
                size = (int(parts[0]), int(parts[1]))
            else:
                raise ValueError(f"Invalid size format: {size}")
        elif isinstance(size, (int, float)):
            # 单个数字，等比例缩放
            scale = size / max(self.img.shape[:2])
            new_width = int(self.img.shape[1] * scale)
            new_height = int(self.img.shape[0] * scale)
            size = (new_width, new_height)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            size = tuple(size)
        else:
            raise ValueError(f"Invalid size parameter: {size}")
        
        # 缩放原图（创建副本）
        resized_img = cv2.resize(self.img.copy(), size, interpolation=interpolation)
        
        # 缩放affordance masks（使用最近邻插值保持mask的离散性）
        mask_interpolation = cv2.INTER_NEAREST if interpolation == cv2.INTER_LINEAR else interpolation
        resized_masks = []
        if len(self.mask) > 0:
            for mask in self.mask:
                if mask is not None and mask.size > 0:
                    resized_mask = cv2.resize(mask.copy(), size, interpolation=mask_interpolation)
                    resized_masks.append(resized_mask)
                else:
                    resized_masks.append(None)
        
        # 缩放obj_mask
        resized_obj_mask = None
        if self.obj_mask is not None and self.obj_mask.size > 0:
            resized_obj_mask = cv2.resize(self.obj_mask.copy(), size, interpolation=mask_interpolation)
        
        # 缩放visible_mask
        resized_visible_mask = None
        if self.visible_mask is not None and self.visible_mask.size > 0:
            resized_visible_mask = cv2.resize(self.visible_mask.copy(), size, interpolation=mask_interpolation)
        
        # 创建新的Image对象
        resized_image = Image(
            img=resized_img,
            obj_type=self.obj_type,
            labels=self.labels.copy() if self.labels else None,
            aff_mask=resized_masks if resized_masks else None,
            obj_mask=resized_obj_mask,
            visible_mask=resized_visible_mask,
            given_id=self.id  # 保持相同的id
        )
        
        # 由于使用了given_id，需要调整计数（因为__init__中已经增加了计数）
        # 将计数恢复到原来的值（因为这是同一个对象的缩放版本，不应该增加计数）
        Image.count[self.obj_type]['ID'] -= 1
        # 如果labels相同，也需要减少label计数（因为__init__中已经增加了）
        if resized_image.labels:
            for label in resized_image.labels:
                if Image.count[resized_image.obj_type][label] > 0:
                    Image.count[resized_image.obj_type][label] -= 1
        
        return resized_image

    @classmethod
    def load_file(cls, filepath, obj_type=None, keep_id: bool=False):
        """
        根据保存结构加载图片和mask
        
        Args:
            filepath: RGB图片路径
            obj_type: 物体类型，如果为None则从文件路径推断
        
        Returns:
            Image对象
        """
        # 确定目录路径（图片的父目录rgb的父目录Image）
        dir_path = os.path.dirname(os.path.dirname(filepath))
        rgb_filename = os.path.basename(filepath)
        
        # 从文件名提取 obj_type 和 id: {obj_type}_{id}.png
        base_name = os.path.splitext(rgb_filename)[0]
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            inferred_obj_type = parts[0]
            inferred_id = parts[1]
        else:
            raise ValueError(f"Cannot parse obj_type and id from filename: {rgb_filename}")
        
        obj_type = obj_type if obj_type is not None else inferred_obj_type
        
        # 加载RGB图片
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        
        # 加载affordance masks
        aff_mask = []
        labels = []
        mask_dir = os.path.join(dir_path, 'mask')
        if os.path.exists(mask_dir):
            # 遍历mask目录下的所有label子目录
            for label_dir in os.listdir(mask_dir):
                label_path = os.path.join(mask_dir, label_dir)
                if not os.path.isdir(label_path):
                    continue
                
                # 查找对应的mask文件: {obj_type}_{id}_{label}.png
                mask_filename = f'{obj_type}_{inferred_id}_{label_dir}.png'
                mask_filepath = os.path.join(label_path, mask_filename)
                
                if os.path.exists(mask_filepath):
                    mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        aff_mask.append(mask)
                        labels.append(label_dir)
        
        # 加载obj_mask
        obj_mask = None
        obj_mask_dir = os.path.join(dir_path, 'obj_mask')
        if os.path.exists(obj_mask_dir):
            obj_mask_path = os.path.join(obj_mask_dir, f'{obj_type}_{inferred_id}_obj_mask.png')
            if os.path.exists(obj_mask_path):
                obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 加载visible_mask
        visible_mask = None
        if os.path.exists(obj_mask_dir):
            vis_mask_path = os.path.join(obj_mask_dir, f'{obj_type}_{inferred_id}_visible_mask.png')
            if os.path.exists(vis_mask_path):
                visible_mask = cv2.imread(vis_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 解析文件名中的id，用于可选的 keep_id
        given_id = int(inferred_id) if keep_id else None

        img_obj = Image(
            img=img,
            obj_type=obj_type,
            labels=labels if labels else None,
            aff_mask=aff_mask if aff_mask else None,
            obj_mask=obj_mask,
            visible_mask=visible_mask,
            given_id=given_id,
        )

        return img_obj

    @classmethod
    def load_all(cls, dataset_root_path, keep_id=False):
        """
        从保存的数据集目录结构中加载所有图片
        
        Args:
            dataset_root_path: 数据集根目录，结构为 {obj_type}/rgb/{obj_type}_{id}.png
            keep_id: 是否保持文件名中的 id，而不是重新分配
        """
        def iterator():
            for obj_type in os.listdir(dataset_root_path):
                obj_type_dir = os.path.join(dataset_root_path, obj_type)
                if not os.path.isdir(obj_type_dir):
                    continue
                
                rgb_dir = os.path.join(obj_type_dir, 'rgb')
                if not os.path.exists(rgb_dir):
                    continue
                
                # 遍历rgb目录下的所有图片文件
                rgb_files = sorted([
                    f for f in os.listdir(rgb_dir)
                    if f.lower().endswith(('.png', '.jpg'))
                ])
                
                for rgb_file in rgb_files:
                    rgb_path = os.path.join(rgb_dir, rgb_file)
                    try:
                        print(f'loading IMG: {rgb_path}')
                        img = cls.load_file(rgb_path, obj_type=obj_type, keep_id=keep_id)
                        yield img
                    except Exception as e:
                        print(f"Failed to load {rgb_path}: {e}")
                        continue
        
        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root, keep_id=False):
        for img in cls.load_all(input_root, keep_id=keep_id):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            img.save_to(dir_path)
            img.free_memory()

    def __del__(self):
        # 更新count
        Image.count[self.obj_type]["ID"] -= 1
        for l in self.labels:
            Image.count[self.obj_type][l] -= 1

        self.free_memory()

    def free_memory(self):
        self.img=None
        self.mask = None
        self.obj_mask = None
        self.visible_mask = None
        self.labels = None


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
                    print(f'loading IMG: {obj['frame_path']}')

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



"""  -------------------------------------------- 文本指令 -----------------------------------------------  """

class Instruction:
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, ins, obj_type:str=None, aff_type:str=None, given_id:int=None):
        self.ins = ins
        self.obj_type = obj_type
        if obj_type is not None:
            Instruction.all[obj_type].append(self)

        self.aff_type = aff_type
        if aff_type is not None:
            Instruction.count[obj_type][aff_type] += 1

        Instruction.count[self.obj_type]['ID'] += 1  # Note: Ins的ID并不是最大的id，仅表示计数
        self.id = Instruction.count[self.obj_type]['ID'] if given_id is None else given_id

    @classmethod
    def sort_by_id(cls):
        for obj_type in cls.all.keys():
            cls.all[obj_type].sort(key=lambda x: x.id)

    @classmethod
    def load(cls, file_path, keep_id=True):
        # 加载csv文件，包含header
        instructions = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ins = row.get('ins')
                obj_type = row.get('obj_type')
                aff_type = row.get('aff_type')
                if not keep_id:
                    given_id = None
                else:
                    given_id = int(row.get('id'))

                instructions.append(cls(ins, obj_type=obj_type, aff_type=aff_type, given_id=given_id))
        return instructions

    @classmethod
    def load_all(cls, dataset_root_path, keep_id=True):
        for obj_type in os.listdir(dataset_root_path):
            file_path = os.path.join(dataset_root_path, obj_type, 'Instruction.csv')
            if os.path.exists(file_path):
                cls.load(file_path, keep_id=keep_id)


    @classmethod
    def save_all(cls, dataset_root_dir, obj_type:list[str]=None):
        """
        Args:
            obj_type: 需要保存的指定的物品list['bag', 'knife',...]，默认保存所有

        """
        # 保存为csv文件，包含header
        fieldnames = ['ins', 'obj_type', 'aff_type', 'id']
        if isinstance(obj_type, str):
            obj_type = [obj_type]

        Instruction.sort_by_id()

        for o in Instruction.all.keys():
            if obj_type is not None and o not in obj_type: continue

            file_path = os.path.join(dataset_root_dir, o, 'Instruction.csv')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file_exists = os.path.exists(file_path)

            # 打开文件：存在则追加，不存在则新建
            with open(file_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 续写时跳过表头
                if not file_exists:
                    writer.writerow(fieldnames)

                for inst in Instruction.all[o]:
                    writer.writerow([
                        f'"{inst.ins}"',  # 强制加上引号，防止出错
                        inst.obj_type,
                        inst.aff_type,
                        inst.id,
                    ])



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