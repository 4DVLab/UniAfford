import os
import json
import warnings
import numpy as np
import open3d as o3d
import cv2
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
    
    def __init__(self, points, obj_type, mask:np.ndarray=None, labels:list=None):
        self.points = points
        self.obj_type = obj_type
        
        PointCloud.all[obj_type].append(self)
        PointCloud.count[obj_type]['ID'] += 1
        self.id = PointCloud.count[obj_type]['ID']

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
    def load_file(filepath, obj_type=None) -> 'PointCloud':
        """加载时重新分配id"""
        obj_type = os.path.basename(os.path.dirname(filepath)) if obj_type is None else obj_type

        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            header = first_line.split(',') if first_line else []
            data = np.loadtxt(f, delimiter=',', skiprows=1)

        if len(header) > 3:
            pc_obj = PointCloud(points = data[:, :3], mask=data[:, 3:], obj_type=obj_type, labels=header[3:])
        else:
            pc_obj = PointCloud(points = data[:, :3], obj_type=obj_type)

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
    def load_all(cls, dataset_root_path):
        def iterator():
            for obj_type in os.listdir(dataset_root_path):
                dir_path = os.path.join(dataset_root_path, obj_type, 'PointCloud')
                if not os.path.isdir(dir_path):
                    continue
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        print(f'loading PC: {file_path}')
                        yield cls.load_file(file_path, obj_type=obj_type)

        return iterator()
    
    @classmethod
    def load_and_save(cls, input_root, output_root):
        for pc in cls.load_all(input_root):
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
                
                # 初始化颜色数组：背景色为深灰色（强烈对比）
                colors = np.full((self.points.shape[0], 3), [0.1, 0.1, 0.1])  # 深灰色背景
                
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
    def load_file(filepath, obj_type=None) -> 'PointCloud':
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
    def load_all(cls, dataset_root_path):
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
    def load_all(cls, dataset_root_path):
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


"""  ----------------------------------------------- Image classes ----------------------------------------------  """

class Image:
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    
    def __init__(self, img:np.ndarray,
            obj_type,
            labels:list=None,
            aff_mask:list[np.ndarray]=None,
            obj_mask:np.ndarray=None,
            visible_mask:np.ndarray=None
        ):
        """ 
        Args:
            aff_mask: affordance区域的标注信息
            obj_mask: 整个物体的区域信息（含被遮挡部分）
        """
        self.img = img
        self.labels = []
        if labels:
            for l in labels:
                Image.count[obj_type][l] += 1
            self.labels = labels

        self.dtype = 'No-mask' if aff_mask is None else 'Segmented'
        self.obj_type = obj_type

        Image.count[self.obj_type]['ID'] += 1
        self.id = Image.count[self.obj_type]['ID']

        self.mask = aff_mask if aff_mask is not None else []
        self.obj_mask = obj_mask
        self.visible_mask = visible_mask

    def save_to(self, dir_path):
        # dir_path 应该是目录，生成2~4个文件：原图 和 aff_mask，并在obj_mask目录下并排保存图片的物体mask和可见部分mask（如有）
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 保存img原图
        rgb_dir = os.path.join(dir_path, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        img_path = os.path.join(rgb_dir, f'{self.obj_type}_{self.id}.png')
        # # 确保img是uint8格式和三通道
        # if img_to_save.dtype != np.uint8:
        #     img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
        # if img_to_save.ndim == 2:  # 灰度图转三通道
        #     img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
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

    @classmethod
    def load_file(cls, filepath, obj_type=None):
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
        
        img_obj = Image(
            img=img,
            obj_type=obj_type,
            labels=labels,
            aff_mask=aff_mask,
            obj_mask=obj_mask,
            visible_mask=visible_mask
        )
        
        return img_obj

    @classmethod
    def load_all(cls, dataset_root_path, with_obj_mask=False):
        """
        从保存的数据集目录结构中加载所有图片
        
        Args:
            dataset_root_path: 数据集根目录，结构为 {obj_type}/rgb/{obj_type}_{id}.png
            with_obj_mask: 是否加载obj_mask（已自动处理，此参数保留兼容性）
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
                        img = cls.load_file(rgb_path, obj_type=obj_type)
                        yield img
                    except Exception as e:
                        print(f"Failed to load {rgb_path}: {e}")
                        continue
        
        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root):
        for img in cls.load_all(input_root):
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
        self.mask = None
        self.obj_mask = None
        self.visible_mask = None
        self.labels = None

class BoxedImage(Image):
    def __init__(self, img, box:np.ndarray=None, labels=None):
        super().__init__(img, labels=labels)

        # 直接将box区域划作mask
        self.mask = np.zeros_like(img)
        self.mask[box[1]:box[3], box[0]:box[2]] = 1
        self.dtype = 'Boxed'

class HeatImage(Image):
    def __init__(self, img:np.ndarray, obj_type, aff_mask:np.ndarray=None, labels=None, obj_mask:np.ndarray=None):
        super().__init__(img, obj_type=obj_type, aff_mask=aff_mask, labels=labels, obj_mask=obj_mask)
        self.dtype = 'HeatMap'

class HANDAL_IMG(Image):
    @classmethod
    def load_file(cls, filepath, obj_type=None):
        raise SyntaxError('懒得写，直接使用 HANDAL_IMG.load_all')

    @classmethod
    def load_all(cls, dir_path, obj_type, aff_type='grasp'):
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
    def load_and_save(cls, input_root, output_root, obj_type, aff_type='grasp'):
        for img in cls.load_all(input_root, obj_type=obj_type, aff_type=aff_type):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            img.save_to(dir_path)

class AGD20k_IMG(Image):...
    # TODO: 热力图的标注转换

class RAGNet(Image):...

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
                         default=None, choices=['AGPIL', 'PIADv2', 'PIAD', 'RAGNet', 'HANDAL', 'AGD20K'])
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
        'Instrument': defaultdict(lambda: defaultdict(int)),
    }

    # 如果要增加某个数据集同时继续编号，则需要指定--info_file加载输出数据集位置下的info.json
    if args.info_file is not None:
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
        else:
            warnings.warn(f"没有找到info.json: {info_file}, 使用初始info_dict")


    # 整理模态输入
    selected_modalities = set(args.modality)
    if 'all' in selected_modalities:
        selected_modalities = {'pc', 'img', 'img_mask', 'ins'}
    

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
                    PointCloud.load_and_save(input_dir, output_dir)
                case 'AGPIL':
                    AGPIL_PC.load_and_save(input_dir, output_dir)
                case 'PIADv2':
                    tmp = list(PIADv2_PC.load_all(input_dir))
                    PointCloud.deduplicate()
                    PointCloud.save_all(output_dir)

                case 'PIAD':
                    ...
                case 'RAGNet':
                    ...
                case e:
                    raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

        if 'img' in selected_modalities:
            match args.dataset:
                case None:
                    Image.load_and_save(input_dir, output_dir)
                case 'HANDAL':
                    assert args.obj_type is not None and args.aff_type is not None
                    HANDAL_IMG.load_and_save(input_dir, output_dir, obj_type=args.obj_type, aff_type=args.aff_type)
                case 'AGD20K' | 'AGD20k': ...

                case e:
                    raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

        if 'ins' in selected_modalities:
            pass


    """  ----------------------------------- 保存信息文件 -------------------------------------  """
    info_file = os.path.join(output_dir, 'info.json')

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

    # 保存信息文件
    with open(info_file, 'w') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)





