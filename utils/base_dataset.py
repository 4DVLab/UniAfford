"""
聚合 Instruction、Image、PointCloud 三元组的数据集类
支持训练集、测试集、验证集的比例分割
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
import warnings
from tqdm import tqdm
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
import cv2
import csv
from utils.common import resolve_path


""" ------------------------------------ Info 信息管理工具 ----------------------------------- """

def create_info_dict() -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    创建一个空的 info_dict 结构
    """
    info_dict = {
        'PointCloud': defaultdict(lambda: defaultdict(int)),
        'Image': defaultdict(lambda: defaultdict(int)),
        'Instruction': defaultdict(lambda: defaultdict(int)),
    }
    # 兼容缩写
    info_dict['pc'] = info_dict['PointCloud']
    info_dict['img'] = info_dict['Image']
    info_dict['ins'] = info_dict['Instruction']
    return info_dict

def load_info(file_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    从 info.json 文件加载数据集统计信息，并恢复各类的计数器
    """
    info_dict = create_info_dict()
   
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_dict = json.load(f)
            for k in info_dict.keys():
                for obj_type, vals in loaded_dict.get(k, {}).items():
                    info_dict[k][obj_type] = defaultdict(int, vals)
           
            # 恢复 cls.count 计数
            PointCloud.count = info_dict['PointCloud']
            Image.count = info_dict['Image']
            Instruction.count = info_dict['Instruction']
       
        print(f"已从 {file_path} 加载统计信息")
    else:
        warnings.warn(f"没有找到 info.json: {file_path}, 使用初始 info_dict")
   
    return info_dict

def save_info(output_dir: str, info_dict: Dict[str, Dict[str, Dict[str, int]]] = None) -> str:
    """
    保存数据集统计信息到 info.json 文件
    
    Args:
        output_dir: 输出目录路径
        info_dict: 可选，已有的 info_dict；如果为 None，则从当前类计数器创建
    
    Returns:
        info_file: 保存的 info.json 文件路径
    """
    if info_dict is None:
        info_dict = create_info_dict()
   
    # 更新 PointCloud.count（保留最大的计数）
    for obj, v in PointCloud.count.items():
        for aff, count in v.items():
            current_max = info_dict['PointCloud'][obj][aff]
            info_dict['PointCloud'][obj][aff] = max(count, current_max)
   
    # 更新 Image.count（保留最大的计数）
    for obj, v in Image.count.items():
        for aff, count in v.items():
            current_max = info_dict['Image'][obj][aff]
            info_dict['Image'][obj][aff] = max(count, current_max)
   
    # 更新 Instruction.count（直接覆盖）
    for obj, v in Instruction.count.items():
        info_dict['Instruction'][obj] = dict(v)
   
    # 转换 defaultdict 为普通 dict 以便 JSON 序列化
    serializable_dict = {}
    for modality, obj_dict in info_dict.items():
        serializable_dict[modality] = {}
        for obj_type, aff_dict in obj_dict.items():
            serializable_dict[modality][obj_type] = dict(aff_dict)
   
    file_path = os.path.join(output_dir, 'info.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
   
    print(f"统计信息已保存至: {file_path}")
    return file_path


""" ------------------------------------ 3种基础模态的支持 ----------------------------------- """
class Modality:
    @classmethod
    def get_by_id(cls, obj_type, idx):
        # 优先使用 id->object 索引表
        if hasattr(cls, "index"):
            obj_map = cls.index.get(obj_type, {})
            if idx in obj_map:
                return obj_map[idx]

        def binary_search(target):
            arr = cls.all[obj_type]
            left, right = 0, len(arr) - 1

            while left <= right:
                # 跳过可能为 None 的左右端点
                while left <= right and arr[left] is None:
                    left += 1
                while left <= right and arr[right] is None:
                    right -= 1
                if left > right:
                    break

                mid = (left + right) // 2
                # 向最近的非 None 处移动
                orig_mid = mid
                while mid >= left and arr[mid] is None:
                    mid -= 1
                if mid < left:
                    mid = orig_mid + 1
                    while mid <= right and arr[mid] is None:
                        mid += 1
                    if mid > right:
                        break

                if arr[mid] is None:
                    break  # 没有可用元素，退出

                if arr[mid].id == target:
                    return mid  # 找到了，返回索引
                elif arr[mid].id < target:
                    left = mid + 1  # 目标在右半部分
                else:
                    right = mid - 1  # 目标在左半部分
           
        res_idx = binary_search(idx)
        if res_idx is not None:
            return cls.all[obj_type][res_idx]
        return None

    @classmethod
    def sort_by_id(cls):
        for obj_type in cls.all.keys():
            cls.all[obj_type].sort(key=lambda x: x.id if x is not None else float('inf'))
    
    @staticmethod
    def normalize_to_set(arg):
        """
        通用辅助方法：将输入参数 (None / str / list) 统一归一化为 set 或 None
        """
        if arg is None:
            return None
        if isinstance(arg, str):
            return {arg}
        return set(arg)
    
    @staticmethod
    def normalize_filter_args(obj_type, target_ids_dict=None):
        """
        统一处理 obj_type, aff_type 和 target_ids_dict 的标准化与交集逻辑
        """
        # 1. 处理 obj_type (调用通用逻辑)
        obj_type_set = Modality.normalize_to_set(obj_type)
        
        # 2. 处理 target_ids_dict 对 obj_type 的限制
        # 如果提供了 target_ids_dict，优先使用其中的 key 作为 obj_set 的基础
        if target_ids_dict is not None:
            target_obj_keys = set(target_ids_dict.keys())
            if obj_type_set is not None:
                obj_type_set = obj_type_set & target_obj_keys # 取交集
            else:
                obj_type_set = target_obj_keys
            
        return obj_type_set

    @staticmethod
    def _normalize_aff_mask_dict(aff_mask_dict):
        """归一化 aff_mask_dict：过滤 None、统一 key 为 str。PointCloud/Image 共用。"""
        if not aff_mask_dict:
            return {}
        normalized = {}
        for aff_type, mask_col in aff_mask_dict.items():
            if aff_type is None or mask_col is None:
                continue
            normalized[str(aff_type)] = np.asarray(mask_col)
        return normalized

    def get_aff_types(self) -> List[str]:
        """获取 affordance 类型列表。有 aff_mask_dict 的用其 keys；否则（如 Instruction）用 aff_type 单值。"""
        aff = getattr(self, 'aff_mask_dict', None)
        if aff:
            return list(aff.keys())
        aff_type = getattr(self, 'aff_type', None)
        return [aff_type] if aff_type else []

    def get_aff_index(self, aff_type: str):
        """返回 aff_type 在 aff_types 列表中的索引，不存在则返回 None。"""
        aff_types = self.get_aff_types()
        if not aff_types:
            return None
        try:
            return aff_types.index(aff_type)
        except ValueError:
            return None

    def get_aff_type_by_index(self, idx: int) -> Optional[str]:
        """根据索引返回 aff_type，越界返回 None。"""
        aff_types = self.get_aff_types()
        if idx is None or idx < 0 or idx >= len(aff_types):
            return None
        return aff_types[idx]

    def get_mask_by_aff(self, aff_type: str) -> Optional[np.ndarray]:
        """根据 aff_type 返回 mask。无 aff_mask_dict 的模态（如 Instruction）返回 None。"""
        if aff_type is None:
            return None
        return getattr(self, 'aff_mask_dict', {}).get(aff_type)

    def get_mask_by_index(self, idx: int):
        """根据索引返回 mask。"""
        aff_type = self.get_aff_type_by_index(idx)
        return self.get_mask_by_aff(aff_type)

    
class PointCloud(Modality):
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    index = defaultdict(dict)
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

    def __init__(self, points, obj_type, aff_mask_dict: Optional[Dict[str, np.ndarray]] = None, given_id:int=None):
        self.points = points
        self.obj_type = obj_type
       
        PointCloud.all[obj_type].append(self)
        PointCloud.count[obj_type]['ID'] += 1
        self.id = PointCloud.count[obj_type]['ID'] if given_id is None else given_id
        PointCloud.index[obj_type][self.id] = self
        self.aff_mask_dict = Modality._normalize_aff_mask_dict(aff_mask_dict)
        for l in self.aff_mask_dict.keys():
            PointCloud.count[obj_type][l] += 1

        self.is_sorted = False
        self._hash = None

    """  ---------------------------------------- 读写相关 ---------------------------------------------  """
    def save_to(self, filepath):
        """统一保存为csv格式，第一行是标签，数据前三列是xyz，后面所有的列分别表示不同aff标注"""
        header = ['x', 'y', 'z']

        # 拼接xyz和mask
        if self.aff_mask_dict:
            labels = list(self.aff_mask_dict.keys())
            mask_matrix = np.column_stack([self.aff_mask_dict[l] for l in labels])
            data = np.concatenate([self.points, mask_matrix], axis=1)
            header += labels
        else:
            data = self.points

        with open(filepath, 'w') as f:
            np.savetxt(f, data, delimiter=',', header=','.join(header))
   
    @classmethod
    def save_all(cls, dataset_root_path, keep_id: bool=False):
        """
        批量保存所有 PointCloud
        
        Args:
            dataset_root_path: 数据集根目录
            keep_id: 是否保持对象的 id，False 时按顺序重新分配 id
        """
        for k, v in cls.all.items():
            dir_path = os.path.join(dataset_root_path, k, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            if keep_id:
                # 使用对象自己的 id
                for e in v:
                    if e is not None:
                        e.save_to(os.path.join(dir_path, f'{e.obj_type}_{e.id}.csv'))
            else:
                # 按顺序重新分配 id
                save_id = None
                for e in v:
                    if e is not None:
                        if save_id is None: save_id = e.id  # 以第一个非 None的pc对象的id作为起始id
                        else: save_id += 1
                        e.save_to(os.path.join(dir_path, f'{e.obj_type}_{save_id}.csv')) # 保存时命名为 {obj_type}_{id}.csv
    
    @staticmethod
    def load_file(filepath,
            obj_type=None,
            aff_type=None,
            keep_id: bool=False,
        ) -> 'PointCloud':
        """
        Args:
            keep_id: 是否保持文件的id，默认False加载时重新分配id
        """
        obj_type = os.path.basename(os.path.dirname(filepath)) if obj_type is None else obj_type

        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            header = first_line.split(',') if first_line else []
            data = np.loadtxt(f, delimiter=',')

        if keep_id:
            file_name = os.path.splitext(os.path.basename(filepath))[0]
            given_id = int(file_name.split('_')[1])
        else:
            given_id = None

        if len(header) > 3:
            labels = header[3:]
            mask = data[:, 3:]

            # 根据 aff_type 过滤列（aff_type 为 None 时保留全部）
            if aff_type is not None:
                aff_set = Modality.normalize_to_set(aff_type)

                keep_indices = [i for i, l in enumerate(labels) if l in aff_set]
                if keep_indices:
                    mask = mask[:, keep_indices]
                    labels = [labels[i] for i in keep_indices]
                else:
                    # 如果没有匹配的列，则置空 mask / labels
                    mask = None
                    labels = None

            aff_mask_dict = {}
            if mask is not None and labels is not None:
                if mask.ndim == 1:
                    aff_mask_dict[str(labels[0])] = mask
                else:
                    for col, label in enumerate(labels):
                        aff_mask_dict[str(label)] = mask[:, col]
            pc_obj = PointCloud(
                points=data[:, :3],
                obj_type=obj_type,
                aff_mask_dict=aff_mask_dict,
                given_id=given_id,
            )
        else:
            pc_obj = PointCloud(points=data[:, :3],
                                obj_type=obj_type,
                                given_id=given_id)

        return pc_obj

    @classmethod
    def load_all(cls,
            dataset_root_path,
            keep_id=False,
            obj_type=None, 
            aff_type=None,
            target_ids_dict: dict = None
        ):
        """
        从统一格式的数据集中批量加载 PointCloud
        
        Args:
            dataset_root_path: 根目录，结构为 {obj_type}/PointCloud/{obj_type}_{id}.csv
            keep_id: 是否保持文件名中的 id，而不是重新分配
            obj_type: 需要加载的物体类型；None 时加载所有，可以是 str 或 list[str]
            aff_type: 需要加载的 affordance 类型；None 时加载所有，可以是 str 或 list[str]
            target_ids_dict: {obj_type: [id1, id2, ...]} 字典。
                             如果提供了此参数，只加载字典中存在的 obj_type 及其对应的 IDs。
        """
        # 确定要加载的文件
        obj_set = cls.normalize_filter_args(obj_type, target_ids_dict)
        all_objs = set([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
        if obj_set is not None:
            all_objs &= obj_set
        
        def iterator():
            for obj_type_name in tqdm(sorted(all_objs), desc='加载PointCloud'):
                dir_path = os.path.join(dataset_root_path, obj_type_name, 'PointCloud')
                if not os.path.exists(dir_path):
                    continue
                
                # 注意：target_ids_dict 的 key 必须与文件夹名完全一致
                if target_ids_dict:
                    files_id_to_load = set()
                    for _aff, target_ids in target_ids_dict.get(obj_type_name, {}).items():
                        for entry in target_ids:
                            files_id_to_load.add(entry)
                    files_to_load = [f"{obj_type_name}_{target_id}.csv" for target_id in sorted(files_id_to_load)]
                else:
                    files_to_load = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            
                for file in tqdm(files_to_load, leave=False, desc=f'PC-{obj_type_name}'):
                    file_path = os.path.join(dir_path, file)
                    if not os.path.isfile(file_path) or not os.path.exists(file_path):
                        warnings.warn(f"File not exist: {file_path}")
                        continue

                    try:
                        pc = cls.load_file(
                            file_path,
                            obj_type=obj_type_name,
                            aff_type=aff_type,
                            keep_id=keep_id,
                        )
                        yield pc
                    except Exception as e:
                        warnings.warn(f"Failed to load PC {file_path}: {e}")
            cls.sort_by_id()
        return iterator()
   
    @classmethod
    def load_and_save(cls, input_root, output_root, keep_id=False):
        id_counter = {}  # 用于记录每个 obj_type 的 id 计数
        for pc in cls.load_all(input_root, keep_id=keep_id):
            dir_path = os.path.join(output_root, pc.obj_type, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
           
            # 根据 keep_id 决定文件名
            if keep_id:
                file_id = pc.id
            else:
                # 按顺序重新分配 id
                if pc.obj_type not in id_counter:
                    id_counter[pc.obj_type] = 0
                id_counter[pc.obj_type] += 1
                file_id = id_counter[pc.obj_type]
           
            pc.save_to(os.path.join(dir_path, f'{pc.obj_type}_{file_id}.csv'))
            pc.free_memory()

    """  ---------------------------------------- 工具 ---------------------------------------------  """
    def __eq__(self, other):
        if not isinstance(other, PointCloud):
            raise TypeError(f"{type(other)}不是点云，不可比较")

        if self.points.ndim != other.points.ndim or self.points.shape != other.points.shape:
            return False

        return hash(self) == hash(other)

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
        import open3d as o3d

        if self.aff_mask_dict:
            for label, mask_col in self.aff_mask_dict.items():
                if selected_labels is not None and label not in selected_labels: continue
                # 未标注区域用浅灰，避免与背景白色混在一起
                base_gray = 0.95
                colors = np.full((self.points.shape[0], 3), base_gray, dtype=np.float32)
                mask_mask = mask_col > 0
                if mask_mask.any():
                    # 归一化 + 轻微对比度增强，让红色渐变更清晰
                    mask_vals = mask_col[mask_mask].astype(np.float32, copy=False)
                    max_val = float(mask_vals.max())
                    if max_val > 1.0:
                        mask_vals = mask_vals / max_val
                    mask_vals = np.clip(mask_vals, 0.0, 1.0)
                    mask_vals = np.sqrt(mask_vals)
                    red = base_gray + (1.0 - base_gray) * mask_vals
                    gb = base_gray * (1.0 - mask_vals)
                    colors[mask_mask, 0] = red
                    colors[mask_mask, 1] = gb
                    colors[mask_mask, 2] = gb
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
            if self.aff_mask_dict is not None:
                for label, mask_col in list(self.aff_mask_dict.items()):
                    if mask_col is not None:
                        self.aff_mask_dict[label] = mask_col[sort_idx]

            self.is_sorted = True
            return True

    def free_memory(self):
        """释放自身占用的内存（不更改计数，用于不重复加载的情况）"""
        # 删除内部数组
        self.points = None
        self.aff_mask_dict = None

    def __del__(self):
        # 更新count
        PointCloud.count[self.obj_type]["ID"] -= 1
        label_source = list(self.aff_mask_dict.keys()) if getattr(self, "aff_mask_dict", None) else []
        if label_source:
            for l in label_source:
                PointCloud.count[self.obj_type][l] -= 1

        self.free_memory()

        # 删除self的记录
        # PointCloud.all[self.obj_type][self.id - 1] = None
        PointCloud.index[self.obj_type].pop(self.id, None)

    def _merge(self, other):
        """合并两个点云标注并更新label、计数（默认点云hash相等）"""
        if isinstance(other, PointCloud):
            if self.aff_mask_dict is None:
                self.aff_mask_dict = {}
            for label, mask_col in other.aff_mask_dict.items():
                if label not in self.aff_mask_dict:
                    self.aff_mask_dict[label] = mask_col
            del other
        return self

    @classmethod
    def deduplicate(cls):
        """根据hash值去重合并数据"""
        for obj_type, ls in tqdm(cls.all.items(), desc='去重PointCloud'):
            loaded = dict()
            for pc in tqdm(ls, leave=False):
                loaded[pc] = pc._merge(loaded.get(pc, None))

class Image(Modality):
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    index = defaultdict(dict)
   
    def __init__(self, img:np.ndarray,
            obj_type,
            aff_mask_dict: Optional[Dict[str, np.ndarray]] = None,
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

        self.aff_mask_dict = Modality._normalize_aff_mask_dict(aff_mask_dict)
        for l in self.aff_mask_dict.keys():
            Image.count[obj_type][l] += 1

        self.dtype = 'No-mask' if not self.aff_mask_dict else 'Segmented'
        self.obj_type = obj_type
        Image.all[obj_type].append(self)

        Image.count[self.obj_type]['ID'] += 1
        self.id = Image.count[self.obj_type]['ID'] if given_id is None else given_id
        Image.index[self.obj_type][self.id] = self

        self.obj_mask = obj_mask
        self.visible_mask = visible_mask

    @classmethod
    def save_all(cls, dataset_root_path, keep_id: bool=False):
        """
        批量保存所有 Image
        
        Args:
            dataset_root_path: 数据集根目录
            keep_id: 是否保持对象的 id，False 时按顺序重新分配 id
        """
        for k, v in cls.all.items():
            dir_path = os.path.join(dataset_root_path, k, 'Image')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            if keep_id:
                # 使用对象自己的 id
                for e in v:
                    if e is not None:
                        e.save_to(dir_path)
            else:
                save_id = None
                for e in v:
                    if e is not None:
                        if save_id is None: save_id = e.id  # 以第一个非 None的pc对象的id作为起始id
                        else: save_id += 1
                        e.save_to(dir_path, file_id=save_id)

    def save_to(self, dir_path, file_id: int=None):
        """
        保存图片和mask到指定目录
        
        Args:
            dir_path: 保存目录
            file_id: 保存时使用的 id；为 None 时使用对象自身的 id
        """
        # dir_path 应该是目录，生成2~4个文件：原图 和 aff_mask，并在obj_mask目录下并排保存图片的物体mask和可见部分mask（如有）
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 确定使用的 id
        save_id = self.id if file_id is None else file_id

        # 保存img原图
        rgb_dir = os.path.join(dir_path, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        img_path = os.path.join(rgb_dir, f'{self.obj_type}_{save_id}.png')
        cv2.imwrite(img_path, self.img)

        # 保存aff_mask
        if self.aff_mask_dict:
            # 每个mask单独保存
            for label, mask in self.aff_mask_dict.items():
                mask_label_dir = os.path.join(dir_path, 'mask', label)
                os.makedirs(mask_label_dir, exist_ok=True)
                single_mask_path = os.path.join(mask_label_dir, f'{self.obj_type}_{save_id}_{label}.png')
                cv2.imwrite(single_mask_path, mask)

        # 保存obj_mask和visib_mask（如有）
        obj_mask_dir = os.path.join(dir_path, 'obj_mask')
        if self.obj_mask is not None and self.obj_mask.size != 0:
            os.makedirs(obj_mask_dir, exist_ok=True)
            obj_mask_path = os.path.join(obj_mask_dir, f'{self.obj_type}_{save_id}_obj_mask.png')
            cv2.imwrite(obj_mask_path, self.obj_mask)
        if self.visible_mask is not None and self.visible_mask.size != 0:
            os.makedirs(obj_mask_dir, exist_ok=True)
            vis_mask_path = os.path.join(obj_mask_dir, f'{self.obj_type}_{save_id}_visible_mask.png')
            cv2.imwrite(vis_mask_path, self.visible_mask)

    @staticmethod
    def _blend_bright_dark(
        img: np.ndarray,
        mask: np.ndarray,
        bright: float = 1.4,
        dark: float = 0.3,
    ) -> np.ndarray:
        """在单张图上混合显示 mask：提亮 mask 区域、压暗其余区域。

        Args:
            img:    uint8 BGR 原图 (H, W, 3)
            mask:   单通道 mask，值域 [0, 255] uint8 或 [0, 1] float
            bright: mask 区域的亮度增益（>1 提亮）
            dark:   非 mask 区域的亮度系数（<1 压暗）
        """
        alpha = mask.astype(np.float32)
        if alpha.max() > 1.0:
            alpha /= 255.0
        weight = dark + (bright - dark) * alpha
        result = img.astype(np.float32) * weight[..., np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_letterbox(
        img: np.ndarray,
        max_width: int = 1920,
        max_height: int = 1080,
        fill_value: int = 0,
    ) -> np.ndarray:
        """将图像等比例缩放并填充黑边至固定尺寸，用于显示。

        Args:
            img: BGR 图像 (H, W, 3)，uint8
            max_width: 目标宽度（默认 1920）
            max_height: 目标高度（默认 1080）
            fill_value: 填充像素值（默认 0，黑色）
        Returns:
            尺寸为 (max_height, max_width, 3) 的图像
        """
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img
        scale = min(max_width / w, max_height / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((max_height, max_width, 3), fill_value, dtype=img.dtype)
        y0 = (max_height - new_h) // 2
        x0 = (max_width - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas

    def show(self, selected_labels: list = None, wait_key=0,
             bright: float = 1.4, dark: float = 0.3,
             display_size: Optional[Tuple[int, int]] = (1080, 720)):
        """
        在单张图上渲染 RGB 原图与 mask：提亮 mask 区域，压暗其余区域。
        显示时固定为 display_size，等比例缩放并黑边填充。

        Args:
            selected_labels: 只显示指定的标签，如果为 None 则显示所有标签
            wait_key: 等待按键的时间，同 cv2.waitKey()；0 表示等待按任意键
            bright: mask 区域亮度增益（默认 1.4）
            dark: 非 mask 区域亮度系数（默认 0.3）
            display_size: 显示窗口固定尺寸 (宽, 高)，等比例缩放并黑边填充，None 表示不缩放
        """
        # 确保图片是uint8格式
        img_display = self.img.copy()
        if img_display.dtype != np.uint8:
            img_display = np.clip(img_display, 0, 255).astype(np.uint8)

        if display_size is not None:
            max_width, max_height = display_size[0], display_size[1]
        else:
            max_width = max_height = None

        # affordance masks
        if self.aff_mask_dict:
            for label, mask in self.aff_mask_dict.items():
                if selected_labels is not None and label not in selected_labels:
                    continue
                mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).clip(0, 255).astype(np.uint8)
                blended = self._blend_bright_dark(img_display, mask_u8, bright, dark)
                if max_width is not None and max_height is not None:
                    blended = self._resize_letterbox(blended, max_width, max_height)
                cv2.imshow(f'{self.obj_type}_{self.id}_{label}', blended)

        # # obj_mask
        # if self.obj_mask is not None and self.obj_mask.size != 0:
        #     m = self.obj_mask
        #     mask_u8 = m if m.dtype == np.uint8 else (m * 255).clip(0, 255).astype(np.uint8)
        #     blended = self._blend_bright_dark(img_display, mask_u8, bright, dark)
        #     cv2.imshow(f'{self.obj_type}_{self.id}_obj_mask', blended)

        # # visible_mask
        # if self.visible_mask is not None and self.visible_mask.size != 0:
        #     m = self.visible_mask
        #     mask_u8 = m if m.dtype == np.uint8 else (m * 255).clip(0, 255).astype(np.uint8)
        #     blended = self._blend_bright_dark(img_display, mask_u8, bright, dark)
        #     cv2.imshow(f'{self.obj_type}_{self.id}_visible_mask', blended)

        if wait_key == 0:
            print(f"显示图片: {self.obj_type}_{self.id} - 按任意键关闭所有窗口")
        cv2.waitKey(wait_key)

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
        resized_aff_mask_dict = {}
        if self.aff_mask_dict:
            for label, mask in self.aff_mask_dict.items():
                if mask is not None and mask.size > 0:
                    resized_mask = cv2.resize(mask.copy(), size, interpolation=mask_interpolation)
                    resized_aff_mask_dict[label] = resized_mask
                else:
                    resized_aff_mask_dict[label] = None
       
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
            aff_mask_dict=resized_aff_mask_dict if resized_aff_mask_dict else None,
            obj_mask=resized_obj_mask,
            visible_mask=resized_visible_mask,
            given_id=self.id  # 保持相同的id
        )
       
        # 由于使用了given_id，需要调整计数（因为__init__中已经增加了计数）
        # 将计数恢复到原来的值（因为这是同一个对象的缩放版本，不应该增加计数）
        Image.count[self.obj_type]['ID'] -= 1
        # 如果labels相同，也需要减少label计数（因为__init__中已经增加了）
        if resized_image.aff_mask_dict:
            for label in resized_image.aff_mask_dict.keys():
                if Image.count[resized_image.obj_type][label] > 0:
                    Image.count[resized_image.obj_type][label] -= 1
       
        return resized_image

    @classmethod
    def load_file(cls,
            filepath,
            obj_type=None,
            aff_type=None,
            keep_id: bool=False,
        ) -> 'Image':
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
       
        if obj_type is None:
            obj_type = inferred_obj_type
       
        # 加载RGB图片
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
       
        # 加载affordance masks
        aff_mask_dict = {}
        mask_dir = os.path.join(dir_path, 'mask')
        if os.path.exists(mask_dir):
            aff_set = Modality.normalize_to_set(aff_type)

            # 遍历mask目录下的所有label子目录
            for aff in os.listdir(mask_dir):
                # aff 过滤：如果指定了 aff_type，则只加载在列表中的子目录
                if aff_set is not None and aff not in aff_set:
                    continue

                label_path = os.path.join(mask_dir, aff)
                if not os.path.isdir(label_path):
                    continue
               
                # 查找对应的mask文件: {obj_type}_{id}_{label}.png
                mask_filename = f'{obj_type}_{inferred_id}_{aff}.png'
                mask_filepath = os.path.join(label_path, mask_filename)
               
                if os.path.exists(mask_filepath):
                    mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        aff_mask_dict[aff] = mask
        # Discard: 暂不使用
        # # 加载obj_mask
        # obj_mask = None
        # obj_mask_dir = os.path.join(dir_path, 'obj_mask')
        # if os.path.exists(obj_mask_dir):
        #     obj_mask_path = os.path.join(obj_mask_dir, f'{obj_type}_{inferred_id}_obj_mask.png')
        #     if os.path.exists(obj_mask_path):
        #         obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
       
        # # 加载visible_mask
        # visible_mask = None
        # if os.path.exists(obj_mask_dir):
        #     vis_mask_path = os.path.join(obj_mask_dir, f'{obj_type}_{inferred_id}_visible_mask.png')
        #     if os.path.exists(vis_mask_path):
        #         visible_mask = cv2.imread(vis_mask_path, cv2.IMREAD_GRAYSCALE)
       
        # 解析文件名中的id，用于可选的 keep_id
        given_id = int(inferred_id) if keep_id else None

        img_obj = Image(
            img=img,
            obj_type=obj_type,
            aff_mask_dict=aff_mask_dict if aff_mask_dict else None,
            # obj_mask=obj_mask,
            # visible_mask=visible_mask,
            given_id=given_id,
        )

        return img_obj

    @classmethod
    def load_all(cls, dataset_root_path, keep_id=False,
                 obj_type=None, aff_type=None, target_ids_dict: dict = None):
        """
        从保存的数据集目录结构中加载所有图片
        
        Args:
            dataset_root_path: 数据集根目录，结构为 {obj_type}/rgb/{obj_type}_{id}.png
            keep_id: 是否保持文件名中的 id，而不是重新分配
            obj_type: 需要加载的物体类型；None 时加载所有，可以是 str 或 list[str]
            aff_type: 需要加载的 affordance 类型；None 时加载所有，可以是 str 或 list[str]
            target_ids_dict: {obj_type: [id1, id2, ...]} 字典。
                             如果提供了此参数，只加载字典中存在的 obj_type 及其对应的 IDs。
        """
        VALID_EXTS = ('.png', '.jpg', '.jpeg') # 定义支持的图片后缀

        # 确定要遍历的物体类型列表
        obj_set = cls.normalize_filter_args(obj_type, target_ids_dict)
        all_objs = set([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
        if obj_set is not None:
            all_objs &= obj_set

        def iterator():
            for obj_type_name in tqdm(sorted(all_objs), desc='加载Image'):
                rgb_dir = os.path.join(dataset_root_path, obj_type_name, 'Image', 'rgb')
                if not os.path.exists(rgb_dir):
                    continue
                
                files_to_load = set()
                # 构造指定id的rgb文件名
                if target_ids_dict is not None:
                    # 这里虽然多了一层循环，但在 ID 确定的情况下，比 os.listdir 依然快得多
                    for _aff, target_ids in target_ids_dict.get(obj_type_name, dict()).items():
                        for target_entry in sorted(target_ids):
                            if isinstance(target_entry, (list, tuple)):
                                if len(target_entry) == 0:
                                    continue
                                target_id = target_entry[0]
                            else:
                                target_id = target_entry
                            found = False
                            # 尝试构造文件名
                            for ext in VALID_EXTS:
                                filename = f"{obj_type_name}_{target_id}{ext}"
                                # 只有文件真实存在时，才加入待加载列表
                                if os.path.exists(os.path.join(rgb_dir, filename)):
                                    files_to_load.add(filename)
                                    found = True
                                    break
                            
                            if not found:
                                warnings.warn(f"Image file not found: {obj_type_name}_{target_id}.png")
                    files_to_load = sorted(files_to_load)
                # 加载整个目录
                else:
                    files_to_load = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(VALID_EXTS)])

                for rgb_file in tqdm(files_to_load, leave=False, desc=f'Img-{obj_type_name}'):
                    file_path = os.path.join(rgb_dir, rgb_file)
                    try:
                        img = cls.load_file(
                            file_path,
                            obj_type=obj_type_name,
                            aff_type=aff_type,
                            keep_id=keep_id,
                        )
                        yield img
                    except Exception as e:
                        warnings.warn(f"Failed to load Img {file_path}: {e}")
            
            cls.sort_by_id()
        return iterator()

    @classmethod
    def load_and_save(cls, input_root, output_root, keep_id=False):
        id_counter = {}  # 用于记录每个 obj_type 的 id 计数
        for img in cls.load_all(input_root, keep_id=keep_id):
            dir_path = os.path.join(output_root, img.obj_type, 'Image')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # 根据 keep_id 决定文件名
            if keep_id:
                file_id = img.id
            else:
                if img.obj_type not in id_counter:
                    id_counter[img.obj_type] = 0
                id_counter[img.obj_type] += 1
                file_id = id_counter[img.obj_type]

            img.save_to(dir_path, file_id=file_id)
            img.free_memory()

    def free_memory(self):
        self.img=None
        self.aff_mask_dict = None
        self.obj_mask = None
        self.visible_mask = None

    def __del__(self):
        # 更新count
        Image.count[self.obj_type]["ID"] -= 1
        label_source = list(self.aff_mask_dict.keys()) if getattr(self, "aff_mask_dict", None) else []
        if label_source:
            for l in label_source:
                Image.count[self.obj_type][l] -= 1

        self.free_memory()

        # 删除self的记录
        # Image.all[self.obj_type][self.id - 1] = None
        Image.index[self.obj_type].pop(self.id, None)

class Instruction(Modality):
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    index = defaultdict(dict)

    def __init__(self, ins, obj_type:str=None, aff_type:str=None, given_id:int=None):
        self.ins = ins
        self.obj_type = obj_type

        self.aff_type = aff_type
        if aff_type is not None:
            Instruction.count[obj_type][aff_type] += 1
        
        # Note: 这里我们假设如果是指定ID加载，外部逻辑会保证ID的唯一性和正确性
        Instruction.count[self.obj_type]['ID'] += 1  # Note: Ins的ID并不是最大的id，仅表示计数
        self.id = Instruction.count[self.obj_type]['ID'] if given_id is None else given_id  # Ins的id和图片的id一一对应

        if self.obj_type is not None:
            Instruction.all[self.obj_type].append(self)
            Instruction.index[self.obj_type][self.id] = self
        
    @classmethod
    def load_file(cls, file_path, aff_type=None, keep_id=True, target_ids: list[int] = None):
        """
        Args:
            target_ids: 指定需要加载的ID列表 (List[int])。如果为 None，则加载所有。
        """
        aff_set = Modality.normalize_to_set(aff_type)
        
        # 将 target_ids 转为 set 以优化查找速度
        target_ids_set = set(target_ids) if target_ids is not None else None

        # 加载csv文件，包含header
        instructions = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:  # 读文件的行很快，不用tqdm
                ins = row.get('ins')
                obj = row.get('obj_type')
                aff = row.get('aff_type')
                
                # 1. Affordance 过滤
                if aff_set is not None and aff not in aff_set: continue

                # 获取当前行ID
                row_id = int(row.get('id'))

                # 2. ID 过滤
                if target_ids_set is not None and row_id not in target_ids_set:
                    continue

                if not keep_id:
                    given_id = None
                else:
                    given_id = row_id

                instructions.append(cls(ins, obj_type=obj, aff_type=aff, given_id=given_id))
        return instructions

    @classmethod
    def load_all(cls, dataset_root_path, obj_type=None, aff_type=None, keep_id=True, target_ids_dict: dict = None):
        """
        Args:
            target_ids_dict: {obj_type: [id1, id2, ...]} 字典。
                             如果提供了此参数，只加载字典中存在的 obj_type 及其对应的 IDs。
        """
        """一次加载一个物体，不使用迭代器"""
        obj_set = cls.normalize_filter_args(obj_type, target_ids_dict)
        # 合并 obj_set 和目录下所有文件夹名，确保 tqdm 数量正确（处理全部可能出现的 obj_type）
        all_objs = set([d for d in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, d))])
        if obj_set is not None:
            all_objs &= obj_set
        
        for obj in tqdm(sorted(all_objs), desc='加载Instruction'):
            file_path = os.path.join(dataset_root_path, obj, 'Instruction.csv')
            if os.path.exists(file_path):
                # 获取当前物体需要加载的具体 ID 列表
                current_target_ids = None
                if target_ids_dict is not None:
                    files_id_to_load = set()
                    for _aff, target_ids in target_ids_dict.get(obj, {}).items():
                        files_id_to_load |= set(target_ids)  # 直接取并集
                    current_target_ids = sorted(files_id_to_load)
                
                cls.load_file(file_path, aff_type=aff_type, keep_id=keep_id, target_ids=current_target_ids)
        cls.sort_by_id()

    @classmethod
    def save_all(cls, dataset_root_dir, obj_type:list[str]=None, keep_id: bool=True):
        """
        Args:
            obj_type: 需要保存的指定的物品list['bag', 'knife',...]，默认保存所有
            keep_id: 是否保持对象的 id，False 时按顺序重新分配 id
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

                if keep_id:
                    # 使用对象自己的 id
                    for inst in Instruction.all[o]:
                        writer.writerow([
                            inst.ins,
                            inst.obj_type,
                            inst.aff_type,
                            inst.id,
                        ])
                else:
                    # 按顺序重新分配 id
                    id_counter = 0
                    for inst in Instruction.all[o]:
                        id_counter += 1
                        writer.writerow([
                            inst.ins,
                            inst.obj_type,
                            inst.aff_type,
                            id_counter,
                        ])


""" --------------------------------------- 聚合数据集 ----------------------------------- """
class JointDataSample:
    """单个数据样本，包含 Instruction、Image、PointCloud 三元组"""
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    start_id = 0  # 所有样本集共用一个id序列编号，不再使用 count[obj_typr]['ID']作为编号

    def __init__(
        self,
        ins: Instruction = None,
        img: Image = None,
        pc: PointCloud = None,
        aff_type: str = None,
        data_source_id: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            ins: Instruction 对象
            img: Image 对象
            pc: PointCloud 对象
            aff_type: 统一的 affordance 类型，当 ins 为 None 时必须传入
        """
        self.ins = ins
        self.img = img
        self.pc = pc
        self.data_source_id = data_source_id
        self.split = None
        

        obj_type, aff_type_val = None, None
        if ins is not None: 
            obj_type = ins.obj_type
            aff_type_val = ins.aff_type
        elif aff_type is not None:
            aff_type_val = aff_type
            if img is not None:
                obj_type = img.obj_type
                if img.get_aff_index(aff_type_val) is None:
                    raise ValueError(f"Image 不包含 aff_type: {aff_type_val}")
            elif pc is not None:
                obj_type = pc.obj_type
                if pc.get_aff_index(aff_type_val) is None:
                    raise ValueError(f"PointCloud 不包含 aff_type: {aff_type_val}")
            else:
                raise ValueError("aff_type 需配合 img 或 pc 使用以确定 obj_type")
        else:
            raise ValueError('需提供 ins 或 aff_type')

        self.aff_type = aff_type_val
        self.obj_type = obj_type

        # 如果没有 ins 参数输入，则使用点云的物体和类别作为 ins
        if ins is None and pc is not None:
            # 生成默认的 instruction 文本
            default_ins_templates = [
                f"Please identify the {aff_type_val} affordance region of the {obj_type}.",
                f"Find the area of the {obj_type} that is related to {aff_type_val} functionality.",
                f"Which part of the {obj_type} provides the {aff_type_val} affordance?",
                f"Locate the {aff_type_val} region on the {obj_type}.",
                f"For this {obj_type}, show the {aff_type_val} functionality region.",
            ]
            default_ins_text = random.choice(default_ins_templates)

            # 创建一个临时的 Instruction 对象
            self.ins = Instruction(
                ins=default_ins_text,
                obj_type=obj_type,
                aff_type=aff_type_val,
            )

        JointDataSample.start_id += 1
        self.id = JointDataSample.start_id

        JointDataSample.all[self.obj_type].append(self)
        JointDataSample.count[self.obj_type][self.aff_type] += 1
        # 模态可见状态：True 表示该模态可见，False 表示不可见（或无数据）
        self.is_available = {
            'ins': self.ins is not None,
            'img': self.img is not None,
            'pc': self.pc is not None,
        }
        if self.data_source_id is None:
            self.data_source_id = {
                'ins_id': self.ins.id if self.ins is not None else None,
                'img_id': self.img.id if self.img is not None else None,
                'pc_id': self.pc.id if self.pc is not None else None,
                'aff_type': self.aff_type,
            }
    
    def apply_mask(self, mask_prob=(0, 0.02, 0.003)) -> 'JointDataSample':
        """对指定模态应用掩码
        Args:
            mask_prob: ['ins', 'img', 'pc'] 每个模态被 mask 的概率（0.0-1.0）
        """
        all_modalities = ['ins', 'img', 'pc']
        
        # 先随机决定每个模态是否被 mask
        candidates_to_mask = []
        for i, mod in enumerate(all_modalities):
            if mask_prob[i] > 0 and random.random() < mask_prob[i]:
                candidates_to_mask.append(mod)
        
        for mod in candidates_to_mask:
            self.is_available[mod] = False
        
        return self
    
    def get_data(self) -> Dict[str, Any]:
        """
        获取数据字典，被 mask 的模态返回 None
        """
        return {
            'ins': self.ins.ins if self.is_available['ins'] else None,
            'img': self.img.img if self.is_available['img'] else None,
            'pc': self.pc.points if self.is_available['pc'] else None,
            'img_gt': self.img.get_mask_by_aff(self.aff_type) if self.is_available['img'] else None,
            'pc_gt': self.pc.get_mask_by_aff(self.aff_type) if self.is_available['pc'] else None,
            'data_source_id': self.data_source_id,
        }


class SplitManager:
    """数据集分割管理器。从已加载的 Instruction/Image/PointCloud 构建 train/val/test，保存为 metadata.json、train.json、val.json、test.json。
    可选：将分割后的源文件复制到指定目录。"""

    def __init__(self, dataset_root: str):
        self.dataset_root = os.path.abspath(dataset_root)

    def split(
        self,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None,
        sample_rate: float = None,
        max_sample_per_group: int = None,
        min_sample_per_group: int = 10,
        random_seed: int = 114514,
        balance_data: bool = False,
        copy_to: Optional[str] = None,
        obj_type: Optional[List[str]] = None,
        aff_type: Optional[List[str]] = None,
        keep_id: bool = True,
    ) -> Tuple['JointDataset', 'JointDataset', 'JointDataset']:
        """
        从已加载的 Instruction/Image/PointCloud 构建 train/val/test 并保存 JSON。

        Args:
            train_ratio, val_ratio, test_ratio: 分割比例
            sample_rate: 采样率 (0~1)，None 表示不采样
            max_sample_per_group: 每组最大采样数
            min_sample_per_group: 每组最小保留数
            random_seed: 随机种子
            balance_data: 是否平衡数据
            copy_to: 若指定，将分割后的源文件复制到该目录，并保存 split_file 至该目录
            obj_type, aff_type, keep_id: 传递给返回的 JointDataset

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        random.seed(random_seed)
        ratios = [train_ratio, val_ratio, test_ratio]
        if any(r is not None for r in ratios):
            none_count = sum(1 for r in ratios if r is None)
            assert none_count < 2, "只允许一个比例为 None"
            if none_count == 1:
                idx_none = ratios.index(None)
                rest = 1.0 - sum(r for r in ratios if r is not None)
                assert rest > 0, "分割比例不合理，剩余比例需大于0"
                if idx_none == 0:
                    train_ratio = rest
                elif idx_none == 1:
                    val_ratio = rest
                else:
                    test_ratio = rest
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
                f"比例之和必须为 1.0，当前为 {train_ratio + val_ratio + test_ratio}"

        text_image_pairs = defaultdict(lambda: defaultdict(list))
        for obj_type_name in tqdm(Instruction.all.keys(), desc="构建图文对"):
            for inst in Instruction.all[obj_type_name]:
                matched_image = Image.get_by_id(obj_type_name, inst.id)
                if matched_image and inst.aff_type in matched_image.aff_mask_dict:
                    pair = (inst.id, matched_image.id)
                    text_image_pairs[obj_type_name][inst.aff_type].append(pair)

        pc_groups = defaultdict(lambda: defaultdict(list))
        for obj_type_name, pcs in PointCloud.all.items():
            for pc in pcs:
                if pc is None or not pc.aff_mask_dict:
                    continue
                for idx, label in enumerate(pc.get_aff_types()):
                    pc_groups[obj_type_name][label].append(((pc.id, idx),))

        train_ids = create_info_dict()
        val_ids = create_info_dict()
        test_ids = create_info_dict()

        def _apply_sampling(unique_ids: list, n_total: int, obj_type_name: str, aff_type_name: str) -> list:
            if sample_rate is None:
                return unique_ids
            group_seed = hash((obj_type_name, aff_type_name, random_seed)) % (2 ** 32)
            rng = random.Random(group_seed)
            n_target = max(min_sample_per_group, int(round(n_total * sample_rate)))
            if max_sample_per_group is not None:
                n_target = min(n_target, max_sample_per_group)
            n_target = min(n_target, n_total)
            return rng.sample(unique_ids, n_target)

        def _split_group(groups, inner):
            nonlocal train_ids, val_ids, test_ids
            for obj_type_name in groups.keys():
                for aff_type_name, inner_ids in groups[obj_type_name].items():
                    if not inner_ids:
                        continue
                    unique_ids = list(set(inner_ids))
                    random.shuffle(unique_ids)
                    n_total = len(unique_ids)
                    unique_ids = _apply_sampling(unique_ids, n_total, obj_type_name, aff_type_name)
                    n_total = len(unique_ids)
                    assert test_ratio > 0 and val_ratio > 0, "test_ratio 和 val_ratio 不能为0"
                    if n_total <= 20:
                        warnings.warn(f"{obj_type_name}-{aff_type_name} 样本数量过少: {n_total}，只训练不评估")
                        # 跳过该数据并删除ids记录
                        del groups[obj_type_name][aff_type_name]
                        continue
                    else:
                        n_test = max(5, int(round(n_total * test_ratio)))
                        n_val = max(5, int(round(n_total * val_ratio)))
                    if n_test + n_val >= n_total:
                        overflow = n_test + n_val - (n_total - 10)
                        while overflow > 0 and (n_val > 0 or n_test > 0):
                            if n_val >= n_test and n_val > 0:
                                n_val -= 1
                            elif n_test > 0:
                                n_test -= 1
                            overflow -= 1
                    test_list = unique_ids[:n_test]
                    val_list = unique_ids[n_test:n_test + n_val]
                    train_list = unique_ids[n_test + n_val:]
                    for idx, modality in enumerate(inner):
                        train_ids[modality][obj_type_name][aff_type_name] = [e[idx] for e in train_list]
                        val_ids[modality][obj_type_name][aff_type_name] = [e[idx] for e in val_list]
                        test_ids[modality][obj_type_name][aff_type_name] = [e[idx] for e in test_list]

        _split_group(text_image_pairs, ('ins', 'img'))
        _split_group(pc_groups, ('pc',))

        train_dataset = JointDataset(
            dataset_root=self.dataset_root,
            sample_ids=train_ids,
            obj_type=obj_type,
            aff_type=aff_type,
            keep_id=keep_id,
            balance_data=balance_data,
        )
        val_dataset = JointDataset(
            dataset_root=self.dataset_root,
            sample_ids=val_ids,
            obj_type=obj_type,
            aff_type=aff_type,
            keep_id=keep_id,
            balance_data=balance_data,
        )
        test_dataset = JointDataset(
            dataset_root=self.dataset_root,
            sample_ids=test_ids,
            obj_type=obj_type,
            aff_type=aff_type,
            keep_id=keep_id,
            balance_data=balance_data,
        )

        def _collect_obj_aff_counts(split_ids):
            split_counts = {}
            for modality in ("Instruction", "Image", "PointCloud"):
                modality_counts = {"total": 0}
                modality_ids = split_ids.get(modality, {})
                for obj_type_name, aff_map in modality_ids.items():
                    obj_counts = {"total": 0}
                    for aff_type_name, entries in aff_map.items():
                        obj_counts[aff_type_name] = len(entries)
                        obj_counts["total"] += len(entries)
                    modality_counts[obj_type_name] = obj_counts
                    modality_counts["total"] += obj_counts["total"]
                split_counts[modality] = modality_counts
            return split_counts

        metadata = {
            'total_sample': len(train_dataset) + len(val_dataset) + len(test_dataset),
            'train_sample': len(train_dataset),
            'val_sample': len(val_dataset),
            'test_sample': len(test_dataset),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed,
            'obj_aff_count_by_split': {
                'train': _collect_obj_aff_counts(train_ids),
                'val': _collect_obj_aff_counts(val_ids),
                'test': _collect_obj_aff_counts(test_ids),
            },
        }
        if sample_rate is not None:
            metadata['sample_rate'] = sample_rate
            metadata['max_sample_per_group'] = max_sample_per_group
            metadata['min_sample_per_group'] = min_sample_per_group
        split_data = {
            'metadata': metadata,
            'train': {k: train_ids[k] for k in ('Instruction', 'Image', 'PointCloud')},
            'val': {k: val_ids[k] for k in ('Instruction', 'Image', 'PointCloud')},
            'test': {k: test_ids[k] for k in ('Instruction', 'Image', 'PointCloud')}
        }

        for t in split_data.keys():
            if t == 'metadata':
                continue
            for m in split_data[t].keys():
                for obj_type_name in list(split_data[t][m].keys()):
                    keys_to_remove = [k for k in split_data[t][m][obj_type_name].keys()
                                    if not split_data[t][m][obj_type_name][k]]
                    for k in keys_to_remove:
                        del split_data[t][m][obj_type_name][k]

        def _normalize_pc_to_ids(entries):
            """PointCloud 与 Image 一致，只保存 id，不保存 index。"""
            return [e[0] if isinstance(e, (list, tuple)) and len(e) > 0 else e for e in entries]

        def _dump_split_part(part: dict, f) -> None:
            """将分割 JSON 写入文件，每个 aff 的 ids 列表保持在同一行。"""
            lines = ['{']
            mod_items = [(k, v) for k, v in part.items() if v]
            for mod_i, (mod_name, mod_data) in enumerate(mod_items):
                obj_items = [(k, {a: v for a, v in aff_map.items() if v}) for k, aff_map in mod_data.items()]
                obj_items = [(k, v) for k, v in obj_items if v]
                if not obj_items:
                    continue
                lines.append(f'  "{mod_name}": {{')
                for obj_i, (obj_name, obj_data) in enumerate(obj_items):
                    lines.append(f'    "{obj_name}": {{')
                    aff_items = [(a, lst) for a, lst in obj_data.items() if lst]
                    for aff_i, (aff_name, aff_list) in enumerate(aff_items):
                        ids_str = json.dumps(aff_list, ensure_ascii=False)
                        comma = ',' if aff_i < len(aff_items) - 1 else ''
                        lines.append(f'      "{aff_name}": {ids_str}{comma}')
                    obj_comma = ',' if obj_i < len(obj_items) - 1 else ''
                    lines.append(f'    }}{obj_comma}')
                mod_comma = ',' if mod_i < len(mod_items) - 1 else ''
                lines.append(f'  }}{mod_comma}')
            lines.append('}')
            f.write('\n'.join(lines))

        def save_split_json(save_dir: str):
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
                part = {}
                for k in ('Instruction', 'Image', 'PointCloud'):
                    mod_data = ids.get(k, {})
                    part[k] = {}
                    for obj, aff_map in mod_data.items():
                        part[k][obj] = {}
                        for aff, entries in aff_map.items():
                            if not entries:
                                continue
                            if k == 'PointCloud':
                                part[k][obj][aff] = _normalize_pc_to_ids(entries)
                            else:
                                part[k][obj][aff] = list(entries)
                with open(os.path.join(save_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
                    _dump_split_part(part, f)
            print(f"分割结果已保存至: {save_dir} (metadata.json, train.json, val.json, test.json)")

        save_split_json(self.dataset_root)

        if copy_to is not None:
            copy_to_abs = os.path.abspath(copy_to)
            os.makedirs(copy_to_abs, exist_ok=True)
            train_dataset.copy_to(os.path.join(copy_to_abs, 'train'))
            val_dataset.copy_to(os.path.join(copy_to_abs, 'val'))
            test_dataset.copy_to(os.path.join(copy_to_abs, 'test'))
            save_split_json(copy_to_abs)

        return train_dataset, val_dataset, test_dataset


class JointDataset:
    """聚合 Instruction、Image、PointCloud 三元组的数据集类。作为 train/val/test 子集使用时，通过 split_file 指定分割 JSON 文件名独立加载。"""

    def __init__(self,
                 dataset_root: str,
                 sample_ids=None,
                 obj_type: list[str] = None,
                 aff_type: list[str] = None,
                 keep_id: bool = True,
                 balance_data: bool = False,
                 split_file: Optional[str] = None):
        """
        初始化数据集。

        Args:
            dataset_root: 数据集根目录
            sample_ids: 可选，已有的样本 ID 结构；若为 None 且 split_file 指定，则从 split_file 加载
            obj_type: 需要加载的物体类型列表，None 时加载所有
            aff_type: 需要加载的 affordance 类型列表，None 时加载所有
            keep_id: 是否保持原有的 id
            balance_data: 是否平衡数据（不建议启用）
            split_file: 分割 JSON 文件名，如 'train.json'、'val.json'、'test.json'。指定时从该文件加载 sample_ids 并自动执行 load_all_data
        """
        self.dataset_root = os.path.abspath(dataset_root)
        self.obj_type = obj_type
        self.aff_type = aff_type
        self.keep_id = keep_id
        self.split_file = split_file

        if sample_ids is not None:
            self.sample_ids = sample_ids
        elif split_file is not None:
            split_path = os.path.join(self.dataset_root, split_file)
            if os.path.exists(split_path):
                self.sample_ids = JointDataset.load_ids_from_split_file(split_path)
            else:
                self.sample_ids = create_info_dict()
        else:
            self.sample_ids = create_info_dict()

        assert not balance_data, "balance_data 不建议启用，容易引起数据分布的改变从而影响模型的训练和评估"
        if balance_data: self.balance_data()
            
        self.samples = []

    
    @staticmethod
    def load_ids_from_split_file(split_json_path: str):
        """从单个分割文件（train.json/val.json/test.json）加载 sample_ids，文件内容为 {Instruction, Image, PointCloud} 结构。"""
        if not os.path.exists(split_json_path):
            raise FileNotFoundError(f"分割JSON文件不存在: {split_json_path}")
        with open(split_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JointDataset._normalize_split_ids(data)

    @staticmethod
    def _normalize_split_ids(sample_ids):
        def _to_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return value

        def _dedup_entries(entries):
            deduped = []
            seen = set()
            for entry in entries:
                key = tuple(entry) if isinstance(entry, (list, tuple)) else entry
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(entry)
            return deduped

        sample_ids = dict(sample_ids)
        sample_ids['ins'] = sample_ids.get('Instruction', sample_ids.get('ins', {}))
        sample_ids['img'] = sample_ids.get('Image', sample_ids.get('img', {}))
        sample_ids['pc'] = sample_ids.get('PointCloud', sample_ids.get('pc', {}))

        for modality in ("ins", "img", "pc"):
            for _, aff_map in sample_ids.get(modality, {}).items():
                for aff_name, entries in list(aff_map.items()):
                    aff_map[aff_name] = _dedup_entries(entries)

        for _, aff_map in sample_ids.get("img", {}).items():
            for aff_name, entries in list(aff_map.items()):
                normalized_img_ids = []
                seen_img = set()
                for entry in entries:
                    if isinstance(entry, (list, tuple)):
                        if len(entry) == 0:
                            continue
                        img_id = _to_int(entry[0])
                    else:
                        img_id = _to_int(entry)
                    key = str(img_id)
                    if key in seen_img:
                        continue
                    seen_img.add(key)
                    normalized_img_ids.append(img_id)
                aff_map[aff_name] = normalized_img_ids

        for _, aff_map in sample_ids.get("pc", {}).items():
            for aff_name, entries in list(aff_map.items()):
                normalized_pc_ids = []
                seen_pc = set()
                for entry in entries:
                    pc_id = entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else entry
                    pc_id = _to_int(pc_id)
                    key = str(pc_id)
                    if key in seen_pc:
                        continue
                    seen_pc.add(key)
                    normalized_pc_ids.append(pc_id)
                aff_map[aff_name] = normalized_pc_ids
        return sample_ids

    def load_all_data(self, filter_by_ids=None):
        """加载 Instruction、Image、PointCloud 数据"""
        import threading
        
        if filter_by_ids is None and self.split_file is not None:
            filter_by_ids=self.sample_ids
        
        def load_pc_wrapper():
            target_ids_dict = filter_by_ids['pc'] if filter_by_ids is not None else None
            for _ in PointCloud.load_all(self.dataset_root, obj_type=self.obj_type, aff_type=self.aff_type, keep_id=self.keep_id, target_ids_dict=target_ids_dict): pass

        def load_img_wrapper():
            target_ids_dict = filter_by_ids['img'] if filter_by_ids is not None else None
            for _ in Image.load_all(self.dataset_root, obj_type=self.obj_type, aff_type=self.aff_type, keep_id=self.keep_id, target_ids_dict=target_ids_dict): pass

        def load_ins_wrapper():
            target_ids_dict = filter_by_ids['ins'] if filter_by_ids is not None else None
            Instruction.load_all(self.dataset_root, obj_type=self.obj_type, aff_type=self.aff_type, keep_id=self.keep_id, target_ids_dict=target_ids_dict)

        # 创建线程
        t1 = threading.Thread(target=load_pc_wrapper)
        t2 = threading.Thread(target=load_img_wrapper)
        t3 = threading.Thread(target=load_ins_wrapper)

        # 启动线程
        t1.start()
        t2.start()
        t3.start()

        # 等待所有线程结束
        t1.join()
        t2.join()
        t3.join()

        self.samples = self.pair_samples()

        print("所有数据加载完成")
        return self

    def copy_to(self, output_dir: str):
        """将当前 sample_ids 对应的源文件复制到指定目录，保持原有目录结构。"""
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        def _collect_ids_per_obj(split_ids):
            ins_ids = defaultdict(set)
            img_ids = defaultdict(set)
            pc_ids = defaultdict(set)
            for modality, key in [('Instruction', 'ins'), ('Image', 'img'), ('PointCloud', 'pc')]:
                mod_data = split_ids.get(modality, split_ids.get(key, {}))
                for obj_type_name, aff_map in mod_data.items():
                    for aff_type_name, entries in aff_map.items():
                        for entry in entries:
                            if modality == 'PointCloud':
                                pc_id = entry[0] if isinstance(entry, (list, tuple)) else entry
                                pc_ids[obj_type_name].add(pc_id)
                            elif modality == 'Instruction':
                                ins_ids[obj_type_name].add(int(entry))
                            else:
                                img_id = entry[0] if isinstance(entry, (list, tuple)) else int(entry)
                                img_ids[obj_type_name].add(img_id)
            return ins_ids, img_ids, pc_ids

        all_ins, all_img, all_pc = _collect_ids_per_obj(self.sample_ids)

        for obj_type_name in tqdm(set(all_ins.keys()) | set(all_img.keys()) | set(all_pc.keys()), desc="复制源文件"):
            obj_dir = os.path.join(self.dataset_root, obj_type_name)
            out_obj_dir = os.path.join(output_dir, obj_type_name)
            if not os.path.isdir(obj_dir):
                continue

            if obj_type_name in all_ins:
                ins_csv = os.path.join(obj_dir, 'Instruction.csv')
                if os.path.exists(ins_csv):
                    ids_to_keep = all_ins[obj_type_name]
                    os.makedirs(out_obj_dir, exist_ok=True)
                    out_ins = os.path.join(out_obj_dir, 'Instruction.csv')
                    with open(ins_csv, 'r', newline='', encoding='utf-8') as f_in:
                        reader = csv.DictReader(f_in)
                        fieldnames = reader.fieldnames
                        rows = [r for r in reader if int(r.get('id', -1)) in ids_to_keep]
                    if rows:
                        with open(out_ins, 'w', newline='', encoding='utf-8') as f_out:
                            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

            if obj_type_name in all_img:
                img_dir = os.path.join(obj_dir, 'Image')
                out_img_dir = os.path.join(out_obj_dir, 'Image')
                if os.path.isdir(img_dir):
                    for img_id in all_img[obj_type_name]:
                        rgb_dir = os.path.join(img_dir, 'rgb')
                        if os.path.isdir(rgb_dir):
                            for ext in ('.png', '.jpg', '.jpeg'):
                                src = os.path.join(rgb_dir, f'{obj_type_name}_{img_id}{ext}')
                                if os.path.exists(src):
                                    dst_dir = os.path.join(out_img_dir, 'rgb')
                                    os.makedirs(dst_dir, exist_ok=True)
                                    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
                                    break
                        mask_dir = os.path.join(img_dir, 'mask')
                        if os.path.isdir(mask_dir):
                            for aff_dir in os.listdir(mask_dir):
                                aff_path = os.path.join(mask_dir, aff_dir)
                                if os.path.isdir(aff_path):
                                    for f in os.listdir(aff_path):
                                        if f.startswith(f'{obj_type_name}_{img_id}_'):
                                            src = os.path.join(aff_path, f)
                                            dst_dir = os.path.join(out_img_dir, 'mask', aff_dir)
                                            os.makedirs(dst_dir, exist_ok=True)
                                            shutil.copy2(src, os.path.join(dst_dir, f))

            if obj_type_name in all_pc:
                pc_dir = os.path.join(obj_dir, 'PointCloud')
                out_pc_dir = os.path.join(out_obj_dir, 'PointCloud')
                if os.path.isdir(pc_dir):
                    os.makedirs(out_pc_dir, exist_ok=True)
                    for pc_id in all_pc[obj_type_name]:
                        src = os.path.join(pc_dir, f'{obj_type_name}_{pc_id}.csv')
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(out_pc_dir, os.path.basename(src)))

        print(f"源文件已复制至: {output_dir}")

    def pair_samples(self):
        """
        将单个分割集（train/val/test）的索引聚合成三元组
        Returns:
            JointDataSample 列表
        """
        samples = []
        
        # 获取所有 obj_type（从三种模态中取并集）
        all_obj_types = set()
        for modality in ['ins', 'img', 'pc']:
            if modality in self.sample_ids:
                all_obj_types.update(self.sample_ids[modality].keys())
        
        for obj_type in all_obj_types:
            # 获取该 obj_type 下所有 aff_type（从三种模态中取并集）
            all_aff_types = set()
            for modality in ['ins', 'img', 'pc']:
                if modality in self.sample_ids and obj_type in self.sample_ids[modality]:
                    all_aff_types.update(self.sample_ids[modality][obj_type].keys())
            
            for aff_type in all_aff_types:
                # 获取各模态的索引列表
                ins_ids = self.sample_ids.get('ins', {}).get(obj_type, {}).get(aff_type, [])
                img_ids = self.sample_ids.get('img', {}).get(obj_type, {}).get(aff_type, [])
                pc_ids = self.sample_ids.get('pc', {}).get(obj_type, {}).get(aff_type, [])
                
                # 计算最大长度
                max_len = max(len(ins_ids), len(img_ids), len(pc_ids))
                
                if max_len == 0:
                    continue
                
                # 按位置配对，不足的用 None 补充
                for i in range(max_len):
                    # 获取 Instruction
                    ins = Instruction.get_by_id(obj_type, ins_ids[i]) if i < len(ins_ids) else None
                    
                    # 获取 Image（按 aff_type 检查是否包含该 affordance）
                    image = None
                    if i < len(img_ids):
                        img_entry = img_ids[i]
                        img_id = img_entry[0] if isinstance(img_entry, (list, tuple)) and len(img_entry) > 0 else img_entry
                        image = Image.get_by_id(obj_type, img_id)
                        if image is not None:
                            has_aff = image.get_aff_index(aff_type) is not None
                            if not has_aff and isinstance(img_entry, (list, tuple)) and len(img_entry) > 1:
                                fallback_idx = int(img_entry[1])
                                if 0 <= fallback_idx < len(image.get_aff_types()) and image.get_aff_type_by_index(fallback_idx) == aff_type:
                                    has_aff = True  # 旧格式兼容
                                else:
                                    warnings.warn(
                                        f"Image 旧格式索引与 aff_type 不一致，已跳过: "
                                        f"{obj_type}-{aff_type}, image_id={img_id}"
                                    )
                            if not has_aff:
                                image = None
                    
                    # 获取 PointCloud（按 aff_type 检查是否包含该 affordance）
                    pc = None
                    if i < len(pc_ids):
                        pc_entry = pc_ids[i]
                        pc_id = pc_entry[0] if isinstance(pc_entry, (list, tuple)) and len(pc_entry) > 0 else pc_entry
                        pc = PointCloud.get_by_id(obj_type, pc_id)
                        if pc is not None:
                            has_aff = pc.get_aff_index(aff_type) is not None
                            if not has_aff and isinstance(pc_entry, (list, tuple)) and len(pc_entry) > 1:
                                # 兼容旧格式 (pc_id, mask_idx)：验证索引对应 aff_type
                                mask_idx = int(pc_entry[1])
                                if 0 <= mask_idx < len(pc.get_aff_types()) and pc.get_aff_type_by_index(mask_idx) == aff_type:
                                    has_aff = True
                                else:
                                    warnings.warn(
                                        f"PointCloud 旧格式索引与 aff_type 不一致，已跳过: "
                                        f"{obj_type}-{aff_type}, pc_id={pc_id}"
                                    )
                            if not has_aff:
                                pc = None
                    
                    # 至少有一个模态有数据才创建样本
                    if (ins or image or pc) is not None:
                        data_source_id = {
                            'ins_id': ins.id if ins is not None else None,
                            'img_id': image.id if image is not None else None,
                            'pc_id': pc.id if pc is not None else None,
                            'aff_type': aff_type,
                        }
                        sample = JointDataSample(
                            ins=ins,
                            img=image,
                            pc=pc,
                            aff_type=aff_type,
                            data_source_id=data_source_id,
                        )
                        samples.append(sample)
            
        return samples
 
    def random_mask(self, samples: List[JointDataSample], mask_prob=(0, 0.01, 0.003)) -> List[JointDataSample]:
        """
        对样本列表应用模态掩码
        
        Args:
            samples: 样本列表
        
        Returns:
            应用掩码后的样本列表（原样本会被修改）
        """
        for sample in samples:
            sample.apply_mask(mask_prob=mask_prob)
        
        return samples
    
    def __len__(self) -> int:
        """返回数据集总样本数"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        根据索引获取单个样本数据
        
        索引范围: [0, len(train) + len(val) + len(test))
        按顺序依次为: train -> val -> test
        
        Args:
            index: 样本索引
        
        Returns:
            包含样本数据的字典
        """
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self.samples)})")

        sample = self.samples[index]
        
        data = sample.get_data()
        data['index'] = index
        data['obj_type'] = sample.obj_type
        data['aff_type'] = sample.aff_type
        return data
    
    def get_batch(self, split: str, batch_size: int, shuffle: bool = True, apply_mask: bool = False, mask_prob: Tuple[float, float, float] = (0.01, 0.02, 0.003)):
        """
        获取批量数据的生成器
        
        Args:
            split: 数据集分割类型 ('train', 'val', 'test')
            batch_size: 批量大小
            shuffle: 是否打乱数据
            apply_mask: 是否应用随机掩码
            mask_prob: 各模态 (ins, img, pc) 被掩码的概率
        
        Yields:
            每次返回一个批次的数据列表
        """
        if split == 'train':
            samples = self.train_samples
        elif split == 'val':
            samples = self.val_samples
        elif split == 'test':
            samples = self.test_samples
        else:
            raise ValueError(f"未知的分割类型: {split}，应为 'train', 'val' 或 'test'")
        
        indices = list(range(len(samples)))
        if shuffle:
            random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_data = []
            
            for idx in batch_indices:
                sample = samples[idx]
                if apply_mask:
                    sample.apply_mask(mask_prob=mask_prob)
                
                data = sample.get_data()
                data['obj_type'] = sample.obj_type
                data['aff_type'] = sample.aff_type
                data['sample_id'] = sample.id
                batch_data.append(data)
            
            yield batch_data
      
    def balance_data(self):
        # 平衡数据集中的 pc 和图文数据
        def _balance_by_copy(data_list: list, target_count: int) -> list:
            """
            通过随机复制将数据列表扩展到目标数量
            """
            if len(data_list) >= target_count:
                return data_list[:target_count]
            
            result = data_list.copy()
            need_count = target_count - len(data_list)
            for _ in range(need_count):
                selected = random.choice(data_list)
                result.append(selected)
            
            return result

        print("平衡各分割集中的 pc 和图文数据...")  # Note: 是否必要？
        
        # 遍历所有 obj_type 和 aff_type 组合
        all_obj_types = set(self.sample_ids['ins'].keys()) | set(self.sample_ids['pc'].keys())
        
        for obj_type in tqdm(all_obj_types, desc='平衡数据'):
            # 获取该 obj_type 下所有的 aff_type
            ins_aff_types = set(self.sample_ids['ins'].get(obj_type, {}).keys())
            pc_aff_types = set(self.sample_ids['pc'].get(obj_type, {}).keys())
            all_aff_types = ins_aff_types & pc_aff_types  # NOTE: 只平衡公共aff的数量
            
            for aff_type in tqdm(all_aff_types, leave=False):
                # 获取当前分割中的 ins/img 和 pc 数量
                ins_list = self.sample_ids['ins'].get(obj_type, {}).get(aff_type, [])
                img_list = self.sample_ids['img'].get(obj_type, {}).get(aff_type, [])
                pc_list = self.sample_ids['pc'].get(obj_type, {}).get(aff_type, [])
                
                # 跳过空列表
                if not ins_list or not pc_list:
                    continue
                
                # 计算目标数量（取最大值）
                target_count = max(len(ins_list), len(pc_list))
                
                if target_count == 0:
                    continue
                
                # 平衡 ins 和 img（它们是配对的，需要同步扩展）
                if ins_list and len(ins_list) < target_count:
                    # 创建配对索引列表
                    pair_indices = list(range(len(ins_list)))
                    balanced_indices = _balance_by_copy(pair_indices, target_count)
                    
                    # 根据平衡后的索引重建列表
                    self.sample_ids['ins'][obj_type][aff_type] = [ins_list[i] for i in balanced_indices]
                    self.sample_ids['img'][obj_type][aff_type] = [img_list[i] for i in balanced_indices]
                
                # 平衡 pc
                if pc_list and len(pc_list) < target_count:
                    self.sample_ids['pc'][obj_type][aff_type] = _balance_by_copy(pc_list, target_count)

        print("数据平衡完成")  


def main():
    """示例用法；支持 -s/--show 作为渲染工具，自动识别 2D/3D 并依次渲染。"""
    import argparse
    parser = argparse.ArgumentParser(description="加载并分割联合数据集；或使用 -s/--show 渲染指定 2D/3D 数据")

    # 数据集分割参数
    parser.add_argument('-d', '--dataset-root', type=str, default=None,
                        help='数据集根目录（非 show 模式时必填）')
    parser.add_argument('-o', '--obj-type', type=str, nargs='+', default=None,
                        help='物体类型列表，默认加载所有')
    parser.add_argument('-a', '--aff-type', type=str, nargs='+', default=None,
                        help='Affordance 类型列表，默认加载所有')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=114514,
                        help='随机种子')
    parser.add_argument('--sample-rate', type=float, default=None,
                        help='采样率 (0~1)，None 表示不采样')
    parser.add_argument('--max-sample-per-group', type=int, default=None,
                        help='每组 (obj_type, aff_type) 最大采样数')
    parser.add_argument('--min-sample-per-group', type=int, default=10,
                        help='每组最小保留数')
    parser.add_argument('--no-balance', action='store_true', default=True,
                        help='禁用数据平衡（默认启用）')
    parser.add_argument('--save-split', action='store_true',
                        help='保存数据集分割结果为JSON文件')
    parser.add_argument('--keep-id', action='store_true', default=True,
                        help='保持原有的ID（默认启用）')
    parser.add_argument('--copy-to', type=str, default=None,
                        help='分割后将源文件复制到指定目录，并保存 split_file 至该目录')

    # 渲染参数，和上述参数不共存
    parser.add_argument('-s', '--show', type=str, nargs='+', default=None,
                        help='渲染模式：指定一个或多个文件/目录路径，自动识别 2D 图像或 3D 点云并依次调用 show()')
    args = parser.parse_args()

    # 渲染模式：-s/--show 时仅作渲染工具，不加载完整数据集
    if args.show:
        PC_EXTS = {'.csv'}
        IMG_EXTS = {'.png', '.jpg', '.jpeg'}

        def _collect_paths(paths):
            """将输入路径展开为文件列表；若为目录则递归收集支持的扩展名。"""
            collected = []
            for p in paths:
                resolved = resolve_path(p)
                if not os.path.exists(resolved):
                    warnings.warn(f"路径不存在，已跳过: {resolved}")
                    continue
                if os.path.isfile(resolved):
                    collected.append(resolved)
                else:
                    for root, _, files in os.walk(resolved):
                        for f in files:
                            ext = os.path.splitext(f)[1].lower()
                            if ext in PC_EXTS or ext in IMG_EXTS:
                                collected.append(os.path.join(root, f))
            return collected

        def _is_point_cloud_path(filepath):
            return os.path.splitext(filepath)[1].lower() in PC_EXTS

        file_paths = _collect_paths(args.show)
        if not file_paths:
            print("未找到可渲染的文件（支持 2D: .png/.jpg/.jpeg，3D 点云: .csv）")
            return

        for file_path in file_paths:
            try:
                if _is_point_cloud_path(file_path):
                    obj = PointCloud.load_file(file_path, keep_id=True)
                    obj.show()
                else:
                    obj = Image.load_file(file_path, keep_id=True)
                    obj.show()
            except Exception as e:
                warnings.warn(f"渲染失败 {file_path}: {e}")
        return

    elif args.dataset_root is not None:
        dataset_root = resolve_path(args.dataset_root)
        # 加载全量数据
        JointDataset(
            dataset_root=dataset_root,
            obj_type=args.obj_type,
            aff_type=args.aff_type,
            keep_id=args.keep_id,
            balance_data=not args.no_balance,
        ).load_all_data()

        # 使用 SplitManager 执行分割
        if args.save_split:
            SplitManager(dataset_root).split(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.seed,
                sample_rate=args.sample_rate,
                max_sample_per_group=args.max_sample_per_group,
                min_sample_per_group=args.min_sample_per_group,
                copy_to=args.copy_to,
            )
    else:
        parser.error("未使用 -s/--show 时需指定 -d/--dataset-root")


if __name__ == "__main__":
    main()
