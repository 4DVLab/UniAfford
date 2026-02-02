"""
聚合 Instruction、Image、PointCloud 三元组的数据集类
支持训练集、测试集、验证集的比例分割
"""
import os
import random
import warnings
from tqdm import tqdm
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
import cv2
import csv


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
        def binary_search(target):
            left, right = 0, len(cls.all[obj_type]) - 1
           
            while left <= right:
                mid = (left + right) // 2  # 取中间索引
               
                if cls.all[obj_type][mid].id == target:
                    return mid  # 找到了，返回索引
                elif cls.all[obj_type][mid].id < target:
                    left = mid + 1  # 目标在右半部分
                else:
                    right = mid - 1  # 目标在左半部分
           
        res_idx = binary_search(idx)
        if res_idx is not None:
            return cls.all[obj_type][res_idx]

    @classmethod
    def sort_by_id(cls):
        for obj_type in cls.all.keys():
            cls.all[obj_type].sort(key=lambda x: x.id)
    
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

    
class PointCloud(Modality):
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

    def __init__(self, points, obj_type, mask:list[np.ndarray]=None, labels:list=None, given_id:int=None):
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
        self._hash = None
       

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
            file_name = os.path.basename(filepath).strip('.csv')
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
            
            mask = [mask[:, col] for col in range(mask.shape[1])]
            pc_obj = PointCloud(points=data[:, :3],
                                mask=mask,
                                obj_type=obj_type,
                                labels=labels,
                                given_id=given_id)
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
                    target_ids = target_ids_dict.get(obj_type_name)
                    files_to_load = [f"{obj_type_name}_{target_id}.csv" for target_id in sorted(target_ids)]
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

    # TODO: 修改颜色条的显示，使得物体和背景（白色），以及标注区域（红色）之间的渐变清晰明了不会混杂
    def show(self, selected_labels:list=None):
        """
        Args:
            selected_labels: 只选择部分标签，否则都显示
        """
        import open3d as o3d

        if self.mask is not None and self.labels is not None and len(self.labels) > 0:
            for idx, label in enumerate(self.labels):
                if selected_labels is not None and label not in selected_labels: continue
                if len(self.mask) <= idx: raise ValueError(f'Error in {self.obj_type}-{self.id}: mask的列数{self.mask.shape}和label的维度 {label} 不同')

                mask_col = self.mask[idx]
               
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

    def free_memory(self):
        """释放自身占用的内存（不更改计数，用于不重复加载的情况）"""
        # 删除内部数组
        self.points = None
        self.mask = None
        self.labels = None

    def __del__(self):
        # 更新count
        PointCloud.count[self.obj_type]["ID"] -= 1
        for l in self.labels:
            PointCloud.count[self.obj_type][l] -= 1

        self.free_memory()

        # 删除self的记录
        PointCloud.all[self.obj_type][self.id - 1] = None

    def _merge(self, other):
        """合并两个点云标注并更新label、计数（默认点云hash相等）"""
        if isinstance(other, PointCloud):
            for i, l in enumerate(other.labels):
                if l not in self.labels:
                    self.labels.append(l)
                    self.mask = np.hstack((self.mask, other.mask[:, [i]]))
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
        if len(self.mask) != 0:
            # 每个mask单独保存
            for idx, mask in enumerate(self.mask):
                mask_label_dir = os.path.join(dir_path, 'mask', self.labels[idx])
                os.makedirs(mask_label_dir, exist_ok=True)
                single_mask_path = os.path.join(mask_label_dir, f'{self.obj_type}_{save_id}_{self.labels[idx]}.png')
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
        aff_mask = []
        labels = []
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
                        aff_mask.append(mask)
                        labels.append(aff)
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
            labels=labels if labels else None,
            aff_mask=aff_mask if aff_mask else None,
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
                
                files_to_load = []
                # 构造指定id的文件名
                if target_ids_dict is not None:
                    target_ids = target_ids_dict.get(obj_type_name, [])
                    # 这里虽然多了一层循环，但在 ID 确定的情况下，比 os.listdir 依然快得多
                    for target_id in sorted(target_ids):
                        found = False
                        # 尝试构造文件名
                        for ext in VALID_EXTS:
                            filename = f"{obj_type_name}_{target_id}{ext}"
                            # 只有文件真实存在时，才加入待加载列表
                            if os.path.exists(os.path.join(rgb_dir, filename)):
                                files_to_load.append(filename)
                                found = True
                                break
                        
                        if not found:
                            warnings.warn(f"Image file not found: {obj_type_name}_{target_id}")
                            
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
        self.mask = None
        self.obj_mask = None
        self.visible_mask = None
        self.labels = None

    def __del__(self):
        # 更新count
        Image.count[self.obj_type]["ID"] -= 1
        for l in self.labels:
            Image.count[self.obj_type][l] -= 1

        self.free_memory()

        # 删除self的记录
        Image.all[self.obj_type][self.id - 1] = None

class Instruction(Modality):
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
        
        # Note: 这里我们假设如果是指定ID加载，外部逻辑会保证ID的唯一性和正确性
        Instruction.count[self.obj_type]['ID'] += 1  # Note: Ins的ID并不是最大的id，仅表示计数
        self.id = Instruction.count[self.obj_type]['ID'] if given_id is None else given_id  # Ins的id和图片的id一一对应

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
            if obj_set is not None and obj not in obj_set: continue

            file_path = os.path.join(dataset_root_path, obj, 'Instruction.csv')
            if os.path.exists(file_path):
                # 获取当前物体需要加载的具体 ID 列表
                current_target_ids = None
                if target_ids_dict is not None:
                    current_target_ids = target_ids_dict.get(obj)
                
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

    def __init__(self, ins: Instruction=None, img: Image=None, pc: PointCloud=None, img_mask_idx=None, pc_mask_idx=None):
        """
        Args:
            ins: Instruction 对象
            imgimg: Image 对象
            pc: PointCloud 对象
        """
        self.ins = ins
        self.img = img
        self.img_mask_idx = img_mask_idx
        self.pc = pc
        self.pc_mask_idx = pc_mask_idx
        

        obj_type, aff_type = None, None
        if ins is not None: 
            obj_type = ins.obj_type
            aff_type = ins.aff_type
        elif img is not None:
            obj_type = img.obj_type
            aff_type = img.labels[img_mask_idx]
        elif pc is not None:
            obj_type = pc.obj_type
            aff_type = pc.labels[pc_mask_idx]
        else:
            raise ValueError('没有任何一条数据包含obj_type或aff_type信息')

        # 如果没有 ins 参数输入，则使用点云的物体和类别作为 ins
        if ins is None and pc is not None:
            # 生成默认的 instruction 文本
            default_ins_templates = [
                f"Please identify the {aff_type} affordance region of the {obj_type}.",
                f"Find the area of the {obj_type} that is related to {aff_type} functionality.",
                f"Which part of the {obj_type} provides the {aff_type} affordance?",
                f"Locate the {aff_type} region on the {obj_type}.",
                f"For this {obj_type}, show the {aff_type} functionality region.",
            ]
            default_ins_text = random.choice(default_ins_templates)

            # 创建一个临时的 Instruction 对象
            self.ins = Instruction(
                ins=default_ins_text,
                obj_type=obj_type,
                aff_type=aff_type,
            )

        self.aff_type = aff_type
        self.obj_type = obj_type
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
            'img_gt': self.img.mask[self.img_mask_idx] if self.is_available['img'] else None,
            'pc_gt': self.pc.mask[self.pc_mask_idx] if self.is_available['pc'] else None,
        }


class JointDataset:
    """聚合 Instruction、Image、PointCloud 三元组的数据集类"""
    
    def __init__(self, 
                 dataset_root: str,
                 obj_type: list[str] = None,
                 aff_type: list[str] = None,
                 train_ratio: float = None,
                 val_ratio: float = None,
                 test_ratio: float = None,
                 random_seed: int = 42,
                 keep_id: bool = True,
                 balance_data: bool = True,
                 mask_prob: float = (0, 0.01, 0.003)):
        """
        初始化数据集
        
        Args:
            dataset_root: 数据集根目录
            obj_type: 需要加载的物体类型列表，None 时加载所有
            aff_type: 需要加载的 affordance 类型列表，None 时加载所有
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            keep_id: 是否保持原有的 id
            balance_data: 是否平衡数据（当点云/图文对数量不一致时，复制较少的数据）
            mask_prob: 获取数据时每个模态被 mask 的概率（0.0-1.0）
        """
        
        # 设置随机种子
        random.seed(random_seed)

        ratios = [train_ratio, val_ratio, test_ratio]
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

        self.dataset_root = os.path.abspath(dataset_root)
        self.obj_type = obj_type
        self.aff_type = aff_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.keep_id = keep_id
        self.balance_data = balance_data
        self.count = 0  # 三元组数据计数

        # 数据集分割
        self.train_samples: List[JointDataSample] = []
        self.val_samples: List[JointDataSample] = []
        self.test_samples: List[JointDataSample] = []
        
        # 原始数据索引（用于保存分割结果）
        self._train_ids = create_info_dict()
        self._val_ids = create_info_dict()
        self._test_ids = create_info_dict()
        
        # 检查是否存在分割JSON文件
        split_json_path = os.path.join(dataset_root, 'dataset_split.json')
        if split_json_path and os.path.exists(split_json_path):
            print(f"从JSON文件加载数据集分割: {split_json_path}")
            self.load_from_split_json(split_json_path)
        else:
            # 加载数据并分割
            self.load_all_data()
            self.split_dataset()
        self.pair_samples()
    
    def load_all_data(self, filter_by_ids=None):
        """加载 Instruction、Image、PointCloud 数据"""
        import threading

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

        print("所有数据加载完成")
            
    def split_dataset(self):
        # 构建图文对（Instruction + Image）
        # 结构: {obj_type: [(inst, img), ...]}
        text_image_pairs = defaultdict(lambda: defaultdict(list))
        for obj_type in tqdm(Instruction.all.keys(), desc="构建图文对"):
            for inst in Instruction.all[obj_type]:
                # 查找匹配的 Image（按 id 匹配）
                matched_image = Image.get_by_id(obj_type, inst.id)
                
                if matched_image:
                    for i, aff in enumerate(matched_image.labels):
                        pair = (inst.id, (matched_image.id, i))  # NOTE: 一张图片可能对应多个inst（不同aff_type），因此使用时需要用对应的aff_type
                        text_image_pairs[obj_type][aff].append(pair)

        # 按 obj_type 和 aff_type 分组点云
        # 结构: {obj_type: {aff_type: [pc_ids]}}
        pc_groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        for obj_type, pcs in PointCloud.all.items():
            for pc in pcs:
                if pc is None or pc.labels is None:
                    continue
                for idx, label in enumerate(pc.labels):
                    pc_groups[obj_type][label].append(((pc.id, idx),)) # 兼容图文对 _split_group
        
        def _split_group(groups, inner, test_ratio, val_ratio):
            # 对每个分组进行分割
            for obj_type in groups.keys():
                for aff_type, inner_ids in groups[obj_type].items():
                    if not inner_ids:
                        continue
                    
                    # 去重并打乱
                    unique_ids = list(set(inner_ids))
                    random.shuffle(unique_ids)
                    
                    n_total = len(unique_ids)
                    # if n_total < 5: continue # 数量过少会导致全部划分到train数据集中(Hack:@Lyh 只训练，不评估，增加鲁棒性？)
                    n_test = int(n_total * test_ratio)
                    n_val = int(n_total * val_ratio)
                    # train 取剩余的，避免舍入误差
                    
                    # 分割ID
                    test_ids = unique_ids[:n_test]
                    val_ids = unique_ids[n_test:n_test + n_val]
                    train_ids = unique_ids[n_test + n_val:]
                    
                    # 记录分割结果
                    for idx, modality in enumerate(inner):
                        self._train_ids[modality][obj_type][aff_type] = [e[idx] for e in train_ids]
                        self._val_ids[modality][obj_type][aff_type] =  [e[idx] for e in val_ids]
                        self._test_ids[modality][obj_type][aff_type] =  [e[idx] for e in test_ids]

        _split_group(text_image_pairs, ('ins', 'img'), self.test_ratio, self.val_ratio)
        _split_group(pc_groups, ('pc',), self.test_ratio, self.val_ratio)


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

        # 平衡每个 train/val/test 中的 pc 和图文数据
        if self.balance_data:
            print("平衡各分割集中的 pc 和图文数据...")  # Note: 是否必要？
            
            # 遍历所有 obj_type 和 aff_type 组合
            all_obj_types = set(self._train_ids['ins'].keys()) | set(self._train_ids['pc'].keys())
            
            for obj_type in tqdm(all_obj_types, desc='平衡数据'):
                # 获取该 obj_type 下所有的 aff_type
                ins_aff_types = set(self._train_ids['ins'].get(obj_type, {}).keys())
                pc_aff_types = set(self._train_ids['pc'].get(obj_type, {}).keys())
                all_aff_types = ins_aff_types & pc_aff_types  # NOTE: 只平衡公共aff的数量
                
                for aff_type in tqdm(all_aff_types, leave=False):
                    # 对 train/val/test 分别进行平衡
                    for split_ids in [self._train_ids, self._val_ids, self._test_ids]:
                        # 获取当前分割中的 ins/img 和 pc 数量
                        ins_list = split_ids['ins'].get(obj_type, {}).get(aff_type, [])
                        img_list = split_ids['img'].get(obj_type, {}).get(aff_type, [])
                        pc_list = split_ids['pc'].get(obj_type, {}).get(aff_type, [])
                        
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
                            split_ids['ins'][obj_type][aff_type] = [ins_list[i] for i in balanced_indices]
                            split_ids['img'][obj_type][aff_type] = [img_list[i] for i in balanced_indices]
                        
                        # 平衡 pc
                        if pc_list and len(pc_list) < target_count:
                            split_ids['pc'][obj_type][aff_type] = _balance_by_copy(pc_list, target_count)

            print("数据平衡完成")
    
    def pair_samples(self):
        """
        将 _train_ids, _val_ids, _test_ids 中的索引聚合成三元组数据
        
        对于每个 (obj_type, aff_type) 组合，将 ins, img, pc 三种模态的数据按位置配对。
        如果某个模态数据不足，用 None 补充。
        
        结果存储在 self.train_samples, self.val_samples, self.test_samples 中
        """
        def _pair_split(split_ids: Dict) -> List[JointDataSample]:
            """
            将单个分割集（train/val/test）的索引聚合成三元组
            
            Args:
                split_ids: 分割索引字典，结构为 {modality: {obj_type: {aff_type: [ids]}}}
            
            Returns:
                JointDataSample 列表
            """
            samples = []
            
            # 获取所有 obj_type（从三种模态中取并集）
            all_obj_types = set()
            for modality in ['ins', 'img', 'pc']:
                if modality in split_ids:
                    all_obj_types.update(split_ids[modality].keys())
            
            for obj_type in all_obj_types:
                # 获取该 obj_type 下所有 aff_type（从三种模态中取并集）
                all_aff_types = set()
                for modality in ['ins', 'img', 'pc']:
                    if modality in split_ids and obj_type in split_ids[modality]:
                        all_aff_types.update(split_ids[modality][obj_type].keys())
                
                for aff_type in all_aff_types:
                    # 获取各模态的索引列表
                    ins_ids = split_ids.get('ins', {}).get(obj_type, {}).get(aff_type, [])
                    img_ids = split_ids.get('img', {}).get(obj_type, {}).get(aff_type, [])
                    pc_ids = split_ids.get('pc', {}).get(obj_type, {}).get(aff_type, [])
                    
                    # 计算最大长度
                    max_len = max(len(ins_ids), len(img_ids), len(pc_ids))
                    
                    if max_len == 0:
                        continue
                    
                    # 按位置配对，不足的用 None 补充
                    for i in range(max_len):
                        # 获取 Instruction
                        ins = Instruction.get_by_id(obj_type, ins_ids[i]) if i < len(ins_ids) else None
                        
                        # 获取 Image 和 GT
                        image, img_mask_idx = None, None 
                        if i < len(img_ids):
                            image = Image.get_by_id(obj_type, img_ids[i][0])
                            img_mask_idx = img_ids[i][1]
                        
                        # 获取 PointCloud 和 GT
                        pc = None
                        pc_mask_idx = None
                        if i < len(pc_ids):
                            pc = PointCloud.get_by_id(obj_type, pc_ids[i][0])
                            pc_mask_idx =  pc_ids[i][1]
                        
                        # 至少有一个模态有数据才创建样本
                        if (ins or image or pc) is not None:
                            sample = JointDataSample(
                                ins=ins,
                                img=image,
                                pc=pc,
                                img_mask_idx=img_mask_idx,
                                pc_mask_idx=pc_mask_idx
                            )
                            samples.append(sample)
            
            return samples
        
        print("聚合三元组数据...")
        self.train_samples = _pair_split(self._train_ids)
        self.val_samples = _pair_split(self._val_ids)
        self.test_samples = _pair_split(self._test_ids)
        
        print(f"三元组聚合完成: train={len(self.train_samples)}, val={len(self.val_samples)}, test={len(self.test_samples)}")

    def load_from_split_json(self, split_json_path: str):
        """
        从分割JSON文件加载数据集
        """
        if not os.path.exists(split_json_path):
            raise FileNotFoundError(f"分割JSON文件不存在: {split_json_path}")
        
        # 加载JSON文件
        with open(split_json_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        # 从metadata中提取参数
        metadata = split_data.get('metadata', {})

        self.train_ratio = metadata['train_ratio']
        self.val_ratio = metadata['val_ratio']
        self.test_ratio = metadata['test_ratio']
        self.random_seed = metadata['random_seed']
        
        # 加载分割索引
        self._train_ids = split_data.get('train', create_info_dict())
        self._val_ids = split_data.get('val', create_info_dict())
        self._test_ids = split_data.get('test', create_info_dict())
        
        # 兼容缩写
        for e in (self._train_ids, self._test_ids, self._val_ids):
            e['ins'] = e['Instruction']
            e['img'] = e['Image']
            e['pc'] = e['PointCloud']

        def merge_split_ids(*splits):
            """
            合并多个 split 字典 (train/val/test)，保持原有的层级结构：
            Modality -> ObjType -> AffType -> List
            """
            # {pc: {obj1: [ids1, ids2]...}}
            merged = defaultdict(lambda: defaultdict(set))
            
            for split in splits:
                if not split: continue
                for mod, obj_dict in split.items():
                    if mod in ('ins', 'Instruction'):
                        for obj, aff_dict in obj_dict.items():
                            for aff, ids in aff_dict.items():
                                merged[mod][obj] |= set(ids)
                    else:
                        for obj, aff_dict in obj_dict.items():
                            for aff, ids in aff_dict.items():
                                merged[mod][obj] |= set([e[0] for e in ids])
            
            return merged
        
        merged_ids = merge_split_ids(self._train_ids, self._test_ids, self._val_ids)
        self.load_all_data(filter_by_ids=merged_ids)

    def save_split_json(self, save_path: str = None) -> str:
        """
        将数据集分割结果保存为JSON文件
        
        保存格式:
        {
            "metadata": {
                "total_sample": 10000,
                "train_sample": 7500,
                "val_sample": 1500,
                "test_sample": 1000,
                "train_ratio": 0.75,
                "val_ratio": 0.15,
                "test_ratio": 0.1,
                "random_seed": 114514
            },
            "train": {
                'Instruction': {obj_type: {aff_type: [id1, id2, ...]}},
                'Image': {obj_type: {aff_type: [(id, mask_idx), ...]}},
                'PointCloud': {obj_type: {aff_type: [(id, mask_idx), ...]}}
            },
            "val": {...},
            "test": {...}
        }
        
        Args:
            save_path: 保存路径，默认为 dataset_root/dataset_split.json
        
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.dataset_root, 'dataset_split.json')
        
        
        # 构建JSON数据
        split_data = {
            'metadata': {
                'total_sample': len(self.train_samples) + len(self.val_samples) + len(self.test_samples),
                'train_sample': len(self.train_samples),
                'val_sample': len(self.val_samples),
                'test_sample': len(self.test_samples),
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'random_seed': self.random_seed
            },
            'train': {k: self._train_ids[k] for k in ('Instruction', 'Image', 'PointCloud')},
            'val': {k: self._val_ids[k] for k in ('Instruction', 'Image', 'PointCloud')},
            'test': {k: self._test_ids[k] for k in ('Instruction', 'Image', 'PointCloud')}
        }

        # 去除值为空的键
        for t in split_data.keys():
            if t == 'metadata': continue
            for m in split_data[t].keys():
                for obj_type in split_data[t][m].keys():
                    keys_to_remove = []
                    for aff_type in split_data[t][m][obj_type].keys():  # 不能同时遍历删除
                        if not split_data[t][m][obj_type][aff_type]:
                            keys_to_remove.append(aff_type)
                    
                    for k in keys_to_remove:
                        del split_data[t][m][obj_type][k]

        def get_compact_json(data):
            json_str = json.dumps(data, indent=4)
            
            output = []
            depth = 0        # 记录方括号嵌套深度
            in_str = False   # 标记是否在字符串内部
            escaped = False  # 标记转义字符

            for char in json_str:
                # 处理字符串状态，防止误判字符串内的括号 (例如 "this is [a] test")
                if char == '"' and not escaped:
                    in_str = not in_str
                
                if in_str:
                    output.append(char)
                    escaped = (char == '\\' and not escaped)
                    continue

                # 核心逻辑：根据括号深度决定是否保留空白字符
                if char == '[':
                    depth += 1
                    output.append(char)
                elif char == ']':
                    depth -= 1
                    output.append(char)
                elif char == '\n':
                    # 如果在列表内，丢弃换行符
                    if depth == 0:
                        output.append(char)
                elif char == ' ':
                    # 如果在列表内，丢弃缩进空格
                    # 但为了可读性，我们可以保留逗号后面的一个空格
                    if depth == 0:
                        output.append(char)
                    elif output and output[-1] == ',':
                        output.append(' ')
                else:
                    output.append(char)

            return "".join(output)

        compact_json = get_compact_json(split_data)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(compact_json)

        
        print(f"数据集分割结果已保存至: {save_path}")
    
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
        return len(self.train_samples) + len(self.val_samples) + len(self.test_samples)
    
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
        train_len = len(self.train_samples)
        val_len = len(self.val_samples)
        
        if index < 0:
            index = len(self) + index
        
        if index < train_len:
            sample = self.train_samples[index]
            split = 'train'
        elif index < train_len + val_len:
            sample = self.val_samples[index - train_len]
            split = 'val'
        else:
            sample = self.test_samples[index - train_len - val_len]
            split = 'test'
        
        data = sample.get_data()
        data['split'] = split
        data['index'] = index
        data['obj_type'] = sample.obj_type
        data['aff_type'] = sample.aff_type
        return data
    
    def get_train_data(self, index: int = None, apply_mask: bool = False, mask_prob: Tuple[float, float, float] = (0.01, 0.02, 0.003)) -> Dict[str, Any]:
        """
        获取训练集数据
        
        Args:
            index: 样本索引，如果为 None 则返回整个训练集
            apply_mask: 是否应用随机掩码
            mask_prob: 各模态 (ins, img, pc) 被掩码的概率
        
        Returns:
            单个样本字典或样本字典列表
        """
        return self._get_split_data(self.train_samples, index, apply_mask, mask_prob)
    
    def get_val_data(self, index: int = None, apply_mask: bool = False, mask_prob: Tuple[float, float, float] = (0.01, 0.02, 0.003)) -> Dict[str, Any]:
        """
        获取验证集数据
        
        Args:
            index: 样本索引，如果为 None 则返回整个验证集
            apply_mask: 是否应用随机掩码
            mask_prob: 各模态 (ins, img, pc) 被掩码的概率
        
        Returns:
            单个样本字典或样本字典列表
        """
        return self._get_split_data(self.val_samples, index, apply_mask, mask_prob)
    
    def get_test_data(self, index: int = None, apply_mask: bool = False, mask_prob: Tuple[float, float, float] = (0.01, 0.02, 0.003)) -> Dict[str, Any]:
        """
        获取测试集数据
        
        Args:
            index: 样本索引，如果为 None 则返回整个测试集
            apply_mask: 是否应用随机掩码
            mask_prob: 各模态 (ins, img, pc) 被掩码的概率
        
        Returns:
            单个样本字典或样本字典列表
        """
        return self._get_split_data(self.test_samples, index, apply_mask, mask_prob)
    
    def _get_split_data(self, samples: List[JointDataSample], index: int = None, apply_mask: bool = False, mask_prob: Tuple[float, float, float] = (0.01, 0.02, 0.003)) -> Dict[str, Any]:
        """
        内部方法：从指定分割集获取数据
        
        Args:
            samples: 样本列表
            index: 样本索引，如果为 None 则返回整个列表
            apply_mask: 是否应用随机掩码
            mask_prob: 各模态 (ins, img, pc) 被掩码的概率
        
        Returns:
            单个样本字典或样本字典列表
        """
        if index is not None:
            # 获取单个样本
            if index < 0:
                index = len(samples) + index
            if index < 0 or index >= len(samples):
                raise IndexError(f"索引 {index} 超出范围 [0, {len(samples)})")
            
            sample = samples[index]
            if apply_mask:
                sample.apply_mask(mask_prob=mask_prob)
            
            data = sample.get_data()
            data['obj_type'] = sample.obj_type
            data['aff_type'] = sample.aff_type
            data['sample_id'] = sample.id
            return data
        else:
            # 获取整个列表
            result = []
            for i, sample in enumerate(samples):
                if apply_mask:
                    sample.apply_mask(mask_prob=mask_prob)
                
                data = sample.get_data()
                data['obj_type'] = sample.obj_type
                data['aff_type'] = sample.aff_type
                data['sample_id'] = sample.id
                result.append(data)
            return result
    
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
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        print(f"数据集根目录: {self.dataset_root}")
        print(f"总样本数: {len(self)}")
        print(f"  - 训练集: {len(self.train_samples)} ({self.train_ratio * 100:.1f}%)")
        print(f"  - 验证集: {len(self.val_samples)} ({self.val_ratio * 100:.1f}%)")
        print(f"  - 测试集: {len(self.test_samples)} ({self.test_ratio * 100:.1f}%)")
        print(f"随机种子: {self.random_seed}")
        print(f"数据平衡: {'启用' if self.balance_data else '禁用'}")
        
        # 统计各物体类型和 affordance 类型的分布
        obj_aff_count = defaultdict(lambda: defaultdict(int))
        for sample in self.train_samples + self.val_samples + self.test_samples:
            obj_aff_count[sample.obj_type][sample.aff_type] += 1
        
        print("\n物体类型和 Affordance 分布:")
        for obj_type, aff_dict in sorted(obj_aff_count.items()):
            print(f"  {obj_type}:")
            for aff_type, count in sorted(aff_dict.items()):
                print(f"    - {aff_type}: {count}")
        print("=" * 60 + "\n")
    
    
def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="加载并分割联合数据集")
    parser.add_argument('-d', '--dataset-root', type=str, required=True,
                       help='数据集根目录')
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
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no-balance', action='store_true',
                       help='禁用数据平衡（默认启用）')
    parser.add_argument('--save-split', action='store_true',
                       help='保存数据集分割结果为JSON文件')
    parser.add_argument('--save-split-path', type=str, default=None,
                       help='保存分割结果的JSON文件路径（默认为 dataset_root/dataset_split.json）')
    parser.add_argument('--keep-id', action='store_true', default=True,
                       help='保持原有的ID（默认启用）')
    
    args = parser.parse_args()
    
    # 创建数据集
    dataset = JointDataset(
        dataset_root=args.dataset_root,
        obj_type=args.obj_type,
        aff_type=args.aff_type,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        keep_id=args.keep_id,
        balance_data=not args.no_balance,
    )
    
    # 保存分割结果
    if args.save_split:
        dataset.save_split_json(args.save_split_path)
    
    # 打印统计信息
    dataset.print_statistics()
    
    # 示例：获取训练集
    train_set = dataset.get_train_data()
    print(f"\n训练集第一个样本: {train_set[0] if train_set else 'None'}")
    
if __name__ == "__main__":
    main()