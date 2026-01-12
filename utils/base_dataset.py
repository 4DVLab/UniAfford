"""
聚合 Instruction、Image、PointCloud 三元组的数据集类
支持训练集、测试集、验证集的比例分割
"""
import os
import random
from tqdm import tqdm
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
import open3d as o3d
import cv2
import csv

""" ------------------------------------ 3种基础模态的支持 ----------------------------------- """
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
            data = np.loadtxt(f, delimiter=',', skiprows=1)

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
                if isinstance(aff_type, str):
                    aff_type = [aff_type]
                aff_set = set(aff_type)

                keep_indices = [i for i, l in enumerate(labels) if l in aff_set]
                if keep_indices:
                    mask = mask[:, keep_indices]
                    labels = [labels[i] for i in keep_indices]
                else:
                    # 如果没有匹配的列，则置空 mask / labels
                    mask = None
                    labels = None

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
    @classmethod
    def load_all(cls,
            dataset_root_path,
            keep_id=False,
            obj_type=None, 
            aff_type=None
        ):
        """
        从统一格式的数据集中批量加载 PointCloud
        
        Args:
            dataset_root_path: 根目录，结构为 {obj_type}/PointCloud/{obj_type}_{id}.csv
            keep_id: 是否保持文件名中的 id，而不是重新分配
            obj_type: 需要加载的物体类型；None 时加载所有，可以是 str 或 list[str]
            aff_type: 需要加载的 affordance 类型；None 时加载所有，可以是 str 或 list[str]
        """
        # 归一化过滤列表
        if obj_type is None:
            obj_type_set = None
        else:
            if isinstance(obj_type, str):
                obj_type = [obj_type]
            obj_type_set = set(obj_type)
        
        if aff_type is None:
            aff_type_set = None
        else:
            if isinstance(aff_type, str):
                aff_type = [aff_type]
            aff_type_set = set(aff_type)

        def iterator():
            for obj_type_name in os.listdir(dataset_root_path):
                # 物体类型过滤
                if obj_type_set is not None and obj_type_name not in obj_type_set:
                    continue

                dir_path = os.path.join(dataset_root_path, obj_type_name, 'PointCloud')
                if not os.path.isdir(dir_path):
                    continue
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if not os.path.isfile(file_path):
                        continue
                    print(f'loading PC: {file_path}')
                    pc = cls.load_file(
                        file_path,
                        obj_type=obj_type_name,
                        aff_type=aff_type,
                        keep_id=keep_id,
                    )
                    yield pc
        
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
        for obj_type, ls in tqdm(cls.all.items()):
            loaded = dict()
            for pc in tqdm(ls):
                loaded[pc] = pc._merge(loaded.get(pc, None))

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
                # 按顺序重新分配 id
                id_counter = 0
                for e in v:
                    if e is not None:
                        id_counter += 1
                        e.save_to(dir_path, file_id=id_counter)

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
        if obj_type is None:
            rgb_filename = os.path.basename(filepath)
            
            # 从文件名提取 obj_type 和 id: {obj_type}_{id}.png
            base_name = os.path.splitext(rgb_filename)[0]
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                inferred_obj_type = parts[0]
                inferred_id = parts[1]
            else:
                raise ValueError(f"Cannot parse obj_type and id from filename: {rgb_filename}")
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
            # 归一化 aff_type 过滤列表（如果提供）
            if aff_type is not None:
                if isinstance(aff_type, str):
                    aff_type = [aff_type]
                aff_set = set(aff_type)
            else:
                aff_set = None

            # 遍历mask目录下的所有label子目录
            for label_dir in os.listdir(mask_dir):
                # aff 过滤：如果指定了 aff_type，则只加载在列表中的子目录
                if aff_set is not None and label_dir not in aff_set:
                    continue

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
    def load_all(cls, dataset_root_path, keep_id=False,
                 obj_type=None, aff_type=None):
        """
        从保存的数据集目录结构中加载所有图片
        
        Args:
            dataset_root_path: 数据集根目录，结构为 {obj_type}/rgb/{obj_type}_{id}.png
            keep_id: 是否保持文件名中的 id，而不是重新分配
            obj_type: 需要加载的物体类型；None 时加载所有，可以是 str 或 list[str]
            aff_type: 需要加载的 affordance 类型；None 时加载所有，可以是 str 或 list[str]
        """
        # 归一化过滤列表
        if obj_type is None:
            obj_type_set = None
        else:
            if isinstance(obj_type, str):
                obj_type = [obj_type]
            obj_type_set = set(obj_type)
        
        if aff_type is None:
            aff_type_set = None
        else:
            if isinstance(aff_type, str):
                aff_type = [aff_type]
            aff_type_set = set(aff_type)

        def iterator():
            for obj_type_name in os.listdir(dataset_root_path):
                # 物体类型过滤
                if obj_type_set is not None and obj_type_name not in obj_type_set:
                    continue

                obj_type_dir = os.path.join(dataset_root_path, obj_type_name)
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
                        img = cls.load_file(
                            rgb_path,
                            obj_type=obj_type_name,
                            aff_type=aff_type,
                            keep_id=keep_id,
                        )
                        yield img
                    except Exception as e:
                        print(f"Failed to load {rgb_path}: {e}")
                        continue
        
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
        self.id = Instruction.count[self.obj_type]['ID'] if given_id is None else given_id  # Ins的id和图片的id一一对应

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
        """一次加载一个物体，不使用迭代器"""
        for obj_type in os.listdir(dataset_root_path):
            file_path = os.path.join(dataset_root_path, obj_type, 'Instruction.csv')
            if os.path.exists(file_path):
                cls.load(file_path, keep_id=keep_id)


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

    def __init__(self, instruction: Instruction, image: Image, pointcloud: PointCloud):
        """
        Args:
            instruction: Instruction 对象
            image: Image 对象
            pointcloud: PointCloud 对象
        """
        self.instruction = instruction
        self.image = image
        self.pointcloud = pointcloud
        
        # 验证一致性
        assert instruction.obj_type == image.obj_type == pointcloud.obj_type, \
            f"obj_type 不一致: {instruction.obj_type}, {image.obj_type}, {pointcloud.obj_type}"
        assert instruction.aff_type == image.aff_type == pointcloud.aff_type, \
            f"obj_type 不一致: {instruction.aff_type}, {image.aff_type}, {pointcloud.aff_type}"

        
        self.aff_type = instruction.aff_type
        self.obj_type = instruction.obj_type
        JointDataSample.start_id += 1
        self.id = JointDataSample.start_id

        JointDataSample.all[self.obj_type].append(self)
        JointDataSample.count[self.obj_type][self.aff_type] += 1
        # 模态掩码状态：True 表示该模态可用，False 表示被 mask
        self._mask_state = {
            'ins': True,
            'img': True,
            'pc': True
        }
    
    def __repr__(self):
        mask_info = ""
        masked_modalities = [k for k, v in self._mask_state.items() if not v]
        if masked_modalities:
            mask_info = f", masked={masked_modalities}"
        return f"JointDataSample(obj_type={self.obj_type}, id={self.id}, aff_type={self.aff_type}{mask_info})"
    
    def apply_mask(self, modalities: List[str] = None, mask_prob: float = None, 
                   min_available: int = 1) -> 'JointDataSample':
        """
        对指定模态应用掩码
        
        Args:
            modalities: 要 mask 的模态列表，可选 ['ins', 'img', 'pc']
                       如果为 None 且 mask_prob 不为 None，则随机选择模态进行 mask
            mask_prob: 每个模态被 mask 的概率（0.0-1.0），仅当 modalities 为 None 时生效
            min_available: 最少保留的可用模态数量（默认至少保留1个模态）
        
        Returns:
            self，支持链式调用
        """
        all_modalities = ['ins', 'img', 'pc']
        
        if modalities is not None:
            # 指定模态进行 mask
            for mod in modalities:
                if mod in all_modalities:
                    self._mask_state[mod] = False
        elif mask_prob is not None:
            # 随机 mask 模态
            # 先随机决定每个模态是否被 mask
            candidates_to_mask = []
            for mod in all_modalities:
                if random.random() < mask_prob:
                    candidates_to_mask.append(mod)
            
            # 确保至少保留 min_available 个模态
            max_mask_count = len(all_modalities) - min_available
            if len(candidates_to_mask) > max_mask_count:
                # 随机选择要 mask 的模态
                candidates_to_mask = random.sample(candidates_to_mask, max_mask_count)
            
            for mod in candidates_to_mask:
                self._mask_state[mod] = False
        
        return self
    
    def reset_mask(self) -> 'JointDataSample':
        """重置所有模态的掩码状态"""
        self._mask_state = {
            'ins': True,
            'img': True,
            'pc': True
        }
        return self
    
    def is_masked(self, modality: str) -> bool:
        """检查指定模态是否被 mask"""
        return not self._mask_state.get(modality, True)
    
    def get_available_modalities(self) -> List[str]:
        """获取当前可用的模态列表"""
        return [k for k, v in self._mask_state.items() if v]
    
    def get_masked_modalities(self) -> List[str]:
        """获取当前被 mask 的模态列表"""
        return [k for k, v in self._mask_state.items() if not v]
    
    def get_modality(self, modality):
        """
        获取指定模态数据（'ins', 'img', 'pc'），如果被 mask 则返回 None
        支持单个字符串或字符串列表输入
        """
        def fetch_one(mod):
            if not self._mask_state.get(mod, True):
                return None
            if mod == 'ins':
                return self.instruction
            elif mod == 'img':
                return self.image
            elif mod == 'pc':
                return self.pointcloud
            else:
                raise ValueError(f"Unknown modality: {mod}")

        if isinstance(modality, list):
            return [fetch_one(mod) for mod in modality]
        else:
            return fetch_one(modality)
        
    def get_data(self) -> Dict[str, Any]:
        """
        获取数据字典，被 mask 的模态返回 None
        """
        return {
            'ins': self.get_modality('ins'),
            'img': self.get_modality('img'),
            'pc': self.get_modality('pc'),
            'obj_type': self.obj_type,
            'aff_type': self.aff_type,
            'id': self.id,
        }


class JointDataset:
    """聚合 Instruction、Image、PointCloud 三元组的数据集类"""
    
    def __init__(self, 
                 dataset_root: str,
                 obj_type: Optional[List[str]] = None,
                 aff_type: Optional[List[str]] = None,
                 train_ratio: float = None,
                 val_ratio: float = None,
                 test_ratio: float = None,
                 random_seed: int = 42,
                 keep_id: bool = True,
                 balance_data: bool = True,
                 mask_prob: float = 0.0,
                 mask_modalities: Optional[List[str]] = None,
                 min_available_modalities: int = 1,
                 split_json_path: Optional[str] = None):
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
            mask_modalities: 固定要 mask 的模态列表，优先级高于 mask_prob
            min_available_modalities: 最少保留的可用模态数量
            split_json_path: 数据集分割JSON文件路径，如果存在则从中加载分割结果
        """
        
        # 设置随机种子
        random.seed(random_seed)
        # 保证 train/val/test 的参数顺序不被搓乱，不改变三者变量的原始变量名（顺序 train, val, test）

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
        self.mask_prob = mask_prob
        self.mask_modalities = mask_modalities
        self.min_available_modalities = min_available_modalities
        self.split_json_path = split_json_path
        self.count = 0  # 三元组数据计数

        # 数据集分割
        self.train_samples: List[JointDataSample] = []
        self.val_samples: List[JointDataSample] = []
        self.test_samples: List[JointDataSample] = []
        
        # 原始数据索引（用于保存分割结果）
        # 结构参考 config/dataset.json，存储物体的id
        self._train_ids: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._val_ids: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._test_ids: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # 检查是否存在分割JSON文件
        if split_json_path and os.path.exists(split_json_path):
            print(f"从JSON文件加载数据集分割: {split_json_path}")
            self._load_from_split_json()
        else:
            # 加载数据并分割
            self.load_all()
            self.split_dataset()
        self.pair_samples()
    
    def load_all(self):
        """加载 Instruction、Image、PointCloud 数据"""
        print(f"开始加载数据集: {self.dataset_root}")
        
        # 加载所有数据
        print("加载 PointCloud...")
        for _ in PointCloud.load_all(self.dataset_root, obj_type=self.obj_type, aff_type=self.aff_type, keep_id=self.keep_id):
            pass  # 消费迭代器
        print("加载 Image...")
        for _ in Image.load_all(self.dataset_root, obj_type=self.obj_type, aff_type=self.aff_type, keep_id=self.keep_id):
            pass  # 消费迭代器
        print("加载 Instruction...")
        Instruction.load_all(self.dataset_root, keep_id=self.keep_id)
        
    def split_dataset(self):
        # 构建图文对（Instruction + Image）
        # 结构: {obj_type: [(inst, img), ...]}
        print("构建图文对...")
        text_image_pairs = defaultdict(list)
        for obj_type in Instruction.all.keys():
            for inst in Instruction.all[obj_type]:
                # 查找匹配的 Image（按 id 匹配）
                matched_image = Image.all[obj_type][inst.id]
                
                if matched_image:
                    for i, aff in enumerate(matched_image.labels):
                        pair = (inst, matched_image.mask[i,:,:])  # NOTE: 一张图片可能对应多个inst（不同aff_type），因此使用时需要用对应的aff_type
                        text_image_pairs[obj_type][aff].append(pair)
                        self.count += 1

        # 按 obj_type 和 aff_type 分组点云
        # 结构: {obj_type: {aff_type: [pc_ids]}}
        pc_groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        for obj_type, pcs in PointCloud.all.items():
            for pc in pcs:
                if pc is None or pc.labels is None:
                    continue
                for label in pc.labels:
                    pc_groups[obj_type][label].append(pc.id)
        
        def _split_group(groups, inner, train_ratio, val_ratio):
            # 对每个分组进行分割
            for obj_type in groups.keys():
                for aff_type, inner_ids in groups[obj_type].items():
                    if not inner_ids:
                        continue
                    
                    # 去重并打乱
                    unique_ids = list(set(inner_ids))
                    random.shuffle(unique_ids)
                    
                    n_total = len(unique_ids)
                    n_train = int(n_total * train_ratio)
                    n_val = int(n_total * val_ratio)
                    # test 取剩余的，避免舍入误差
                    
                    # 分割ID
                    train_ids = unique_ids[:n_train]
                    val_ids = unique_ids[n_train:n_train + n_val]
                    test_ids = unique_ids[n_train + n_val:]
                    
                    # 记录分割结果
                    for idx, modality in enumerate(inner):
                        self._train_ids[modality][obj_type][aff_type] = [e[idx] for e in train_ids]
                        self._val_ids[modality][obj_type][aff_type] =  [e[idx] for e in val_ids]
                        self._test_ids[modality][obj_type][aff_type] =  [e[idx] for e in test_ids]

        _split_group(text_image_pairs, ('ins', 'img'), self.train_ratio, self.val_ratio)
        _split_group(pc_groups, ('pc',), self.train_ratio, self.val_ratio)


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
                "ins": {"obj_type": {"aff_type": [id1, id2, ...]}},
                "img": {"obj_type": {"aff_type": [id1, id2, ...]}},
                "pc": {"obj_type": {"aff_type": [id1, id2, ...]}}
            },
            "val": {...},
            "test": {...}
        }
        
        Args:
            save_path: 保存路径，默认为 dataset_root/dataset.json
        
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.dataset_root, 'dataset.json')
        
        # 构建各集合的模态索引结构
        def build_modality_index(samples: List[JointDataSample]) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
            """
            构建模态索引结构: {modality: {obj_type: {aff_type: [ids]}}}
            """
            result = {
                'ins': defaultdict(lambda: defaultdict(list)),
                'img': defaultdict(lambda: defaultdict(list)),
                'pc': defaultdict(lambda: defaultdict(list))
            }
            
            for sample in samples:
                obj_type = sample.obj_type
                aff_type = sample.aff_type
                
                # 记录各模态的ID
                if sample.instruction is not None:
                    result['ins'][obj_type][aff_type].append(sample.instruction.id)
                if sample.image is not None:
                    result['img'][obj_type][aff_type].append(sample.image.id)
                if sample.pointcloud is not None:
                    result['pc'][obj_type][aff_type].append(sample.pointcloud.id)
            
            # 转换为普通dict以便JSON序列化
            return {
                modality: {
                    obj: dict(affs) for obj, affs in objs.items()
                } for modality, objs in result.items()
            }
        
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
            'train': build_modality_index(self.train_samples),
            'val': build_modality_index(self.val_samples),
            'test': build_modality_index(self.test_samples)
        }
        
        # 保存JSON文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=4, ensure_ascii=False)
        
        print(f"数据集分割结果已保存至: {save_path}")
        return save_path
    
    def _apply_mask_to_samples(self, samples: List[JointDataSample]) -> List[JointDataSample]:
        """
        对样本列表应用模态掩码
        
        Args:
            samples: 样本列表
        
        Returns:
            应用掩码后的样本列表（原样本会被修改）
        """
        for sample in samples:
            # 先重置掩码状态
            sample.reset_mask()
            
            # 应用掩码
            if self.mask_modalities is not None:
                # 固定掩码指定的模态
                sample.apply_mask(modalities=self.mask_modalities, 
                                  min_available=self.min_available_modalities)
            elif self.mask_prob > 0:
                # 按概率随机掩码
                sample.apply_mask(mask_prob=self.mask_prob,
                                  min_available=self.min_available_modalities)
        
        return samples
    
    def get_sample_with_mask(self, sample: JointDataSample, 
                             modalities: List[str] = None,
                             mask_prob: float = None) -> JointDataSample:
        """
        获取单个样本并应用指定的掩码
        
        Args:
            sample: 要处理的样本
            modalities: 要 mask 的模态列表
            mask_prob: 每个模态被 mask 的概率
        
        Returns:
            应用掩码后的样本
        """
        sample.reset_mask()
        if modalities is not None:
            sample.apply_mask(modalities=modalities, 
                              min_available=self.min_available_modalities)
        elif mask_prob is not None:
            sample.apply_mask(mask_prob=mask_prob,
                              min_available=self.min_available_modalities)
        return sample
    
    def reset_all_masks(self):
        """重置所有样本的掩码状态"""
        for sample in self.samples:
            sample.reset_mask()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.samples),
            'train_samples': len(self.train_samples),
            'val_samples': len(self.val_samples),
            'test_samples': len(self.test_samples),
            'obj_types': defaultdict(int),
            'aff_types': defaultdict(int),
        }
        
        for sample in self.samples:
            stats['obj_types'][sample.obj_type] += 1
            stats['aff_types'][sample.aff_type] += 1
        
        return stats
    

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
    parser.add_argument('--mask-prob', type=float, default=0.0,
                       help='每个模态被 mask 的概率（0.0-1.0）')
    parser.add_argument('--mask-modalities', type=str, nargs='+', default=None,
                       choices=['ins', 'img', 'pc'],
                       help='固定要 mask 的模态列表')
    parser.add_argument('--min-available', type=int, default=1,
                       help='最少保留的可用模态数量')
    parser.add_argument('--split-json', type=str, default=None,
                       help='数据集分割JSON文件路径，如果存在则从中加载分割结果')
    parser.add_argument('--save-split', action='store_true',
                       help='保存数据集分割结果为JSON文件')
    parser.add_argument('--save-split-path', type=str, default=None,
                       help='保存分割结果的JSON文件路径（默认为 dataset_root/dataset.json）')
    parser.add_argument('--keep-id', action='store_true', default=True,
                       help='保持原有的ID（默认启用）')
    parser.add_argument('--no-keep-id', action='store_false', dest='keep_id',
                       help='不保持原有ID，重新分配')
    
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
        mask_prob=args.mask_prob,
        mask_modalities=args.mask_modalities,
        min_available_modalities=args.min_available,
        split_json_path=args.split_json
    )
    
    # 保存分割结果
    if args.save_split:
        dataset.save_split_json(args.save_split_path)
    
    # 打印统计信息
    dataset.print_statistics()
    
    # 示例：获取训练集
    train_set = dataset.get_train_set(apply_mask=args.mask_prob > 0 or args.mask_modalities is not None)
    print(f"\n训练集第一个样本: {train_set[0] if train_set else 'None'}")
    
    # 如果启用了 mask，显示 mask 信息
    if train_set and (args.mask_prob > 0 or args.mask_modalities is not None):
        sample = train_set[0]
        print(f"  可用模态: {sample.get_available_modalities()}")
        print(f"  被 mask 的模态: {sample.get_masked_modalities()}")


if __name__ == "__main__":
    main()

