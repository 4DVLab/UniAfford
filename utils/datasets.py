import os
import json
import numpy as np
import open3d as o3d
import cv2
from collections import defaultdict


# 自定义参数
DEFAULT_OUTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 输出的数据集位置
DEFAULT_INTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"  # 加载的数据集位置

# 全局信息文件路径与缓存，模块导入时即初始化
info_root = DEFAULT_OUTPUT_DIR
info_file = os.path.join(info_root, 'info.json')
info_dict = defaultdict(dict)



class PointCloud:
    all = defaultdict(list)
    count = defaultdict(lambda: defaultdict(int))
    
    def __init__(self, points, obj_type, mask:np.ndarray=None, labels:list=None):
        self.points = points
        self.obj_type = obj_type
        
        PointCloud.all[obj_type].append(self)
        PointCloud.count[obj_type]['ID'] += 1
        self.id = PointCloud.count[obj_type]['ID']

        for l in labels:
            PointCloud.count[obj_type][l] += 1

        self.mask = mask       # 对应点的aff的值
        self.labels = labels   # aff_mask对应列的标签


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
 
        pc_obj = PointCloud(points = data[:, :3], obj_type=obj_type)
        if data.shape[1] > 3:
            pc_obj.mask = data[:, 3:] 
        if len(header) > 3:
            pc_obj.labels = header[3:] 

        return pc_obj

    @classmethod
    def save_all(cls, dataset_root_path):
        for k, v in cls.all.items():
            dir_path = os.path.join(dataset_root_path, k, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            for e in v:
                e.save_to(os.path.join(dir_path, f'{e.obj_type}_{e.id}.csv')) # 保存时命名为 {obj_type}_{id}.csv
    
    @classmethod
    def load_all(cls, dataset_root_path):
        def iterator():
            for dir in os.listdir(dataset_root_path):
                dir_path = os.path.join(dataset_root_path, dir)
                if not os.path.isdir(dir_path):
                    continue
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        print(f'loading {file_path}')
                        pc = cls.load_file(file_path)
                        yield pc

        return iterator()
    
    @classmethod
    def load_and_save(cls, input_root, output_root):
        for pc in cls.load_all(input_root):
            dir_path = os.path.join(output_root, pc.obj_type, 'PointCloud')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            pc.save_to(os.path.join(dir_path, f'{pc.obj_type}_{pc.id}.csv'))

    def show(self):
        if self.mask is not None and self.labels is not None and len(self.labels) > 0:
            for idx, label in enumerate(self.labels):
                if self.mask.shape[1] <= idx: break

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

class AGPIL_PC(PointCloud):
    aff_type = [
        'grasp', 'contain', 'lift', 'open', 'lay',
        'sit', 'support', 'wrapgrasp', 'pour', 'move',
        'display', 'push', 'listen', 'wear', 'press',
        'cut', 'stab',     
    ]
    all = {
        k: list() for k in [
            'Bag', 'Bed', 'Bottle', 'Bowl', 'Chair',
            'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',
            'Faucet', 'Hat', 'Keyboard', 'Knife', 'Laptop',
            'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture',
            'Table', 'TrashCan', 'Vase',
        ]
    }

    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type, mask, labels)
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
        pc_obj = AGPIL_PC(points = data[:, :3], obj_type=obj_type)
        
        # 筛选出 mask 中全 0 的列索引（按列判断，忽略前三列 xyz）
        zero_col_idx = np.where(np.all(data[:, 3:] == 0, axis=0))[0]
        # 根据列索引过滤掉对应的标签；此处先用 header 作为占位标签
        pc_obj.labels = [label for idx, label in enumerate(AGPIL_PC.aff_type) if idx not in zero_col_idx]

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
                                    print(f'loading {file_path}')
                                    pc = cls.load_file(file_path, obj_type=obj_type)
                                    yield pc
        return iterator()

class PIADv2_PC(PointCloud):
    aff_type = [
        'grasp', 'contain', 'lift', 'open', 'lay',
        'sit', 'support', 'wrapgrasp', 'pour', 'move',
        'display', 'push', 'listen', 'wear', 'press',
        'cut', 'stab',
    ]
    all = {
        k: list() for k in [
            'Bag', 'Bed', 'Bottle', 'Bowl', 'Chair',
            'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',
            'Faucet', 'Hat', 'Keyboard', 'Knife', 'Laptop',
            'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture',
            'Table', 'TrashCan', 'Vase',
        ]
    }

    count = defaultdict(lambda: defaultdict(int))

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type, mask, labels)
        AGPIL_PC.all[obj_type].append(self)
        AGPIL_PC.count[obj_type]['ID'] += 1

    @staticmethod
    def load_file(filepath, obj_type=None) -> 'PointCloud':
        """
        TODO
        """

        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                data.append(list(map(float, parts[2:])))

        data = np.asarray(data, dtype=float)
        pc_obj = PIADv2_PC(points=data[:, :3], obj_type=obj_type)


        return pc_obj

    @classmethod
    def load_all(cls, dataset_root_path):
        """
        Args:
            dataset_root_path:
        """
        def iterator():
            ...

        return iterator()




class Image:
    all = defaultdict(list)
    id = defaultdict(int)
    
    def __init__(self, img:np.ndarray,
            obj_type,
            labels=None,
            aff_mask:list[np.ndarray]=None,
            obj_mask=None,
            visible_mask=None
        ):
        """ 
        Args:
            aff_mask: affordance区域的标注信息
            obj_mask: 整个物体的区域信息（含被遮挡部分）
        """
        self.img = img
        self.labels=labels if labels is not None else []
        self.dtype = 'No-mask' if aff_mask is None else 'Segmented'
        self.obj_type = obj_type

        Image.id[self.obj_type] += 1
        self.id = Image.id[self.obj_type]

        self.mask = aff_mask if aff_mask is not None else []
        self.obj_mask = obj_mask if obj_mask is not None else np.asarray([])
        self.visible_mask = visible_mask if visible_mask is not None else np.asarray([])

    def save_to(self, dir_path):
        # TODO: 修改保存目录、id匹配
        # dir_path 应该是目录，生成两个文件：原图 和 mask 合成图
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 保存img原图
        img_path = os.path.join(dir_path, 'rgb', f'{self.obj_type}_{self.id}.png')
        img_to_save = self.img
        # # 确保img是uint8格式和三通道
        # if img_to_save.dtype != np.uint8:
        #     img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
        # if img_to_save.ndim == 2:  # 灰度图转三通道
        #     img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(img_path, img_to_save)

        # 保存aff_mask
        mask_path = os.path.join(dir_path, 'mask', f'{self.obj_type}_{self.id}_mask.png')
        if len(self.mask) != 0:
            # 若mask为list，每个mask单独保存 TODO
            if isinstance(self.mask, list):
                for idx, mask in enumerate(self.mask):
                    single_mask_path = os.path.join(
                        dir_path, 'mask', f'{self.obj_type}_{self.id}_mask_{idx}.png'
                    )
                    cv2.imwrite(single_mask_path, mask)
            else:
                cv2.imwrite(mask_path, self.mask)

        # 保存obj_mask和visib_mask（如有）
        if self.obj_mask is not None and self.obj_mask.size != 0:
            obj_mask_path = os.path.join(dir_path, 'mask', f'{self.obj_type}_{self.id}_obj_mask.png')
            cv2.imwrite(obj_mask_path, self.obj_mask)
        if self.visible_mask is not None and self.visible_mask.size != 0:
            vis_mask_path = os.path.join(dir_path, 'mask', f'{self.obj_type}_{self.id}_visible_mask.png')
            cv2.imwrite(vis_mask_path, self.visible_mask)


    @classmethod
    def load_file(cls, filepath, obj_type=None):
        ...

    @classmethod
    def load_all(cls, dir_path):
        ...

class BoxedImage(Image):
    def __init__(self, img, box:np.ndarray=None, labels=None):
        super().__init__(img, labels=labels)

        # 直接将box区域划作mask
        self.mask = np.zeros_like(img)
        self.mask[box[1]:box[3], box[0]:box[2]] = 1
        self.dtype = 'Boxed'

class HeatImage(Image):
    def __init__(self, img:np.ndarray, aff_mask:np.ndarray=None, labels=None, obj_mask:np.ndarray=None):
        super().__init__(img, aff_mask=aff_mask, labels=labels, obj_mask=obj_mask)
        self.dtype = 'HeatMap'


class HANDAL_IMG(Image):
    all = {}
    @classmethod
    def load_file(cls, filepath, obj_type):...

    @classmethod
    def load_all(cls, dir_path, obj_type, aff_type):
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
                            aff_mask=aff_mask,
                            obj_mask=obj_mask,
                            visible_mask=visib_mask,
                        )

                        yield obj

        return iterator()


class RAGNet(Image):...



def resolve_path(path_str: str):
    """兼容相对/绝对路径，返回绝对路径。"""
    if path_str is None: return None
    return path_str if os.path.isabs(path_str) else os.path.abspath(os.path.join(os.getcwd(), path_str))


def load_info(output_dir=DEFAULT_OUTPUT_DIR, rewrite=False):
    """
    载入 info.json，并按需恢复各类 id 计数器。
    在模块导入时即可调用，方便包内函数直接使用全局缓存。
    """
    global info_root, info_file, info_dict
    info_root = output_dir
    info_file = os.path.join(info_root, 'info.json')

    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info_dict = defaultdict(dict, json.load(f))
            if rewrite:
                if pc_id_dict := info_dict.get('PointCloud', {}):
                    PointCloud.count = defaultdict(lambda: defaultdict(int), pc_id_dict)
                if img_id_dict := info_dict.get('Image', {}):
                    Image.id = defaultdict(lambda: defaultdict(int), img_id_dict)
    else:
        info_dict = defaultdict(lambda: defaultdict(int))


def update_info():
    global info_dict
    # 保留最大的id
    for k, v in PointCloud.count.items():
        info_dict['PointCloud'][k] = max(v["ID"], info_dict['PointCloud'][k])

    # 保存信息文件
    with open(info_file, 'w') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    # 单独运行时作为数据处理工具
    import argparse

    parser = argparse.ArgumentParser(description="根据不同的数据集选择不同的处理方式，整合为同一个数据集")
    parser.add_argument("-i", "--input_dir", type=str, help="输入数据集位置", required=True)
    parser.add_argument('-o', "--output_dir", type=str, help="输出数据集的绝对位置", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-a", "--aff_name", type=str, help="affordance种类", default=None)
    parser.add_argument("-t", "--type_of_obj", type=str, help="物体类型", default=None)
    parser.add_argument("-m", "--modality", type=str, nargs="+", help="手动添加数据的模态，可选一个或多个",
                         default=['all'], choices=['pc', 'img', 'img_mask', 'ins', 'all'])
    parser.add_argument("-d", "--dataset", type=str, help="按照预设定数据集整理",
                         default=None, choices=['AGPIL', 'PIADv2', 'PIAD', 'RAGNet', 'HANDAL'])
    parser.add_argument('-s', '--show', type=str, nargs="+", help='直接渲染点云文件的路径，选择时只执行渲染操作', default=[])
    parser.add_argument('-r', '--rewrite', action='store_true', help='是否按id从1开始重写已有数据集')

    args = parser.parse_args()

    # 兼容相对/绝对路径并载入信息文件
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    load_info(output_dir, args.rewrite)

    # 整理模态输入
    selected_modalities = set(args.modality)
    if 'all' in selected_modalities:
        selected_modalities = {'pc', 'img', 'img_mask', 'ins'}
    

    if args.show:
        for f in args.show:
            file_path = resolve_path(f)
            match args.dataset:
                case 'AGPIL':
                    pc = AGPIL_PC.load_file(file_path)
                case 'PIADv2': ...
                case None:
                    pc = PointCloud.load_file(file_path)
                case e:
                    raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')
            pc.show()

    else:
        match args.dataset:
            case 'AGPIL':
                AGPIL_PC.load_and_save(input_dir, output_dir)
            case 'PIADv2': ...
            case 'PIAD': ...
            case 'RAGNet': ...
            case e:
                raise TypeError(f'Selected dataset "{args.dataset}" is not supported!!')

        if 'pc' in selected_modalities:
            pass
        if 'img' in selected_modalities:
            pass
        if 'ins' in selected_modalities:
            pass

    # 保存信息文件
    update_info()




