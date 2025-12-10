from csv import Error
import os
import json
import numpy as np
import open3d as o3d
from collections import defaultdict


# 自定义参数
DEAFULT_OUTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"
DEAFULT_INTPUT_DIR = "/mnt/data/datasets/2D-3DJointAffordance"



class PointCloud:
    all = defaultdict(list)
    id = defaultdict(int)
    
    def __init__(self, points, obj_type, mask:np.ndarray=None, labels:list=None):
        self.points = points
        self.obj_type = obj_type
        
        PointCloud.all[obj_type].append(self)
        PointCloud.id[obj_type] += 1
        self.id = PointCloud.id[obj_type]

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
    def load_file(filepath) -> 'PointCloud':
        """加载时重新分配id"""
        obj_type = os.path.basename(os.path.dirname(filepath))

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
    header = [
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

    id = defaultdict(int)

    def __init__(self, points, obj_type, mask: np.ndarray = None, labels: list = None):
        super().__init__(points, obj_type, mask, labels)
        AGPIL_PC.all[obj_type].append(self)
        AGPIL_PC.id[obj_type] += 1


    @staticmethod
    def load_file(filepath, obj_type) -> 'PointCloud':
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
        pc_obj.labels = [label for idx, label in enumerate(AGPIL_PC.header) if idx not in zero_col_idx]

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

class Image:
    all = defaultdict(list)
    
    def __init__(self, img:np.ndarray):
        self.img = img

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.img)
    
class BoxedImage(Image):
    def __init__(self, img, box:np.ndarray=None):
        super().__init__(img)

        # 直接将box区域划作mask
        self.mask = np.zeros_like(img)
        self.mask[box[1]:box[3], box[0]:box[2]] = 1
        self.dtype = 'boxed'

class SegmentedImage(Image):
    def __init__(self, data, mask:np.ndarray=None):
        self.data = data
        self.mask = mask
        self.dtype = 'segemented'



def resolve_path(path_str: str):
    """兼容相对/绝对路径，返回绝对路径。"""
    if path_str is None:
        return None
    return path_str if os.path.isabs(path_str) else os.path.abspath(os.path.join(os.getcwd(), path_str))



if __name__=="__main__":
    # 单独运行时作为数据处理工具
    import argparse

    parser = argparse.ArgumentParser(description="根据不同的数据集选择不同的处理方式，整合为同一个数据集")
    parser.add_argument("-i", "--input_dir", type=str, help="输入数据集位置", required=True)                                                                                 
    parser.add_argument('-o', "--output_dir", type=str, help="输出数据集的绝对位置", default=DEAFULT_OUTPUT_DIR)
    parser.add_argument("-a", "--aff_name", type=str, help="affordance种类", default=None)
    parser.add_argument("-t", "--type_of_obj", type=str, help="物体类型", default=None)
    parser.add_argument("-m", "--modality", type=str, nargs="+", help="手动添加数据的模态，可选一个或多个",
                         default=['all'], choices=['pc', 'img', 'img_mask', 'ins', 'all'])
    parser.add_argument("-d", "--dataset", type=str, help="按照预设定数据集整理",
                         default=None, choices=['AGPIL', 'PIADv2', 'PIAD', 'RAGNet'])
    parser.add_argument('-s', '--show', type=str, nargs="+", help='直接渲染点云文件的路径，选择时只执行渲染操作', default=[])
    parser.add_argument('-r', '--rewrite', type=str, help='是否按id从1开始重写已有数据集', default=False)

    args = parser.parse_args()

    # 兼容相对/绝对路径
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)


    # 记录每种模态数据的对应数据的id信息
    info_file = os.path.join(output_dir, 'info.json')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info_dict = json.load(f)
            if args.rewrite:
                # 重新初始化类起始id
                pc_id_dict = info_dict.get('PointCloud', {})
                if pc_id_dict:
                    PointCloud.id = defaultdict(int, pc_id_dict)
    else:
        info_dict = {}

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
                    raise Error(f'Selected dataset "{args.dataset}" is not supported!!')
            pc.show()

    else:
        match args.dataset:
            case 'AGPIL':
                AGPIL_PC.load_and_save(input_dir, output_dir)
            case 'PIADv2': ...
            case 'PIAD': ...
            case 'RAGNet': ...
            case e:...


        if 'pc' in selected_modalities:
            pass
        if 'img' in selected_modalities:
            pass
        if 'ins' in selected_modalities:
            pass


        # 保留最大的id
        for k, v in PointCloud.all.items():
            info_dict['PointCloud'][k] = max(v[-1], info_dict.get('PointCloud', {}).get(k, 0))


        # 保存信息文件
        with open(info_file, 'w') as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=2)



