"""
PyTorch DataLoader 适配器

使用方式:
    # 创建数据集管理器（只加载一次数据）
    dataset_manager = DatasetManager(dataset_dir, ...)
    
    # 获取各分割的 PyTorch Dataset
    train_dataset = dataset_manager.get_train_dataset()
    val_dataset = dataset_manager.get_val_dataset()
    test_dataset = dataset_manager.get_test_dataset()
"""
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.base_dataset import JointDataset, JointDataSample
from model.llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)


class DatasetManager:
    """
    数据集管理器，负责创建和管理 JointDataset，并提供获取各分割数据集的接口。
    确保 JointDataset 只被创建一次。
    """
    
    def __init__(self, dataset_dir: str, tokenizer=None, vision_tower: str = None,
                 samples_per_epoch: int = 10000, precision: str = "bf16",
                 image_size: int = 1024, num_classes_per_sample: int = 3,
                 exclude_val: bool = False, dataset: str = None,
                 sample_rate: List[float] = None, sem_seg_data: str = None,
                 refer_seg_data: str = None, vqa_data: str = None,
                 reason_seg_data: str = None, explanatory: float = 0.1,
                 num_points: int = 2048,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, random_seed: int = 42):
        """
        初始化数据集管理器
        
        Args:
            dataset_dir: 数据集根目录
            tokenizer: 分词器
            vision_tower: 视觉塔配置
            samples_per_epoch: 每个 epoch 的样本数
            precision: 精度设置
            image_size: 图像大小
            num_points: 点云点数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.image_size = image_size
        self.num_points = num_points
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 只创建一次 JointDataset
        self.joint_dataset = JointDataset(
            dataset_root=dataset_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        
        # 缓存已创建的数据集
        self._train_dataset: Optional[HybridDataset] = None
        self._val_dataset: Optional[ValDataset] = None
        self._test_dataset: Optional[TestDataset] = None
    
    def get_train_dataset(self, samples_per_epoch: int = None) -> 'HybridDataset':
        """获取训练数据集"""
        if self._train_dataset is None:
            self._train_dataset = HybridDataset(
                samples=self.joint_dataset.train_samples,
                samples_per_epoch=samples_per_epoch or self.samples_per_epoch,
                image_size=self.image_size,
                num_points=self.num_points,
                tokenizer=self.tokenizer,
                precision=self.precision,
            )
        return self._train_dataset
    
    def get_val_dataset(self) -> 'ValDataset':
        """获取验证数据集"""
        if self._val_dataset is None:
            self._val_dataset = ValDataset(
                samples=self.joint_dataset.val_samples,
                image_size=self.image_size,
                num_points=self.num_points,
            )
        return self._val_dataset
    
    def get_test_dataset(self) -> 'TestDataset':
        """获取测试数据集"""
        if self._test_dataset is None:
            self._test_dataset = TestDataset(
                samples=self.joint_dataset.test_samples,
                image_size=self.image_size,
                num_points=self.num_points,
            )
        return self._test_dataset
    
    def get_all_datasets(self) -> Tuple['HybridDataset', 'ValDataset', 'TestDataset']:
        """获取所有数据集（训练、验证、测试）"""
        return self.get_train_dataset(), self.get_val_dataset(), self.get_test_dataset()


class _BaseDataset(Dataset):
    """数据集基类，提供通用的数据处理方法"""
    
    def __init__(self, samples: List[JointDataSample], image_size: int = 1024, num_points: int = 2048):
        self.samples = samples
        self.image_size = image_size
        self.num_points = num_points
    
    def _process_sample(self, sample: JointDataSample) -> Dict[str, Any]:
        """处理单个样本，返回标准化的数据字典"""
        import cv2
        data = sample.get_data()
        result = {
            'instruction': data['ins'] or "", 
            'obj_type': sample.obj_type, 
            'aff_type': sample.aff_type,
            'has_image': False,
            'has_point_cloud': False,
        }
        
        # 处理图像数据
        if data['img'] is not None:
            img = cv2.resize(data['img'], (self.image_size, self.image_size)) if data['img'].shape[:2] != (self.image_size, self.image_size) else data['img']
            result['images'] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
            result['has_image'] = True
            if data['img_gt'] is not None:
                mask = cv2.resize(data['img_gt'], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST) if data['img_gt'].shape[:2] != (self.image_size, self.image_size) else data['img_gt']
                result['masks'] = torch.from_numpy((mask.astype(np.float32) / 255.0) if mask.max() > 1 else mask.astype(np.float32)).float()
        
        # 处理点云数据
        if data['pc'] is not None:
            pts, n = data['pc'], len(data['pc'])
            idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
            sp = pts[idx] - np.mean(pts[idx], axis=0)
            md = np.max(np.sqrt(np.sum(sp ** 2, axis=1)))
            result['point_clouds'] = torch.from_numpy((sp / md) if md > 0 else sp).float()
            result['has_point_cloud'] = True
            if data['pc_gt'] is not None:
                pm = data['pc_gt'][idx]
                result['pc_masks'] = torch.from_numpy((pm.astype(np.float32) / 255.0) if isinstance(pm, np.ndarray) and pm.max() > 1 else pm.astype(np.float32)).float()
        
        return result


class HybridDataset(_BaseDataset):
    """训练数据集，支持样本重复和打乱"""
    
    def __init__(self, samples: List[JointDataSample], samples_per_epoch: int = 10000,
                 image_size: int = 1024, num_points: int = 2048,
                 tokenizer=None, precision: str = "bf16"):
        super().__init__(samples, image_size, num_points)
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        
        # 构建样本索引（支持重复采样）
        if len(self.samples) < samples_per_epoch:
            repeat = (samples_per_epoch // len(self.samples)) + 1
            self.sample_indices = (list(range(len(self.samples))) * repeat)[:samples_per_epoch]
        else:
            self.sample_indices = list(range(len(self.samples)))
        random.shuffle(self.sample_indices)
    
    def __len__(self): 
        return self.samples_per_epoch
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[self.sample_indices[index % len(self.sample_indices)]]
        return self._process_sample(sample)


class ValDataset(_BaseDataset):
    """验证数据集"""
    
    def __init__(self, samples: List[JointDataSample], image_size: int = 1024, num_points: int = 2048):
        super().__init__(samples, image_size, num_points)
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[index])


class TestDataset(_BaseDataset):
    """测试数据集"""
    
    def __init__(self, samples: List[JointDataSample], image_size: int = 1024, num_points: int = 2048):
        super().__init__(samples, image_size, num_points)
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[index])


def collate_fn(batch: List[Dict], tokenizer=None, conv_type: str = None,
               use_mm_start_end: bool = False, local_rank: int = 0) -> Dict[str, Any]:
    """
    批量数据整理函数
    
    将数据集返回的样本整理成模型所需的输入格式，支持图像和点云两种模态。
    
    Args:
        batch: 样本列表
        tokenizer: 分词器
        conv_type: 对话类型
        use_mm_start_end: 是否使用多模态开始/结束标记
        local_rank: 本地进程排名
        
    Returns:
        包含模型输入的字典
    """
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    
    # 初始化结果字典
    batch_size = len(batch)
    
    # 收集各模态数据
    images_list = []
    images_clip_list = []
    masks_list = []
    point_clouds_list = []
    pc_masks_list = []
    conversations = []
    
    has_image_flags = []
    has_pc_flags = []
    
    for sample in batch:
        has_image = sample.get('has_image', False)
        has_pc = sample.get('has_point_cloud', False)
        has_image_flags.append(has_image)
        has_pc_flags.append(has_pc)
        
        # 收集图像数据
        if has_image:
            img = sample.get('images')
            if img is not None:
                images_list.append(img)
                images_clip_list.append(img)  # CLIP 使用相同的图像
            mask = sample.get('masks')
            if mask is not None:
                masks_list.append(mask.unsqueeze(0) if mask.dim() == 2 else mask)
        
        # 收集点云数据
        if has_pc:
            pc = sample.get('point_clouds')
            if pc is not None:
                point_clouds_list.append(pc)
            pc_mask = sample.get('pc_masks')
            if pc_mask is not None:
                pc_masks_list.append(pc_mask)
        
        # 构建对话
        instruction = sample.get('instruction', '')
        obj_type = sample.get('obj_type', '')
        aff_type = sample.get('aff_type', '')
        
        # 构建问题和回答
        if instruction:
            question = instruction
        else:
            question = f"Please identify the {aff_type} affordance region of the {obj_type}."
        
        # 根据模态构建回答
        answer_parts = []
        if has_image:
            answer_parts.append("The 2D affordance region is [SEG].")
        if has_pc:
            answer_parts.append("The 3D affordance region is [AFF].")
        
        if not answer_parts:
            answer_parts.append("I cannot identify the affordance region without visual input.")
        
        answer = " ".join(answer_parts)
        
        # 添加图像标记
        if has_image:
            if use_mm_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
        
        conversations.append((question, answer))
    
    # 处理对话并生成 input_ids 和 labels
    input_ids_list = []
    labels_list = []
    
    conv = conversation_lib.conv_templates[conv_type].copy() if conv_type else conversation_lib.default_conversation.copy()
    
    for question, answer in conversations:
        conv_instance = conv.copy()
        conv_instance.append_message(conv_instance.roles[0], question)
        conv_instance.append_message(conv_instance.roles[1], answer)
        prompt = conv_instance.get_prompt()
        
        # 分词
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # 创建标签（只计算回答部分的损失）
        labels = input_ids.clone()
        
        # 找到回答开始的位置，将问题部分的标签设为 IGNORE_INDEX
        sep = conv_instance.sep + conv_instance.roles[1] + ": "
        sep_ids = tokenizer(sep, add_special_tokens=False).input_ids
        
        # 简化处理：将前半部分设为 IGNORE_INDEX
        total_len = len(input_ids)
        answer_start = total_len // 2  # 粗略估计
        labels[:answer_start] = IGNORE_INDEX
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    # 填充 input_ids 和 labels
    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        padding_len = max_len - len(input_ids)
        padded_input_ids.append(torch.cat([input_ids, torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labels, torch.full((padding_len,), IGNORE_INDEX, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)]))
    
    # 构建结果字典
    result = {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_masks': torch.stack(attention_masks),
        'offset': torch.tensor([0] + [i + 1 for i in range(batch_size)], dtype=torch.long),
    }
    
    # 处理图像数据
    if images_list:
        result['images'] = torch.stack(images_list)
        result['images_clip'] = torch.stack(images_clip_list)
        result['masks_list'] = masks_list
        result['resize_list'] = [(img.shape[1], img.shape[2]) for img in images_list]
        result['label_list'] = [mask.shape[-2:] for mask in masks_list] if masks_list else []
    else:
        # 创建占位符
        result['images'] = torch.zeros(batch_size, 3, 1024, 1024)
        result['images_clip'] = torch.zeros(batch_size, 3, 224, 224)
        result['masks_list'] = []
        result['resize_list'] = [(1024, 1024)] * batch_size
        result['label_list'] = []
    
    # 处理点云数据
    if point_clouds_list:
        result['point_clouds'] = torch.stack(point_clouds_list)
        result['point_masks_list'] = pc_masks_list if pc_masks_list else None
    else:
        result['point_clouds'] = None
        result['point_masks_list'] = None
    
    # 添加有效性标记
    result['image_valid_mask'] = torch.tensor(has_image_flags, dtype=torch.bool)
    result['pc_valid_mask'] = torch.tensor(has_pc_flags, dtype=torch.bool)
    
    return result
