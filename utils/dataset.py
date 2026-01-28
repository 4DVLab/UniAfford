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
        """处理单个样本，返回标准化的数据字典，增加缓存以加速重复访问"""
        import cv2

        # 用于存储缓存的成员变量
        if not hasattr(self, '_sample_cache'):
            self._sample_cache = {}

        sample_id = id(sample)
        if sample_id in self._sample_cache:
            return self._sample_cache[sample_id]

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
            pts = data['pc']
            # 确保 pts 是 (N, 3) 形状
            if pts.ndim != 2 or pts.shape[1] != 3:
                pts = pts.reshape(-1, 3)
            n = pts.shape[0]
            idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
            sampled_pts = pts[idx]  # (num_points, 3)
            sp = sampled_pts - np.mean(sampled_pts, axis=0)
            md = np.max(np.sqrt(np.sum(sp ** 2, axis=1)))
            normed_sp = (sp / md) if md > 0 else sp
            result['point_clouds'] = torch.from_numpy(normed_sp).float()  # (num_points, 3)
            result['has_point_cloud'] = True
            if data['pc_gt'] is not None:
                pc_gt = data['pc_gt']
                # 兼容 pc_gt 是 (N,) 或者 (N, 1) 形状
                if pc_gt.ndim == 2 and pc_gt.shape[1] == 1:
                    pc_gt = pc_gt[:, 0]
                pm = pc_gt[idx]  # (num_points,)
                pm = pm.astype(np.float32)
                if pm.max() > 1:
                    pm = pm / 255.0
                result['pc_masks'] = torch.from_numpy(pm).float()  # (num_points,)
        # 写入缓存
        self._sample_cache[sample_id] = result

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
        self.num_samples = len(self.samples)
        
        # 只创建一个 epoch 大小的打乱索引，避免内存爆炸
        # 当 samples_per_epoch 很大时，使用动态随机采样而不是预先创建大列表
        self._shuffle_indices()
    
    def _shuffle_indices(self):
        """创建或重新打乱索引（只保留一个 epoch 的样本数量）"""
        self.sample_indices = list(range(self.num_samples))
        random.shuffle(self.sample_indices)
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 使用取模运算动态映射到实际样本，避免创建大列表
        actual_index = self.sample_indices[index % self.num_samples]
        sample = self.samples[actual_index]
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
    批量数据整理函数（支持动态batch size和模态padding）
    
    将数据集返回的样本整理成模型所需的输入格式，支持图像和点云两种模态。
    自动处理不同模态数据的padding和对齐，确保批次内数据形状一致。
    
    功能特性：
    1. 动态batch size：自动适应不同大小的批次
    2. 模态padding：对图像和点云数据进行智能padding
    3. 序列对齐：对文本序列进行padding和attention mask生成
    4. 有效性标记：标记每个样本包含的模态类型（类似attention mask）
    
    Args:
        batch: 样本列表，每个样本是一个字典
        tokenizer: 分词器
        conv_type: 对话类型
        use_mm_start_end: 是否使用多模态开始/结束标记
        local_rank: 本地进程排名
        
    Returns:
        包含模型输入的字典，包括：
        - input_ids: [Batch, SeqLen] 文本输入
        - labels: [Batch, SeqLen] 标签
        - attention_masks: [Batch, SeqLen] 文本注意力掩码（标记有效token）
        - images: [Batch, 3, H, W] 图像数据（如果有）
        - images_clip: [Batch, 3, H, W] CLIP图像数据（如果有）
        - masks_list: List[Tensor] 图像分割标注掩码（Ground Truth）
        - point_clouds: [Batch, N, 3] 点云数据（如果有）
        - point_masks_list: List[Tensor] 点云分割标注掩码（Ground Truth）
        - image_valid_mask: [Batch] 图像模态有效性标记（标记哪些样本有图像）
        - pc_valid_mask: [Batch] 点云模态有效性标记（标记哪些样本有点云）
        - point_valid_lengths: [Batch] 每个样本的有效点数（用于处理padding）
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
            raise RuntimeError("没有获取到输入指令") # debug
            question = f"Please identify the {aff_type} affordance region of the {obj_type}."
        
        # 根据模态构建回答（增加随机性）
        answer_parts = []
        
        if has_image:
            # 2D 图像回答模板（随机选择）
            image_templates = [
                f"The {aff_type} affordance region of the {obj_type} is [SEG].",
                f"Here is the {aff_type} region: [SEG].",
                f"The {aff_type} area for the {obj_type} is highlighted as [SEG].",
                f"I've identified the {aff_type} affordance: [SEG].",
                f"The region for {aff_type} interaction is [SEG].",
                f"[SEG] shows the {aff_type} affordance of the {obj_type}.",
            ]
            answer_parts.append(random.choice(image_templates))
        
        if has_pc:
            # 3D 点云回答模板（随机选择）
            pc_templates = [
                f"The 3D {aff_type} affordance region is [AFF].",
                f"In 3D space, the {aff_type} region is [AFF].",
                f"The point cloud shows the {aff_type} area as [AFF].",
                f"[AFF] represents the 3D {aff_type} affordance.",
                f"The {aff_type} region in the point cloud is [AFF].",
                f"For 3D interaction, the {aff_type} area is [AFF].",
            ]
            answer_parts.append(random.choice(pc_templates))
        
        if not answer_parts:
            # 无输入时的回答模板（随机选择）
            no_input_templates = [
                "I cannot identify the affordance region without visual input.",
                "I need visual information to identify the affordance region.",
                "Please provide an image or point cloud to analyze the affordance.",
                "Visual input is required to determine the affordance region.",
            ]
            answer_parts.append(random.choice(no_input_templates))
        
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
        
        # 分别构建问题和回答的 prompt
        conv_instance.append_message(conv_instance.roles[0], question)
        conv_instance.append_message(conv_instance.roles[1], None)  # 先不添加回答内容
        question_prompt = conv_instance.get_prompt()
        
        # 分别对问题和回答进行分词
        question_ids = tokenizer_image_token(question_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # 对回答进行分词（不包含特殊的对话格式）
        answer_ids = tokenizer_image_token(answer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # 合并 input_ids：问题 + 回答
        input_ids = torch.cat([question_ids, answer_ids], dim=0)
        
        # 创建 labels：问题部分设为 IGNORE_INDEX，回答部分保留原始 ids
        question_labels = torch.full_like(question_ids, IGNORE_INDEX)
        answer_labels = answer_ids.clone()
        labels = torch.cat([question_labels, answer_labels], dim=0)
        
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
    
    # 处理图像数据（支持不同尺寸的padding）
    if images_list:
        # 检查所有图像是否具有相同的尺寸
        image_shapes = [img.shape for img in images_list]
        
        if len(set(image_shapes)) > 1:
            # 尺寸不一致，需要padding到最大尺寸
            max_h = max(img.shape[1] for img in images_list)
            max_w = max(img.shape[2] for img in images_list)
            
            padded_images = []
            padded_images_clip = []
            padded_masks = []
            resize_list = []
            original_size_list = []
            
            for i, img in enumerate(images_list):
                c, h, w = img.shape
                
                # Padding图像
                if h < max_h or w < max_w:
                    pad_h = max_h - h
                    pad_w = max_w - w
                    # 使用零填充（黑色背景）
                    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    padded_img = img
                padded_images.append(padded_img)
                padded_images_clip.append(padded_img)  # CLIP使用相同的padding
                
                # Padding掩码
                if masks_list and i < len(masks_list):
                    mask = masks_list[i]
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
                    
                    mask_h, mask_w = mask.shape[1], mask.shape[2]
                    if mask_h < max_h or mask_w < max_w:
                        pad_h = max_h - mask_h
                        pad_w = max_w - mask_w
                        padded_mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    else:
                        padded_mask = mask
                    padded_masks.append(padded_mask)
                    original_size_list.append((mask_h, mask_w))
                
                resize_list.append((h, w))
            
            result['images'] = torch.stack(padded_images)  # [Batch, 3, MaxH, MaxW]
            result['images_clip'] = torch.stack(padded_images_clip)  # [Batch, 3, MaxH, MaxW]
            result['masks_list'] = padded_masks if padded_masks else None
            result['resize_list'] = resize_list  # 记录每个样本的原始尺寸
            result['original_size_list'] = original_size_list if original_size_list else []
        else:
            # 尺寸一致，直接stack
            result['images'] = torch.stack(images_list)
            result['images_clip'] = torch.stack(images_clip_list)
            result['masks_list'] = masks_list
            result['resize_list'] = [(img.shape[1], img.shape[2]) for img in images_list]
            result['original_size_list'] = [mask.shape[-2:] for mask in masks_list] if masks_list else []
    else:
        # 创建占位符
        result['images'] = None
        result['images_clip'] = None
        result['masks_list'] = None
        result['resize_list'] = None
        result['original_size_list'] = None
    
    # 处理点云数据（支持不同点数的padding）
    if point_clouds_list:
        # 检查所有点云是否具有相同的点数
        point_nums = [pc.shape[0] for pc in point_clouds_list]
        max_points = max(point_nums)
        
        if len(set(point_nums)) > 1:
            # 点数不一致，需要padding
            padded_pcs = []
            padded_pc_masks = []
            
            for i, pc in enumerate(point_clouds_list):
                num_points = pc.shape[0]
                if num_points < max_points:
                    # Padding点云：使用零填充
                    padding = torch.zeros(max_points - num_points, 3, dtype=pc.dtype)
                    padded_pc = torch.cat([pc, padding], dim=0)
                else:
                    padded_pc = pc
                padded_pcs.append(padded_pc)
                
                # Padding掩码
                if pc_masks_list and i < len(pc_masks_list):
                    pc_mask = pc_masks_list[i]
                    if num_points < max_points:
                        mask_padding = torch.zeros(max_points - num_points, dtype=pc_mask.dtype)
                        padded_mask = torch.cat([pc_mask, mask_padding], dim=0)
                    else:
                        padded_mask = pc_mask
                    padded_pc_masks.append(padded_mask)
            
            result['point_clouds'] = torch.stack(padded_pcs)  # [Batch, MaxPoints, 3]
            result['point_masks_list'] = padded_pc_masks if padded_pc_masks else None
            result['point_valid_lengths'] = torch.tensor(point_nums, dtype=torch.long)  # 记录每个样本的有效点数
        else:
            # 点数一致，直接stack
            result['point_clouds'] = torch.stack(point_clouds_list)  # [Batch, NumPoints, 3]
            result['point_masks_list'] = pc_masks_list if pc_masks_list else None
            result['point_valid_lengths'] = torch.tensor(point_nums, dtype=torch.long)
    else:
        result['point_clouds'] = None
        result['point_masks_list'] = None
        result['point_valid_lengths'] = None
    
    # 添加有效性标记
    result['image_valid_mask'] = torch.tensor(has_image_flags, dtype=torch.bool)
    result['pc_valid_mask'] = torch.tensor(has_pc_flags, dtype=torch.bool)
    
    return result
