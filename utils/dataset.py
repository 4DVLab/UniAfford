"""
PyTorch DataLoader 适配器
"""
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.base_dataset import JointDataset, JointDataSample
from model.llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)


class HybridDataset(Dataset):
    """兼容原有接口的混合数据集"""
    
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
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.num_points = num_points
        self.precision = precision
        
        self.joint_dataset = JointDataset(
            dataset_root=dataset_dir, train_ratio=train_ratio,
            val_ratio=val_ratio, test_ratio=test_ratio,
            random_seed=random_seed,
        )
        self.samples = self.joint_dataset.train_samples
        
        if len(self.samples) < samples_per_epoch:
            repeat = (samples_per_epoch // len(self.samples)) + 1
            self.sample_indices = (list(range(len(self.samples))) * repeat)[:samples_per_epoch]
        else:
            self.sample_indices = list(range(len(self.samples)))
        random.shuffle(self.sample_indices)
    
    def __len__(self): return self.samples_per_epoch
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        import cv2
        sample = self.samples[self.sample_indices[index % len(self.sample_indices)]]
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


class ValDataset(Dataset):
    """验证数据集"""
    
    def __init__(self, dataset_dir: str, tokenizer=None, vision_tower: str = None,
                 val_dataset: str = None, image_size: int = 1024, num_points: int = 2048,
                 joint_dataset: JointDataset = None,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, random_seed: int = 42):
        self.image_size, self.num_points = image_size, num_points
        self.joint_dataset = joint_dataset or JointDataset(
            dataset_root=dataset_dir, train_ratio=train_ratio, val_ratio=val_ratio,
            test_ratio=test_ratio, random_seed=random_seed,
        )
        self.samples = self.joint_dataset.val_samples
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        import cv2
        sample, data = self.samples[index], self.samples[index].get_data()
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
