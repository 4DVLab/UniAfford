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
import cv2
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.base_dataset import JointDataset, JointDataSample
from utils.common import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from PIL import Image
from model.qwenvl.data.rope2d import get_rope_index_3
from model.qwenvl.data.data_processor import IGNORE_INDEX as QWEN_IGNORE_INDEX


class BaseDataset(Dataset):
    """数据集基类，提供通用的数据处理方法"""
    
    def __init__(self, samples: List[JointDataSample], tokenizer, image_size: int = 1024, num_points: int = 2048, 
                 precision = torch.float32, conv_type: str = None, use_mm_start_end: bool = False, use_sample_cache: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.num_points = num_points
        self.conv_type = conv_type
        self.use_mm_start_end = use_mm_start_end
        self.tokenizer = tokenizer
        self.use_sample_cache = use_sample_cache  # 缓存开关
        self.precision = precision  # 添加精度属性
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[index])

    def _process_sample(self, sample: JointDataSample) -> Dict[str, Any]:
        """
        处理单个样本，返回标准化的数据字典
        
        优化策略：
        1. 在缓存阶段完成精度转换，避免 collate_fn 中重复转换
        2. 在缓存阶段构建 conversation，避免 collate_fn 中重复构建
        3. 使用连续内存布局提高传输效率
        4. 缓存处理后的张量以避免重复计算
        """
        # 用于存储缓存的成员变量（仅在启用缓存时使用）
        if self.use_sample_cache:
            if not hasattr(self, '_sample_cache'):
                self._sample_cache = {}

            sample_id = id(sample)
            if sample_id in self._sample_cache:
                return self._sample_cache[sample_id]

        data = sample.get_data()
        
        # 先确定模态信息
        has_image = data['img'] is not None
        has_pc = data['pc'] is not None
        
        result = {
            'obj_type': sample.obj_type, 
            'aff_type': sample.aff_type,
            'has_image': has_image,
            'has_point_cloud': has_pc,
        }
        
        """ --------------------- 构建 conversation ---------------------"""
        instruction = data['ins'] or ""
        obj_type = sample.obj_type
        aff_type = sample.aff_type
        
        # 构建问题
        if instruction:
            question = instruction
        else:
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
                f"The 3D {aff_type} affordance region is [SEG].",
                f"In 3D space, the {aff_type} region is [SEG].",
                f"The point cloud shows the {aff_type} area as [SEG].",
                f"[SEG] represents the 3D {aff_type} affordance.",
                f"The {aff_type} region in the point cloud is [SEG].",
                f"For 3D interaction, the {aff_type} area is [SEG].",
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
            if self.use_mm_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
        
        from model.llava import conversation as conversation_lib
        from model.llava.mm_utils import tokenizer_image_token

        # 处理对话并生成 input_ids 和 labels
        if self.conv_type and self.conv_type in conversation_lib.conv_templates:
            conv = conversation_lib.conv_templates[self.conv_type].copy()
        else:
            conv = conversation_lib.default_conversation.copy()
        
        # 分别构建问题和回答的 prompt
        conv.append_message(conv.roles[0], question)
        # 在这里不要给 assistant 填 None，要填一个占位符，否则 get_prompt 会报错（或逻辑异常）
        conv.append_message(conv.roles[1], "")  
        question_prompt = conv.get_prompt()
        
        # 分别对问题和回答进行分词
        question_ids = tokenizer_image_token(question_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        answer_ids = tokenizer_image_token(answer, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # 合并 input_ids：问题 + 回答
        input_ids = torch.cat([question_ids, answer_ids], dim=0)
        
        # 创建 labels：问题部分设为 IGNORE_INDEX，回答部分保留原始 ids
        question_labels = torch.full_like(question_ids, IGNORE_INDEX)
        answer_labels = answer_ids.clone()
        labels = torch.cat([question_labels, answer_labels], dim=0)

        result['input_ids'] = input_ids
        result['labels'] = labels

        
        """ --------------------- 处理图像数据（优化：直接转换为目标精度）---------------------"""
        if data['img'] is not None:
            img = cv2.resize(data['img'], (self.image_size[0], self.image_size[1])) if data['img'].shape[:2] != (self.image_size[0], self.image_size[1]) else data['img']
            # 优化：一次性完成转换、归一化和精度转换
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=self.precision).div_(255.0).contiguous()
            result['images'] = img_tensor
            
            if data['img_gt'] is not None:
                mask = cv2.resize(data['img_gt'], (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_NEAREST) if data['img_gt'].shape[:2] != (self.image_size[0], self.image_size[1]) else data['img_gt']
                mask_tensor = torch.from_numpy(mask).to(dtype=self.precision).contiguous()
                if mask_tensor.max() > 1:
                    mask_tensor.div_(255.0)
                result['img_gt'] = mask_tensor
        
        """ --------------------- 处理点云数据 ---------------------"""
        if data['pc'] is not None:
            pts = data['pc']
            # 确保 pts 是 (N, 3) 形状
            if pts.ndim != 2 or pts.shape[1] != 3:
                pts = pts.reshape(-1, 3)
            n = pts.shape[0]
            idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
            sampled_pts = pts[idx]  # (num_points, 3)
            # 直接在张量上进行中心化和归一化，简化流程
            pc_tensor = torch.from_numpy(sampled_pts).to(dtype=self.precision)
            pc_tensor = pc_tensor - pc_tensor.mean(dim=0)
            md = torch.norm(pc_tensor, dim=1).max()
            if md > 0:
                pc_tensor = pc_tensor / md
            pc_tensor = pc_tensor.contiguous()
            result['point_clouds'] = pc_tensor
            
            if data['pc_gt'] is not None:
                pc_gt = data['pc_gt']
                # 兼容 pc_gt 是 (N,) 或者 (N, 1) 形状
                if pc_gt.ndim == 2 and pc_gt.shape[1] == 1:
                    pc_gt = pc_gt[:, 0]
                pm = pc_gt[idx]
                if pm.max() > 1:
                    pm = pm / 255.0
                pm_tensor = torch.from_numpy(pm).to(dtype=self.precision).contiguous()
                result['pc_gt'] = pm_tensor
        
        # 写入缓存（已经是目标精度，conversation 已构建）
        if self.use_sample_cache:
            self._sample_cache[sample_id] = result

        return result

class TrainDataset(BaseDataset):
    """训练数据集，支持样本重复和打乱"""
    
    def __init__(self, samples: List[JointDataSample], tokenizer, image_size: int = 1024, num_points: int = 2048, 
                 precision = torch.float32, conv_type: str = None, use_mm_start_end: bool = False, 
                 samples_per_epoch: int = 10000, use_sample_cache: bool = True,
                 ):
        super().__init__(samples, tokenizer, image_size, num_points, precision, conv_type, use_mm_start_end, use_sample_cache)
        self.samples_per_epoch = samples_per_epoch
        self.num_samples = len(self.samples)
        self.current_epoch = 0
        
        # 为当前 epoch 生成随机采样索引
        self._generate_epoch_indices()
    
    def _generate_epoch_indices(self):
        """为当前 epoch 生成随机采样索引"""
        # 如果 samples_per_epoch 大于实际样本数，则进行重复采样
        if self.samples_per_epoch <= self.num_samples:
            # 不重复采样：从所有样本中随机选择 samples_per_epoch 个
            self.epoch_indices = random.sample(range(self.num_samples), self.samples_per_epoch)
        else:
            # 需要重复采样：使用 random.choices 允许重复
            self.epoch_indices = random.choices(range(self.num_samples), k=self.samples_per_epoch)
    
    def set_epoch(self, epoch: int):
        """
        设置当前 epoch，用于在每个 epoch 开始时重新生成随机索引
        
        Args:
            epoch: 当前 epoch 编号
        """
        self.current_epoch = epoch
        self._generate_epoch_indices()
    
    def __len__(self): 
        return self.samples_per_epoch
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 直接使用预生成的索引
        actual_index = self.epoch_indices[index]
        sample = self.samples[actual_index]
        return self._process_sample(sample)


def _pad_and_cat_position_ids(position_ids_list: List[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in position_ids_list)
    padded_tensors = []
    for tensor in position_ids_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


class Qwen3VLDataset(Dataset):
    """Qwen3-VL 输入格式的数据集适配。"""

    def __init__(
        self,
        samples: List[JointDataSample],
        processor,
        image_size=(1024, 1024),
        num_points: int = 2048,
        precision=torch.float32,
        use_sample_cache: bool = False,
    ):
        self.samples = samples
        self.image_size = image_size
        self.num_points = num_points
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.use_sample_cache = use_sample_cache
        self.precision = precision
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[index])

    def _process_sample(self, sample: JointDataSample) -> Dict[str, Any]:
        if self.use_sample_cache:
            if not hasattr(self, "_sample_cache"):
                self._sample_cache = {}
            sample_id = id(sample)
            if sample_id in self._sample_cache:
                return self._sample_cache[sample_id]

        data = sample.get_data()
        has_image = data["img"] is not None
        has_pc = data["pc"] is not None

        result = {
            "obj_type": sample.obj_type,
            "aff_type": sample.aff_type,
            "has_image": has_image,
            "has_point_cloud": has_pc,
        }

        instruction = data["ins"] or ""
        obj_type = sample.obj_type
        aff_type = sample.aff_type
        if instruction:
            question = instruction
        else:
            question = f"Please identify the {aff_type} affordance region of the {obj_type}."

        answer_parts = []
        if has_image:
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
            pc_templates = [
                f"The 3D {aff_type} affordance region is [SEG].",
                f"In 3D space, the {aff_type} region is [SEG].",
                f"The point cloud shows the {aff_type} area as [SEG].",
                f"[SEG] represents the 3D {aff_type} affordance.",
                f"The {aff_type} region in the point cloud is [SEG].",
                f"For 3D interaction, the {aff_type} area is [SEG].",
            ]
            answer_parts.append(random.choice(pc_templates))
        if not answer_parts:
            no_input_templates = [
                "I cannot identify the affordance region without visual input.",
                "I need visual information to identify the affordance region.",
                "Please provide an image or point cloud to analyze the affordance.",
                "Visual input is required to determine the affordance region.",
            ]
            answer_parts.append(random.choice(no_input_templates))
        answer = " ".join(answer_parts)

        # [ZeRO-3 兼容] 无论样本是否有真实图片，都提供图片输入。
        # 没有真实图片的样本使用 28×28 黑色占位图，确保 Qwen VL
        # 的视觉编码器在所有 rank 上始终执行，避免 NCCL all-gather 死锁。
        if has_image:
            img_rgb = cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = Image.new("RGB", (28, 28), color=(0, 0, 0))

        messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": question},
                ],
            }
        )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

        full_result = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = full_result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        labels = torch.full_like(input_ids, QWEN_IGNORE_INDEX)

        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

        result["input_ids"] = input_ids
        result["labels"] = labels

        if "image_grid_thw" in full_result:
            grid_thw = full_result.get("image_grid_thw")
            if not isinstance(grid_thw, (list, tuple)):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        position_ids, _ = get_rope_index_3(
            self.merge_size,
            input_ids,
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        result["position_ids"] = position_ids

        if "pixel_values" in full_result:
            result["pixel_values"] = full_result["pixel_values"]
        if "image_grid_thw" in full_result:
            result["image_grid_thw"] = full_result["image_grid_thw"]

        if data["img"] is not None:
            img = cv2.resize(data["img"], (self.image_size[0], self.image_size[1])) if data["img"].shape[:2] != (self.image_size[0], self.image_size[1]) else data["img"]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=self.precision).div_(255.0).contiguous()
            result["images"] = img_tensor

            if data["img_gt"] is not None:
                mask = cv2.resize(data["img_gt"], (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_NEAREST) if data["img_gt"].shape[:2] != (self.image_size[0], self.image_size[1]) else data["img_gt"]
                mask_tensor = torch.from_numpy(mask).to(dtype=self.precision).contiguous()
                if mask_tensor.max() > 1:
                    mask_tensor.div_(255.0)
                result["img_gt"] = mask_tensor

        if data["pc"] is not None:
            pts = data["pc"]
            if pts.ndim != 2 or pts.shape[1] != 3:
                pts = pts.reshape(-1, 3)
            n = pts.shape[0]
            idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
            sampled_pts = pts[idx]
            pc_tensor = torch.from_numpy(sampled_pts).to(dtype=self.precision)
            pc_tensor = pc_tensor - pc_tensor.mean(dim=0)
            md = torch.norm(pc_tensor, dim=1).max()
            if md > 0:
                pc_tensor = pc_tensor / md
            pc_tensor = pc_tensor.contiguous()
            result["point_clouds"] = pc_tensor

            if data["pc_gt"] is not None:
                pc_gt = data["pc_gt"]
                if pc_gt.ndim == 2 and pc_gt.shape[1] == 1:
                    pc_gt = pc_gt[:, 0]
                pm = pc_gt[idx]
                if pm.max() > 1:
                    pm = pm / 255.0
                pm_tensor = torch.from_numpy(pm).to(dtype=self.precision).contiguous()
                result["pc_gt"] = pm_tensor

        if self.use_sample_cache:
            self._sample_cache[sample_id] = result
        return result


class Qwen3VLTrainDataset(Qwen3VLDataset):
    """训练数据集，支持样本重复和打乱（Qwen3-VL 格式）。"""

    def __init__(
        self,
        samples: List[JointDataSample],
        processor,
        image_size=(1024, 1024),
        num_points: int = 2048,
        precision=torch.float32,
        samples_per_epoch: int = 10000,
        use_sample_cache: bool = True,
    ):
        super().__init__(
            samples=samples,
            processor=processor,
            image_size=image_size,
            num_points=num_points,
            precision=precision,
            use_sample_cache=use_sample_cache,
        )
        self.samples_per_epoch = samples_per_epoch
        self.num_samples = len(self.samples)
        self.current_epoch = 0
        self._generate_epoch_indices()

    def _generate_epoch_indices(self):
        if self.samples_per_epoch <= self.num_samples:
            self.epoch_indices = random.sample(range(self.num_samples), self.samples_per_epoch)
        else:
            self.epoch_indices = random.choices(range(self.num_samples), k=self.samples_per_epoch)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self._generate_epoch_indices()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        actual_index = self.epoch_indices[index]
        sample = self.samples[actual_index]
        return self._process_sample(sample)


def qwen3vl_collate_fn(
        batch: List[Dict],
        tokenizer=None,
        output_image_size=(1024,1024),
        output_point_nums=2048,
        precision=torch.float32
    ) -> Dict[str, Any]:
    """
    Returns:
        {
            # mllm输入
            "input_ids"            : [B, L]           # 文本 token ids（含图像 token）
            "labels"               : [B, L]           # 语言建模标签（非目标位置为 IGNORE_INDEX）
            "attention_mask"       : [B, L]           # 文本注意力 mask
            "position_ids"         : [B, 3, L]        # Qwen3-VL RoPE 位置编码
            "pixel_values"         : [B, 3, H, W]     # Qwen3-VL 视觉输入（占位样本用复用值回填）
            "image_grid_thw"       : [B, 3]           # Qwen3-VL 图像网格 (T, H, W)
            
            # 2D 分割输入
            "images"               : [B, 3, H, W]     # 2D 分割输入图像（统一 padding）
            "img_gt_tensor"        : [B, H, W]        # 2D 分割 GT（无效样本为 0）
            "original_size_list"   : list[(h, w)]     # 原始 GT 尺寸（与图像一致）
            "img_valid_mask"       : [B]              # 是否有有效图像分割监督

            # 3D 分割输入
            "point_clouds"         : [B, N, 3]        # 点云输入（统一 padding）
            "pc_gt_tensor"         : [B, N]           # 点云分割 GT（无效样本为 0）
            "pc_valid_lengths"     : [B]              # 有效点数量（0 表示无效样本）
        }
    """
    # -------- 1) 收集 batch 内各模态原始数据 --------
    batch_size = len(batch)

    input_ids_list, labels_list, position_ids_list = [], [], []
    pixel_values_list, image_grid_thw_list = [], []
    pixel_values_valid_flags = []

    images_list, img_gt_masks = [], []
    point_clouds_list, pc_gt_masks = [], []
    has_image_flags = []

    for sample in batch:
        # 文本与 Qwen3-VL 位置编码
        input_ids_list.append(sample["input_ids"].squeeze(0))
        labels_list.append(sample["labels"].squeeze(0))
        position_ids_list.append(sample["position_ids"])

        # Qwen3-VL 视觉输入（可能缺失，后续统一回填）
        pixel_values = sample.get("pixel_values")
        image_grid_thw = sample.get("image_grid_thw")
        pixel_values_list.append(pixel_values)
        image_grid_thw_list.append(image_grid_thw)
        pixel_values_valid_flags.append(pixel_values is not None and image_grid_thw is not None)

        # 2D/3D 分割输入
        has_image_flags.append(sample.get("has_image", False))
        images_list.append(sample.get("images"))
        img_gt_masks.append(sample.get("img_gt"))
        point_clouds_list.append(sample.get("point_clouds"))
        pc_gt_masks.append(sample.get("pc_gt"))

    # -------- 2) 文本输入 padding --------
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=QWEN_IGNORE_INDEX
    )
    position_ids = _pad_and_cat_position_ids(position_ids_list)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    batch_out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # -------- 3) Qwen3-VL 视觉输入补齐（保证 ZeRO-3 各 rank 统一前向）--------
    # 视觉输入非空是必要条件；缺失时用占位数据回填。
    if any(pixel_values_valid_flags):
        fallback_pixel_values = next(
            pv for pv, ok in zip(pixel_values_list, pixel_values_valid_flags) if ok
        )
        fallback_grid_thw = next(
            g for g, ok in zip(image_grid_thw_list, pixel_values_valid_flags) if ok
        )
    else:
        # 理论上不会发生（dataset 已提供占位图），若触发说明数据异常
        import warnings
        dummy_h, dummy_w = output_image_size
        warnings.warn(
            "qwen3vl_collate_fn: pixel_values 全为空，已使用占位图回填。"
            "请检查数据集/预处理逻辑。",
            RuntimeWarning,
        )
        fallback_pixel_values = torch.zeros(
            1, 3, dummy_h, dummy_w, dtype=precision
        )
        fallback_grid_thw = torch.ones(1, 3, dtype=torch.long)

    fixed_pixel_values = [
        pv if pv is not None else fallback_pixel_values for pv in pixel_values_list
    ]
    fixed_grid_thw = [
        g if g is not None else fallback_grid_thw for g in image_grid_thw_list
    ]
    batch_out["pixel_values"] = torch.cat(fixed_pixel_values, dim=0)
    batch_out["image_grid_thw"] = torch.cat(fixed_grid_thw, dim=0)

    # -------- 4) 2D 图像与 GT padding --------
    valid_images = [img for img in images_list if img is not None]
    if len(valid_images) == 0:
        dummy_h, dummy_w = output_image_size
        batch_out["images"] = torch.zeros(batch_size, 3, dummy_h, dummy_w, dtype=precision)
        batch_out["img_gt_tensor"] = torch.zeros(batch_size, dummy_h, dummy_w, dtype=precision)
        batch_out["original_size_list"] = [(dummy_h, dummy_w)] * batch_size
    else:
        max_h, max_w = output_image_size
        padded_images = []
        padded_masks = []
        original_size_list = []

        for i, img in enumerate(images_list):
            if img is None:
                padded_images.append(torch.zeros(3, max_h, max_w, dtype=precision))
            else:
                c, h, w = img.shape
                if h < max_h or w < max_w:
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
                else:
                    padded_img = img
                if padded_img.dtype != precision:
                    padded_img = padded_img.to(precision)
                padded_images.append(padded_img)

            mask = img_gt_masks[i] if i < len(img_gt_masks) else None
            if mask is None:
                padded_masks.append(torch.zeros(max_h, max_w, dtype=precision))
                original_size_list.append((max_h, max_w))
            else:
                mask_h, mask_w = mask.shape[0], mask.shape[1]
                original_size_list.append((mask_h, mask_w))
                if mask_h < max_h or mask_w < max_w:
                    pad_h = max_h - mask_h
                    pad_w = max_w - mask_w
                    padded_mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
                else:
                    padded_mask = mask
                if padded_mask.dtype != precision:
                    padded_mask = padded_mask.to(precision)
                padded_masks.append(padded_mask)

        stacked_images = torch.stack(padded_images)
        batch_out["images"] = stacked_images
        batch_out["img_gt_tensor"] = torch.stack(padded_masks)
        batch_out["original_size_list"] = original_size_list

    # 2D 有效样本标记（供上游屏蔽无效样本梯度）
    batch_out["img_valid_mask"] = torch.tensor(has_image_flags, dtype=torch.bool)

    # -------- 5) 3D 点云与 GT padding --------
    valid_pcs = [pc for pc in point_clouds_list if pc is not None]
    if len(valid_pcs) == 0:
        batch_out["point_clouds"] = torch.zeros(batch_size, output_point_nums, 3, dtype=precision)
        batch_out["pc_gt_tensor"] = torch.zeros(batch_size, output_point_nums, dtype=precision)
        batch_out["pc_valid_lengths"] = torch.zeros(batch_size, dtype=torch.long)
    else:
        point_nums = []
        padded_pcs = []
        padded_pc_masks = []
        for i, pc in enumerate(point_clouds_list):
            if pc is None:
                padded_pcs.append(torch.zeros(output_point_nums, 3, dtype=precision))
                point_nums.append(0)
            else:
                num_points = min(pc.shape[0], output_point_nums)
                pc_cut = pc[:num_points]
                point_nums.append(num_points)
                if num_points < output_point_nums:
                    padding = torch.zeros(output_point_nums - num_points, 3, dtype=pc_cut.dtype)
                    padded_pc = torch.cat([pc_cut, padding], dim=0)
                else:
                    padded_pc = pc_cut
                if padded_pc.dtype != precision:
                    padded_pc = padded_pc.to(precision)
                padded_pcs.append(padded_pc)

            pc_mask = pc_gt_masks[i] if i < len(pc_gt_masks) else None
            if pc_mask is None:
                padded_pc_masks.append(torch.zeros(output_point_nums, dtype=precision))
            else:
                num_mask_points = pc_mask.shape[0]
                if num_mask_points < output_point_nums:
                    mask_padding = torch.zeros(output_point_nums - num_mask_points, dtype=pc_mask.dtype)
                    padded_mask = torch.cat([pc_mask[:num_mask_points], mask_padding], dim=0)
                else:
                    padded_mask = pc_mask[:output_point_nums]
                if padded_mask.dtype != precision:
                    padded_mask = padded_mask.to(precision)
                padded_pc_masks.append(padded_mask)

        batch_out["point_clouds"] = torch.stack(padded_pcs)
        batch_out["pc_gt_tensor"] = torch.stack(padded_pc_masks)
        batch_out["pc_valid_lengths"] = torch.tensor(point_nums, dtype=torch.long)

    return batch_out

