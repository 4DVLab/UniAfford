"""JointAffordanceModel 的 PyTorch Dataset 适配层。"""

import random
import warnings
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model.qwenvl.data.rope2d import get_rope_index_3
from utils.base_dataset import JointDataSample
from utils.common import resolve_dtype, SEG_TOKEN, AFF_TOKEN, IGNORE_INDEX


def _pad_and_cat_position_ids(position_ids_list: List[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in position_ids_list)
    padded_tensors = []
    for tensor in position_ids_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded_tensors, dim=1)


class JointAffordanceTorchDataset(Dataset):
    """将 JointDataSample 预构建为 CPU 张量样本。"""

    def __init__(
        self,
        samples: List[JointDataSample],
        processor,
        image_size=(1024, 1024),
        num_points: int = 2048,
        mllm_precision="bf16",
        image_precision="fp32",
        point_precision="fp32",
        use_sample_cache: bool = True,
    ):
        self.samples = samples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_size = image_size
        self.num_points = num_points
        self.mllm_precision = resolve_dtype(mllm_precision) or torch.bfloat16
        self.image_precision = resolve_dtype(image_precision) or torch.float32
        self.point_precision = resolve_dtype(point_precision) or torch.float32
        self.use_sample_cache = use_sample_cache
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self._sample_cache = {}

        if self.use_sample_cache:
            for idx, sample in enumerate(self.samples):
                self._sample_cache[idx] = self._build_sample(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.use_sample_cache:
            return self._sample_cache[index]
        return self._build_sample(self.samples[index])

    def _build_text(self, sample: JointDataSample, has_image: bool, has_pc: bool, instruction: str) -> tuple[str, str]:
        obj_type = sample.obj_type
        aff_type = sample.aff_type
        question = instruction or f"Please identify the {aff_type} affordance region of the {obj_type}."

        answer_parts = []
        if has_image:
            image_templates = [
                f"The {aff_type} affordance region of the {obj_type} is {SEG_TOKEN}.",
                f"Here is the {aff_type} region: {SEG_TOKEN}.",
                f"The {aff_type} area for the {obj_type} is highlighted as {SEG_TOKEN}.",
                f"I've identified the {aff_type} affordance: {SEG_TOKEN}.",
                f"The region for {aff_type} interaction is {SEG_TOKEN}.",
                f"{SEG_TOKEN} shows the {aff_type} affordance of the {obj_type}.",
            ]
            answer_parts.append(random.choice(image_templates))
        if has_pc:
            pc_templates = [
                f"The 3D {aff_type} affordance region is {SEG_TOKEN}.",
                f"In 3D space, the {aff_type} region is {SEG_TOKEN}.",
                f"The point cloud shows the {aff_type} area as {SEG_TOKEN}.",
                f"{SEG_TOKEN} represents the 3D {aff_type} affordance.",
                f"The {aff_type} region in the point cloud is {SEG_TOKEN}.",
                f"For 3D interaction, the {aff_type} area is {SEG_TOKEN}.",
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

        return question, answer

    def _build_qwen_inputs(self, question: str, answer: str, pil_img: Image.Image) -> Dict[str, Any]:
        """
        构建单样本 Qwen 输入，并统一返回 CPU 张量。

        Returns:
            {
                "input_ids":                文本 token ids（含图像 token）                  [1, L] int64
                "labels":                   语言建模标签（非监督位为 QWEN_IGNORE_INDEX）     [1, L] int64
                "position_ids":             Qwen3-VL RoPE 位置编码                          [1, 3, L] int64
                "pixel_values (可选)":      Qwen3-VL 视觉输入                               [N, C, H, W] mllm_precision
                "image_grid_thw (可选)":    Qwen3-VL 图像网格 (T, H, W)                     [N, 3] int64
            }
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        full_result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)

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

        full_result["labels"] = labels
        full_result["input_ids"] = input_ids

        grid_thw = full_result.get("image_grid_thw")
        if grid_thw is None:
            cat_grid_thw = None
        elif isinstance(grid_thw, (list, tuple)):
            cat_grid_thw = torch.cat(grid_thw, dim=0)
        else:
            cat_grid_thw = grid_thw

        # position_ids, _ = get_rope_index_3(
        #     self.merge_size,
        #     input_ids,
        #     image_grid_thw=cat_grid_thw,
        #     video_grid_thw=None,
        #     second_per_grid_ts=None,
        # )

        # 固定输出：文本 token、监督标签、RoPE 位置编码
        out = {
            "input_ids": input_ids.cpu(),
            "labels": labels.cpu(),
            # "position_ids": position_ids.cpu(),
        }
        # 可选输出：视觉分支输入（由 processor 决定是否返回）
        if "pixel_values" in full_result:
            out["pixel_values"] = full_result["pixel_values"].to(dtype=self.mllm_precision).cpu()
        if "image_grid_thw" in full_result:
            grid = full_result["image_grid_thw"]
            out["image_grid_thw"] = [g.cpu() for g in grid] if isinstance(grid, (list, tuple)) else grid.cpu()
        return out

    def _build_sample(self, sample: JointDataSample) -> Dict[str, Any]:
        """
        构建 JointAffordanceModel 的单样本输入，并统一返回 CPU 张量。

        Returns:
            {
                "input_ids":                 文本 token ids（含图像 token）                  [1, L] int64
                "labels":                    语言建模标签（非监督位为 QWEN_IGNORE_INDEX）     [1, L] int64
                "position_ids":              Qwen3-VL RoPE 位置编码                          [1, 3, L] int64
                "pixel_values (可选)":       Qwen3-VL 视觉输入                               [N, C, H, W] mllm_precision
                "image_grid_thw (可选)":     Qwen3-VL 图像网格 (T, H, W)                     [N, 3] int64

                "images (可选)":             2D 分支输入图像                                 [3, H, W] image_precision
                "img_gt (可选)":             2D 分支监督掩码                                 [H, W] image_precision
                
                "point_clouds (可选)":       3D 分支输入点云                                 [N, 3] point_precision
                "pc_gt (可选)":              3D 分支监督掩码                                 [N] point_precision
            }
        """
        data = sample.get_data()
        has_image = data["img"] is not None
        has_pc = data["pc"] is not None

        result = {}

        question, answer = self._build_text(sample, has_image, has_pc, data.get("ins") or "")

        # ZeRO-3 兼容：始终给 Qwen 提供一张图片
        if has_image:
            # 为了抑制 Qwen 视觉 token 的批间波动，Qwen 输入也统一到固定尺寸。
            # 否则原图分辨率差异过大时，某些 batch 会触发显存峰值。
            qwen_img = data["img"]
            if qwen_img.shape[:2] != (self.image_size[0], self.image_size[1]):
                qwen_img = cv2.resize(qwen_img, (self.image_size[0], self.image_size[1]))
            qwen_img_rgb = cv2.cvtColor(qwen_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(qwen_img_rgb)
        else:
            pil_img = Image.new("RGB", (28, 28), color=(0, 0, 0))

        result.update(self._build_qwen_inputs(question, answer, pil_img))

        # 2D 输入（CPU + image_precision）
        if has_image:
            img = data["img"]
            if img.shape[:2] != (self.image_size[0], self.image_size[1]):
                img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=self.image_precision).div_(255.0).contiguous()
            result["images"] = img_tensor.cpu()

            if data["img_gt"] is not None:
                mask = data["img_gt"]
                if mask.shape[:2] != (self.image_size[0], self.image_size[1]):
                    mask = cv2.resize(mask, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(mask).to(dtype=self.image_precision).contiguous()
                if mask_tensor.max() > 1:
                    mask_tensor.div_(255.0)
                result["img_gt"] = mask_tensor.cpu()

        # 3D 输入（CPU + point_precision）
        if has_pc:
            pts = data["pc"]
            if pts.ndim != 2 or pts.shape[1] != 3:
                pts = pts.reshape(-1, 3)
            n = pts.shape[0]
            idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
            sampled_pts = pts[idx]
            pc_tensor = torch.from_numpy(sampled_pts).to(dtype=self.point_precision)
            pc_tensor = pc_tensor - pc_tensor.mean(dim=0)
            md = torch.norm(pc_tensor, dim=1).max()
            if md > 0:
                pc_tensor = pc_tensor / md
            result["point_clouds"] = pc_tensor.contiguous().cpu()

            if data["pc_gt"] is not None:
                pc_gt = data["pc_gt"]
                if pc_gt.ndim == 2 and pc_gt.shape[1] == 1:
                    pc_gt = pc_gt[:, 0]
                pm = pc_gt[idx]
                if pm.max() > 1:
                    pm = pm / 255.0
                result["pc_gt"] = torch.from_numpy(pm).to(dtype=self.point_precision).contiguous().cpu()

        return result


class JointAffordanceTrainDataset(JointAffordanceTorchDataset):
    """训练阶段随机采样 epoch 子集。"""

    def __init__(
        self,
        samples: List[JointDataSample],
        processor,
        image_size=(1024, 1024),
        num_points: int = 2048,
        samples_per_epoch: int = 10000,
        mllm_precision="bf16",
        image_precision="fp32",
        point_precision="fp32",
        use_sample_cache: bool = True,
    ):
        super().__init__(
            samples=samples,
            processor=processor,
            image_size=image_size,
            num_points=num_points,
            mllm_precision=mllm_precision,
            image_precision=image_precision,
            point_precision=point_precision,
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
        return super().__getitem__(actual_index)


def joint_affordance_collate_fn(
    batch: List[Dict],
    tokenizer=None,
    output_image_size=(1024, 1024),
    output_point_nums=2048,
    # 这里的精度控制填充和占位张量和实际数据类型保持一致
    mllm_precision="bf16",
    image_precision="fp32",
    point_precision="fp32",
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
    mllm_precision = resolve_dtype(mllm_precision) or torch.bfloat16
    image_precision = resolve_dtype(image_precision) or torch.float32
    point_precision = resolve_dtype(point_precision) or torch.float32

    # -------- 1) 收集 batch 内各模态原始数据 --------
    batch_size = len(batch)

    input_ids_list, labels_list = [], []
    pixel_values_list, image_grid_thw_list = [], []
    pixel_values_valid_flags = []

    images_list, img_gt_masks = [], []
    point_clouds_list, pc_gt_masks = [], []
    for sample in batch:
        # 文本与 Qwen3-VL 位置编码
        input_ids_list.append(sample["input_ids"].squeeze(0))
        labels_list.append(sample["labels"].squeeze(0))
        # position_ids_list.append(sample["position_ids"])

        # Qwen3-VL 视觉输入（可能缺失，后续统一回填）
        pixel_values = sample.get("pixel_values")
        image_grid_thw = sample.get("image_grid_thw")
        pixel_values_list.append(pixel_values)
        image_grid_thw_list.append(image_grid_thw)
        pixel_values_valid_flags.append(pixel_values is not None and image_grid_thw is not None)

        # 2D/3D 分割输入
        images_list.append(sample.get("images"))
        img_gt_masks.append(sample.get("img_gt"))
        point_clouds_list.append(sample.get("point_clouds"))
        pc_gt_masks.append(sample.get("pc_gt"))

    # -------- 2) 文本输入 padding --------
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    batch_out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
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
        dummy_h, dummy_w = output_image_size
        warnings.warn(
            "joint_affordance_collate_fn: pixel_values 全为空，已使用占位图回填。"
            "请检查数据集/预处理逻辑。",
            RuntimeWarning,
        )
        fallback_pixel_values = torch.zeros(
            1, 3, dummy_h, dummy_w, dtype=mllm_precision
        )
        fallback_grid_thw = torch.ones(1, 3, dtype=torch.long)

    fixed_pixel_values = [
        (pv if pv is not None else fallback_pixel_values)
        for pv in pixel_values_list
    ]
    fixed_grid_thw = []
    for g in image_grid_thw_list:
        if g is None:
            fixed_grid_thw.append(fallback_grid_thw)
        elif isinstance(g, (list, tuple)):
            fixed_grid_thw.append(torch.cat(g, dim=0))
        else:
            fixed_grid_thw.append(g)
    batch_out["pixel_values"] = torch.cat(fixed_pixel_values, dim=0)
    batch_out["image_grid_thw"] = torch.cat(fixed_grid_thw, dim=0)

    # -------- 4) 2D 图像与 GT padding --------
    valid_images = [img for img in images_list if img is not None]
    if len(valid_images) == 0:
        dummy_h, dummy_w = output_image_size
        batch_out["images"] = torch.zeros(batch_size, 3, dummy_h, dummy_w, dtype=image_precision)
        batch_out["img_gt_tensor"] = torch.zeros(batch_size, dummy_h, dummy_w, dtype=image_precision)
        batch_out["original_size_list"] = [(dummy_h, dummy_w)] * batch_size
    else:
        max_h, max_w = output_image_size
        padded_images = []
        padded_masks = []
        original_size_list = []

        for i, img in enumerate(images_list):
            if img is None:
                padded_images.append(torch.zeros(3, max_h, max_w, dtype=image_precision))
            else:
                _, h, w = img.shape
                if h < max_h or w < max_w:
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
                else:
                    padded_img = img
                if padded_img.dtype != image_precision:
                    padded_img = padded_img.to(image_precision)
                padded_images.append(padded_img)

            mask = img_gt_masks[i] if i < len(img_gt_masks) else None
            if mask is None:
                padded_masks.append(torch.zeros(max_h, max_w, dtype=image_precision))
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
                if padded_mask.dtype != image_precision:
                    padded_mask = padded_mask.to(image_precision)
                padded_masks.append(padded_mask)

        stacked_images = torch.stack(padded_images)
        batch_out["images"] = stacked_images
        batch_out["img_gt_tensor"] = torch.stack(padded_masks)
        batch_out["original_size_list"] = original_size_list

    # 2D 有效样本标记（供上游屏蔽无效样本梯度）
    batch_out["img_valid_mask"] = torch.tensor(
        [img is not None for img in images_list], dtype=torch.bool
    )

    # -------- 5) 3D 点云与 GT padding --------
    valid_pcs = [pc for pc in point_clouds_list if pc is not None]
    if len(valid_pcs) == 0:
        batch_out["point_clouds"] = torch.zeros(batch_size, output_point_nums, 3, dtype=point_precision)
        batch_out["pc_gt_tensor"] = torch.zeros(batch_size, output_point_nums, dtype=point_precision)
        batch_out["pc_valid_lengths"] = torch.zeros(batch_size, dtype=torch.long)
    else:
        point_nums = []
        padded_pcs = []
        padded_pc_masks = []
        for i, pc in enumerate(point_clouds_list):
            if pc is None:
                padded_pcs.append(torch.zeros(output_point_nums, 3, dtype=point_precision))
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
                if padded_pc.dtype != point_precision:
                    padded_pc = padded_pc.to(point_precision)
                padded_pcs.append(padded_pc)

            pc_mask = pc_gt_masks[i] if i < len(pc_gt_masks) else None
            if pc_mask is None:
                padded_pc_masks.append(torch.zeros(output_point_nums, dtype=point_precision))
            else:
                num_mask_points = pc_mask.shape[0]
                if num_mask_points < output_point_nums:
                    mask_padding = torch.zeros(output_point_nums - num_mask_points, dtype=pc_mask.dtype)
                    padded_mask = torch.cat([pc_mask[:num_mask_points], mask_padding], dim=0)
                else:
                    padded_mask = pc_mask[:output_point_nums]
                if padded_mask.dtype != point_precision:
                    padded_mask = padded_mask.to(point_precision)
                padded_pc_masks.append(padded_mask)

        batch_out["point_clouds"] = torch.stack(padded_pcs)
        batch_out["pc_gt_tensor"] = torch.stack(padded_pc_masks)
        batch_out["pc_valid_lengths"] = torch.tensor(point_nums, dtype=torch.long)

    return batch_out


__all__ = [
    "JointAffordanceTorchDataset",
    "JointAffordanceTrainDataset",
    "joint_affordance_collate_fn",
]

