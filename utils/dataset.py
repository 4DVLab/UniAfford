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
from utils.common import (
    resolve_dtype,
    FUNCTIONAL_TOKENS,
    IGNORE_INDEX,
    DEFAULT_PC_TOKEN,
)


def build_functional_tokens_from_samples(samples: List[JointDataSample]) -> Dict[str, Dict[str, str]]:
    """
    兼容保留：router 架构下不再按 obj-aff 动态扩展功能 token。
    仅返回通用占位 token，避免继续注入 <img_obj_aff>/<pc_obj_aff>。

    Returns:
        {
            "img": {"img_aff_token": "<img_aff>"},
            "pc":  {"pc_aff_token": "<pc_aff>"},
        }
    """
    _ = samples
    return {
        "img": {"img_aff_token": "<img_aff>"},
        "pc": {"pc_aff_token": "<pc_aff>"},
    }


def build_functional_tokens_from_sample_ids(sample_ids: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    从 split/sample_ids 结构构建功能 token（无需加载大样本到内存）。
    仅根据是否存在对应模态 id 判断是否注册 img/pc token。
    """
    token_map: Dict[str, Dict[str, str]] = {"img": {}, "pc": {}}
    ins_map = sample_ids.get("Instruction", sample_ids.get("ins", {})) or {}
    img_map = sample_ids.get("Image", sample_ids.get("img", {})) or {}
    pc_map = sample_ids.get("PointCloud", sample_ids.get("pc", {})) or {}
    all_obj = set(ins_map.keys()) | set(img_map.keys()) | set(pc_map.keys())
    for obj_type in all_obj:
        all_aff = set(ins_map.get(obj_type, {}).keys()) | set(img_map.get(obj_type, {}).keys()) | set(pc_map.get(obj_type, {}).keys())
        for aff_type in all_aff:
            pair_key = f"{obj_type}_{aff_type}"
            if len(img_map.get(obj_type, {}).get(aff_type, [])) > 0:
                img_name = f"img_{pair_key}"
                token_map["img"][img_name] = f"<{img_name}>"
            if len(pc_map.get(obj_type, {}).get(aff_type, [])) > 0:
                pc_name = f"pc_{pair_key}"
                token_map["pc"][pc_name] = f"<{pc_name}>"
    return token_map


class JointAffordanceTorchDataset(Dataset):
    """将 JointDataSample 预构建为 CPU 张量样本。"""

    def __init__(
        self,
        samples,
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
        # 懒加载源（如 JointDataset.lazy_load=True）下，禁止预缓存，避免初始化阶段全量读盘
        if getattr(samples, "lazy_load", False) and use_sample_cache:
            warnings.warn(
                "检测到 lazy_load 数据源，已自动禁用 use_sample_cache，避免初始化阶段全量加载。",
                UserWarning,
            )
            use_sample_cache = False
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

    def _build_text(self, obj_type, aff_type, has_image: bool, has_pc: bool, instruction: str) -> tuple[str, str]:
        """使用通用占位 token 构建回答，作为 router 的 token-level 监督标签。
        - 2D 分支占位：<img_aff>
        - 3D 分支占位：<pc_aff>
        不再在训练文本中使用 <img_obj_aff>/<pc_obj_aff> 这类按类别展开 token。"""
        img_token = "<img_aff>"
        pc_token = "<pc_aff>"
        question = instruction or f"Please identify the {aff_type} affordance region of the {obj_type}."
        # 点云输入占位：使用单锚点 <pointcloud>，在 MLLM 前向时动态展开为 K_i 个点云 token。
        if has_pc:
            question = (
                f"{question}\n"
                f"Point cloud input: {DEFAULT_PC_TOKEN}"
            )

        answer_parts = []
        if has_image:
            image_templates = [
                f"The {aff_type} affordance region of the {obj_type} is {img_token}.",
                f"Here is the {aff_type} region of the {obj_type}: {img_token}.",
                f"For the {obj_type}, the {aff_type} area is highlighted as {img_token}.",
                f"I've identified the {aff_type} affordance of the {obj_type}: {img_token}.",
                f"On the {obj_type}, the region for {aff_type} interaction is {img_token}.",
                f"This token {img_token} marks the {aff_type} affordance of the {obj_type}.",
            ]
            answer_parts.append(random.choice(image_templates))
        if has_pc:
            pc_templates = [
                f"The 3D {aff_type} affordance region of the {obj_type} is {pc_token}.",
                f"In 3D space, the {aff_type} region of the {obj_type} is {pc_token}.",
                f"Within the point cloud, the {aff_type} area of the {obj_type} is {pc_token}.",
                f"This token {pc_token} represents the 3D {aff_type} affordance of the {obj_type}.",
                f"For the {obj_type}, the {aff_type} region in the point cloud is {pc_token}.",
                f"For 3D interaction with the {obj_type}, the {aff_type} area is {pc_token}.",
            ]
            answer_parts.append(random.choice(pc_templates))
        if not answer_parts:
            no_input_templates = [
                f"I cannot identify the {aff_type} affordance region of the {obj_type} without visual input.",
                f"I need visual information to identify the {aff_type} affordance region of the {obj_type}.",
                f"Please provide an image or point cloud to analyze the {aff_type} affordance of the {obj_type}.",
                f"For the {obj_type}, visual input is required to determine the {aff_type} affordance region.",
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

        # 不依赖硬编码 token id，按 chat template 真实边界匹配 assistant 监督区间：
        # <|im_start|>assistant\n ... <|im_end|>
        input_ids_flat = input_ids[0].tolist()
        assistant_prefix = self.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        im_end = self.tokenizer.encode(
            "<|im_end|>\n",
            add_special_tokens=False,
        )

        def _find_subseq(seq, pat, start=0):
            if not pat:
                return -1
            end = len(seq) - len(pat) + 1
            for i in range(max(0, start), max(0, end)):
                if seq[i : i + len(pat)] == pat:
                    return i
            return -1

        pos = 0
        while True:
            prefix_pos = _find_subseq(input_ids_flat, assistant_prefix, start=pos)
            if prefix_pos < 0:
                break
            ans_start = prefix_pos + len(assistant_prefix)
            end_pos = _find_subseq(input_ids_flat, im_end, start=ans_start)
            if end_pos < 0:
                # 模板异常时至少监督 assistant 内容尾部，避免全 -100
                labels[0, ans_start:] = input_ids[0, ans_start:]
                break

            ans_end = end_pos + len(im_end)
            labels[0, ans_start:ans_end] = input_ids[0, ans_start:ans_end]
            pos = ans_end

        full_result["labels"] = labels
        full_result["input_ids"] = input_ids

        # grid_thw = full_result.get("image_grid_thw")
        # if grid_thw is None:
        #     cat_grid_thw = None
        # elif isinstance(grid_thw, (list, tuple)):
        #     cat_grid_thw = torch.cat(grid_thw, dim=0)
        # else:
        #     cat_grid_thw = grid_thw

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

    def _build_sample(self, sample: Any) -> Dict[str, Any]:
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
        is_joint_sample = hasattr(sample, "get_data")
        data = sample.get_data() if is_joint_sample else sample
        has_image = data["img"] is not None
        has_pc = data["pc"] is not None

        result = {}
        # 供验证/可视化阶段直接使用的样本级元信息
        obj_type = sample.obj_type if is_joint_sample else data.get("obj_type")
        aff_type = sample.aff_type if is_joint_sample else data.get("aff_type")
        sample_id = sample.id if is_joint_sample else data.get("sample_id", data.get("index", -1))
        result["sample_id"] = sample_id
        result["obj_type"] = obj_type
        result["aff_type"] = aff_type

        question, answer = self._build_text(obj_type, aff_type, has_image, has_pc, data.get("ins") or "")

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
            # 无图样本也使用与训练一致的分辨率，避免同一 batch 内视觉 token 长度不一致
            pil_img = Image.new("RGB", (self.image_size[0], self.image_size[1]), color=(0, 0, 0))

        result.update(self._build_qwen_inputs(question, answer, pil_img))

        # 2D 输入（CPU + image_precision）
        if has_image:
            img = data["img"]
            orig_h, orig_w = int(img.shape[0]), int(img.shape[1])
            result["original_size"] = (orig_h, orig_w)
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
            "labels"               : [B, L]           # 语言建模标签（不需要预测的位置为 IGNORE_INDEX）
            "attention_mask"       : [B, L]           # 文本注意力，指示每个token是否为有效输入（即不是pad的token）
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
    sample_ids, obj_types, aff_types = [], [], []
    original_size_per_sample = []
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
        sample_ids.append(sample.get("sample_id"))
        obj_types.append(sample.get("obj_type"))
        aff_types.append(sample.get("aff_type"))
        original_size_per_sample.append(sample.get("original_size"))

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
        "sample_id": sample_ids,
        "obj_type": obj_types,
        "aff_type": aff_types,
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
            orig_size = original_size_per_sample[i] if i < len(original_size_per_sample) else None
            if orig_size is not None and len(orig_size) >= 2:
                original_size_list.append((int(orig_size[0]), int(orig_size[1])))
            elif mask is not None:
                original_size_list.append((mask.shape[0], mask.shape[1]))
            else:
                original_size_list.append((max_h, max_w))
            if mask is None:
                padded_masks.append(torch.zeros(max_h, max_w, dtype=image_precision))
            else:
                mask_h, mask_w = mask.shape[0], mask.shape[1]
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

