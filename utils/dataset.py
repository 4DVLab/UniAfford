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
    router 架构下不再按 obj-aff 动态扩展 token；
    仅根据是否存在对应模态样本，决定是否保留通用占位 token。
    """
    return {
        "img": {"img_aff_token": "<img_aff>"},
        "pc": {"pc_aff_token": "<pc_aff>"},
    }


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
        use_simple_answer_template: bool = True,
    ):
        self.samples = samples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_size = image_size
        self.num_points = num_points
        self.mllm_precision = resolve_dtype(mllm_precision) or torch.bfloat16
        self.image_precision = resolve_dtype(image_precision) or torch.float32
        self.point_precision = resolve_dtype(point_precision) or torch.float32
        self.use_simple_answer_template = bool(use_simple_answer_template)
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

    def _build_simple_answer(
        self,
        obj_type: str,
        aff_type: str,
        has_image: bool,
        has_pc: bool,
        img_token: str,
        pc_token: str,
    ) -> str:
        """构造短回答模板。

        Args:
            obj_type: 物体类别名称，用于保留回答中的语义监督。
            aff_type: affordance 类别名称，用于保留回答中的语义监督。
            has_image: 当前样本是否包含 2D 图像输入。
            has_pc: 当前样本是否包含 3D 点云输入。
            img_token: 2D affordance 分支的占位 token。
            pc_token: 3D affordance 分支的占位 token。

        Returns:
            str: 固定且低自由度的 assistant answer。相比原丰富模板，它减少随机表达；
            相比纯 token 模板，它保留 obj/aff 语义，避免语言监督过弱。
        """
        if has_image and has_pc:
            return (
                f"In the 2D image, the {aff_type} affordance region of the {obj_type} is {img_token}; "
                f"in the 3D point cloud, it is {pc_token}."
            )
        if has_image:
            return f"In the 2D image, the {aff_type} affordance region of the {obj_type} is {img_token}."
        if has_pc:
            return f"In the 3D point cloud, the {aff_type} affordance region of the {obj_type} is {pc_token}."
        return "No visual input."

    def _build_text(self, obj_type, aff_type, has_image: bool, has_pc: bool, instruction: str) -> tuple[str, str]:
        """使用通用占位 token 构建回答，作为 router 的 token-level 监督标签。
        - 2D 分支占位：<img_aff>
        - 3D 分支占位：<pc_aff>
        不再在训练文本中使用 <img_obj_aff>/<pc_obj_aff> 这类按类别展开 token。"""
        img_token = "<img_aff>"
        pc_token = "<pc_aff>"
        if instruction:
            question = instruction
        else:
            if has_image and has_pc:
                question_templates = [
                    f"Using the provided 2D image and 3D point cloud, identify the {aff_type} affordance region of the {obj_type} in both modalities.",
                    f"Given both the image and point cloud of the {obj_type}, locate the {aff_type} affordance area in 2D and 3D.",
                    f"Please analyze the 2D image together with the 3D point cloud and mark the {aff_type} affordance region of the {obj_type}.",
                    f"From the available image and point cloud inputs, determine where the {obj_type} supports the {aff_type} affordance.",
                    f"Inspect both visual modalities for the {obj_type} and find the {aff_type} affordance region in the image and point cloud.",
                    f"Locate the {aff_type} affordance of the {obj_type} by using the supplied 2D image and 3D point cloud.",
                    f"With access to an image and a point cloud, identify the {aff_type} region on the {obj_type}.",
                    f"Show the {aff_type} affordance area of the {obj_type} for the 2D view and the 3D point cloud.",
                    f"Analyze the {obj_type} across its image and point cloud inputs, then indicate the {aff_type} affordance region.",
                    f"Use the paired 2D and 3D observations to find the {aff_type} affordance region of the {obj_type}.",
                    f"Based on the image plus the point cloud, where is the {aff_type} affordance area on the {obj_type}?",
                    f"For this {obj_type}, identify the {aff_type} affordance region using both the 2D visual input and the 3D point cloud.",
                    f"Determine the {aff_type} affordance location of the {obj_type} in the provided image and point cloud.",
                    f"Refer to the 2D image and the 3D point cloud to segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using all provided modalities, mark the {aff_type} affordance region associated with the {obj_type}.",
                    f"Find the {aff_type} interaction area of the {obj_type} in both the image plane and the point cloud.",
                    f"Examine the image and 3D points for the {obj_type}, then identify its {aff_type} affordance region.",
                    f"Combine the 2D image evidence with the 3D point cloud evidence to locate the {aff_type} affordance of the {obj_type}.",
                    f"Please return the {aff_type} affordance region for the {obj_type} using the available image and point cloud.",
                    f"Across the 2D and 3D inputs, indicate the area of the {obj_type} that affords {aff_type}.",
                    f"Identify where the {obj_type} can be used for {aff_type}, considering both the image and the point cloud.",
                    f"From the paired visual inputs, segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using the 2D view and 3D geometry, point out the {aff_type} affordance region on the {obj_type}.",
                    f"Evaluate the {obj_type} in the image and point cloud, and locate the region relevant to {aff_type}.",
                    f"Mark the {aff_type} affordance area for the {obj_type} with guidance from both available modalities.",
                    f"Please find the {obj_type}'s {aff_type} affordance region in the provided image and 3D point cloud.",
                    f"Rely on the image appearance and the point cloud structure to identify the {aff_type} region of the {obj_type}.",
                    f"Using multimodal input, locate the {aff_type} affordance area on the {obj_type}.",
                    f"Tell me which part of the {obj_type} corresponds to the {aff_type} affordance in 2D and 3D.",
                    f"Detect the {aff_type} affordance region of the {obj_type} from the supplied image-point-cloud pair.",
                ]
            elif has_image:
                question_templates = [
                    f"Using the provided 2D image, identify the {aff_type} affordance region of the {obj_type}.",
                    f"Given the image of the {obj_type}, locate the {aff_type} affordance area in the 2D view.",
                    f"Please analyze the image and mark the {aff_type} affordance region of the {obj_type}.",
                    f"From the available 2D visual input, determine where the {obj_type} supports the {aff_type} affordance.",
                    f"Inspect the image for the {obj_type} and find the visible {aff_type} affordance region.",
                    f"Locate the {aff_type} affordance of the {obj_type} using only the supplied image.",
                    f"With access to the 2D image, identify the {aff_type} area on the {obj_type}.",
                    f"Show the {aff_type} affordance region of the {obj_type} in the image.",
                    f"Analyze the {obj_type} in the provided picture, then indicate the {aff_type} affordance region.",
                    f"Use the 2D observation to find the {aff_type} affordance region of the {obj_type}.",
                    f"Based on this image, where is the {aff_type} affordance area on the {obj_type}?",
                    f"For this {obj_type}, identify the {aff_type} affordance region from the 2D visual input.",
                    f"Determine the image-space location of the {aff_type} affordance on the {obj_type}.",
                    f"Refer to the 2D image to segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using the provided image modality, mark the {aff_type} affordance region associated with the {obj_type}.",
                    f"Find the {aff_type} interaction area of the {obj_type} in the image plane.",
                    f"Examine the picture of the {obj_type}, then identify its {aff_type} affordance region.",
                    f"Use the image evidence to locate the {aff_type} affordance of the {obj_type}.",
                    f"Please return the 2D {aff_type} affordance region for the {obj_type}.",
                    f"In the image, indicate the area of the {obj_type} that affords {aff_type}.",
                    f"Identify where the {obj_type} can be used for {aff_type}, considering the 2D image.",
                    f"From the visual input, segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using the 2D view, point out the {aff_type} affordance region on the {obj_type}.",
                    f"Evaluate the image of the {obj_type} and locate the region relevant to {aff_type}.",
                    f"Mark the {aff_type} affordance area for the {obj_type} with guidance from the image.",
                    f"Please find the {obj_type}'s {aff_type} affordance region in the provided image.",
                    f"Rely on the image appearance to identify the {aff_type} region of the {obj_type}.",
                    f"Using the 2D input, locate the {aff_type} affordance area on the {obj_type}.",
                    f"Tell me which visible part of the {obj_type} corresponds to the {aff_type} affordance.",
                    f"Detect the {aff_type} affordance region of the {obj_type} from the supplied image.",
                ]
            elif has_pc:
                question_templates = [
                    f"Using the provided 3D point cloud, identify the {aff_type} affordance region of the {obj_type}.",
                    f"Given the point cloud of the {obj_type}, locate the {aff_type} affordance area in 3D.",
                    f"Please analyze the point cloud and mark the {aff_type} affordance region of the {obj_type}.",
                    f"From the available 3D input, determine where the {obj_type} supports the {aff_type} affordance.",
                    f"Inspect the point cloud for the {obj_type} and find the {aff_type} affordance region.",
                    f"Locate the {aff_type} affordance of the {obj_type} using only the supplied 3D point cloud.",
                    f"With access to the point cloud, identify the {aff_type} area on the {obj_type}.",
                    f"Show the {aff_type} affordance region of the {obj_type} in the 3D point cloud.",
                    f"Analyze the {obj_type} in the provided point cloud, then indicate the {aff_type} affordance region.",
                    f"Use the 3D observation to find the {aff_type} affordance region of the {obj_type}.",
                    f"Based on this point cloud, where is the {aff_type} affordance area on the {obj_type}?",
                    f"For this {obj_type}, identify the {aff_type} affordance region from the 3D point cloud input.",
                    f"Determine the point-cloud location of the {aff_type} affordance on the {obj_type}.",
                    f"Refer to the 3D point cloud to segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using the provided point-cloud modality, mark the {aff_type} affordance region associated with the {obj_type}.",
                    f"Find the {aff_type} interaction area of the {obj_type} in 3D space.",
                    f"Examine the point cloud of the {obj_type}, then identify its {aff_type} affordance region.",
                    f"Use the 3D geometric evidence to locate the {aff_type} affordance of the {obj_type}.",
                    f"Please return the 3D {aff_type} affordance region for the {obj_type}.",
                    f"In the point cloud, indicate the area of the {obj_type} that affords {aff_type}.",
                    f"Identify where the {obj_type} can be used for {aff_type}, considering the point cloud.",
                    f"From the 3D input, segment the {aff_type} affordance region of the {obj_type}.",
                    f"Using the point cloud, point out the {aff_type} affordance region on the {obj_type}.",
                    f"Evaluate the 3D structure of the {obj_type} and locate the region relevant to {aff_type}.",
                    f"Mark the {aff_type} affordance area for the {obj_type} with guidance from the point cloud.",
                    f"Please find the {obj_type}'s {aff_type} affordance region in the provided 3D point cloud.",
                    f"Rely on the point cloud geometry to identify the {aff_type} region of the {obj_type}.",
                    f"Using the 3D input, locate the {aff_type} affordance area on the {obj_type}.",
                    f"Tell me which 3D part of the {obj_type} corresponds to the {aff_type} affordance.",
                    f"Detect the {aff_type} affordance region of the {obj_type} from the supplied point cloud.",
                ]
            else:
                question_templates = [
                    f"Identify the {aff_type} affordance region of the {obj_type}, but note that no image or point cloud input is provided.",
                    f"Please locate the {aff_type} affordance area of the {obj_type}; no 2D or 3D modality is available.",
                    f"Determine where the {obj_type} supports {aff_type} without any provided visual input.",
                    f"Analyze the {obj_type} for its {aff_type} affordance region, although neither image nor point cloud data is present.",
                    f"Find the {aff_type} affordance of the {obj_type} with no supplied image or 3D point cloud.",
                    f"Report the {aff_type} affordance region for the {obj_type}; the sample contains no visual modality.",
                    f"Indicate the {aff_type} area on the {obj_type}, despite the absence of 2D and 3D inputs.",
                    f"Try to identify the {aff_type} affordance region of the {obj_type} when no visual data is available.",
                    f"Provide the {aff_type} affordance region of the {obj_type}, noting that both image and point cloud are missing.",
                    f"Locate the {aff_type} interaction area of the {obj_type} without access to image or point cloud evidence.",
                    f"Can you find the {aff_type} affordance area of the {obj_type} without visual input?",
                    f"For this {obj_type}, identify the {aff_type} region even though no 2D image or 3D point cloud is provided.",
                    f"Assess the {aff_type} affordance of the {obj_type} in a sample with no available modality data.",
                    f"Please mark the {aff_type} affordance of the {obj_type}; no image or point cloud accompanies the request.",
                    f"Use the request alone to address the {aff_type} affordance region of the {obj_type}.",
                    f"Specify the {aff_type} affordance area for the {obj_type} without any visual modality.",
                    f"Attempt to segment the {aff_type} affordance region of the {obj_type} while no input data is supplied.",
                    f"Describe where the {obj_type} affords {aff_type}, given that the visual inputs are unavailable.",
                    f"Return the {aff_type} affordance region for the {obj_type}; there is no 2D or 3D observation.",
                    f"Identify the part of the {obj_type} related to {aff_type}, but the image and point cloud are absent.",
                    f"Without image or point cloud information, determine the {aff_type} affordance region of the {obj_type}.",
                    f"Given no visual modality, locate the {aff_type} affordance region of the {obj_type}.",
                    f"Please infer the {aff_type} affordance area for the {obj_type} from the text request only.",
                    f"Find the region of the {obj_type} used for {aff_type}, noting that no 2D or 3D input exists.",
                    f"State the {aff_type} affordance region of the {obj_type} with no supporting visual evidence.",
                    f"Look for the {aff_type} affordance of the {obj_type}, although the sample lacks image and point cloud data.",
                    f"Tell me the {aff_type} affordance region of the {obj_type}; no modality input is attached.",
                    f"Resolve the {aff_type} affordance localization request for the {obj_type} without visual observations.",
                    f"Consider the {obj_type} and its {aff_type} affordance, but recognize that no image or point cloud is available.",
                    f"Handle this {aff_type} affordance query for the {obj_type} in the absence of visual data.",
                ]
            question = random.choice(question_templates)
        # 点云输入占位：使用单锚点 <pointcloud>，在 MLLM 前向时动态展开为 K_i 个点云 token。
        if has_pc:
            question = (
                f"{question}\n"
                f"Point cloud input: {DEFAULT_PC_TOKEN}"
            )
        # 默认启用短回答模板：训练目标只保留必要的任务 token，降低 generate 阶段格式漂移。
        if self.use_simple_answer_template:
            return question, self._build_simple_answer(obj_type, aff_type, has_image, has_pc, img_token, pc_token)

        answer_prefixes = [
            "",
            "",
            "Sure, ",
            "Certainly, ",
            "Yes, ",
            "Okay, ",
            "Alright, ",
            "Got it, ",
            "En, ",
            "In response, ",
            "Based on the input, ",
        ]

        def _with_answer_prefix(text: str) -> str:            
            def _lower_answer_start_after_prefix(text: str) -> str:
                if not text:
                    return text
                first_word = text.split(maxsplit=1)[0]
                # Keep first-person starts grammatical after prefixes, e.g. "Sure, I ..."
                if first_word in {"I", "I've", "I'd", "I'll", "I'm"}:
                    return text
                return text[0].lower() + text[1:]
            
            prefix = random.choice(answer_prefixes)
            if not prefix:
                return text
            return f"{prefix}{_lower_answer_start_after_prefix(text)}"

        answer_parts = []
        if has_image and has_pc:
            joint_templates = [
                f"In the 2D image, the {aff_type} affordance region of the {obj_type} is {img_token}; in the 3D point cloud, it is {pc_token}.",
                f"The {obj_type}'s {aff_type} affordance is marked as {img_token} for the image and {pc_token} for the point cloud.",
                f"For the {obj_type}, {img_token} denotes the 2D {aff_type} region, while {pc_token} denotes the matching 3D region.",
                f"Image modality: {img_token} identifies the {aff_type} area of the {obj_type}. Point-cloud modality: {pc_token} identifies it in 3D.",
                f"I located the {aff_type} affordance of the {obj_type} in both modalities: {img_token} in 2D and {pc_token} in 3D.",
                f"The image-space {aff_type} region for the {obj_type} is {img_token}, and the point-cloud region is {pc_token}.",
                f"Use {img_token} for the 2D {aff_type} affordance of the {obj_type} and {pc_token} for its 3D counterpart.",
                f"Across the paired inputs, the {aff_type} affordance of the {obj_type} appears as {img_token} in the image and {pc_token} in the point cloud.",
                f"For this {obj_type}, the visible 2D {aff_type} area is {img_token}, and the 3D {aff_type} area is {pc_token}.",
                f"The {aff_type} interaction region is separated by modality: image result {img_token}, point-cloud result {pc_token}.",
                f"Here are the {obj_type}'s {aff_type} affordance regions: {img_token} for the image and {pc_token} for the 3D points.",
                f"On the 2D input, {img_token} marks the {aff_type} region of the {obj_type}; on the 3D input, {pc_token} marks it.",
                f"The image token {img_token} highlights the {obj_type}'s {aff_type} affordance, and the point-cloud token {pc_token} highlights the 3D affordance.",
                f"In 2D, the {obj_type} region for {aff_type} is {img_token}; in 3D, the corresponding point-cloud region is {pc_token}.",
                f"Both modalities have been localized: {img_token} gives the image {aff_type} region, and {pc_token} gives the point-cloud region for the {obj_type}.",
                f"The {aff_type} affordance area of the {obj_type} is represented by {img_token} in the image and by {pc_token} in the point cloud.",
                f"Visual appearance points to {img_token} for the 2D {aff_type} region, while 3D geometry points to {pc_token} for the {obj_type}.",
                f"Result for the {obj_type}: 2D {aff_type} affordance = {img_token}; 3D {aff_type} affordance = {pc_token}.",
                f"The image prediction for the {aff_type} affordance is {img_token}, and the point-cloud prediction for the {obj_type} is {pc_token}.",
                f"Considering both inputs, I mark the {obj_type}'s {aff_type} region as {img_token} in 2D and {pc_token} in 3D.",
                f"The {aff_type} region can be found at {img_token} in the image modality and at {pc_token} in the point-cloud modality.",
                f"For the image, the {obj_type}'s {aff_type} affordance is {img_token}; for the 3D point cloud, it is {pc_token}.",
                f"Two modality-specific regions are returned for the {obj_type}: {img_token} for 2D {aff_type} and {pc_token} for 3D {aff_type}.",
                f"The 2D token {img_token} and the 3D token {pc_token} respectively indicate the {aff_type} affordance of the {obj_type}.",
                f"On the provided inputs, {img_token} marks the image area and {pc_token} marks the point-cloud area for the {obj_type}'s {aff_type} affordance.",
                f"Image evidence localizes the {aff_type} affordance of the {obj_type} to {img_token}; point-cloud evidence localizes it to {pc_token}.",
                f"The {obj_type} affords {aff_type} at {img_token} in the 2D view and at {pc_token} in the 3D point cloud.",
                f"Modality-specific answer: {img_token} is the 2D {aff_type} region, and {pc_token} is the 3D {aff_type} region for the {obj_type}.",
                f"I return {img_token} as the image affordance mask and {pc_token} as the point-cloud affordance mask for the {obj_type}'s {aff_type} function.",
                f"Detected regions for {aff_type} on the {obj_type}: {img_token} in image coordinates and {pc_token} in point-cloud coordinates.",
            ]
            answer_parts.append(_with_answer_prefix(random.choice(joint_templates)))
        elif has_image:
            image_templates = [
                f"The {aff_type} affordance region of the {obj_type} is {img_token}.",
                f"Here is the {aff_type} region of the {obj_type}: {img_token}.",
                f"For the {obj_type}, the {aff_type} area is highlighted as {img_token}.",
                f"I've identified the {aff_type} affordance of the {obj_type}: {img_token}.",
                f"On the {obj_type}, the region for {aff_type} interaction is {img_token}.",
                f"This token {img_token} marks the {aff_type} affordance of the {obj_type}.",
                f"In the 2D image, {img_token} marks the {aff_type} affordance region of the {obj_type}.",
                f"The image-based {aff_type} area for the {obj_type} is {img_token}.",
                f"Using the image modality, I locate the {obj_type}'s {aff_type} affordance at {img_token}.",
                f"The visible {aff_type} interaction region on the {obj_type} is {img_token}.",
                f"For the provided image, {img_token} indicates where the {obj_type} affords {aff_type}.",
                f"Image result: the {aff_type} affordance region of the {obj_type} is {img_token}.",
                f"The 2D affordance mask for {aff_type} on the {obj_type} is represented by {img_token}.",
                f"Within the image plane, the {obj_type}'s {aff_type} region corresponds to {img_token}.",
                f"I mark the image-space {aff_type} affordance of the {obj_type} as {img_token}.",
                f"The token {img_token} highlights the 2D {aff_type} affordance area of the {obj_type}.",
                f"On the visual input, the {aff_type} region for the {obj_type} is {img_token}.",
                f"The {obj_type} has its {aff_type} affordance localized in the image as {img_token}.",
                f"From the 2D view, the {aff_type} area of the {obj_type} is {img_token}.",
                f"This image-only answer uses {img_token} for the {obj_type}'s {aff_type} affordance.",
                f"Based on the image, {img_token} is the {aff_type} affordance region of the {obj_type}.",
                f"The requested 2D {aff_type} affordance region on the {obj_type} is {img_token}.",
                f"For image segmentation, {img_token} denotes the {aff_type} affordance of the {obj_type}.",
                f"The {aff_type} region visible on the {obj_type} is encoded as {img_token}.",
                f"Here, {img_token} is the image token for the {obj_type}'s {aff_type} affordance.",
                f"The {obj_type}'s image-level {aff_type} affordance is identified by {img_token}.",
                f"In the supplied 2D modality, the {aff_type} affordance is {img_token}.",
                f"The 2D region of the {obj_type} relevant to {aff_type} is {img_token}.",
                f"Visual appearance localizes the {aff_type} affordance of the {obj_type} to {img_token}.",
                f"Detected in the image, the {aff_type} affordance area of the {obj_type} is {img_token}.",
            ]
            answer_parts.append(_with_answer_prefix(random.choice(image_templates)))
        elif has_pc:
            pc_templates = [
                f"The 3D {aff_type} affordance region of the {obj_type} is {pc_token}.",
                f"In 3D space, the {aff_type} region of the {obj_type} is {pc_token}.",
                f"Within the point cloud, the {aff_type} area of the {obj_type} is {pc_token}.",
                f"This token {pc_token} represents the 3D {aff_type} affordance of the {obj_type}.",
                f"For the {obj_type}, the {aff_type} region in the point cloud is {pc_token}.",
                f"For 3D interaction with the {obj_type}, the {aff_type} area is {pc_token}.",
                f"Using the point-cloud modality, I locate the {obj_type}'s {aff_type} affordance at {pc_token}.",
                f"The point-cloud-based {aff_type} area for the {obj_type} is {pc_token}.",
                f"In the 3D input, {pc_token} marks the {aff_type} affordance region of the {obj_type}.",
                f"The geometric {aff_type} interaction region on the {obj_type} is {pc_token}.",
                f"For the provided point cloud, {pc_token} indicates where the {obj_type} affords {aff_type}.",
                f"Point-cloud result: the {aff_type} affordance region of the {obj_type} is {pc_token}.",
                f"The 3D affordance mask for {aff_type} on the {obj_type} is represented by {pc_token}.",
                f"Within the point-cloud coordinates, the {obj_type}'s {aff_type} region corresponds to {pc_token}.",
                f"I mark the 3D {aff_type} affordance of the {obj_type} as {pc_token}.",
                f"The token {pc_token} highlights the point-cloud {aff_type} affordance area of the {obj_type}.",
                f"On the 3D visual input, the {aff_type} region for the {obj_type} is {pc_token}.",
                f"The {obj_type} has its {aff_type} affordance localized in the point cloud as {pc_token}.",
                f"From the 3D view, the {aff_type} area of the {obj_type} is {pc_token}.",
                f"This point-cloud-only answer uses {pc_token} for the {obj_type}'s {aff_type} affordance.",
                f"Based on the point cloud, {pc_token} is the {aff_type} affordance region of the {obj_type}.",
                f"The requested 3D {aff_type} affordance region on the {obj_type} is {pc_token}.",
                f"For point-cloud segmentation, {pc_token} denotes the {aff_type} affordance of the {obj_type}.",
                f"The {aff_type} region in the 3D structure of the {obj_type} is encoded as {pc_token}.",
                f"Here, {pc_token} is the point-cloud token for the {obj_type}'s {aff_type} affordance.",
                f"The {obj_type}'s point-level {aff_type} affordance is identified by {pc_token}.",
                f"In the supplied 3D modality, the {aff_type} affordance is {pc_token}.",
                f"The 3D region of the {obj_type} relevant to {aff_type} is {pc_token}.",
                f"Geometric evidence localizes the {aff_type} affordance of the {obj_type} to {pc_token}.",
                f"Detected in the point cloud, the {aff_type} affordance area of the {obj_type} is {pc_token}.",
            ]
            answer_parts.append(_with_answer_prefix(random.choice(pc_templates)))
        else:
            no_input_templates = [
                f"I cannot identify the {aff_type} affordance region of the {obj_type} without visual input.",
                f"I need visual information to identify the {aff_type} affordance region of the {obj_type}.",
                f"Please provide an image or point cloud to analyze the {aff_type} affordance of the {obj_type}.",
                f"For the {obj_type}, visual input is required to determine the {aff_type} affordance region.",
                f"No 2D image or 3D point cloud is available, so the {aff_type} affordance region of the {obj_type} cannot be localized.",
                f"Without either modality, I cannot mark the {obj_type}'s {aff_type} affordance region.",
                f"The {aff_type} affordance of the {obj_type} requires image or point-cloud evidence, which is missing here.",
                f"There is no visual data to support locating the {aff_type} region of the {obj_type}.",
                f"I cannot produce an affordance token for the {obj_type}'s {aff_type} region because no modality input was provided.",
                f"Since both the image and point cloud are absent, the {aff_type} affordance area of the {obj_type} is unavailable.",
                f"The request cannot be grounded: no 2D or 3D input is present for the {obj_type}'s {aff_type} affordance.",
                f"Image and point-cloud inputs are both missing, so I cannot identify where the {obj_type} affords {aff_type}.",
                f"A valid {aff_type} affordance region for the {obj_type} cannot be determined without visual observations.",
                f"Because no image or point cloud accompanies the {obj_type}, I cannot localize its {aff_type} affordance.",
                f"The {aff_type} region of the {obj_type} is not identifiable from text alone in this sample.",
                f"With no modality data, there is no reliable region to assign for the {obj_type}'s {aff_type} affordance.",
                f"I would need a 2D image or 3D point cloud before marking the {aff_type} area of the {obj_type}.",
                f"The supplied sample lacks visual input, so the {aff_type} affordance of the {obj_type} cannot be segmented.",
                f"Cannot locate the {aff_type} affordance region of the {obj_type}: both image and point cloud are unavailable.",
                f"No image token or point-cloud token can be assigned for the {obj_type}'s {aff_type} affordance without input data.",
                f"The {obj_type}'s {aff_type} affordance region remains unknown because no 2D or 3D evidence is provided.",
                f"Please include an image or point cloud to localize the {aff_type} affordance region on the {obj_type}.",
                f"Absent visual modalities prevent identifying the {aff_type} interaction area of the {obj_type}.",
                f"The model has no visual basis for deciding the {aff_type} affordance region of the {obj_type}.",
                f"Neither the image modality nor the point-cloud modality is present, so the {aff_type} region cannot be marked.",
                f"I cannot return {img_token} or {pc_token} for the {obj_type}'s {aff_type} affordance because no visual input exists.",
                f"Localization of the {aff_type} affordance on the {obj_type} requires at least one provided modality.",
                f"The {aff_type} affordance query for the {obj_type} cannot be answered with a region when visual data is missing.",
                f"Since the sample has no image and no point cloud, the {obj_type}'s {aff_type} affordance area cannot be identified.",
                f"No modality-specific evidence is available to determine the {aff_type} affordance of the {obj_type}.",
            ]
            answer_parts.append(_with_answer_prefix(random.choice(no_input_templates)))
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
        result["data_source_id"] = data.get("data_source_id", {})

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
            pil_img = Image.new("RGB", (28, 28), color=(0, 0, 0))

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
        use_simple_answer_template: bool = True,
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
            use_simple_answer_template=use_simple_answer_template,
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
    sample_ids, obj_types, aff_types, data_source_ids = [], [], [], []
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
        data_source_ids.append(sample.get("data_source_id", {}))
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
        "data_source_id": data_source_ids,
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

