#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""
LLaVA (Large Language and Vision Assistant) 架构实现
定义了多模态语言模型的基础架构，包括视觉编码器和多模态投影器
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower


class LlavaMetaModel:
    """
    LLaVA 元模型类
    定义了视觉编码器和多模态投影器的基本结构
    """
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # 如果配置中包含视觉编码器设置，则初始化
        if hasattr(config, "mm_vision_tower"):
            # 延迟加载视觉编码器（节省内存）
            self.vision_tower = build_vision_tower(config, delay_load=True)
            # 多模态投影器：将视觉特征投影到语言模型的隐藏空间
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        """
        获取视觉编码器
        
        Returns:
            视觉编码器对象
        """
        vision_tower = getattr(self, "vision_tower", None)
        # 如果是列表（FSDP 情况），返回第一个
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        初始化视觉模块（视觉编码器和多模态投影器）
        
        Args:
            model_args: 模型参数对象
            fsdp: FSDP（Fully Sharded Data Parallel）相关参数
        """
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer  # 选择哪一层的特征
        mm_vision_select_feature = model_args.mm_vision_select_feature  # 选择哪种特征（patch/cls）
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter  # 预训练的 MLP 适配器路径

        self.config.mm_vision_tower = vision_tower

        # 构建视觉编码器
        vision_tower = build_vision_tower(model_args)

        # FSDP 处理：将视觉编码器包装在列表中
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        # 配置多模态投影相关参数
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # 如果还没有投影器，创建一个
        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        # 如果提供了预训练的 MLP 适配器，加载权重
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                """从权重字典中提取特定关键字对应的权重"""
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    """
    LLaVA 因果语言模型抽象基类
    定义了多模态因果语言模型的基本接口
    """
    @abstractmethod
    def get_model(self):
        """获取模型主体（抽象方法，子类必须实现）"""
        pass

    def get_vision_tower(self):
        """获取视觉编码器"""
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        编码图像：使用视觉编码器提取特征，然后投影到语言模型空间
        
        Args:
            images: 输入图像张量
            
        Returns:
            投影后的图像特征
        """
        # 使用视觉编码器提取特征
        image_features = self.get_model().get_vision_tower()(images)
        # 使用多模态投影器将视觉特征投影到语言模型隐藏空间
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        """
        为多模态输入准备嵌入和标签
        将图像 token 替换为实际的图像特征嵌入
        
        Args:
            input_ids: 输入 token IDs，包含 IMAGE_TOKEN_INDEX 占位符
            attention_mask: 注意力掩码
            past_key_values: 过去的键值对（用于生成）
            labels: 标签（用于计算损失）
            images: 输入图像
            
        Returns:
            input_ids: None（已替换为嵌入）
            attention_mask: 更新后的注意力掩码
            past_key_values: 过去的键值对
            new_input_embeds: 新的输入嵌入（包含图像特征）
            new_labels: 新的标签（图像部分设为 IGNORE_INDEX）
        """
        vision_tower = self.get_vision_tower()
        # 如果没有视觉编码器、图像或序列长度为 1（生成模式），直接返回
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # 生成模式：更新注意力掩码
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        # 处理多个图像的情况
        if type(images) is list or images.ndim == 5:
            # 拼接所有图像
            concat_images = torch.cat([image for image in images], dim=0)
            # 编码图像特征
            image_features = self.encode_images(concat_images)
            # 按原始图像数量分割
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # 展平空间维度
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            # 单个图像：直接编码
            image_features = self.encode_images(images)

        # 构建新的输入嵌入和标签
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # 遍历批次中的每个样本
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 如果当前样本没有图像 token，直接使用文本嵌入
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # 多模态 LLM，但当前样本不是多模态的
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # 添加一个虚拟项以保持计算图（用于梯度流）
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # 找到所有图像 token 的位置
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # 处理所有图像 token
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                # 如果微调 MLP 适配器且使用图像开始/结束标记
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    # 添加图像特征嵌入
                    cur_new_input_embeds.append(cur_image_features)
                    # 添加图像结束标记嵌入
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        # 添加图像前的标签
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # 图像部分的标签设为 IGNORE_INDEX（不计算损失）
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        # 添加图像结束标记的标签
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                # 如果使用图像开始/结束标记（但不微调 MLP）
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                # 不使用图像开始/结束标记
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                # 更新 input_ids，移除已处理的图像 token
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                # 查找下一个图像 token
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # 处理剩余的文本 token
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            # 将嵌入移动到正确的设备并拼接
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # 如果批次中不同样本的长度不同，需要填充对齐
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                # 用零填充到最大长度
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
