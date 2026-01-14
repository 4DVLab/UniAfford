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
LLaVA + LLaMA 模型实现
结合了 LLaVA 多模态架构和 LLaMA 语言模型
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    """LLaVA 配置类，继承自 LLaMA 配置"""
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """
    LLaVA + LLaMA 模型主体
    继承自 LlavaMetaModel（多模态支持）和 LlamaModel（语言模型）
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    LLaVA + LLaMA 因果语言模型
    继承自 LlamaForCausalLM（语言模型）和 LlavaMetaForCausalLM（多模态支持）
    """
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        # 创建模型主体
        self.model = LlavaLlamaModel(config)

        # 语言模型头：将隐藏状态映射到词汇表
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_model(self):
        """获取模型主体"""
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            past_key_values: 过去的键值对（用于生成）
            inputs_embeds: 输入嵌入（如果提供，将替代 input_ids）
            labels: 标签（用于计算损失）
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            images: 输入图像
            return_dict: 是否返回字典格式
            
        Returns:
            模型输出（包含损失、logits、隐藏状态等）
        """
        # 设置输出选项
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 准备多模态输入：将图像 token 替换为图像特征嵌入
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # 解码器输出包含 (dec_features, layer_state, dec_hidden, dec_attn)

        # 通过模型主体进行前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态并计算 logits
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 移位：token < n 预测 token n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平 token
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 启用模型/管道并行
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 根据 return_dict 决定返回格式
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 训练时返回所有隐藏状态，推理时只返回最后一层
        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        """
        为生成准备输入
        
        Args:
            input_ids: 输入 token IDs
            past_key_values: 过去的键值对
            attention_mask: 注意力掩码
            inputs_embeds: 输入嵌入
            images: 图像
            **kwargs: 其他参数
            
        Returns:
            准备好的模型输入字典
        """
        # 如果有 past_key_values，只需要最后一个 token
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 如果提供了 inputs_embeds，只在第一步生成时使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        """
        为生成准备输入
        
        Args:
            input_ids: 输入 token IDs
            past_key_values: 过去的键值对
            attention_mask: 注意力掩码
            inputs_embeds: 输入嵌入
            images: 图像
            **kwargs: 其他参数
            
        Returns:
            准备好的模型输入字典
        """
        # 如果有 past_key_values，只需要最后一个 token
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 如果提供了 inputs_embeds，只在第一步生成时使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
