from typing import Optional, Dict, Tuple
from pathlib import Path
from transformers import AutoProcessor
import torch
import torch.nn as nn
from configs import MLLMConfigs
from model.pointcept import PointCloudPrefixEncoder
from utils.common import IGNORE_INDEX


class MLLMBackbone(nn.Module):
    """MLLM 主干实现（Qwen3-VL）。"""

    def __init__(self, config: MLLMConfigs):
        super().__init__()
        self.config = config
        self.model = self._build_qwen_model(config)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.vocab_size = self.model.config.text_config.vocab_size

        if self.config.hidden_size != self.hidden_size:
            print(f"Warning: hidden_size mismatch, config={self.config.hidden_size}, model={self.hidden_size}")
            self.config.hidden_size = self.hidden_size

        # 统一使用 AutoProcessor（包含 tokenizer + image_processor），
        # 避免 tokenizer 和 processor 分开创建导致 special token 不同步。
        self.processor = AutoProcessor.from_pretrained(
            self.config.qwen_model_name_or_path,
        )
        self.processor.tokenizer.padding_side = "right"
        self.tokenizer = self.processor.tokenizer

        self.functional_tokens, token_modality = self._normalize_functional_tokens(
            self.config.functional_tokens
        )
        self.functional_token_ids = self._ensure_special_tokens(
            self.functional_tokens, token_modality
        )

        # 特殊 token 注入后，词表大小可能变化，这里以模型实际词表为准回写配置。
        self.vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        if self.config.vocab_size != self.vocab_size:
            print(f"Warning: vocab_size mismatch, config={self.config.vocab_size}, model={self.vocab_size}")
            self.config.vocab_size = self.vocab_size

        self.point_prefix_encoder = None
        if getattr(self.config, "enable_pc_prefix", False):
            self.point_prefix_encoder = PointCloudPrefixEncoder(
                out_hidden_size=self.hidden_size,
                compute_dtype=self.config.compute_dtype,
                backbone_kwargs=getattr(self.config, "point_prefix_backbone_kwargs", None),
            )

        self.to(dtype=self.config.compute_dtype)

    def _normalize_functional_tokens(self, candidate_tokens: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        兼容两种输入：
        1) 扁平映射：token_name -> token_str
        2) 分模态映射：{"img": {...}, "pc": {...}}（内部可含正反向，函数只提取 name->token_str）
        """
        flat: Dict[str, str] = {}
        token_modality: Dict[str, str] = {}

        if isinstance(candidate_tokens, dict) and "img" in candidate_tokens and "pc" in candidate_tokens:
            for modality in ("img", "pc"):
                sub = candidate_tokens.get(modality, {})
                if not isinstance(sub, dict):
                    continue
                for token_name, token in sub.items():
                    if isinstance(token_name, str) and isinstance(token, str) and token.startswith("<") and token.endswith(">"):
                        flat[token_name] = token
                        token_modality[token_name] = modality
        else:
            for token_name, token in candidate_tokens.items():
                if isinstance(token_name, str) and isinstance(token, str) and token.startswith("<") and token.endswith(">"):
                    flat[token_name] = token
                    token_modality[token_name] = "img" if token_name.lower().startswith("img_") or token_name == "img_aff_token" else "pc"

        return flat, token_modality

    def _ensure_special_tokens(self, candidate_tokens: Dict[str, str], token_modality: Dict[str, str]):
        """
        确保配置中的分割 token 已加入 tokenizer，并与 MLLM embedding 对齐。
        若 special token 已被占用（如已预训练好的模型、继续微调），则直接使用其现有 id，无需重复添加。
        """
        tokens_to_add = []
        # Qwen tokenizer unknown token id
        unk_id = self.tokenizer.unk_token_id

        for _, token in candidate_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            # token 不存在或被识别为 unk，视为需要添加
            if token_id is None or (unk_id is not None and token_id == unk_id):
                tokens_to_add.append(token)
            # 如果token已存在，认为是兼容已预训练好的模型，无需报错也无需重新添加，兼容性处理

        # 只有需要添加时才扩充tokenizer和embedding
        if len(tokens_to_add) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            self.model.resize_token_embeddings(len(self.tokenizer))
            # NOTE: embedding resize 可能会导致新 token embedding 随机初始化。
            # 若继续微调，建议加入相关 embedding warmup 策略。

        # 必须在 add_special_tokens 之后重新读取 token_id，避免返回旧值/unk_id
        # 双向映射并按模态分组：
        # {
        #   "img": {token_name: token_id, token_id: token_name},
        #   "pc":  {token_name: token_id, token_id: token_name},
        # }
        functional_token_ids = {"img": {}, "pc": {}}
        id_to_token_info = dict()
        for token_name, token in candidate_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or (unk_id is not None and token_id == unk_id):
                raise ValueError(f"功能 token 注册失败: name={token_name}, token={token}")
            tid = int(token_id)
            modality = token_modality.get(token_name, "img" if token_name.lower().startswith("img_") or token_name == "img_aff_token" else "pc")
            functional_token_ids[modality][token_name] = tid
            functional_token_ids[modality][tid] = token_name
            id_to_token_info[tid] = {"name": token_name, "token": token, "modality": modality}
        self.id_to_token_info = id_to_token_info
        return functional_token_ids

    def _build_qwen_model(self, config: MLLMConfigs):
        model_name = config.qwen_model_name_or_path
        model_name_lower = model_name.lower()
        dtype = config.compute_dtype

        # choose QwenVL version
        if "qwen3" in model_name_lower and "a" in Path(model_name.rstrip("/")).name.lower():
            from transformers import Qwen3VLMoeForConditionalGeneration
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        elif "qwen3" in model_name_lower:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        elif "qwen2.5" in model_name_lower:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation=config.qwen_attn_implementation,
                dtype=dtype,
            )

        # 二次兜底：确保参数实际 dtype 与训练配置一致，避免被预训练权重默认 dtype 影响。
        if dtype is not None:
            model = model.to(dtype=dtype)

        model.config.use_cache = False
        
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        # 统一约束：始终走 inputs_embeds 路径，保证多模态缺失/齐全时前向形式一致。
        # 注意：Qwen-VL 在存在 image token 时仍需要 input_ids 参与占位匹配与 RoPE 计算。
        if input_ids is None:
            raise ValueError("MLLMBackbone.forward 统一使用 inputs_embeds 模式时，input_ids 不能为空。")

        # 1) 基础文本 embedding（无论是否有点云，都会构建）
        #    这一步确保即便纯文本/文本+图像样本，也和点云样本共享同一前向入口。
        token_embeds = self.model.get_input_embeddings()(input_ids)

        # 2) attention_mask 兜底（若外部未提供）
        #    - 有 pad_token_id: 以非 pad 位置为有效 token
        #    - 无 pad_token_id: 退化为全 1
        if attention_mask is None:
            if self.tokenizer.pad_token_id is not None:
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # 默认不加前缀：保持 inputs_embeds 与 input_ids 对齐
        final_input_ids = input_ids
        final_inputs_embeds = token_embeds
        final_attention_mask = attention_mask
        final_labels = labels

        # 3) 点云前缀（可选）：若编码器可用且传入点云，则把前缀拼到序列前面。
        #    即使某个样本无有效点云，prefix_mask 会把该样本前缀位标记为无效（attention=0）。
        if self.point_prefix_encoder is not None and point_clouds is not None:
            prefix_embeds, prefix_mask = self.point_prefix_encoder(
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
            )
            if prefix_embeds is not None and prefix_mask is not None:
                # 对齐 dtype，避免 cat 时精度不一致。
                token_embeds = token_embeds.to(dtype=prefix_embeds.dtype)
                final_inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

                # attention_mask 前缀位由 prefix_mask 控制（有效前缀=1，无效填充前缀=0）
                prefix_attn = prefix_mask.to(dtype=attention_mask.dtype)
                final_attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)

                # labels 前缀位统一忽略，不参与语言建模 CE。
                if labels is not None:
                    prefix_labels = torch.full(
                        (labels.shape[0], prefix_embeds.shape[1]),
                        IGNORE_INDEX,
                        device=labels.device,
                        dtype=labels.dtype,
                    )
                    final_labels = torch.cat([prefix_labels, labels], dim=1)

                # Qwen-VL 仍依赖 input_ids 识别图像占位 token。
                # 前缀对应位置填充为 pad/eos，不会命中功能 token 与 image token。
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                prefix_ids = torch.full(
                    (input_ids.shape[0], prefix_embeds.shape[1]),
                    int(pad_id),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                final_input_ids = torch.cat([prefix_ids, input_ids], dim=1)

        # 4) 统一封装模型输入：无论点云是否缺失，都会显式提供 inputs_embeds。
        model_inputs = {
            "input_ids": final_input_ids,
            "inputs_embeds": final_inputs_embeds,
            "attention_mask": final_attention_mask,
            "labels": final_labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        model_inputs["output_hidden_states"] = True
        model_inputs["return_dict"] = True
        outputs = self.model(**model_inputs)

        # TODO: check outputs
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # 确保输出为 [B, L, C]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        return {"hidden_states": hidden_states, "output": outputs}

