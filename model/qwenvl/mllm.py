from typing import Optional, Dict, Tuple
from pathlib import Path
from transformers import AutoProcessor
import torch
import torch.nn as nn
from configs import MLLMConfigs


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
    ) -> dict:
        # 构建 Qwen 模型输入，自动过滤 None 值
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
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

