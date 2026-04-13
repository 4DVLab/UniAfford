from typing import Optional, Dict, Tuple
from pathlib import Path
from transformers import AutoProcessor
import torch
import torch.nn as nn
from configs import MLLMConfigs
from model.pointcept import PointCloudEncoder
from utils.common import (
    IGNORE_INDEX,
    DEFAULT_PC_TOKEN,
    DEFAULT_PC_PATCH_TOKEN,
    DEFAULT_PC_START_TOKEN,
    DEFAULT_PC_END_TOKEN,
)


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
        self._ensure_pointcloud_tokens()

        # 特殊 token 注入后，词表大小可能变化，这里以模型实际词表为准回写配置。
        self.vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        if self.config.vocab_size != self.vocab_size:
            print(f"Warning: vocab_size mismatch, config={self.config.vocab_size}, model={self.vocab_size}")
            self.config.vocab_size = self.vocab_size

        self.point_encoder = None
        if getattr(self.config, "enable_point_encoder", False):
            point_encoder_ckpt = getattr(self.config, "point_encoder_pretrained", None)
            point_encoder_cfg = getattr(self.config, "point_encoder_pretrained_config", None)
            if point_encoder_ckpt:
                self.point_encoder = PointCloudEncoder.from_pretrained(
                    checkpoint_path=point_encoder_ckpt,
                    out_hidden_size=self.hidden_size,
                    compute_dtype=self.config.compute_dtype,
                    backbone_config=getattr(self.config, "point_encoder_backbone", None),
                    pretrained_config_path=point_encoder_cfg,
                )
            else:
                self.point_encoder = PointCloudEncoder(
                    out_hidden_size=self.hidden_size,
                    compute_dtype=self.config.compute_dtype,
                    backbone_config=getattr(self.config, "point_encoder_backbone", None),
                )
        self.pc_anchor_token_id = self._resolve_token_id(DEFAULT_PC_TOKEN)
        self.pc_patch_token_id = self._resolve_token_id(DEFAULT_PC_PATCH_TOKEN)

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

    def _ensure_pointcloud_tokens(self):
        """
        注册点云输入占位 token（与 Qwen 视觉 token 风格一致），
        便于在 input_ids 中定位并注入点云 embedding。
        """
        candidates = [
            DEFAULT_PC_TOKEN,
            DEFAULT_PC_PATCH_TOKEN,
            DEFAULT_PC_START_TOKEN,
            DEFAULT_PC_END_TOKEN,
        ]
        unk_id = self.tokenizer.unk_token_id
        tokens_to_add = []
        for tok in candidates:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None or (unk_id is not None and int(tid) == int(unk_id)):
                tokens_to_add.append(tok)
        if tokens_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _resolve_token_id(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or (unk_id is not None and int(tid) == int(unk_id)):
            raise ValueError(f"token 未注册: {token}")
        return int(tid)

    def _inject_pointcloud_embeddings(
        self,
        input_ids: torch.Tensor,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        point_clouds: Optional[torch.Tensor],
        pc_valid_lengths: Optional[torch.Tensor],
        point_token_embeds: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        token-to-token 对齐注入（仿照 Qwen 视觉位点替换思路）：
        - 文本中使用 <pointcloud> 作为锚点
        - 前向时将该锚点替换为点云编码输出的 K_i 个 token embedding（每样本可变长）
        - 同步扩展 input_ids/attention_mask/labels，保持与 inputs_embeds 严格对齐
        """
        if point_token_embeds is not None and point_token_mask is not None:
            mllm_point_tokens, mllm_point_token_mask = point_token_embeds, point_token_mask
        else:
            if self.point_encoder is None or point_clouds is None:
                return input_ids, token_embeds, attention_mask, labels
            mllm_point_tokens, mllm_point_token_mask = self.point_encoder(
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
            )
        if mllm_point_tokens is None or mllm_point_token_mask is None:
            return input_ids, token_embeds, attention_mask, labels

        B, L, C = token_embeds.shape
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0
        anchor_id = int(self.pc_anchor_token_id)
        patch_id = int(self.pc_patch_token_id)

        seq_ids_list = []
        seq_emb_list = []
        seq_attn_list = []
        seq_lbl_list = [] if labels is not None else None
        out_lens = []

        for i in range(B):
            cur_ids = input_ids[i]
            cur_emb = token_embeds[i]
            cur_attn = attention_mask[i]
            cur_lbl = labels[i] if labels is not None else None

            anchor_pos = (cur_ids == anchor_id).nonzero(as_tuple=False).view(-1)
            valid_pc = True
            if pc_valid_lengths is not None:
                valid_pc = bool(pc_valid_lengths[i].item() > 0)

            if anchor_pos.numel() == 0 or not valid_pc:
                seq_ids = cur_ids
                seq_emb = cur_emb
                seq_attn = cur_attn
                seq_lbl = cur_lbl
            else:
                # 仅替换第一个锚点，避免模板里重复锚点带来歧义
                pos = int(anchor_pos[0].item())
                valid_k = int(mllm_point_token_mask[i].sum().item())
                if valid_k <= 0:
                    seq_ids = cur_ids
                    seq_emb = cur_emb
                    seq_attn = cur_attn
                    seq_lbl = cur_lbl
                else:
                    pc_tok = mllm_point_tokens[i, :valid_k].to(dtype=cur_emb.dtype)
                    pc_ids = torch.full((valid_k,), patch_id, dtype=cur_ids.dtype, device=cur_ids.device)
                    pc_attn = torch.ones((valid_k,), dtype=cur_attn.dtype, device=cur_attn.device)
                    if cur_lbl is not None:
                        pc_lbl = torch.full((valid_k,), IGNORE_INDEX, dtype=cur_lbl.dtype, device=cur_lbl.device)

                    seq_ids = torch.cat([cur_ids[:pos], pc_ids, cur_ids[pos + 1 :]], dim=0)
                    seq_emb = torch.cat([cur_emb[:pos], pc_tok, cur_emb[pos + 1 :]], dim=0)
                    seq_attn = torch.cat([cur_attn[:pos], pc_attn, cur_attn[pos + 1 :]], dim=0)
                    if cur_lbl is not None:
                        seq_lbl = torch.cat([cur_lbl[:pos], pc_lbl, cur_lbl[pos + 1 :]], dim=0)
                    else:
                        seq_lbl = None

            seq_ids_list.append(seq_ids)
            seq_emb_list.append(seq_emb)
            seq_attn_list.append(seq_attn)
            if seq_lbl_list is not None:
                seq_lbl_list.append(seq_lbl)
            out_lens.append(int(seq_ids.shape[0]))

        max_len = max(out_lens) if out_lens else L
        out_ids = input_ids.new_full((B, max_len), pad_id)
        # 关键：padding 位的 embedding 使用“真实 pad token embedding”，
        # 避免在 Qwen get_placeholder_mask(input_ids=None) 中被误判为 image token。
        pad_embed = self.model.get_input_embeddings()(
            torch.tensor([pad_id], device=token_embeds.device, dtype=input_ids.dtype)
        )[0].to(dtype=token_embeds.dtype)
        out_emb = pad_embed.view(1, 1, C).expand(B, max_len, C).clone()
        out_attn = attention_mask.new_zeros((B, max_len))
        out_lbl = None
        if labels is not None:
            out_lbl = labels.new_full((B, max_len), IGNORE_INDEX)

        for i in range(B):
            cur_len = seq_ids_list[i].shape[0]
            out_ids[i, :cur_len] = seq_ids_list[i]
            out_emb[i, :cur_len] = seq_emb_list[i]
            out_attn[i, :cur_len] = seq_attn_list[i]
            if out_lbl is not None and seq_lbl_list is not None:
                out_lbl[i, :cur_len] = seq_lbl_list[i]

        return out_ids, out_emb, out_attn, out_lbl

    def _compute_multimodal_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        使用 Qwen-VL 内部 get_rope_index 计算位置编码。

        说明：
        - 为了满足“仅 inputs_embeds 前向”的要求，这里在外部先计算好 position_ids，
          然后传给 self.model(...)，避免内部在缺少 input_ids 时退化为纯文本位置策略。
        - 这样可以保留图像 token 的 3D 视觉位置建模（time/height/width）。
        """
        vl_core = getattr(self.model, "model", None)
        if vl_core is None or not hasattr(vl_core, "get_rope_index"):
            return None

        try:
            # Qwen2-VL / Qwen3-VL 接口（不含 second_per_grid_ts）
            position_ids, rope_deltas = vl_core.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        except TypeError:
            # Qwen2.5-VL 接口（含 second_per_grid_ts）
            position_ids, rope_deltas = vl_core.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            )
            

        # 与官方实现保持一致：缓存 rope_deltas 供后续增量推理使用。
        if hasattr(vl_core, "rope_deltas"):
            vl_core.rope_deltas = rope_deltas
        return position_ids

    def _sanitize_image_placeholder_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        在 inputs_embeds 路径下，Qwen 会通过“向量相等”识别 image token。
        该函数用 input_ids 作为真值，消除非 image 位点被误判成 image token 的情况。
        """
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            return inputs_embeds

        image_token_id = int(image_token_id)
        expected_mask = input_ids.eq(image_token_id)  # [B, L]
        image_embed = self.model.get_input_embeddings()(
            torch.tensor([image_token_id], device=inputs_embeds.device, dtype=input_ids.dtype)
        )[0].to(dtype=inputs_embeds.dtype)
        observed_mask = inputs_embeds.eq(image_embed.view(1, 1, -1)).all(dim=-1)  # [B, L]

        extra_mask = observed_mask & (~expected_mask)
        if not extra_mask.any():
            return inputs_embeds

        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0
        pad_embed = self.model.get_input_embeddings()(
            torch.tensor([pad_id], device=inputs_embeds.device, dtype=input_ids.dtype)
        )[0].to(dtype=inputs_embeds.dtype)
        return torch.where(extra_mask.unsqueeze(-1), pad_embed.view(1, 1, -1), inputs_embeds)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        point_token_embeds: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        # 统一约束：始终走 inputs_embeds 路径，保证多模态缺失/齐全时前向形式一致。
        # 与之前不同：不再把 input_ids 直接传入 self.model(...)，而是仅用于外部构造 embedding 与位置编码。
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

        # 默认保持 inputs_embeds 与 input_ids 序列对齐
        final_input_ids = input_ids
        final_inputs_embeds = token_embeds
        final_attention_mask = attention_mask
        final_labels = labels

        # 3) 点云输入注入：<pointcloud> 锚点 -> K_i 个 <|point_pad|> 对齐 token embedding。
        final_input_ids, final_inputs_embeds, final_attention_mask, final_labels = self._inject_pointcloud_embeddings(
            input_ids=final_input_ids,
            token_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
            point_token_embeds=point_token_embeds,
            point_token_mask=point_token_mask,
        )
        # 对齐校正：确保 image placeholder 计数以 input_ids 为准，避免 embeddings 等值误判。
        final_inputs_embeds = self._sanitize_image_placeholder_embeddings(
            input_ids=final_input_ids,
            inputs_embeds=final_inputs_embeds,
        )

        # 4) 计算多模态 position_ids（保留视觉 RoPE 建模）。
        position_ids = self._compute_multimodal_position_ids(
            input_ids=final_input_ids,
            attention_mask=final_attention_mask,
            image_grid_thw=image_grid_thw,
        )

        # 5) 统一封装模型输入：只传 inputs_embeds，不传 input_ids。
        model_inputs = {
            "inputs_embeds": final_inputs_embeds,
            "attention_mask": final_attention_mask,
            "position_ids": position_ids,
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
        return {
            "hidden_states": hidden_states,
            "output": outputs,
            "aligned_labels": final_labels,
            "aligned_attention_mask": final_attention_mask,
        }

