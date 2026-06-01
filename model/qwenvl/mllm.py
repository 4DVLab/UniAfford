from typing import Optional, Dict, Tuple, Any
from collections import OrderedDict
from pathlib import Path
import tempfile
from transformers import AutoConfig, AutoProcessor
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
        self.processor = self._build_processor(config)
        self.processor.tokenizer.padding_side = "right"
        self.tokenizer = self.processor.tokenizer

        self.task_placeholder_tokens = self._build_task_placeholder_tokens(config)
        self.functional_tokens, token_modality = self._normalize_functional_tokens(self.config.functional_tokens)
        self._merge_task_placeholders(self.functional_tokens, token_modality)
        self.functional_token_ids = self._ensure_special_tokens(
            self.functional_tokens, token_modality
        )
        self._ensure_pointcloud_tokens()
        self.task_placeholder_ids = self._resolve_task_placeholder_ids(self.task_placeholder_tokens)
        self.task_id_by_name = {name: idx for idx, name in enumerate(self.task_placeholder_tokens.keys())}
        self.task_name_by_id = {idx: name for name, idx in self.task_id_by_name.items()}
        self.placeholder_id_to_task_id = {
            token_id: self.task_id_by_name[name]
            for name, token_id in self.task_placeholder_ids.items()
        }
        self.non_text_placeholder_ids = {
            token_id
            for name, token_id in self.task_placeholder_ids.items()
            if name != "text"
        }
        self.text_token = self.task_placeholder_tokens.get("text", "<text>")
        self.text_token_id = self.task_placeholder_ids.get("text")
        self.latent_token = self.task_placeholder_tokens.get("latent")
        self.latent_token_id = self.task_placeholder_ids.get("latent")

        # 特殊 token 注入后，词表大小可能变化，这里以模型实际词表为准回写配置。
        self.vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        if self.config.vocab_size != self.vocab_size:
            print(f"Warning: vocab_size mismatch, config={self.config.vocab_size}, model={self.vocab_size}")
            self.config.vocab_size = self.vocab_size

        self.point_encoder = None
        if getattr(self.config, "enable_point_encoder", False):
            point_encoder_ckpt = getattr(self.config, "point_encoder_pretrained", None)
            point_encoder_cfg = getattr(self.config, "point_encoder_pretrained_config", None)
            if getattr(self.config, "restore_from_checkpoint", False):
                self.point_encoder = PointCloudEncoder(
                    out_hidden_size=self.hidden_size,
                    compute_dtype=self.config.compute_dtype,
                    backbone_config=getattr(self.config, "point_encoder_backbone", None),
                )
            elif point_encoder_ckpt:
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
            self._sync_point_encoder_config()
        self.pc_anchor_token_id = self._resolve_token_id(DEFAULT_PC_TOKEN)
        self.pc_patch_token_id = self._resolve_token_id(DEFAULT_PC_PATCH_TOKEN)

        self.to(dtype=self.config.compute_dtype)

    def _sync_point_encoder_config(self):
        point_encoder = getattr(self, "point_encoder", None)
        backbone_cfg = getattr(self.config, "point_encoder_backbone", None)
        if point_encoder is None or backbone_cfg is None:
            return

        if hasattr(backbone_cfg, "update") and hasattr(point_encoder, "backbone_config_dict"):
            backbone_cfg.update(point_encoder.backbone_config_dict)
        elif hasattr(point_encoder, "backbone_config_dict"):
            self.config.point_encoder_backbone = point_encoder.backbone_config_dict

        pretrained_info = getattr(point_encoder, "pretrained_info", None)
        if pretrained_info and pretrained_info.get("config_path"):
            self.config.point_encoder_pretrained_config = pretrained_info["config_path"]

    @staticmethod
    def _write_temp_assets(root_dir: str, file_map: Optional[Dict[str, bytes]]) -> None:
        for rel_path, content in (file_map or {}).items():
            file_path = Path(root_dir) / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

    def _build_processor(self, config: MLLMConfigs):
        serialized_files = getattr(config, "serialized_processor_files", None)
        if getattr(config, "restore_from_checkpoint", False) and serialized_files:
            with tempfile.TemporaryDirectory(prefix="ja_processor_restore_") as tmpdir:
                self._write_temp_assets(tmpdir, serialized_files)
                return AutoProcessor.from_pretrained(tmpdir, local_files_only=True)
        return AutoProcessor.from_pretrained(
            self.config.qwen_model_name_or_path,
        )

    @staticmethod
    def _resolve_qwen_model_class(class_name: Optional[str], model_name: str):
        model_name_lower = (model_name or "").lower()
        normalized = (class_name or "").strip()
        if normalized == "Qwen3VLMoeForConditionalGeneration":
            from transformers import Qwen3VLMoeForConditionalGeneration
            return Qwen3VLMoeForConditionalGeneration
        if normalized == "Qwen3VLForConditionalGeneration":
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        if normalized == "Qwen2_5_VLForConditionalGeneration":
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
        if normalized == "Qwen2VLForConditionalGeneration":
            from transformers import Qwen2VLForConditionalGeneration
            return Qwen2VLForConditionalGeneration

        if "qwen3" in model_name_lower and "a" in Path(model_name.rstrip("/")).name.lower():
            from transformers import Qwen3VLMoeForConditionalGeneration
            return Qwen3VLMoeForConditionalGeneration
        if "qwen3" in model_name_lower:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        if "qwen2.5" in model_name_lower:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration

    def _build_qwen_from_serialized_config(self, config: MLLMConfigs):
        serialized_files = getattr(config, "serialized_model_config_files", None)
        if not serialized_files:
            return None

        with tempfile.TemporaryDirectory(prefix="ja_qwen_restore_") as tmpdir:
            self._write_temp_assets(tmpdir, serialized_files)
            config_obj = AutoConfig.from_pretrained(tmpdir, local_files_only=True)
            if config.qwen_attn_implementation is not None:
                setattr(config_obj, "_attn_implementation", config.qwen_attn_implementation)
                setattr(config_obj, "attn_implementation", config.qwen_attn_implementation)
            model_cls = self._resolve_qwen_model_class(
                getattr(config, "serialized_model_class_name", None),
                getattr(config, "qwen_model_name_or_path", ""),
            )
            return model_cls(config_obj)

    def _normalize_functional_tokens(self, candidate_tokens: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        只接受显式分任务映射：{"task": {token_name: token_str}}。
        任务归属由外层 key 决定，不再从 token_name 字符串推断。
        """
        flat: Dict[str, str] = {}
        token_modality: Dict[str, str] = {}

        if not isinstance(candidate_tokens, dict):
            return flat, token_modality
        for task_name, sub in candidate_tokens.items():
            if not isinstance(sub, dict):
                raise ValueError("functional_tokens 必须使用 {'task': {'token_name': '<token>'}} 结构。")
            for token_name, token in sub.items():
                if isinstance(token_name, str) and isinstance(token, str) and token.startswith("<") and token.endswith(">"):
                    flat[token_name] = token
                    token_modality[token_name] = task_name

        return flat, token_modality

    def _build_task_placeholder_tokens(self, config: MLLMConfigs) -> "OrderedDict[str, str]":
        raw = getattr(config, "task_placeholders", None)
        if raw is None:
            raise ValueError("MLLMConfigs.task_placeholders 不能为空，需显式声明任务 placeholder。")

        ordered: "OrderedDict[str, str]" = OrderedDict()
        if "text" not in raw:
            ordered["text"] = "<text>"
        for task_name, token in raw.items():
            if not isinstance(task_name, str) or not task_name:
                raise ValueError(f"任务 placeholder 名称必须是非空字符串，当前为: {task_name}")
            if not isinstance(token, str) or not token.startswith("<") or not token.endswith(">"):
                raise ValueError(f"{task_name} placeholder 必须是形如 <...> 的 token，当前为: {token}")
            ordered[task_name] = token
        if "text" in raw:
            ordered.move_to_end("text", last=False)
        return ordered

    def _merge_task_placeholders(
        self,
        functional_tokens: Dict[str, str],
        token_modality: Dict[str, str],
    ) -> None:
        for task_name, token in self.task_placeholder_tokens.items():
            token_name = f"{task_name}_token"
            functional_tokens[token_name] = token
            token_modality[token_name] = task_name

    def _resolve_task_placeholder_ids(self, task_placeholders: Dict[str, str]) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for task_name, token in task_placeholders.items():
            ids[task_name] = self._resolve_token_id(token)
        return ids

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
        functional_token_ids = {task_name: {} for task_name in self.task_placeholder_tokens.keys()}
        id_to_token_info = dict()
        for token_name, token in candidate_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or (unk_id is not None and token_id == unk_id):
                raise ValueError(f"功能 token 注册失败: name={token_name}, token={token}")
            tid = int(token_id)
            if token_name not in token_modality:
                raise ValueError(f"功能 token 缺少显式任务归属: {token_name}")
            modality = token_modality[token_name]
            if modality not in functional_token_ids:
                raise ValueError(f"功能 token 任务未在 task_placeholders 中声明: {modality}")
            functional_token_ids[modality][token_name] = tid
            functional_token_ids[modality][tid] = token_name
            id_to_token_info[tid] = {"name": token_name, "token": token, "modality": modality}
        self.id_to_token_info = id_to_token_info
        return functional_token_ids

    def _build_qwen_model(self, config: MLLMConfigs):
        model_name = config.qwen_model_name_or_path
        dtype = config.compute_dtype
        restore_mode = bool(getattr(config, "restore_from_checkpoint", False))

        model = None
        if restore_mode:
            model = self._build_qwen_from_serialized_config(config)
        if model is None:
            model_cls = self._resolve_qwen_model_class(
                getattr(config, "serialized_model_class_name", None),
                model_name,
            )
            model = model_cls.from_pretrained(
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

    def _resolve_optional_token_id(self, token: Optional[str]) -> Optional[int]:
        if not token:
            return None
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or (unk_id is not None and int(tid) == int(unk_id)):
            return None
        return int(tid)

    def _validate_qwen_hidden_states(
        self,
        hidden_states: Any,
        *,
        source: str,
        expected_batch: Optional[int] = None,
        expected_seq_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if hidden_states.dim() != 3:
            return None
        if hidden_states.shape[-1] != int(self.hidden_size):
            return None
        if expected_batch is not None and hidden_states.shape[0] != expected_batch:
            return None
        if expected_seq_len is not None and hidden_states.shape[1] != expected_seq_len:
            return None
        return hidden_states

    def _extract_qwen_hidden_states(
        self,
        outputs: Any,
        model_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        expected_batch = model_inputs["inputs_embeds"].shape[0]
        expected_seq_len = model_inputs["inputs_embeds"].shape[1]

        hidden_stack = getattr(outputs, "hidden_states", None)
        if hidden_stack is not None and len(hidden_stack) > 0:
            hidden_states = self._validate_qwen_hidden_states(
                hidden_stack[-1],
                source="outputs.hidden_states[-1]",
                expected_batch=expected_batch,
                expected_seq_len=expected_seq_len,
            )
            if hidden_states is not None:
                return hidden_states

        for source, candidate in (
            ("outputs.last_hidden_state", getattr(outputs, "last_hidden_state", None)),
            ("outputs[0]", outputs[0] if isinstance(outputs, (tuple, list)) and len(outputs) > 0 else None),
        ):
            hidden_states = self._validate_qwen_hidden_states(
                candidate,
                source=source,
                expected_batch=expected_batch,
                expected_seq_len=expected_seq_len,
            )
            if hidden_states is not None:
                return hidden_states

        fallback = self._fallback_qwen_core_hidden_states(model_inputs)
        if fallback is not None:
            return fallback

        logits = getattr(outputs, "logits", None)
        logits_shape = None if logits is None else tuple(logits.shape)
        raise RuntimeError(
            "无法从 Qwen 输出中提取合法 hidden states；已拒绝使用可能是 logits 的 fallback。"
            f" expected_hidden_size={self.hidden_size}, logits_shape={logits_shape}"
        )

    def _fallback_qwen_core_hidden_states(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        qwen_core = getattr(self.model, "model", None)
        if qwen_core is None:
            return None
        core_inputs = {
            k: v
            for k, v in model_inputs.items()
            if k not in {"labels", "return_dict"}
        }
        core_inputs["output_hidden_states"] = True
        core_inputs["return_dict"] = True
        try:
            core_outputs = qwen_core(**core_inputs)
        except Exception:
            return None
        hidden_stack = getattr(core_outputs, "hidden_states", None)
        candidate = hidden_stack[-1] if hidden_stack is not None and len(hidden_stack) > 0 else getattr(core_outputs, "last_hidden_state", None)
        return self._validate_qwen_hidden_states(
            candidate,
            source="qwen_core.hidden_states[-1]",
            expected_batch=model_inputs["inputs_embeds"].shape[0],
            expected_seq_len=model_inputs["inputs_embeds"].shape[1],
        )

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
        hidden_states = self._extract_qwen_hidden_states(outputs, model_inputs)
        return {
            "hidden_states": hidden_states,
            "output": outputs,
            "aligned_labels": final_labels,
            "aligned_attention_mask": final_attention_mask,
        }

    def autoregressive_forward_with_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        point_token_embeds: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        router: Optional[nn.Module] = None,
        generation_config: Optional[Any] = None,
    ) -> dict:
        """
        一阶段自回归 latent feedback 入口。

        router 判为 text 的位置沿用 LM logits -> token id -> embedding；其它任务位置直接把
        route 前 hidden state 作为下一步输入 embedding。
        生成长度和停止条件沿用 Qwen 的 generation_config（max_length/eos_token_id）。
        """
        if input_ids is None:
            raise ValueError("autoregressive_forward_with_latents 需要 input_ids。")

        if attention_mask is None:
            if self.tokenizer.pad_token_id is not None:
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        attention_mask = attention_mask.clone()
        generated_ids, inputs_embeds, attention_mask, aligned_labels = self._inject_pointcloud_embeddings(
            input_ids=input_ids,
            token_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
            point_token_embeds=point_token_embeds,
            point_token_mask=point_token_mask,
        )
        hidden_history = []
        logits_history = []
        route_history = []

        full_ids = generated_ids
        full_embeds = inputs_embeds
        full_attention = attention_mask
        if aligned_labels is not None:
            supervised = aligned_labels.ne(IGNORE_INDEX)
            has_supervision = supervised.any(dim=1)
            if bool(has_supervision.any().item()):
                first_supervised = torch.where(
                    has_supervision,
                    supervised.float().argmax(dim=1).to(dtype=torch.long),
                    torch.full_like(has_supervision.to(dtype=torch.long), aligned_labels.shape[1] - 1),
                )
                prefix_len = int(first_supervised.max().item())
            else:
                prefix_len = max(1, int(full_ids.shape[1]) - 1)
            prefix_len = max(1, min(prefix_len, int(full_ids.shape[1]) - 1))
            generated_ids = full_ids[:, :prefix_len]
            inputs_embeds = full_embeds[:, :prefix_len]
            attention_mask = full_attention[:, :prefix_len]
            target_length = int(aligned_labels.shape[1])
            eos_token_ids = None
            pad_token_id = 0
            unfinished = torch.ones((generated_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
        else:
            generation_config = generation_config or getattr(self.model, "generation_config", None)
            target_length = int(getattr(generation_config, "max_length", getattr(self.config, "model_max_length", 512)))
            target_length = max(target_length, int(generated_ids.shape[1]) + 1)
            eos_token_id = getattr(generation_config, "eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
            pad_token_id = getattr(generation_config, "pad_token_id", getattr(self.tokenizer, "pad_token_id", None))
            if isinstance(eos_token_id, int):
                eos_token_ids = torch.tensor([eos_token_id], device=input_ids.device, dtype=input_ids.dtype)
            elif eos_token_id is None:
                eos_token_ids = None
            else:
                eos_token_ids = torch.tensor(list(eos_token_id), device=input_ids.device, dtype=input_ids.dtype)
            pad_token_id = int(pad_token_id) if pad_token_id is not None else 0
            unfinished = torch.ones((generated_ids.shape[0],), dtype=torch.bool, device=input_ids.device)

        
        def _detach_past_key_values(past_key_values: Any) -> Any:
            """递归 detach KV cache，避免训练时 cache 挂住历史 step 的计算图。"""
            if past_key_values is None:
                return None
            if isinstance(past_key_values, torch.Tensor):
                return past_key_values.detach()
            if isinstance(past_key_values, tuple):
                return tuple(self._detach_past_key_values(item) for item in past_key_values)
            if isinstance(past_key_values, list):
                return [self._detach_past_key_values(item) for item in past_key_values]
            if isinstance(past_key_values, dict):
                return {key: self._detach_past_key_values(value) for key, value in past_key_values.items()}
            if hasattr(past_key_values, "detach"):
                return past_key_values.detach()
            return past_key_values

        def _qwen_core_forward(model_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Any]:
            """调用 Qwen backbone 获取最后层 hidden state，再手动过 lm_head 得到 logits。"""
            qwen_core = getattr(self.model, "model", None)
            if qwen_core is None:
                raise RuntimeError("当前 Qwen 模型缺少 .model backbone，无法执行 core forward。")
            core_inputs = {
                k: v
                for k, v in model_inputs.items()
                if k not in {"labels", "logits_to_keep"}
            }
            core_inputs["return_dict"] = True
            core_outputs = qwen_core(**core_inputs)
            hidden_states = getattr(core_outputs, "last_hidden_state", None)
            hidden_states = self._validate_qwen_hidden_states(
                hidden_states,
                source="qwen_core.last_hidden_state",
                expected_batch=model_inputs["inputs_embeds"].shape[0],
                expected_seq_len=model_inputs["inputs_embeds"].shape[1],
            )
            if hidden_states is None:
                raise RuntimeError("Qwen core forward 未返回合法 last_hidden_state。")
            logits = self.model.lm_head(hidden_states[:, -1:, :])
            return hidden_states, logits, getattr(core_outputs, "past_key_values", None)

        past_key_values = None
        while generated_ids.shape[1] < target_length and bool(unfinished.any().item()):
            next_pos = int(generated_ids.shape[1])
            position_ids = self._compute_multimodal_position_ids(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
            step_inputs_embeds = inputs_embeds if past_key_values is None else inputs_embeds[:, -1:, :]
            if position_ids is not None and past_key_values is not None:
                position_ids = position_ids[:, :, -1:]
            model_inputs = {
                "inputs_embeds": self._sanitize_image_placeholder_embeddings(
                    generated_ids if past_key_values is None else generated_ids[:, -1:],
                    step_inputs_embeds,
                ),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "pixel_values": pixel_values if past_key_values is None else None,
                "image_grid_thw": image_grid_thw if past_key_values is None else None,
                "use_cache": True,
                "return_dict": True,
            }
            model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
            hidden_states, logits, past_key_values = _qwen_core_forward(model_inputs)
            past_key_values = _detach_past_key_values(past_key_values)

            step_hidden = hidden_states[:, -1, :]
            step_logits = logits[:, -1, :]
            if router is None:
                hard_route = torch.full(
                    (input_ids.shape[0],),
                    0,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                _, _, routed = router.route_hidden_states(step_hidden.unsqueeze(1))
                hard_route = routed[:, 0]

            text_mask = hard_route.eq(getattr(router, "route_text_idx", 0) if router is not None else 0)
            next_token_ids = step_logits.argmax(dim=-1)
            text_next_embeds = self.model.get_input_embeddings()(next_token_ids)
            # 与标准 LLM 自回归一致：下一步输入不反传回“生成该输入”的上一步状态。
            latent_source = step_hidden.detach()
            latent_next_embeds = latent_source.to(dtype=text_next_embeds.dtype)
            next_embeds = torch.where(text_mask.view(-1, 1), text_next_embeds, latent_next_embeds)
            next_ids = next_token_ids.clone()
            if router is not None:
                for task_name, route_idx in router.task_id_by_name.items():
                    if task_name == "text":
                        continue
                    task_mask = hard_route.eq(route_idx)
                    placeholder_id = int(router.task_placeholder_ids[task_name])
                    next_ids = torch.where(task_mask, torch.full_like(next_ids, placeholder_id), next_ids)

            next_attention = attention_mask.new_ones((attention_mask.shape[0], 1))
            if aligned_labels is not None and next_pos < full_ids.shape[1]:
                # 无监督位置属于 prompt/多模态占位，继续喂 teacher embedding；监督位置使用路由后的生成 embedding。
                teacher_mask = aligned_labels[:, next_pos].eq(IGNORE_INDEX)
                next_embeds = torch.where(teacher_mask.view(-1, 1), full_embeds[:, next_pos, :], next_embeds)
                next_ids = torch.where(teacher_mask, full_ids[:, next_pos], next_ids)
                next_attention = full_attention[:, next_pos].view(-1, 1)

            next_ids = torch.where(unfinished, next_ids, torch.full_like(next_ids, pad_token_id))
            inputs_embeds = torch.cat([inputs_embeds, next_embeds.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)
            generated_ids = torch.cat([generated_ids, next_ids.unsqueeze(1)], dim=1)
            hidden_history.append(step_hidden)
            logits_history.append(step_logits)
            route_history.append(hard_route)
            if eos_token_ids is not None:
                is_eos = next_ids.unsqueeze(-1).eq(eos_token_ids.view(1, -1)).any(dim=-1)
                unfinished = unfinished & (~is_eos)

        if aligned_labels is not None:
            step_labels = aligned_labels[:, prefix_len : prefix_len + len(hidden_history)]
            aligned_labels = torch.cat(
                [
                    aligned_labels.new_full((aligned_labels.shape[0], 1), IGNORE_INDEX),
                    step_labels,
                ],
                dim=1,
            )

        return {
            "input_ids": generated_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "step_hidden_states": torch.stack(hidden_history, dim=1) if hidden_history else None,
            "step_logits": torch.stack(logits_history, dim=1) if logits_history else None,
            "step_routes": torch.stack(route_history, dim=1) if route_history else None,
            "aligned_labels": aligned_labels,
        }
