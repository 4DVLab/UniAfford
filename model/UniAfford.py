"""
UniAfford模型骨架，子架构分布到其他model中并作为模块导入
"""
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import UniAffordConfig
from model.pointcept import PointCloudHiddenStateDecoder
from model.segment_anything import ImageHiddenStateDecoder
from model.qwenvl import MLLMBackbone
from model.HeadRouter import HeadRouter
from utils.common import IGNORE_INDEX


class UniAffordModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    @staticmethod
    def _apply_sample_mask(logits: Optional[torch.Tensor], sample_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if logits is None or sample_mask is None:
            return logits
        view_shape = [sample_mask.shape[0]] + [1] * (logits.dim() - 1)
        return logits * sample_mask.bool().view(*view_shape).to(logits.dtype)

    @staticmethod
    def _apply_query_mask(logits: Optional[torch.Tensor], query_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        将无效 query 的 decoder 输出置零。

        HeadRouter 在没有选中 img/pc token 时会返回一个全零占位 query，并用 query_mask=False 标记。
        这里保留 batch 张量形状，但不让该占位 query 产生有效预测。
        """
        if logits is None or query_mask is None or logits.dim() < 3:
            return logits
        if logits.shape[0] != query_mask.shape[0] or logits.shape[1] != query_mask.shape[1]:
            return logits
        view_shape = list(query_mask.shape) + [1] * (logits.dim() - 2)
        valid_mask = query_mask.bool().view(*view_shape)
        # 无效 query 仍保持 logits 语义，但 sigmoid 后应接近 0，避免低阈值下变成整图正样本。
        invalid_logits = torch.full_like(logits, -30.0)
        return torch.where(valid_mask, logits, invalid_logits)

    def _sync_point_decoder_config(self):
        point_decoder_cfg = self.config.point_decoder
        if point_decoder_cfg is None:
            return

        point_decoder_cfg.backbone_mode = str(point_decoder_cfg.backbone_mode).lower()
        if point_decoder_cfg.backbone_mode == "shared":
            encoder_backbone_cfg = self.config.mllm.point_encoder_backbone.to_dict()
            decoder_backbone_cfg = dict(encoder_backbone_cfg)
            decoder_backbone_cfg["enc_mode"] = False
            point_decoder_cfg.backbone_kwargs = decoder_backbone_cfg
            point_decoder_cfg.backbone_out_channels = int(
                decoder_backbone_cfg.get("dec_channels", (64,))[0]
            )
        else:
            point_decoder = getattr(self, "point_decoder", None)
            if point_decoder is not None and hasattr(point_decoder, "config"):
                point_decoder_cfg.backbone_kwargs = dict(point_decoder.config.backbone_kwargs)
                point_decoder_cfg.backbone_out_channels = int(point_decoder.config.backbone_out_channels)

    def _build_prompt_only_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        validate/generate 推理专用：从 full input_ids 中截掉 GT answer，只保留 prompt。

        labels 只用于定位 answer 起点，不把 GT answer token 传入 MLLM。
        """
        if input_ids is None:
            raise ValueError("generate 推理需要 input_ids。")
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = input_ids.ne(int(pad_id))

        batch_size, seq_len = input_ids.shape
        prompt_lens = []
        for i in range(batch_size):
            if labels is not None:
                supervised = labels[i].ne(IGNORE_INDEX)
                if bool(supervised.any().item()):
                    prompt_len = int(supervised.float().argmax().item())
                else:
                    prompt_len = int(attention_mask[i].sum().item())
            else:
                prompt_len = int(attention_mask[i].sum().item())
            prompt_lens.append(max(1, min(prompt_len, seq_len)))

        max_prompt_len = max(prompt_lens)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        prompt_ids = input_ids.new_full((batch_size, max_prompt_len), pad_id)
        prompt_attention = attention_mask.new_zeros((batch_size, max_prompt_len))
        for i, prompt_len in enumerate(prompt_lens):
            prompt_ids[i, :prompt_len] = input_ids[i, :prompt_len]
            prompt_attention[i, :prompt_len] = attention_mask[i, :prompt_len]
        return prompt_ids, prompt_attention

    def _decode_with_route_out(
        self,
        route_out: Dict[str, torch.Tensor],
        point_encoder_outputs: Optional[Dict[str, torch.Tensor]],
        images: Optional[torch.Tensor],
        img_valid_mask: Optional[torch.Tensor],
        point_clouds: Optional[torch.Tensor],
        pc_valid_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """复用 routed query 执行 2D/3D decoder。"""
        image_logits = None
        if images is not None:
            image_embeddings = self.image_decoder.get_visual_embs(images)
            input_size = (images.shape[-2], images.shape[-1])
            all_image_logits = self.image_decoder(
                route_out["img_query_tokens"],
                image_embeddings,
                input_size,
                input_size,
                query_mask=route_out.get("img_query_mask"),
            )
            all_image_logits = self._apply_query_mask(all_image_logits, route_out.get("img_query_mask"))
            image_logits = self._apply_sample_mask(all_image_logits, img_valid_mask)

        has_per_point_features = (
            point_encoder_outputs is not None
            and point_encoder_outputs.get("per_point_features") is not None
            and point_encoder_outputs.get("per_point_mask") is not None
        )
        if point_clouds is not None and (has_per_point_features or not self.point_decoder_uses_shared_backbone):
            all_point_logits = self.point_decoder(
                point_clouds=point_clouds,
                per_point_features=None if not has_per_point_features else point_encoder_outputs.get("per_point_features"),
                per_point_mask=None if not has_per_point_features else point_encoder_outputs.get("per_point_mask"),
                query_embeddings=route_out.get("pc_query_tokens"),
                query_mask=route_out.get("pc_query_mask"),
            )
        else:
            all_point_logits = None

        if all_point_logits is None:
            point_logits = None
        elif pc_valid_lengths is not None:
            all_point_logits = self._apply_query_mask(all_point_logits, route_out.get("pc_query_mask"))
            point_logits = self._apply_sample_mask(all_point_logits, pc_valid_lengths > 0)
        else:
            all_point_logits = self._apply_query_mask(all_point_logits, route_out.get("pc_query_mask"))
            point_logits = all_point_logits
        return image_logits, point_logits

    @staticmethod
    def _apply_missing_query_fallback(
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        branch_token_emb: torch.Tensor,
        route_probs: Optional[torch.Tensor],
        route_idx: Optional[int],
        sample_available: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """为 generate 阶段缺失的分支 query 补充最高路由概率的 hidden state。

        Args:
            query_tokens: 已按 hard route 打包后的分支 query，形状为 ``[B, K, C]``。
            query_mask: query 是否有效的掩码，形状为 ``[B, K]``。
            branch_token_emb: 对应分支投影后的 generated hidden，形状为 ``[B, T, C]``。
            route_probs: 每个生成 step 的路由概率，形状为 ``[B, T, R]``。
            route_idx: 当前分支在 router 中的类别 id。
            sample_available: 当前样本是否具备该模态输入，形状为 ``[B]``。

        Returns:
            Tuple[Tensor, Tensor, Tensor, int]: fallback 后的 query、query mask、
            fallback 位置 mask，以及实际补充的样本数。
        """
        if (
            query_tokens is None
            or query_mask is None
            or branch_token_emb is None
            or route_probs is None
            or route_idx is None
        ):
            fallback_mask = torch.zeros_like(query_mask) if query_mask is not None else None
            return query_tokens, query_mask, fallback_mask, 0
        if query_tokens.dim() != 3 or query_mask.dim() != 2 or branch_token_emb.dim() != 3:
            return query_tokens, query_mask, torch.zeros_like(query_mask), 0
        if route_probs.dim() != 3 or route_probs.shape[:2] != branch_token_emb.shape[:2]:
            return query_tokens, query_mask, torch.zeros_like(query_mask), 0

        batch_size = min(query_tokens.shape[0], query_mask.shape[0], branch_token_emb.shape[0])
        if sample_available is None:
            available = torch.ones(batch_size, dtype=torch.bool, device=query_tokens.device)
        else:
            available = sample_available[:batch_size].bool().to(query_tokens.device)

        out_tokens = query_tokens.clone()
        out_mask = query_mask.clone()
        fallback_mask = torch.zeros_like(query_mask)
        fallback_count = 0
        for batch_idx in range(batch_size):
            if not bool(available[batch_idx].item()):
                continue
            if bool(out_mask[batch_idx].any().item()):
                continue
            branch_scores = route_probs[batch_idx, :, int(route_idx)]
            if not bool(torch.isfinite(branch_scores).any().item()):
                continue
            best_pos = int(branch_scores.argmax().item())
            out_tokens[batch_idx, 0, :] = branch_token_emb[batch_idx, best_pos, :].to(out_tokens.dtype)
            out_mask[batch_idx, 0] = True
            fallback_mask[batch_idx, 0] = True
            fallback_count += 1
        return out_tokens, out_mask, fallback_mask, fallback_count

    @staticmethod
    def _build_decoder_query_pairs(
        img_query_tokens: Optional[torch.Tensor],
        img_query_mask: Optional[torch.Tensor],
        img_fallback_mask: Optional[torch.Tensor],
        pc_query_tokens: Optional[torch.Tensor],
        pc_query_mask: Optional[torch.Tensor],
        pc_fallback_mask: Optional[torch.Tensor],
    ) -> List[List[Tuple[str, torch.Tensor, bool]]]:
        """记录实际输入 decoder 的 query，用于验证表分析。

        Args:
            img_query_tokens: 2D decoder 使用的 query，形状为 ``[B, K_img, C]``。
            img_query_mask: 2D query 是否有效。
            img_fallback_mask: 2D query 是否由 fallback 补充。
            pc_query_tokens: 3D decoder 使用的 query，形状为 ``[B, K_pc, C]``。
            pc_query_mask: 3D query 是否有效。
            pc_fallback_mask: 3D query 是否由 fallback 补充。

        Returns:
            List[List[Tuple[str, Tensor, bool]]]: 每个样本的实际 decoder query 列表，
            tuple 依次为分支名、query embedding、是否 fallback。
        """
        batch_size = 0
        for tensor in (img_query_tokens, pc_query_tokens):
            if isinstance(tensor, torch.Tensor):
                batch_size = max(batch_size, int(tensor.shape[0]))
        pairs: List[List[Tuple[str, torch.Tensor, bool]]] = [[] for _ in range(batch_size)]

        def _append_branch(branch: str, tokens, mask, fallback_mask):
            if tokens is None or mask is None:
                return
            cur_bsz = min(batch_size, int(tokens.shape[0]), int(mask.shape[0]))
            fallback = torch.zeros_like(mask) if fallback_mask is None else fallback_mask
            for batch_idx in range(cur_bsz):
                valid_pos = torch.nonzero(mask[batch_idx].bool(), as_tuple=False).view(-1)
                for pos in valid_pos.tolist():
                    is_fallback = bool(fallback[batch_idx, pos].item()) if pos < fallback.shape[1] else False
                    pairs[batch_idx].append((branch, tokens[batch_idx, pos, :], is_fallback))

        _append_branch("img", img_query_tokens, img_query_mask, img_fallback_mask)
        _append_branch("pc", pc_query_tokens, pc_query_mask, pc_fallback_mask)
        return pairs

    @staticmethod
    def _build_decoder_query_hidden_pairs(
        hidden_states: torch.Tensor,
        hard_route: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        route_probs: Optional[torch.Tensor],
        img_route_idx: Optional[int],
        img_fallback_mask: Optional[torch.Tensor],
        pc_route_idx: Optional[int],
        pc_fallback_mask: Optional[torch.Tensor],
    ) -> List[List[Tuple[str, torch.Tensor, bool]]]:
        """记录实际 decoder query 对应的原始 LLM hidden state。

        Args:
            hidden_states: generate 每一步的原始 LLM hidden，形状为 ``[B, T, C]``。
            hard_route: generate 每一步的离散路由结果，形状为 ``[B, T]``。
            attention_mask: generate step 是否有效的掩码，形状为 ``[B, T]``。
            route_probs: generate 每一步的路由概率，形状为 ``[B, T, R]``。
            img_route_idx: img 分支在 router 中的类别 id。
            img_fallback_mask: img query 是否由 fallback 补充，形状为 ``[B, K_img]``。
            pc_route_idx: pc 分支在 router 中的类别 id。
            pc_fallback_mask: pc query 是否由 fallback 补充，形状为 ``[B, K_pc]``。

        Returns:
            List[List[Tuple[str, Tensor, bool]]]: 每个样本的 query 记录，tuple 依次为
            分支名、投影前的 LLM hidden state、是否 fallback。
        """
        if hidden_states is None or hard_route is None:
            return []
        batch_size, seq_len = hidden_states.shape[:2]
        pairs: List[List[Tuple[str, torch.Tensor, bool]]] = [[] for _ in range(batch_size)]
        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
        else:
            valid_mask = attention_mask.bool().to(hidden_states.device)

        def _append_branch(branch: str, route_idx: Optional[int], fallback_mask: Optional[torch.Tensor]):
            if route_idx is None:
                return
            for batch_idx in range(batch_size):
                route_pos = torch.nonzero(
                    hard_route[batch_idx].eq(int(route_idx)) & valid_mask[batch_idx],
                    as_tuple=False,
                ).view(-1)
                for pos in route_pos.tolist():
                    pairs[batch_idx].append((branch, hidden_states[batch_idx, pos, :], False))

                has_fallback = (
                    fallback_mask is not None
                    and batch_idx < fallback_mask.shape[0]
                    and bool(fallback_mask[batch_idx].any().item())
                )
                if has_fallback and route_probs is not None:
                    # fallback query 由最高 route probability 的 step 投影得到；
                    # 这里用于语言空间反投影，因此记录投影前的原始 hidden state。
                    branch_probs = route_probs[batch_idx, :, int(route_idx)].masked_fill(
                        ~valid_mask[batch_idx],
                        float("-inf"),
                    )
                    best_pos = int(branch_probs.argmax().item())
                    pairs[batch_idx].append((branch, hidden_states[batch_idx, best_pos, :], True))

        _append_branch("img", img_route_idx, img_fallback_mask)
        _append_branch("pc", pc_route_idx, pc_fallback_mask)
        return pairs

    def __init__(self, config: Optional[UniAffordConfig] = None):
        super().__init__()
        self.config = config or UniAffordConfig()

        self.mllm = MLLMBackbone(self.config.mllm)
        self.functional_tokens = self.mllm.functional_tokens
        self.functional_token_ids = self.mllm.functional_token_ids

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder, self.config.mllm.hidden_size)

        self.point_encoder = getattr(self.mllm, "point_encoder", None)
        if self.point_encoder is not None:
            point_feature_size = int(getattr(self.point_encoder, "point_feature_size"))
        else:
            point_feature_size = int(self.config.mllm.point_encoder_backbone.dec_channels[0])

        self.point_decoder_uses_shared_backbone = str(self.config.point_decoder.backbone_mode).lower() == "shared"
        self.point_decoder = PointCloudHiddenStateDecoder(
            self.config.point_decoder,
            self.config.mllm.hidden_size,
            point_feature_size=point_feature_size if self.point_decoder_uses_shared_backbone else None,
        )
        self._sync_point_decoder_config()

        hidden_size = int(self.config.mllm.hidden_size)
        self.router = HeadRouter(
            hidden_size=hidden_size,
            tokenizer=self.tokenizer,
            task_placeholder_tokens=self.mllm.task_placeholder_tokens,
        )
        # 本地 feedback generate 运行在 MLLMBackbone 内部，但路由器属于外层联合模型。
        # 因此 HeadRouter 创建完成后，需要把它作为运行时引用挂到 MLLM 上；
        # 这样推理每一步才能用同一个 router 判断 text / 非 text 写回方式。
        self.mllm.set_generation_feedback_router(self.router)
        # 只有在执行路由设计相关的“消融”操作时，才应将此开关设置为“fixed_anchor”状态。
        self.routing_design_ablation = None
        assert self.routing_design_ablation is None, "手动注释这行以启用 fixed-anchor 的路由模式"

        self.task_placeholder_tokens = self.router.task_placeholder_tokens
        self.task_placeholder_ids = self.router.task_placeholder_ids
        self.placeholder_id_to_task_id = self.router.placeholder_id_to_task_id


    @property
    def tokenizer(self): return self.mllm.tokenizer

    @property
    def processor(self): return self.mllm.processor

    def _compute_text_ce_without_task_tokens(
        self,
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        route_out: Optional[Dict[str, torch.Tensor]] = None,
        ignore_index: int = -100,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        自定义 CE：
        - 仍使用标准 next-token CE（shift 对齐）
        - 忽略标签中 <img_aff>/<pc_aff>/<latent> 等非 text 任务 token
        - 若提供 route_out，则进一步只保留 router 判为 text 类的位置
        """
        if logits is None or labels is None:
            return None, 0
        if logits.dim() != 3 or labels.dim() != 2:
            return None, 0
        if logits.shape[0] != labels.shape[0]:
            return None, 0

        if labels.shape[1] <= 1:
            return logits.new_zeros(()), 0
        pred_len = min(logits.shape[1], labels.shape[1] - 1)
        shift_logits = logits[:, :pred_len, :].contiguous()       # [B, P, V]
        shift_labels = labels[:, 1 : 1 + pred_len].contiguous()   # [B, P]
        valid = shift_labels.ne(ignore_index)
        placeholder_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        non_text_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for task_name, placeholder_id in self.task_placeholder_ids.items():
            cur_mask = shift_labels.eq(int(placeholder_id))
            placeholder_mask = placeholder_mask | cur_mask
            if task_name != "text":
                non_text_mask = non_text_mask | cur_mask

        if route_out is not None and route_out.get("hard_route") is not None:
            hard_route = route_out["hard_route"]
            pred_len = min(valid.shape[1], hard_route.shape[1])
            route_text = hard_route[:, :pred_len].eq(self.router.route_text_idx)
            valid[:, :pred_len] = valid[:, :pred_len] & route_text
            if pred_len < valid.shape[1]:
                valid[:, pred_len:] = False

        # placeholder token 只用于路由监督；普通文本 token 才走语言 CE。
        ignored_tokens = int((valid & placeholder_mask).sum().item())
        valid = valid & (~placeholder_mask) & (~non_text_mask)
        if not valid.any():
            return shift_logits.new_zeros(()), ignored_tokens

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view_as(shift_labels)
        valid_f = valid.to(token_loss.dtype)
        return (token_loss * valid_f).sum() / valid_f.sum().clamp_min(1.0), ignored_tokens

    def _compute_router_classification_stats(
        self,
        labels: Optional[torch.Tensor],
        route_out: Optional[Dict[str, torch.Tensor]],
        ignore_index: int = -100,
    ) -> Dict[str, int]:
        """统计 router 的逐 token 分类结果。

        Args:
            labels: 训练用语言标签，形状为 ``[B, L]``。这里沿用 causal LM 的
                shift 规则，用 ``labels[t + 1]`` 作为 ``hard_route[t]`` 的监督目标。
            route_out: HeadRouter 输出，必须包含 ``hard_route``。
            ignore_index: 不参与语言监督的位置，同样不参与 router 分类统计。

        Returns:
            Dict[str, int]: batch 级计数。字段命名使用 ``*_wrong`` / ``*_total``，
            日志里展示为 ``wrong/total``，避免小 batch 下准确率抖动掩盖真实数量。
        """
        empty_stats = {
            "route_wrong": 0,
            "route_total": 0,
            "route_text_wrong": 0,
            "route_text_total": 0,
            "route_img_wrong": 0,
            "route_img_total": 0,
            "route_pc_wrong": 0,
            "route_pc_total": 0,
            "route_aff_wrong": 0,
            "route_aff_total": 0,
            "route_text_as_aff": 0,
            "route_aff_as_text": 0,
        }
        if labels is None or route_out is None or route_out.get("hard_route") is None:
            return empty_stats
        hard_route = route_out["hard_route"]
        if labels.dim() != 2 or hard_route.dim() != 2 or labels.shape[0] != hard_route.shape[0]:
            return empty_stats
        if labels.shape[1] <= 1:
            return empty_stats

        # Router 与语言模型一样采用 next-token 对齐：hidden[t] 应预测 labels[t + 1] 的语义角色。
        pred_len = min(hard_route.shape[1], labels.shape[1] - 1)
        if pred_len <= 0:
            return empty_stats
        shifted_labels = labels[:, 1 : 1 + pred_len].to(hard_route.device)
        pred_routes = hard_route[:, :pred_len]
        valid = shifted_labels.ne(ignore_index)

        target_routes = torch.full_like(pred_routes, int(self.router.route_text_idx))
        for placeholder_id, task_id in self.router.placeholder_id_to_task_id.items():
            target_routes = torch.where(
                shifted_labels.eq(int(placeholder_id)),
                torch.full_like(target_routes, int(task_id)),
                target_routes,
            )

        wrong = valid & pred_routes.ne(target_routes)
        text_target = valid & target_routes.eq(int(self.router.route_text_idx))
        aff_target = valid & target_routes.ne(int(self.router.route_text_idx))
        pred_aff = pred_routes.ne(int(self.router.route_text_idx))

        stats = dict(empty_stats)
        stats["route_wrong"] = int(wrong.sum().item())
        stats["route_total"] = int(valid.sum().item())
        stats["route_text_wrong"] = int((wrong & text_target).sum().item())
        stats["route_text_total"] = int(text_target.sum().item())
        stats["route_aff_wrong"] = int((wrong & aff_target).sum().item())
        stats["route_aff_total"] = int(aff_target.sum().item())
        stats["route_text_as_aff"] = int((text_target & pred_aff).sum().item())
        stats["route_aff_as_text"] = int((aff_target & pred_routes.eq(int(self.router.route_text_idx))).sum().item())

        if self.router.route_img_idx is not None:
            img_target = valid & target_routes.eq(int(self.router.route_img_idx))
            stats["route_img_wrong"] = int((wrong & img_target).sum().item())
            stats["route_img_total"] = int(img_target.sum().item())
        if self.router.route_pc_idx is not None:
            pc_target = valid & target_routes.eq(int(self.router.route_pc_idx))
            stats["route_pc_wrong"] = int((wrong & pc_target).sum().item())
            stats["route_pc_total"] = int(pc_target.sum().item())
        return stats

    def generate_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        original_size_list: Optional[List] = None,
        img_valid_mask: Optional[torch.Tensor] = None,
        img_gt_tensor: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        pc_gt_tensor: Optional[torch.Tensor] = None,
        obj_type: Optional[List[str]] = None,
        aff_type: Optional[List[str]] = None,
        return_hidden_states: bool = False,
        return_mllm_output: bool = False,
        max_new_tokens: Optional[int] = None,
        generate_query_fallback: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        validate 推理入口：只用 prompt 进行 generate，GT answer 仅用于评估。

        生成过程由 MLLMBackbone 本地 greedy feedback loop 记录每步 hidden/route；下游 decoder 使用
        route 到 img/pc 的 generated hidden，而不是 teacher-forced GT answer hidden。
        """
        prompt_ids, prompt_attention = self._build_prompt_only_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        point_encoder_outputs = None
        if self.point_encoder is not None and point_clouds is not None:
            point_encoder_outputs = self.point_encoder.encode_shared(
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
            )

        if max_new_tokens is None and labels is not None:
            answer_counts = labels.ne(IGNORE_INDEX).sum(dim=1)
            max_new_tokens = max(1, int(answer_counts.max().item()) + 8)

        img_available = img_valid_mask if img_valid_mask is not None else None
        pc_available = (pc_valid_lengths > 0) if pc_valid_lengths is not None else None
        mllm_out = self.mllm.generate_with_router_feedback(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
            point_token_embeds=None if point_encoder_outputs is None else point_encoder_outputs.get("mllm_point_tokens"),
            point_token_mask=None if point_encoder_outputs is None else point_encoder_outputs.get("mllm_point_token_mask"),
            img_available=img_available,
            pc_available=pc_available,
            max_new_tokens=max_new_tokens,
            generation_config=kwargs.pop("generation_config", None),
        )

        hidden_states = mllm_out["step_hidden_states"]
        hard_route = mllm_out["step_routes"]
        route_logits = mllm_out.get("step_route_logits")
        route_probs = mllm_out.get("step_route_probs")
        generated_token_ids = mllm_out.get("generated_token_ids")
        generated_attention = mllm_out.get("step_attention_mask")
        if generated_attention is None:
            generated_attention = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            generated_attention = generated_attention.bool().to(hidden_states.device)
            if generated_attention.shape != hidden_states.shape[:2]:
                generated_attention = torch.ones(
                    hidden_states.shape[:2],
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
        fallback_route_probs = route_probs
        if route_probs is not None:
            # fallback 只能从尚未结束的生成 step 中选择，避免 EOS 后 pad step 的 route 污染 query。
            fallback_route_probs = route_probs.masked_fill(~generated_attention[:, :, None], float("-inf"))

        img_token_emb = self.router.img_branch_head(hidden_states)
        pc_token_emb = self.router.pc_branch_head(hidden_states)
        img_query_tokens, img_query_mask, pc_query_tokens, pc_query_mask = self.router.build_branch_query_tokens(
            img_token_emb=img_token_emb,
            pc_token_emb=pc_token_emb,
            hard_route=hard_route,
            attention_mask=generated_attention,
            route_mask=None,
        )
        fallback_stats = {"img_query_fallback_count": 0, "pc_query_fallback_count": 0}
        img_query_fallback_mask = torch.zeros_like(img_query_mask)
        pc_query_fallback_mask = torch.zeros_like(pc_query_mask)
        if generate_query_fallback and fallback_route_probs is not None:
            (
                img_query_tokens,
                img_query_mask,
                img_query_fallback_mask,
                fallback_stats["img_query_fallback_count"],
            ) = self._apply_missing_query_fallback(
                query_tokens=img_query_tokens,
                query_mask=img_query_mask,
                branch_token_emb=img_token_emb,
                route_probs=fallback_route_probs,
                route_idx=self.router.route_img_idx,
                sample_available=img_available,
            )
            (
                pc_query_tokens,
                pc_query_mask,
                pc_query_fallback_mask,
                fallback_stats["pc_query_fallback_count"],
            ) = self._apply_missing_query_fallback(
                query_tokens=pc_query_tokens,
                query_mask=pc_query_mask,
                branch_token_emb=pc_token_emb,
                route_probs=fallback_route_probs,
                route_idx=self.router.route_pc_idx,
                sample_available=pc_available,
            )
        decoder_query_pairs = self._build_decoder_query_hidden_pairs(
            hidden_states=hidden_states,
            hard_route=hard_route,
            attention_mask=generated_attention,
            route_probs=route_probs,
            img_route_idx=self.router.route_img_idx,
            img_fallback_mask=img_query_fallback_mask,
            pc_route_idx=self.router.route_pc_idx,
            pc_fallback_mask=pc_query_fallback_mask,
        )
        routed_token_ids = self.router.build_routed_token_ids(
            base_token_ids=generated_token_ids,
            hard_route=hard_route,
            route_mask=None,
        )
        aff_token_pairs = self.router.build_aff_token_pairs(
            hard_route=hard_route,
            token_hidden_states=hidden_states,
            attention_mask=generated_attention,
            route_mask=None,
        )
        if route_probs is not None:
            structure_signals = self.router.build_structure_signals(
                route_probs=route_probs,
                attention_mask=generated_attention,
                route_mask=None,
            )
        else:
            structure_signals = {
                "img_any_prob": None,
                "pc_any_prob": None,
                "img_expected_count": None,
                "pc_expected_count": None,
            }
        route_out = {
            "route_logits": route_logits,
            "route_probs": route_probs,
            "hard_route": hard_route,
            "img_query_tokens": img_query_tokens,
            "img_query_mask": img_query_mask,
            "pc_query_tokens": pc_query_tokens,
            "pc_query_mask": pc_query_mask,
            "routed_token_ids": routed_token_ids,
            "aff_token_pairs": aff_token_pairs,
            **structure_signals,
        }

        image_logits, point_logits = self._decode_with_route_out(
            route_out=route_out,
            point_encoder_outputs=point_encoder_outputs,
            images=images,
            img_valid_mask=img_valid_mask,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
        )

        zero_loss = hidden_states.new_zeros(())
        output_dict = {
            "hidden_states": hidden_states if return_hidden_states else None,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "token_ids": routed_token_ids,
            "img_query_mask": img_query_mask,
            "pc_query_mask": pc_query_mask,
            "generated_token_ids": generated_token_ids,
            "generated_ids": mllm_out.get("generated_ids"),
            "generated_logits": mllm_out.get("step_logits"),
            "hard_route": hard_route,
            "labels": labels,
            "attention_mask": attention_mask,
            "ce_loss": zero_loss,
            "output": mllm_out.get("output") if return_mllm_output else None,
            "aff_token_pairs": aff_token_pairs,
            "decoder_query_pairs": decoder_query_pairs,
            "route_logits": route_logits,
            "route_probs": route_probs,
            "img_any_prob": route_out["img_any_prob"],
            "pc_any_prob": route_out["pc_any_prob"],
            "img_expected_count": route_out["img_expected_count"],
            "pc_expected_count": route_out["pc_expected_count"],
            "query_fallback_stats": fallback_stats,
            "task_placeholder_ids": self.router.task_placeholder_ids,
            "placeholder_id_to_task_id": self.router.placeholder_id_to_task_id,
            "ce_ignored_token_count": 0,
            "route_classification_stats": self._compute_router_classification_stats(None, None),
        }
        return output_dict

    def forward(
        self,
        # Qwen 推理所需
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        # 图像分割所需
        images: Optional[torch.Tensor] = None,
        original_size_list: Optional[List] = None,
        img_valid_mask: Optional[torch.Tensor] = None,
        img_gt_tensor: Optional[torch.Tensor] = None,  # 后续可能支持，暂且保留
        # 点云分割所需
        point_clouds: Optional[torch.Tensor] = None,
        pc_valid_lengths: Optional[torch.Tensor] = None,
        pc_gt_tensor: Optional[torch.Tensor] = None,  # 后续可能支持，暂且保留
        obj_type: Optional[List[str]] = None,
        aff_type: Optional[List[str]] = None,

        return_hidden_states: bool = False,
        return_mllm_output: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if bool(kwargs.pop("inference_generate", False)):
            return self.generate_forward(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                images=images,
                original_size_list=original_size_list,
                img_valid_mask=img_valid_mask,
                img_gt_tensor=img_gt_tensor,
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
                pc_gt_tensor=pc_gt_tensor,
                obj_type=obj_type,
                aff_type=aff_type,
                return_hidden_states=return_hidden_states,
                return_mllm_output=return_mllm_output,
                **kwargs,
            )

        # ---- 0. 点云编码（单次 backbone，产出 token级 + 逐点级 两路特征）----
        point_encoder_outputs = None
        if self.point_encoder is not None and point_clouds is not None:
            point_encoder_outputs = self.point_encoder.encode_shared(
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
            )

        # ---- 1. MLLM 前向 ----
        # 训练/普通 forward 保持 teacher-forcing 并行前向；validate 的无泄露推理由 generate_forward 负责。
        mllm_out = self.mllm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
            point_token_embeds=None if point_encoder_outputs is None else point_encoder_outputs.get("mllm_point_tokens"),
            point_token_mask=None if point_encoder_outputs is None else point_encoder_outputs.get("mllm_point_token_mask"),
        )
        hidden_states = mllm_out["hidden_states"]  # [B, L, C]
        output_obj = mllm_out.get("output")
        # 关键：当启用 pc prefix 时，使用与 logits 同长度的对齐标签，避免 CE 维度不一致
        model_labels = mllm_out.get("aligned_labels", labels)
        model_attention_mask = mllm_out.get("aligned_attention_mask", attention_mask)
        logits_token_ids = None
        output_logits = None
        if output_obj is not None:
            # 从 logits 中取 token_ids（用于可视化与占位 token 回写，不参与路由决策）
            if getattr(output_obj, "logits", None) is not None:
                logits_token_ids = output_obj.logits.argmax(dim=-1)
                output_logits = output_obj.logits
        B,L,C = hidden_states.shape

        ce_loss = None
        ce_ignored_token_count = 0
        route_classification_stats = self._compute_router_classification_stats(None, None)
        
        if hidden_states is not None:
            img_available = img_valid_mask if img_valid_mask is not None else None
            pc_available = (pc_valid_lengths > 0) if pc_valid_lengths is not None else None
            route_kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=model_attention_mask,
                img_available=img_available,
                pc_available=pc_available,
                labels=model_labels,
                base_token_ids=logits_token_ids,
            )
            if self.routing_design_ablation == "fixed_anchor":
                route_out = self.router.fixed_anchor_forward(**route_kwargs)
            else:
                route_out = self.router(**route_kwargs)
            route_classification_stats = self._compute_router_classification_stats(
                labels=model_labels,
                route_out=route_out,
                ignore_index=-100,
            )

            if output_logits is not None and model_labels is not None:
                ce_loss, ce_ignored_token_count = self._compute_text_ce_without_task_tokens(
                    logits=output_logits,
                    labels=model_labels,
                    route_out=route_out,
                    ignore_index=-100,
                )
            # 兜底：无 logits 时沿用底座返回 loss
            if ce_loss is None and output_obj is not None and getattr(output_obj, "loss", None) is not None:
                ce_loss = output_obj.loss

            # ---- 3. 2D 图像分割 ----
            image_logits = None
            if images is not None:
                image_embeddings = self.image_decoder.get_visual_embs(images)
                input_size = (images.shape[-2], images.shape[-1])
                # 训练时 decoder 输出需与 img_gt_tensor 一致（均为 padding 后的 input_size）
                # 推理保存时再按 original_size_list 缩放还原
                original_size = input_size

                all_image_logits = self.image_decoder(
                    route_out["img_query_tokens"],
                    image_embeddings,
                    input_size,
                    original_size,
                    query_mask=route_out.get("img_query_mask"),
                )
                all_image_logits = self._apply_query_mask(all_image_logits, route_out.get("img_query_mask"))

                # 将无效样本的输出置零（不影响 loss 计算）
                image_logits = self._apply_sample_mask(all_image_logits, img_valid_mask)

            # ---- 4. 3D 点云分割 ----
            has_per_point_features = (
                point_encoder_outputs is not None
                and point_encoder_outputs.get("per_point_features") is not None
                and point_encoder_outputs.get("per_point_mask") is not None
            )
            if point_clouds is not None and (has_per_point_features or not self.point_decoder_uses_shared_backbone):
                all_point_logits = self.point_decoder(
                    point_clouds=point_clouds,
                    per_point_features=None if not has_per_point_features else point_encoder_outputs.get("per_point_features"),
                    per_point_mask=None if not has_per_point_features else point_encoder_outputs.get("per_point_mask"),
                    query_embeddings=route_out.get("pc_query_tokens"),
                    query_mask=route_out.get("pc_query_mask"),
                )
            else:
                all_point_logits = None

            # 将无效样本的输出置零
            if all_point_logits is None:
                point_logits = None
            elif pc_valid_lengths is not None:
                all_point_logits = self._apply_query_mask(all_point_logits, route_out.get("pc_query_mask"))
                point_logits = self._apply_sample_mask(all_point_logits, pc_valid_lengths > 0)
            else:
                all_point_logits = self._apply_query_mask(all_point_logits, route_out.get("pc_query_mask"))
                point_logits = all_point_logits


        output_dict = {
            "hidden_states": None,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "token_ids": route_out["routed_token_ids"],
            "img_query_mask": route_out.get("img_query_mask"),
            "pc_query_mask": route_out.get("pc_query_mask"),
            "hard_route": route_out.get("hard_route"),
            "labels": model_labels,
            "attention_mask": model_attention_mask,
            # 语言模型交叉熵损失（若未提供 labels 或模型未返回 loss，则为 None）
            "ce_loss": ce_loss,
            "output": None,
            # 用于下游分支的 token 名称与向量（供 validate 等记录）
            # 格式: List[List[Tuple[str, Tensor]]]，每样本 [("<img-x>", emb), ("<pc-x>", emb), ...]
            "aff_token_pairs": None,
            # 路由监督辅助输出
            "route_logits": route_out["route_logits"],
            "route_probs": route_out["route_probs"],
            "img_any_prob": route_out["img_any_prob"],
            "pc_any_prob": route_out["pc_any_prob"],
            "img_expected_count": route_out["img_expected_count"],
            "pc_expected_count": route_out["pc_expected_count"],
            "task_placeholder_ids": self.router.task_placeholder_ids,
            "placeholder_id_to_task_id": self.router.placeholder_id_to_task_id,
            # batch 级统计：CE 中被忽略的非 text 任务 token 数
            "ce_ignored_token_count": ce_ignored_token_count,
            # batch 级 router 分类统计：日志中展示为 wrong/total，便于直接观察误判规模。
            "route_classification_stats": route_classification_stats,
        }

        if hidden_states is not None:
            output_dict["aff_token_pairs"] = route_out["aff_token_pairs"]
        if return_hidden_states:
            output_dict["hidden_states"] = hidden_states
        if return_mllm_output:
            output_dict["output"] = mllm_out.get("output")

        return output_dict


__all__ = [
    "UniAffordModel",
    "MLLMBackbone",
    "ImageHiddenStateDecoder",
    "PointCloudHiddenStateDecoder",
]
