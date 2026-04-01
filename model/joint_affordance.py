"""
JointAffordance模型骨架，子架构分布到其他model中并作为模块导入
"""
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import JointAffordanceConfig
# from model.pointnet2 import PointCloudHiddenStateDecoder
from model.pointcept import PointCloudHiddenStateDecoder
from model.segment_anything import ImageHiddenStateDecoder
from model.qwenvl import MLLMBackbone


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()

        self.mllm = MLLMBackbone(self.config.mllm)
        self.functional_tokens = self.mllm.functional_tokens
        self.functional_token_ids = self.mllm.functional_token_ids

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder, self.config.mllm.hidden_size)
        self.point_decoder = PointCloudHiddenStateDecoder(self.config.point_decoder, self.config.mllm.hidden_size)

        hidden_size = int(self.config.mllm.hidden_size)
        # text/img/pc 三路 router：不再依赖 <img-obj-aff>/<pc-obj-aff> 字符串规则
        self.route_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        # 两个分支专用头（低风险改造：位于 JointAffordance，不侵入 Qwen 内核）
        self.img_branch_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pc_branch_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.route_text_idx = 0
        self.route_img_idx = 1
        self.route_pc_idx = 2
        self.img_placeholder_token = "<img_aff>"
        self.pc_placeholder_token = "<pc_aff>"
        self.img_placeholder_id = self._resolve_token_id(self.img_placeholder_token)
        self.pc_placeholder_id = self._resolve_token_id(self.pc_placeholder_token)


    @property
    def tokenizer(self): return self.mllm.tokenizer

    @property
    def processor(self): return self.mllm.processor

    def _resolve_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and int(token_id) == int(unk_id)):
            raise ValueError(f"占位 token 未注册到 tokenizer: {token}")
        return int(token_id)

    def _route_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        img_available: Optional[torch.Tensor] = None,
        pc_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        token-level 路由计算（text/img/pc 三类）。

        Args:
            hidden_states: MLLM 最后一层隐藏状态，形状 [B, L, C]。
            attention_mask: 有效 token 掩码，形状 [B, L]；无效位置会被强制设为 text 路由。
            img_available: 每样本是否可用 2D 分支，形状 [B]；不可用样本会屏蔽 img 路由。
            pc_available: 每样本是否可用 3D 分支，形状 [B]；不可用样本会屏蔽 pc 路由。

        Returns:
            route_logits: 路由 logits，形状 [B, L, 3]（text/img/pc）。
            route_probs: 路由概率（softmax 后），形状 [B, L, 3]。
            hard_route: 硬路由索引（argmax），形状 [B, L]。
        """
        # route_logits/probs: [B, L, 3]，三类分别为 text/img/pc
        route_logits = self.route_head(hidden_states)
        # 按样本可用模态屏蔽路由专家，避免无监督分支被误激活并主导路由
        if img_available is not None:
            img_mask = img_available.bool().view(-1, 1)  # [B,1]
            route_logits[:, :, self.route_img_idx] = torch.where(
                img_mask,
                route_logits[:, :, self.route_img_idx],
                torch.full_like(route_logits[:, :, self.route_img_idx], -1e4),
            )
        if pc_available is not None:
            pc_mask = pc_available.bool().view(-1, 1)  # [B,1]
            route_logits[:, :, self.route_pc_idx] = torch.where(
                pc_mask,
                route_logits[:, :, self.route_pc_idx],
                torch.full_like(route_logits[:, :, self.route_pc_idx], -1e4),
            )
        route_probs = torch.softmax(route_logits, dim=-1)
        hard_route = route_logits.argmax(dim=-1)
        if attention_mask is not None:
            valid = attention_mask.bool()
            hard_route = torch.where(valid, hard_route, torch.full_like(hard_route, self.route_text_idx))
            route_probs = route_probs * valid.unsqueeze(-1).to(route_probs.dtype)
        return route_logits, route_probs, hard_route

    def _build_branch_embeddings(
        self,
        hidden_states: torch.Tensor,
        route_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于软路由概率构造分支 query（样本级）与 token 级分支特征。

        Args:
            hidden_states: MLLM 隐藏状态，形状 [B, L, C]。
            route_probs: 路由概率，形状 [B, L, 3]。

        Returns:
            img_emb: 2D 分支样本级 query，形状 [B, C]。
            pc_emb: 3D 分支样本级 query，形状 [B, C]。
            img_token_emb: 2D 专用头后的 token 特征，形状 [B, L, C]。
            pc_token_emb: 3D 专用头后的 token 特征，形状 [B, L, C]。
        """
        # 软路由聚合，保证路由器与专用头可从下游损失获得梯度
        img_token_emb = self.img_branch_head(hidden_states)  # [B, L, C]
        pc_token_emb = self.pc_branch_head(hidden_states)    # [B, L, C]

        img_w = route_probs[:, :, self.route_img_idx:self.route_img_idx + 1]  # [B, L, 1]
        pc_w = route_probs[:, :, self.route_pc_idx:self.route_pc_idx + 1]      # [B, L, 1]

        img_den = img_w.sum(dim=1).clamp_min(1e-6)
        pc_den = pc_w.sum(dim=1).clamp_min(1e-6)
        img_emb = (img_token_emb * img_w).sum(dim=1) / img_den
        pc_emb = (pc_token_emb * pc_w).sum(dim=1) / pc_den
        return img_emb, pc_emb, img_token_emb, pc_token_emb

    def _build_routed_token_ids(
        self,
        base_token_ids: Optional[torch.Tensor],
        hard_route: torch.Tensor,
        route_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        依据硬路由结果回写占位 token（保持原 token 顺序）。

        Args:
            base_token_ids: 基础 token ids（通常为 logits argmax），形状 [B, L]。
            hard_route: 硬路由索引，形状 [B, L]。
            route_mask: 可选位置掩码，形状 [B, L]；仅在 True 位置执行占位替换。

        Returns:
            routed_token_ids: 替换后的 token ids，形状 [B, L]；若 base_token_ids 为空则返回 None。
        """
        if base_token_ids is None:
            return None
        routed = base_token_ids.clone()
        img_sel = (hard_route == self.route_img_idx)
        pc_sel = (hard_route == self.route_pc_idx)
        if route_mask is not None:
            img_sel = img_sel & route_mask
            pc_sel = pc_sel & route_mask
        routed = torch.where(
            img_sel,
            torch.full_like(routed, self.img_placeholder_id),
            routed,
        )
        routed = torch.where(
            pc_sel,
            torch.full_like(routed, self.pc_placeholder_id),
            routed,
        )
        return routed

    def _build_aff_token_pairs(
        self,
        hard_route: torch.Tensor,
        img_token_emb: torch.Tensor,
        pc_token_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[str, torch.Tensor]]]:
        """
        构造每样本的路由 token 列表（用于验证记录与可解释性分析）。

        Args:
            hard_route: 硬路由索引，形状 [B, L]。
            img_token_emb: 2D 专用头 token 特征，形状 [B, L, C]。
            pc_token_emb: 3D 专用头 token 特征，形状 [B, L, C]。
            attention_mask: 可选有效 token 掩码，形状 [B, L]。
            route_mask: 可选位置掩码，形状 [B, L]；仅记录 True 位置。

        Returns:
            aff_token_pairs: 长度为 B 的列表；每个元素为 [(token_name, emb), ...]。
        """
        B, L = hard_route.shape
        aff_token_pairs: List[List[Tuple[str, torch.Tensor]]] = [[] for _ in range(B)]
        for i in range(B):
            seq_len = int(attention_mask[i].sum().item()) if attention_mask is not None else L
            for pos in range(seq_len):
                if route_mask is not None and not bool(route_mask[i, pos].item()):
                    continue
                rid = int(hard_route[i, pos].item())
                if rid == self.route_img_idx:
                    aff_token_pairs[i].append((self.img_placeholder_token, img_token_emb[i, pos, :]))
                elif rid == self.route_pc_idx:
                    aff_token_pairs[i].append((self.pc_placeholder_token, pc_token_emb[i, pos, :]))
        return aff_token_pairs

    @staticmethod
    def _build_route_mask_from_labels(
        labels: Optional[torch.Tensor],
        seq_len: int,
        ignore_index: int = -100,
    ) -> Optional[torch.Tensor]:
        """
        构造与 validate 中 pred_token_ids 对齐的位置掩码（p-1 对齐规则）。

        Args:
            labels: 语言监督标签，形状 [B, L]，非监督位置为 ignore_index。
            seq_len: 目标序列长度（通常取 hidden_states 的 L）。
            ignore_index: 非监督标签值，默认 -100。

        Returns:
            route_mask: 布尔掩码，形状 [B, seq_len]；若 labels 不可用则返回 None。
        """
        if labels is None or labels.dim() != 2:
            return None
        bsz, label_len = labels.shape
        use_len = min(seq_len, label_len)
        answer_mask = labels[:, :use_len].ne(ignore_index)
        route_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=labels.device)
        if use_len <= 1:
            return route_mask
        valid_answer_pos = answer_mask[:, 1:use_len]  # p>=1
        route_mask[:, :use_len - 1] = valid_answer_pos
        return route_mask

    def _compute_text_ce_without_aff_placeholders(
        self,
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        ignore_index: int = -100,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        自定义 CE：
        - 仍使用标准 next-token CE（shift 对齐）
        - 但忽略标签中 <img_aff>/<pc_aff> 的位置，不监督 MLLM 必须生成这两个占位 token
        - 这些位置由 router + 下游解码器损失来学习
        """
        if logits is None or labels is None:
            return None, 0
        if logits.dim() != 3 or labels.dim() != 2:
            return None, 0
        if logits.shape[0] != labels.shape[0]:
            return None, 0

        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()      # [B, L-1]
        valid = shift_labels.ne(ignore_index)
        aff_placeholder_mask = shift_labels.eq(self.img_placeholder_id) | shift_labels.eq(self.pc_placeholder_id)
        # 忽略路由占位 token 标签位：不让 CE 直接监督它们
        ignored_tokens = int((valid & aff_placeholder_mask).sum().item())
        valid = valid & (~aff_placeholder_mask)
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

        # ---- 1. MLLM 前向 ----
        mllm_out = self.mllm(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            point_clouds=point_clouds,
            pc_valid_lengths=pc_valid_lengths,
        )
        hidden_states = mllm_out["hidden_states"]  # [B, L, C]
        output_obj = mllm_out.get("output")
        B,L,C = hidden_states.shape

        logits_token_ids = None
        routed_token_ids = None
        ce_loss = None
        ce_ignored_token_count = 0
        if output_obj is not None:
            # 从 logits 中取 token_ids（用于可视化与占位 token 回写，不参与路由决策）
            if getattr(output_obj, "logits", None) is not None:
                logits_token_ids = output_obj.logits.argmax(dim=-1)
                # CE 改为“忽略 <img_aff>/<pc_aff> 标签位”的版本
                ce_loss, ce_ignored_token_count = self._compute_text_ce_without_aff_placeholders(
                    logits=output_obj.logits,
                    labels=labels,
                    ignore_index=-100,
                )
            # 兜底：无 logits 时沿用底座返回 loss
            if ce_loss is None and getattr(output_obj, "loss", None) is not None:
                ce_loss = output_obj.loss
        
        image_logits = None
        point_logits = None
        route_logits = None
        route_probs = None

        if hidden_states is not None:
            img_available = img_valid_mask if img_valid_mask is not None else None
            pc_available = (pc_valid_lengths > 0) if pc_valid_lengths is not None else None
            route_logits, route_probs, hard_route = self._route_hidden_states(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                img_available=img_available,
                pc_available=pc_available,
            )
            img_emb, pc_emb, img_token_emb, pc_token_emb = self._build_branch_embeddings(
                hidden_states=hidden_states,
                route_probs=route_probs,
            )
            route_mask = self._build_route_mask_from_labels(labels=labels, seq_len=hidden_states.shape[1])
            # 路由后 token_ids 仅基于模型 logits，避免回退到输入文本造成评估/可视化偏差
            routed_token_ids = self._build_routed_token_ids(
                logits_token_ids, hard_route, route_mask=route_mask
            )
            aff_token_pairs = self._build_aff_token_pairs(
                hard_route=hard_route,
                img_token_emb=img_token_emb,
                pc_token_emb=pc_token_emb,
                attention_mask=attention_mask,
                route_mask=route_mask,
            )

            image_pred_emb = self.image_decoder.project_hidden_states(img_emb)
            point_pred_emb = self.point_decoder.project_hidden_states(pc_emb)

            # ---- 3. 2D 图像分割 ----
            image_embeddings = self.image_decoder.get_visual_embs(images)
            input_size = (images.shape[-2], images.shape[-1])
            # 训练时 decoder 输出需与 img_gt_tensor 一致（均为 padding 后的 input_size）
            # 推理保存时再按 original_size_list 缩放还原
            original_size = input_size

            all_image_logits = self.image_decoder(image_pred_emb, image_embeddings, input_size, original_size)

            # 将无效样本的输出置零（不影响 loss 计算）
            if img_valid_mask is not None:
                mask_2d = img_valid_mask.bool().view(B, 1, 1).to(all_image_logits.dtype)
                image_logits = all_image_logits * mask_2d
            else:
                image_logits = all_image_logits

            # ---- 4. 3D 点云分割 ----
            all_point_logits = self.point_decoder(point_pred_emb, point_clouds)

            # 将无效样本的输出置零
            if pc_valid_lengths is not None:
                mask_3d = (pc_valid_lengths > 0).to(all_point_logits.dtype).unsqueeze(-1)
                point_logits = all_point_logits * mask_3d
            else:
                point_logits = all_point_logits


        output_dict = {
            "hidden_states": None,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "token_ids": routed_token_ids,
            "labels": labels,
            # 语言模型交叉熵损失（若未提供 labels 或模型未返回 loss，则为 None）
            "ce_loss": ce_loss,
            "output": None,
            # 用于下游分支的 token 名称与向量（供 validate 等记录）
            # 格式: List[List[Tuple[str, Tensor]]]，每样本 [("<img-x>", emb), ("<pc-x>", emb), ...]
            "aff_token_pairs": None,
            # 路由监督辅助输出
            "route_logits": route_logits,
            "route_probs": route_probs,
            "img_placeholder_id": self.img_placeholder_id,
            "pc_placeholder_id": self.pc_placeholder_id,
            # batch 级统计：CE 中被忽略的 <img_aff>/<pc_aff> 标签 token 数
            "ce_ignored_token_count": ce_ignored_token_count,
        }

        if hidden_states is not None:
            output_dict["aff_token_pairs"] = aff_token_pairs
        if return_hidden_states:
            output_dict["hidden_states"] = hidden_states
        if return_mllm_output:
            output_dict["output"] = mllm_out.get("output")

        return output_dict


__all__ = [
    "JointAffordanceModel",
    "MLLMBackbone",
    "ImageHiddenStateDecoder",
    "PointCloudHiddenStateDecoder",
]
