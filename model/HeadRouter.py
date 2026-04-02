from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn


class HeadRouter(nn.Module):
    """
    token 级三路路由器（text/img/pc）。

    当前实现使用“硬路由执行”：
    - 先对每个 token 计算三路概率；
    - 再取 argmax 分配路由；
    - 最终仅使用分配到该分支的 token 构造分支 query。
    """

    def __init__(
        self,
        hidden_size: int,
        tokenizer,
        img_placeholder_token: str = "<img_aff>",
        pc_placeholder_token: str = "<pc_aff>",
    ):
        super().__init__()
        self.route_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
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

        self.img_placeholder_token = img_placeholder_token
        self.pc_placeholder_token = pc_placeholder_token
        self.img_placeholder_id = self._resolve_token_id(tokenizer, self.img_placeholder_token)
        self.pc_placeholder_id = self._resolve_token_id(tokenizer, self.pc_placeholder_token)

    @staticmethod
    def _resolve_token_id(tokenizer, token: str) -> int:
        """
        解析占位 token 在 tokenizer 词表中的 id。

        如果 token 不存在或被映射到 unk，则立刻抛错，避免训练中才发现监督目标不一致。
        """
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and int(token_id) == int(unk_id)):
            raise ValueError(f"占位 token 未注册到 tokenizer: {token}")
        return int(token_id)

    def route_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        img_available: Optional[torch.Tensor] = None,
        pc_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 token 级路由结果。

        Args:
            hidden_states: [B, L, C]，来自主干 MLLM 的 token 表征。
            attention_mask: [B, L]，无效位置会被强制到 text 类。
            img_available: [B]，样本是否可用 2D 分支；不可用则屏蔽 img 类。
            pc_available: [B]，样本是否可用 3D 分支；不可用则屏蔽 pc 类。

        Returns:
            route_logits: [B, L, 3]，三类路由 logits。
            route_probs: [B, L, 3]，softmax 概率。
            hard_route: [B, L]，argmax 离散路由索引。
        """
        route_logits = self.route_head(hidden_states)  # [B, L, 3]

        # 屏蔽样本不可用模态，防止路由误激活无监督分支
        if img_available is not None:
            img_mask = img_available.bool().view(-1, 1)
            route_logits[:, :, self.route_img_idx] = torch.where(
                img_mask,
                route_logits[:, :, self.route_img_idx],
                torch.full_like(route_logits[:, :, self.route_img_idx], -1e4),
            )
        if pc_available is not None:
            pc_mask = pc_available.bool().view(-1, 1)
            route_logits[:, :, self.route_pc_idx] = torch.where(
                pc_mask,
                route_logits[:, :, self.route_pc_idx],
                torch.full_like(route_logits[:, :, self.route_pc_idx], -1e4),
            )

        route_probs = torch.softmax(route_logits, dim=-1)
        hard_route = route_logits.argmax(dim=-1)

        # 无效 token 位置统一归到 text，避免 padding 位置污染分支聚合
        if attention_mask is not None:
            valid = attention_mask.bool()
            hard_route = torch.where(valid, hard_route, torch.full_like(hard_route, self.route_text_idx))
            route_probs = route_probs * valid.unsqueeze(-1).to(route_probs.dtype)

        return route_logits, route_probs, hard_route

    def build_branch_embeddings(
        self,
        hidden_states: torch.Tensor,
        hard_route: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行硬路由聚合，构造分支样本级 query。

        关键点：
        - 仅使用 hard_route 命中分支的 token；
        - 分支没有命中 token 时，query 退化为零向量（通过 clamp_min 防 NaN）。
        """
        img_token_emb = self.img_branch_head(hidden_states)  # [B, L, C]
        pc_token_emb = self.pc_branch_head(hidden_states)    # [B, L, C]

        img_sel = hard_route.eq(self.route_img_idx)
        pc_sel = hard_route.eq(self.route_pc_idx)
        if attention_mask is not None:
            valid = attention_mask.bool()
            img_sel = img_sel & valid
            pc_sel = pc_sel & valid

        img_w = img_sel.unsqueeze(-1).to(img_token_emb.dtype)
        pc_w = pc_sel.unsqueeze(-1).to(pc_token_emb.dtype)
        img_den = img_w.sum(dim=1).clamp_min(1.0)
        pc_den = pc_w.sum(dim=1).clamp_min(1.0)
        img_emb = (img_token_emb * img_w).sum(dim=1) / img_den
        pc_emb = (pc_token_emb * pc_w).sum(dim=1) / pc_den
        return img_emb, pc_emb, img_token_emb, pc_token_emb

    def build_routed_token_ids(
        self,
        base_token_ids: Optional[torch.Tensor],
        hard_route: torch.Tensor,
        route_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        将 hard route 结果回写到 token ids（仅用于日志/可视化）。

        - 命中 img 的位置替换为 <img_aff>
        - 命中 pc 的位置替换为 <pc_aff>
        """
        if base_token_ids is None:
            return None
        routed = base_token_ids.clone()
        img_sel = hard_route.eq(self.route_img_idx)
        pc_sel = hard_route.eq(self.route_pc_idx)
        if route_mask is not None:
            img_sel = img_sel & route_mask
            pc_sel = pc_sel & route_mask
        routed = torch.where(img_sel, torch.full_like(routed, self.img_placeholder_id), routed)
        routed = torch.where(pc_sel, torch.full_like(routed, self.pc_placeholder_id), routed)
        return routed

    def build_aff_token_pairs(
        self,
        hard_route: torch.Tensor,
        img_token_emb: torch.Tensor,
        pc_token_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[str, torch.Tensor]]]:
        """
        构造每样本路由结果明细（token 名称 + token 向量）。

        该结构用于 validate 阶段记录与后续可解释性分析，不参与主损失计算。
        """
        bsz, seq_len = hard_route.shape
        pairs: List[List[Tuple[str, torch.Tensor]]] = [[] for _ in range(bsz)]
        for i in range(bsz):
            cur_len = int(attention_mask[i].sum().item()) if attention_mask is not None else seq_len
            for pos in range(cur_len):
                if route_mask is not None and not bool(route_mask[i, pos].item()):
                    continue
                rid = int(hard_route[i, pos].item())
                if rid == self.route_img_idx:
                    pairs[i].append((self.img_placeholder_token, img_token_emb[i, pos, :]))
                elif rid == self.route_pc_idx:
                    pairs[i].append((self.pc_placeholder_token, pc_token_emb[i, pos, :]))
        return pairs

    @staticmethod
    def build_route_mask_from_labels(
        labels: Optional[torch.Tensor],
        seq_len: int,
        ignore_index: int = -100,
    ) -> Optional[torch.Tensor]:
        """
        依据 labels 构造路由有效位掩码（next-token 对齐）。

        因为 logits[t] 预测的是 labels[t+1]，这里同样采用 p-1 对齐规则。
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        img_available: Optional[torch.Tensor] = None,
        pc_available: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        base_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        路由统一前向入口（供 JointAffordance 直接调用）。

        一次调用完成：
        1) token 路由计算；
        2) 硬路由分支聚合；
        3) route mask 构造；
        4) token 回写与可解释性 pair 构造。
        """
        route_logits, route_probs, hard_route = self.route_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            img_available=img_available,
            pc_available=pc_available,
        )
        img_emb, pc_emb, img_token_emb, pc_token_emb = self.build_branch_embeddings(
            hidden_states=hidden_states,
            hard_route=hard_route,
            attention_mask=attention_mask,
        )
        route_mask = self.build_route_mask_from_labels(
            labels=labels,
            seq_len=hidden_states.shape[1],
        )
        routed_token_ids = self.build_routed_token_ids(
            base_token_ids=base_token_ids,
            hard_route=hard_route,
            route_mask=route_mask,
        )
        aff_token_pairs = self.build_aff_token_pairs(
            hard_route=hard_route,
            img_token_emb=img_token_emb,
            pc_token_emb=pc_token_emb,
            attention_mask=attention_mask,
            route_mask=route_mask,
        )

        return {
            "route_logits": route_logits,
            "route_probs": route_probs,
            "hard_route": hard_route,
            "route_mask": route_mask,
            "img_emb": img_emb,
            "pc_emb": pc_emb,
            "img_token_emb": img_token_emb,
            "pc_token_emb": pc_token_emb,
            "routed_token_ids": routed_token_ids,
            "aff_token_pairs": aff_token_pairs,
        }
