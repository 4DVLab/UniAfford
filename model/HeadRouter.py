from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadRouter(nn.Module):
    """
    token 级三路路由器（text/img/pc）。

    当前 router 的目标不是做 MoE 专家负载均衡，而是做 token 语义角色判别：
    - text: 普通文本 token
    - img : 用于 2D affordance 解码的 token
    - pc  : 用于 3D affordance 解码的 token

    训练时：
    - 路由概率使用 softmax，便于结构损失（存在性/稀疏性）直接回传梯度；
    - 下游执行仍采用 hard argmax，保持“一个 token 只解释成一个分支”的约束。
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

    def build_structure_signals(
        self,
        route_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        基于 softmax 概率构造“可微”的样本结构信号。

        为什么要单独构造这组量：
        - 我们希望模型学到“有图时至少出现一个 img token、无图时不要出现 img token”
        - 这类结构约束若直接定义在 argmax/计数上是离散的，梯度无法稳定回传
        - 因此训练时使用 route_probs 构造连续近似量，再用 BCE / 稀疏损失监督

        这里输出两类信号：
        1) any_prob:
           - noisy-or 近似“至少存在一个该类 token”的概率
           - 公式: 1 - prod(1 - p_t)
        2) expected_count:
           - soft 期望个数，等价于 sum_t p_t(class)

        这些量都定义在 answer 对齐后的有效 token 上：
        - 优先使用 route_mask（与 next-token 监督对齐）
        - 若 route_mask 不存在，则退化为 attention_mask
        """
        if route_mask is not None:
            valid_mask = route_mask.bool()
        elif attention_mask is not None:
            valid_mask = attention_mask.bool()
        else:
            valid_mask = torch.ones(
                route_probs.shape[0],
                route_probs.shape[1],
                dtype=torch.bool,
                device=route_probs.device,
            )

        valid_f = valid_mask.to(route_probs.dtype)
        img_probs = route_probs[:, :, self.route_img_idx] * valid_f
        pc_probs = route_probs[:, :, self.route_pc_idx] * valid_f

        # noisy-or: 至少存在一个 token 命中对应分支的连续近似概率
        img_any_prob = 1.0 - torch.exp(torch.sum(torch.log1p(-img_probs.clamp(max=1 - 1e-6)), dim=1))
        pc_any_prob = 1.0 - torch.exp(torch.sum(torch.log1p(-pc_probs.clamp(max=1 - 1e-6)), dim=1))

        # soft 期望个数：用于控制 affordance token 不要过多
        img_expected_count = img_probs.sum(dim=1)
        pc_expected_count = pc_probs.sum(dim=1)

        return {
            "structure_valid_mask": valid_mask,
            "img_any_prob": img_any_prob.clamp(0.0, 1.0),
            "pc_any_prob": pc_any_prob.clamp(0.0, 1.0),
            "img_expected_count": img_expected_count,
            "pc_expected_count": pc_expected_count,
        }

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
        structure_signals = self.build_structure_signals(
            route_probs=route_probs,
            attention_mask=attention_mask,
            route_mask=route_mask,
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
            **structure_signals,
        }
