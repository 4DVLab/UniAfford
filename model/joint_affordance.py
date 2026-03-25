"""
JointAffordance模型骨架，子架构分布到其他model中并作为模块导入
"""
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # route_logits/probs: [B, L, 3]，三类分别为 text/img/pc
        route_logits = self.route_head(hidden_states)
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
    ) -> Optional[torch.Tensor]:
        if base_token_ids is None:
            return None
        routed = base_token_ids.clone()
        routed = torch.where(
            hard_route == self.route_img_idx,
            torch.full_like(routed, self.img_placeholder_id),
            routed,
        )
        routed = torch.where(
            hard_route == self.route_pc_idx,
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
    ) -> List[List[Tuple[str, torch.Tensor]]]:
        B, L = hard_route.shape
        aff_token_pairs: List[List[Tuple[str, torch.Tensor]]] = [[] for _ in range(B)]
        for i in range(B):
            seq_len = int(attention_mask[i].sum().item()) if attention_mask is not None else L
            for pos in range(seq_len):
                rid = int(hard_route[i, pos].item())
                if rid == self.route_img_idx:
                    aff_token_pairs[i].append((self.img_placeholder_token, img_token_emb[i, pos, :]))
                elif rid == self.route_pc_idx:
                    aff_token_pairs[i].append((self.pc_placeholder_token, pc_token_emb[i, pos, :]))
        return aff_token_pairs

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
        )
        hidden_states = mllm_out["hidden_states"]  # [B, L, C]
        output_obj = mllm_out.get("output")
        B,L,C = hidden_states.shape

        logits_token_ids = None
        routed_token_ids = None
        ce_loss = None
        if output_obj is not None:
            # 从 logits 中取 token_ids（用于可视化与占位 token 回写，不参与路由决策）
            if getattr(output_obj, "logits", None) is not None:
                logits_token_ids = output_obj.logits.argmax(dim=-1)
            # 传入 labels 时，Qwen output.loss 即语言模型 CE
            if getattr(output_obj, "loss", None) is not None:
                ce_loss = output_obj.loss
        
        image_logits = None
        point_logits = None
        route_logits = None
        route_probs = None

        if hidden_states is not None:
            route_logits, route_probs, hard_route = self._route_hidden_states(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            img_emb, pc_emb, img_token_emb, pc_token_emb = self._build_branch_embeddings(
                hidden_states=hidden_states,
                route_probs=route_probs,
            )
            # 路由后 token_ids 仅基于模型 logits，避免回退到输入文本造成评估/可视化偏差
            routed_token_ids = self._build_routed_token_ids(logits_token_ids, hard_route)
            aff_token_pairs = self._build_aff_token_pairs(
                hard_route=hard_route,
                img_token_emb=img_token_emb,
                pc_token_emb=pc_token_emb,
                attention_mask=attention_mask,
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
