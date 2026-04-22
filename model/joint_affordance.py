"""
JointAffordance模型骨架，子架构分布到其他model中并作为模块导入
"""
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import JointAffordanceConfig
from model.pointcept import PointCloudHiddenStateDecoder
from model.segment_anything import ImageHiddenStateDecoder
from model.qwenvl import MLLMBackbone
from model.HeadRouter import HeadRouter


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def _sync_point_decoder_config(self):
        point_decoder_cfg = getattr(self.config, "point_decoder", None)
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

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()

        self.mllm = MLLMBackbone(self.config.mllm)
        self.functional_tokens = self.mllm.functional_tokens
        self.functional_token_ids = self.mllm.functional_token_ids

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder, self.config.mllm.hidden_size)

        self.point_encoder = getattr(self.mllm, "point_encoder", None)
        if self.point_encoder is not None:
            point_feature_size = int(getattr(self.point_encoder, "point_feature_size"))
        else:
            point_feature_size = int(getattr(self.config.mllm.point_encoder_backbone, "dec_channels", (64,))[0])

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
            img_placeholder_token="<img_aff>",
            pc_placeholder_token="<pc_aff>",
        )
        # 兼容现有训练/日志代码使用的字段名
        self.img_placeholder_token = self.router.img_placeholder_token
        self.pc_placeholder_token = self.router.pc_placeholder_token
        self.img_placeholder_id = self.router.img_placeholder_id
        self.pc_placeholder_id = self.router.pc_placeholder_id


    @property
    def tokenizer(self): return self.mllm.tokenizer

    @property
    def processor(self): return self.mllm.processor

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

        # ---- 0. 点云编码（单次 backbone，产出 token级 + 逐点级 两路特征）----
        point_encoder_outputs = None
        if self.point_encoder is not None and point_clouds is not None:
            point_encoder_outputs = self.point_encoder.encode_shared(
                point_clouds=point_clouds,
                pc_valid_lengths=pc_valid_lengths,
            )

        # ---- 1. MLLM 前向 ----
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
        B,L,C = hidden_states.shape

        logits_token_ids = None
        ce_loss = None
        ce_ignored_token_count = 0
        if output_obj is not None:
            # 从 logits 中取 token_ids（用于可视化与占位 token 回写，不参与路由决策）
            if getattr(output_obj, "logits", None) is not None:
                logits_token_ids = output_obj.logits.argmax(dim=-1)
                # CE 改为“忽略 <img_aff>/<pc_aff> 标签位”的版本
                ce_loss, ce_ignored_token_count = self._compute_text_ce_without_aff_placeholders(
                    logits=output_obj.logits,
                    labels=model_labels,
                    ignore_index=-100,
                )
            # 兜底：无 logits 时沿用底座返回 loss
            if ce_loss is None and getattr(output_obj, "loss", None) is not None:
                ce_loss = output_obj.loss
        
        if hidden_states is not None:
            img_available = img_valid_mask if img_valid_mask is not None else None
            pc_available = (pc_valid_lengths > 0) if pc_valid_lengths is not None else None
            route_out = self.router(
                hidden_states=hidden_states,
                attention_mask=model_attention_mask,
                img_available=img_available,
                pc_available=pc_available,
                labels=model_labels,
                base_token_ids=logits_token_ids,
            )

            # ---- 3. 2D 图像分割 ----
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

            # 将无效样本的输出置零（不影响 loss 计算）
            if img_valid_mask is not None:
                mask_2d = img_valid_mask.bool().view(B, 1, 1).to(all_image_logits.dtype)
                image_logits = all_image_logits * mask_2d
            else:
                image_logits = all_image_logits

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
                mask_3d = (pc_valid_lengths > 0).to(all_point_logits.dtype).unsqueeze(-1)
                point_logits = all_point_logits * mask_3d
            else:
                point_logits = all_point_logits


        output_dict = {
            "hidden_states": None,
            "image_logits": image_logits,
            "point_logits": point_logits,
            "token_ids": route_out["routed_token_ids"],
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
            "img_placeholder_id": self.router.img_placeholder_id,
            "pc_placeholder_id": self.router.pc_placeholder_id,
            # batch 级统计：CE 中被忽略的 <img_aff>/<pc_aff> 标签 token 数
            "ce_ignored_token_count": ce_ignored_token_count,
        }

        if hidden_states is not None:
            output_dict["aff_token_pairs"] = route_out["aff_token_pairs"]
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
