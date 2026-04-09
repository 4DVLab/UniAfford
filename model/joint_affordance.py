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

from utils.debug import decode_token_ids


class JointAffordanceModel(nn.Module):
    """模型管理基座，负责加载配置并组织各模块。"""

    def __init__(self, config: Optional[JointAffordanceConfig] = None):
        super().__init__()
        self.config = config or JointAffordanceConfig()

        self.mllm = MLLMBackbone(self.config.mllm)
        self.functional_tokens = self.mllm.functional_tokens
        self.functional_token_ids = self.mllm.functional_token_ids
        # 双向映射：token_id -> {name, modality}，用于从 output_ids 中快速查找功能 token
        self.id_to_token_info = getattr(self.mllm, "id_to_token_info", {})

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder, self.config.mllm.hidden_size)
        self.point_decoder = PointCloudHiddenStateDecoder(self.config.point_decoder, self.config.mllm.hidden_size)
        self.point_encoder = getattr(self.mllm, "point_encoder", None)


    @property
    def tokenizer(self): return self.mllm.tokenizer

    @property
    def processor(self): return self.mllm.processor

    def _extract_aff_from_output_ids(
        self,
        hidden_states: torch.Tensor,
        output_ids: torch.Tensor,
        id_to_token_info: dict,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[List[Tuple[str, torch.Tensor]]]]:
        """
        从 output_ids 中按顺序查找功能 token（在 id_to_token_info 中），提取 hidden state。
        按模态分组，下游 2D/3D 分支对其对应所有 token 推理。

        Returns:
            Dict[str, List[List[Tuple[str, Tensor]]]]:
                - "img": 每样本 [(token_str, emb), ...] 该样本所有 img 模态 token
                - "pc": 每样本 [(token_str, emb), ...] 该样本所有 pc 模态 token
        """
        B, L, C = hidden_states.shape
        aff_emb_dict: Dict[str, List[List[Tuple[str, torch.Tensor]]]] = {
            "img": [[] for _ in range(B)],
            "pc": [[] for _ in range(B)],
        }

        for i in range(B):
            if attention_mask is not None:
                seq_len = int(attention_mask[i].sum().item())
            else:
                seq_len = min(L, output_ids.shape[1])
            for pos in range(seq_len):
                tid = int(output_ids[i, pos].item())
                if tid not in id_to_token_info:
                    continue
                info = id_to_token_info[tid]
                emb = hidden_states[i, pos, :]
                pair = (info["token"], emb)
                modality = info.get("modality", "pc")
                if modality not in aff_emb_dict:
                    continue
                aff_emb_dict[modality][i].append(pair)

        return aff_emb_dict

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
        model_labels = mllm_out.get("aligned_labels", labels)
        model_attention_mask = mllm_out.get("aligned_attention_mask", attention_mask)
        B,L,C = hidden_states.shape

        # output_ids 选择策略：
        # - 训练：优先使用 logits_token_ids，缺失时允许回退到 input_ids（teacher-forcing 更稳定）
        # - 推理：仅使用 logits_token_ids，不回退用户输入
        logits_token_ids = None
        ce_loss = None
        if output_obj is not None:
            # 1）从 logits 中取出 token_ids，供下游 [SEG] / [AFF] token 提取使用
            if getattr(output_obj, "logits", None) is not None:
                logits_token_ids = output_obj.logits.argmax(dim=-1)
            # 2）若传入了 labels，Qwen 的 output.loss 即为语言模型交叉熵损失
            if getattr(output_obj, "loss", None) is not None:
                # 注意：这里不做缩放，交由 calculator.compute_losses 中的 ce_loss_weight 控制
                ce_loss = output_obj.loss
        
        image_logits = None
        point_logits = None

        if hidden_states is not None:
            if self.training:
                output_ids = logits_token_ids if logits_token_ids is not None else input_ids
            else:
                output_ids = logits_token_ids
            if output_ids is not None and self.id_to_token_info:
                aff_dict = self._extract_aff_from_output_ids(
                    hidden_states, output_ids, self.id_to_token_info, attention_mask=model_attention_mask
                )
                # 每样本所有 token 的 emb 做 mean pool，得到 [B, C] 供 decoder；
                # 注意避免使用 new_zeros + in-place 赋值，确保 2D/3D loss 可回传到 MLLM hidden_states。
                img_emb_list = []
                pc_emb_list = []
                for i in range(B):
                    if aff_dict["img"][i]:
                        img_emb_i = torch.stack([p[1] for p in aff_dict["img"][i]], dim=0).mean(dim=0)
                    else:
                        img_emb_i = hidden_states.new_zeros(C)
                    if aff_dict["pc"][i]:
                        pc_emb_i = torch.stack([p[1] for p in aff_dict["pc"][i]], dim=0).mean(dim=0)
                    else:
                        pc_emb_i = hidden_states.new_zeros(C)
                    img_emb_list.append(img_emb_i)
                    pc_emb_list.append(pc_emb_i)
                img_emb = torch.stack(img_emb_list, dim=0)
                pc_emb = torch.stack(pc_emb_list, dim=0)
                # 供 validate 的 per-sample 列表：img-aff + pc-aff

                aff_token_pairs = [
                    aff_dict["img"][i] + aff_dict["pc"][i]
                    for i in range(B)
                ]
            else:
                img_emb = hidden_states.new_zeros(B, C)
                pc_emb = hidden_states.new_zeros(B, C)
                aff_token_pairs = [[] for _ in range(B)]

            image_pred_emb = self.image_decoder.project_hidden_states(img_emb)

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
            has_per_point_features = (
                point_encoder_outputs is not None
                and point_encoder_outputs.get("per_point_features") is not None
                and point_encoder_outputs.get("per_point_mask") is not None
            )
            if has_per_point_features and point_clouds is not None:
                all_point_logits = self.point_decoder(
                    pred_embeddings=pc_emb,
                    per_point_features=point_encoder_outputs.get("per_point_features"),
                    per_point_mask=point_encoder_outputs.get("per_point_mask"),
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
            "token_ids": logits_token_ids,
            "labels": model_labels,
            "attention_mask": model_attention_mask,
            # 语言模型交叉熵损失（若未提供 labels 或模型未返回 loss，则为 None）
            "ce_loss": ce_loss,
            "output": None,
            # 用于下游分支的 token 名称与向量（供 validate 等记录）
            # 格式: List[List[Tuple[str, Tensor]]]，每样本 [("<img-x>", emb), ("<pc-x>", emb), ...]
            "aff_token_pairs": None,
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
