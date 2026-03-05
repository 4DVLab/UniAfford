"""
JointAffordance模型骨架，子架构分布到其他model中并作为模块导入
"""
from typing import Optional, Dict, List
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
        # 以 mllm 初始化后的 token 索引为准。
        self.functional_tokens = self.mllm.functional_tokens
        self.functional_token_ids = self.mllm.functional_token_ids

        self.image_decoder = ImageHiddenStateDecoder(self.config.image_decoder, self.config.mllm.hidden_size)
        self.point_decoder = PointCloudHiddenStateDecoder(self.config.point_decoder, self.config.mllm.hidden_size)


    @property
    def tokenizer(self): return self.mllm.tokenizer

    @property
    def processor(self): return self.mllm.processor

    def _extract_token_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        token_idx: int,
    ) -> torch.Tensor:
        """
        从隐藏状态中提取匹配特殊 token（如 [SEG]）位置的嵌入。

        当前假设每个样本恰好有 1 个 [SEG] token。
        TODO: 若后续支持多 [SEG]（一个 SEG 对应一组图像/点云），
              应返回 [B, N_seg, C] 并在下游逐 SEG 生成 mask。

        Args:
            last_hidden_state: [B, L', C] 投影后的隐藏状态。
            token_ids: [B, L] 的 token id 序列（通常来自模型输出 logits 的 argmax）。
            token_idx: 特殊 token 的词汇表索引。

        Returns:
            token_embeddings: [B, C] — 每个样本中第一个匹配 token 的嵌入。
                若某样本无匹配 token，对应行为零向量。
        """
        B, _, C = last_hidden_state.shape
        if token_ids is None or token_idx is None:
            return last_hidden_state.new_zeros(B, C)

        token_mask = (token_ids == int(token_idx))

        # 若长度不一致，截断到较短的公共长度
        if last_hidden_state.shape[1] != token_mask.shape[1]:
            min_len = min(last_hidden_state.shape[1], token_mask.shape[1])
            token_mask = token_mask[:, :min_len]
            last_hidden_state = last_hidden_state[:, :min_len, :]

        # 取每个样本中第一个匹配位置的嵌入（无匹配时返回零向量）
        # has_token: [B], first_idx: [B]（无匹配时 first_idx 为 0，但会被 has_token 置零）
        has_token = token_mask.any(dim=1)                                    # [B]
        first_idx = token_mask.to(torch.long).argmax(dim=1)                  # [B]
        # 用 gather 提取：[B, L, C] → [B, 1, C] → [B, C]
        embeddings = last_hidden_state.gather(
            1, first_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, C)
        ).squeeze(1)                                                         # [B, C]
        # 无匹配 token 的样本置零
        embeddings = embeddings * has_token.unsqueeze(-1).to(embeddings.dtype)
        return embeddings

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

        return_hidden_states: bool = False,
        return_mllm_output: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        B = input_ids.shape[0] if input_ids is not None else 1

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
        
        
        # 训练稳定性保底：
        # 若某样本在 logits_token_ids 中未命中 [SEG]，则回退该样本到 input_ids 进行 [SEG] hidden 提取。
        token_ids_for_seg = logits_token_ids
        if input_ids is not None:
            if token_ids_for_seg is None:
                token_ids_for_seg = input_ids
            else:
                token_ids_for_seg = token_ids_for_seg.clone()
                min_len = min(token_ids_for_seg.shape[1], input_ids.shape[1])
                pred_has_seg = (token_ids_for_seg[:, :min_len] == int(self.img_aff_token)).any(dim=1)
                input_has_seg = (input_ids[:, :min_len] == int(self.img_aff_token)).any(dim=1)
                fallback_rows = (~pred_has_seg) & input_has_seg
                if fallback_rows.any():
                    token_ids_for_seg[fallback_rows, :min_len] = input_ids[fallback_rows, :min_len]

        image_logits = None
        point_logits = None

        if hidden_states is not None:
            # ---- 2. 先提取 SEG token 的单 token hidden_state，再分别做投影 ----
            # 分离2D\3D任务的token语义空间
            img_aff_token = self._extract_token_embeddings(
                hidden_states, token_ids_for_seg, self.functional_token_ids["img_aff_token"]
            )  # [B, C]
            pc_aff_token = self._extract_token_embeddings(
                hidden_states, token_ids_for_seg, self.functional_token_ids["pc_aff_token"]
            )  # [B, C]
            image_pred_emb = self.image_decoder.project_hidden_states(img_aff_token)
            point_pred_emb = self.point_decoder.project_hidden_states(pc_aff_token)

            # ---- 3. 2D 图像分割 ----
            image_embeddings = self.image_decoder.get_visual_embs(images)
            input_size = (images.shape[-2], images.shape[-1])
            original_size = tuple(original_size_list[0]) if original_size_list else input_size

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
            "token_ids": logits_token_ids,
            "labels": labels,
            # 语言模型交叉熵损失（若未提供 labels 或模型未返回 loss，则为 None）
            "ce_loss": ce_loss,
            "output": None,
        }

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
