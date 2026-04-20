from configs import ImageDecoderConfigs
import torch
import torch.nn as nn
from collections import OrderedDict
from .build_sam import build_sam_vit_h


class ImageHiddenStateDecoder(nn.Module):
    """使用文本隐藏状态查询图像特征的解码器。"""

    def __init__(
        self,
        config: ImageDecoderConfigs,
        text_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.visual_model = build_sam_vit_h(getattr(config, "vision_pretrained", None))

        text_fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(text_hidden_size, 2*text_hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2*text_hidden_size, self.config.hidden_size)),
            # ("dropout", nn.Dropout(0.0)),
        ]))
        self.text_hidden_fcs = nn.ModuleList([text_fc])
        for param in self.parameters():
            param.requires_grad = True

        self.image_encoder = self.visual_model.image_encoder

        self.to(dtype=self.config.compute_dtype)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 LLM 隐藏状态投影到 SAM prompt 空间。"""
        # assert len(self.text_hidden_fcs) == 1
        hidden_states = hidden_states.to(self.config.compute_dtype)
        projected = self.text_hidden_fcs[0](hidden_states)
        return projected

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """使用 SAM image_encoder 提取图像特征，不计算梯度。"""
        pixel_values = pixel_values.to(dtype=self.config.compute_dtype)
        with torch.no_grad():
            return self.visual_model.image_encoder(pixel_values)

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        input_size: tuple,
        original_size: tuple,
    ) -> torch.Tensor:
        """
        批量生成 2D 分割掩码。

        Args:
            pred_embeddings: [B, C] — 每个样本的 SEG token 投影嵌入（已 mean-pool）
            image_embeddings: [B, C_sam, H_emb, W_emb] — SAM 图像编码特征
            input_size: (H, W) — 输入图像尺寸（batch 内统一）
            original_size: (H, W) — 原始图像尺寸（batch 内统一）

        Returns:
            pred_masks: [B, H_orig, W_orig]
        """
        pred_embeddings = self.project_hidden_states(pred_embeddings).to(self.config.compute_dtype)
        image_embeddings = image_embeddings.to(self.config.compute_dtype)

        # [B, C] → [B, 1, C]，作为 prompt_encoder 的 text_embeds 输入
        text_embeds = pred_embeddings.unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
            points=None, boxes=None, masks=None,
            text_embeds=text_embeds,
        )
        sparse_embeddings = sparse_embeddings.to(pred_embeddings.dtype)

        # mask_decoder 已支持 batch：image_embeddings [B,...] 与 sparse [B,...] 一一对应
        low_res_masks, _ = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: [B, 1, H_low, W_low]

        pred_masks = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        # [B, 1, H, W] → [B, H, W]
        return pred_masks[:, 0]

