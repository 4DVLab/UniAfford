"""代码架构未完成，不使用"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from en_decoder import TextEncoder, ImageEncoder, PointEncoder, ImageHeatDecoder, PointHeatDecoder


class JointAff(nn.Module):
    """
    文本-图像-点云联合affordance预测模型。

    - 文本（Instruction）经 TextEncoder 得到对象/动作语义嵌入 T_o
    - 图像经 ImageEncoder 得到多尺度特征，用于生成图像token I_h
    - 点云经 PointEncoder（PointNet++ 层级结构）得到分层特征 encoder_p
    - PointHeatDecoder 参考 PointNet++，利用 T_o、I_h 和 encoder_p 生成每点 3D affordance 预测
    - 预留 ImageHeatDecoder 接口用于 2D heatmap 预测
    """

    def __init__(self,
                 pretrained_weight: dict | None = None,
                 use_image: bool = True,
                 use_pointcloud: bool = True):
        super().__init__()

        self.use_image = use_image
        self.use_pointcloud = use_pointcloud

        # 文本编码器：将指令/文本编码为对象语义向量 T_o
        # 具体结构在 TextEncoder 内部实现
        self.text_encoder = TextEncoder(
            pretrained_weight=None if pretrained_weight is None else pretrained_weight.get("text_encoder")
        )

        # 图像编码器：ResNet backbone，输出多尺度特征
        if self.use_image:
            self.img_encoder = ImageEncoder(
                pretrained_weight=None if pretrained_weight is None else pretrained_weight.get("image_encoder")
            )
            # 2D heatmap decoder（当前在 en_decoder 中尚未实现具体结构，这里仅占位）
            self.img_decoder = ImageHeatDecoder(
                pretrained_weight=None if pretrained_weight is None else pretrained_weight.get("image_decoder")
            )

        # 点云编码器：PointNet++ 多尺度集合采样 + 特征传播
        if self.use_pointcloud:
            self.point_encoder = PointEncoder(
                pretrained_weight=None if pretrained_weight is None else pretrained_weight.get("point_encoder")
            )
            # 3D heat decoder，已在 en_decoder 中实现基于 PointNet++ 的上采样与融合
            self.point_decoder = PointHeatDecoder(
                pretrained_weight=None if pretrained_weight is None else pretrained_weight.get("point_decoder")
            )

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        编码文本指令。

        Args:
            text_tokens: 文本输入，形状和类型由 TextEncoder 自行约定（例如 [B, L] 或 [B, L, D]）

        Returns:
            T_o: 文本/对象语义嵌入，形状 [B, C_t]
        """
        return self.text_encoder(text_tokens)

    def encode_image(self, img: torch.Tensor):
        """
        图像编码。

        Args:
            img: [B, 3, H, W]

        Returns:
            down_0, down_1, down_2, down_3: 多尺度特征
        """
        assert self.use_image, "use_image=False 时不应调用 encode_image"
        return self.img_encoder(img)

    def image_to_tokens(self, down_3: torch.Tensor) -> torch.Tensor:
        """
        将图像最高层特征 map 展开为 token 序列，供 PointHeatDecoder / ImageHeatDecoder 使用。

        Args:
            down_3: [B, C, H, W]，来自 ImageEncoder 的最高层特征

        Returns:
            I_h: [B, N_i, C] 的图像token序列
        """
        B, C, H, W = down_3.shape
        I_h = down_3.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, N_i, C]
        return I_h

    def encode_pointcloud(self, points: torch.Tensor):
        """
        点云编码（PointNet++ 层级表示）。

        Args:
            points: [B, C, N]，通常 C>=3，前3维为 xyz

        Returns:
            encoder_p: 层级特征列表 [[l0_xyz, l0_points], ..., [l3_xyz, l3_points]]
        """
        assert self.use_pointcloud, "use_pointcloud=False 时不应调用 encode_pointcloud"
        return self.point_encoder(points)

    def forward(self,
                text: torch.Tensor,
                img: torch.Tensor | None = None,
                points: torch.Tensor | None = None,
                gt_aff2d: torch.Tensor | None = None,
                gt_aff3d: torch.Tensor | None = None,
                img_mask: torch.Tensor | None = None,
                pc_mask: torch.Tensor | None = None,
                return_loss: bool = False):
        """
        联合前向过程。

        Args:
            text: 文本输入（Instruction），传给 TextEncoder
            img: 图像张量 [B, 3, H, W]，可选
            points: 点云张量 [B, C, N]，可选
            gt_aff2d: 2D GT 热力图 / mask，形状与输出一致，可选
            gt_aff3d: 3D GT 每点 affordance，形状 [B, N, 1] 或 [B, N]，可选
            img_mask: 图像区域 mask，可选（例如仅在物体区域监督）
            pc_mask: 点云 mask，可选（例如仅在可见点上监督）
            return_loss: 若为 True，则同时返回 loss 字典

        Returns:
            outputs: dict，可能包含：
                - 'aff2d': [B, 1, H', W'] 或 [B, N_i, 1]（视 ImageHeatDecoder 实现而定）
                - 'aff3d': [B, N_p, 1]
                - 'loss': 标量 loss（若 return_loss=True 且提供 GT）
                - 'loss_2d': 2D 分支 loss
                - 'loss_3d': 3D 分支 loss
        """
        outputs: dict[str, torch.Tensor] = {}

        # 1. 文本编码
        T_o = self.encode_text(text)  # 期望形状 [B, C_t]

        # 2. 图像编码（可选）
        if self.use_image and img is not None:
            down_0, down_1, down_2, down_3 = self.encode_image(img)
            I_h = self.image_to_tokens(down_3)  # [B, N_i, C]
        else:
            down_0 = down_1 = down_2 = down_3 = None
            I_h = None

        # 3. 点云编码（可选）
        if self.use_pointcloud and points is not None:
            encoder_p = self.encode_pointcloud(points)
        else:
            encoder_p = None

        # 4. 3D affordance 预测（基于 PointNet++）
        loss_3d = None
        if self.use_pointcloud and encoder_p is not None and I_h is not None:
            # PointHeatDecoder 接口：forward(self, T_o, I_h, encoder_p)
            aff3d = self.point_decoder(T_o, I_h, encoder_p)  # [B, N_p, 1]，Sigmoid 后的概率
            outputs["aff3d"] = aff3d

            if gt_aff3d is not None and return_loss:
                # 对齐形状 [B, N_p, 1]
                if gt_aff3d.dim() == 2:
                    gt_aff3d = gt_aff3d.unsqueeze(-1)
                if pc_mask is not None:
                    # 仅在有效点上计算 loss
                    while pc_mask.dim() < gt_aff3d.dim():
                        pc_mask = pc_mask.unsqueeze(-1)
                    mask = pc_mask.bool()
                    pred = aff3d[mask]
                    target = gt_aff3d[mask]
                else:
                    pred = aff3d
                    target = gt_aff3d

                loss_3d = F.binary_cross_entropy(pred, target)
                outputs["loss_3d"] = loss_3d

        # 5. 2D affordance 预测（预留，视 ImageHeatDecoder 具体实现）
        # 当前 ImageHeatDecoder 为空实现，这里只给出接口和 loss 计算示例
        loss_2d = None
        if self.use_image and img is not None and hasattr(self, "img_decoder"):
            # TODO: 根据 ImageHeatDecoder 实现调整接口
            # 示例：假设 img_decoder 接收 (T_o, encoder_feats) 并输出 [B, 1, H', W']
            # aff2d = self.img_decoder(T_o, (down_0, down_1, down_2, down_3))
            # outputs["aff2d"] = aff2d
            aff2d = None  # 占位

            if aff2d is not None and gt_aff2d is not None and return_loss:
                pred2d = aff2d
                target2d = gt_aff2d
                if img_mask is not None:
                    while img_mask.dim() < target2d.dim():
                        img_mask = img_mask.unsqueeze(1)
                    mask2d = img_mask.bool()
                    pred2d = pred2d[mask2d]
                    target2d = target2d[mask2d]
                loss_2d = F.binary_cross_entropy(pred2d, target2d)
                outputs["loss_2d"] = loss_2d

        # 6. 汇总 loss
        if return_loss and (loss_2d is not None or loss_3d is not None):
            total = 0.0
            if loss_2d is not None:
                total = total + loss_2d
            if loss_3d is not None:
                total = total + loss_3d
            outputs["loss"] = total

        return outputs

__all__ = []