import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation


class TextEncoder(nn.Module):
    def __init__(self, pretrained_weight=None):
        super().__init__()

    def forward(self, text): ...


class PointEncoder(nn.Module):
    def __init__(self, n_p1=512, n_p2=128, n_p3=64, pretrained_weight=None):
        super().__init__()
        self.n_p = [0, n_p1, n_p2, n_p3]

        self.sa1 = PointNetSetAbstractionMsg(n_p1, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(n_p2, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(n_p3, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])


    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]


class ImageEncoder(nn.Module):
    def __init__(self, pretrained_weight=None):
        super().__init__()

        if pretrained_weight is None:
            self.model = models.resnet18(pretrained=False)
            self.model.relu = nn.ReLU()

    def forward(self, img):
        """HACK: copied from GREAT
        TODO:? 要不要对灰度增加一个关注度
        Args:
            img: 输入的图像，三通道RGB图片
        """

        # B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))

        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        down_1 = self.model.layer2(out)
        down_2 = self.model.layer3(down_1)
        down_3 = self.model.layer4(down_2)

        return down_3

