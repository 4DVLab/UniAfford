import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz): ...


class PointEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz):...


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz): ...

