import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

try:
    # Qwen3 文本编码依赖 transformers
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # transformers 未安装时的降级处理
    AutoModel = None
    AutoTokenizer = None


class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


""" ---------------------------------------------- encoders ------------------------------------------------- """
class TextEncoder(nn.Module):
    """
    使用 Qwen3 作为文本编码器的 TextEncoder 封装。

    - 默认使用 Qwen3 系列模型（可通过 pretrained_weight / model_name 指定）
    - 输出一个句子级别的向量表示（平均池化 last_hidden_state 后再线性投影）
    """

    def __init__(
        self,
        pretrained_weight: str | None = None,
        model_name: str = "Qwen/Qwen2.5-1.8B-Instruct",
        proj_dim: int = 1024,
        device: str | None = None,
    ):
        """
        Args:
            pretrained_weight: 可选，用于覆盖默认的 model_name，
                例如传入本地 Qwen3 权重路径或 HuggingFace 上的模型名；
                若为 None 则使用默认 model_name。
            model_name: 默认的 Qwen3 模型名（当 pretrained_weight 为 None 时使用）。
            proj_dim: 将 Qwen3 的 hidden_size 投影到的维度；
                若希望保持原始 hidden_size，可将 proj_dim 设为 -1。
            device: 运行设备，默认自动选择 cuda / cpu。
        """
        super().__init__()

        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                "未检测到 transformers 库或其依赖，无法使用 Qwen3 文本编码器。\n"
                "请先安装：pip install transformers accelerate\n"
                "并确保可以从 HuggingFace 或本地加载 Qwen3 模型。"
            )

        model_name = pretrained_weight or model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer & backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.backbone.to(self.device)

        hidden_size = self.backbone.config.hidden_size
        if proj_dim is None or proj_dim <= 0 or proj_dim == hidden_size:
            self.proj = nn.Identity()
            self.out_dim = hidden_size
        else:
            self.proj = nn.Linear(hidden_size, proj_dim)
            self.out_dim = proj_dim

    def forward(self, text):
        """
        Args:
            text:
                - List[str]: 原始文本列表
                - torch.LongTensor: 已经 tokenized 的 input_ids，[B, L]
                - dict: 已经准备好的 tokenizer 输出（包含 input_ids / attention_mask 等）

        Returns:
            emb: [B, out_dim] 的句子级别 embedding
        """
        # 1) 准备输入
        if isinstance(text, dict) and "input_ids" in text:
            inputs = {k: v.to(self.device) for k, v in text.items()}
        elif isinstance(text, torch.Tensor):
            # 认为是 input_ids
            input_ids = text.to(self.device)
            if self.tokenizer.pad_token_id is None:
                pad_id = 0
            else:
                pad_id = self.tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long()
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            # 假定是 List[str]
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2) 编码
        outputs = self.backbone(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        attention_mask = inputs.get("attention_mask", None)

        # 3) 池化为句子向量（mask average pooling）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (hidden_states * mask).sum(dim=1)  # [B, H]
            lengths = mask.sum(dim=1).clamp(min=1.0)    # [B, 1]
            pooled = summed / lengths
        else:
            pooled = hidden_states.mean(dim=1)

        # 4) 线性投影到目标维度
        emb = self.proj(pooled)  # [B, out_dim]
        return emb


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

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 3, npoint_sa1] --- [B, 320, npoint_sa1]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, npoint_sa2] --- [B, 512, npoint_sa2]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, N_p]        --- [B, 512, N_p]

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
        down_0 = self.model.layer1(out)
        down_1 = self.model.layer2(down_0)
        down_2 = self.model.layer3(down_1)
        down_3 = self.model.layer4(down_2)

        return down_0, down_1, down_2, down_3


""" ---------------------------------------------- decoders ------------------------------------------------- """
class TextDecoder(nn.Module):
    def __init__(self, pretrained_weight=None):
        super().__init__()

    def forward(self, token):...

class PointHeatDecoder(nn.Module):
    def __init__(self, n_p1=512, n_p2=128, n_p3=64, pretrained_weight=None):
        super().__init__()
        self.n_p = [0, n_p1, n_p2, n_p3]  # num of points

        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        # upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + self.emb_dim, mlp=[768, 512])
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512])
        self.fp1 = PointNetFeaturePropagation(in_channel=518 + additional_channel, mlp=[512, 512])

        self.cmff = Cross_Modal_Feature_Fusion(emb_dim, proj_dim)
        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )
        self.reshape = nn.Sequential(
            nn.Linear(49, 49 * 8),
            SwapAxes(),
            nn.BatchNorm1d(49 * 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(49 * 8, 2048),
        )
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )


    def forward(self, xyz_token):...

    def forward(self, T_o, I_h, encoder_p):
        '''
        T_o --->object knowledge embedding
        I_h ---> [B, N_i, C]
        encoder_p  ---> [Hierarchy feature]
        '''
        B, _, _ = I_h.shape

        p_0, p_1, p_2, p_3 = encoder_p

        p_3[1] = self.cmff(T_o, p_3[1].transpose(-2, -1))
        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])

        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)

        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]], 1), up_sample)

        F_I = self.reshape(I_h.permute(0, 2, 1))

        F_j = torch.cat((F_I, up_sample), dim=1)
        F_j_fusion = self.fusion(F_j)

        aff3d = self.out_head(F_j_fusion.permute(0, 2, 1))
        aff3d = self.sigmoid(aff3d)

        return aff3d

class ImageHeatDecoder(nn.Module):
    def __init__(self, pretrained_weight=None):
        super().__init__()


