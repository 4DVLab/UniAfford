"""
训练工具函数
"""
import torch
import torch.distributed as dist
from enum import Enum

# 常量定义
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# 点云相关常量
DEFAULT_PC_TOKEN = "<point_cloud>"
DEFAULT_PC_START_TOKEN = "<pc_start>"
DEFAULT_PC_END_TOKEN = "<pc_end>"


class Summary(Enum):
    """统计汇总类型"""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """分布式训练时同步所有进程的统计值"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """进度显示器"""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    """将字典中的所有张量移动到 CUDA 设备"""
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            input_dict[k] = [item.cuda(non_blocking=True) for item in v]
    return input_dict


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """
    计算交集和并集（GPU 版本）
    
    Args:
        output: 预测结果
        target: 真实标签
        K: 类别数
        ignore_index: 忽略的索引值
        
    Returns:
        intersection: 交集
        union: 并集
        target_area: 目标区域面积
    """
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnion3D(pred_mask, gt_mask, threshold=0.5):
    """
    计算 3D 点云掩码的交集和并集
    
    Args:
        pred_mask: 预测掩码 [N] 或 [B, N]，值在 [0, 1] 之间
        gt_mask: 真实掩码 [N] 或 [B, N]，二值
        threshold: 二值化阈值
        
    Returns:
        intersection: 交集点数
        union: 并集点数
        iou: IoU 值
    """
    # 二值化预测掩码
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    # 计算交集和并集
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    # 计算 IoU
    iou = intersection / (union + 1e-8)
    
    return intersection, union, iou


def get_model_device(model):
    """获取模型所在的设备"""
    return next(model.parameters()).device


def count_parameters(model, trainable_only=True):
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', 0)
