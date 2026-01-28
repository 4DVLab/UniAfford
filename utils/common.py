

import os
import torch

"""  ----------------------------------------------- utils functions ----------------------------------------------  """

def resolve_path(path_str: str):
    """兼容相对/绝对路径，返回绝对路径。"""
    if path_str is None: return None
    return path_str if os.path.isabs(path_str) else os.path.abspath(os.path.join(os.getcwd(), path_str))


def clean_quotes(value:str):
    """去除字段两边的引号"""
    if not value: return ''
    while len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value

"""
训练工具函数
"""
import torch
import torch.distributed as dist

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

