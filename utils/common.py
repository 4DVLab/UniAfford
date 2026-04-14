import os
import shutil
import logging
from datetime import datetime

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# 常量定义
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<image_pad>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Qwen-VL 常见视觉相关 token（不同版本命名可能略有差异，保留便于比对）
QWEN_VL_VISION_START_TOKEN = "<|vision_start|>"
QWEN_VL_VISION_END_TOKEN = "<|vision_end|>"
QWEN_VL_IMAGE_PAD_TOKEN = "<|image_pad|>"
QWEN_VL_VIDEO_PAD_TOKEN = "<|video_pad|>"

# 点云相关常量
DEFAULT_PC_TOKEN = "<pointcloud>"
# 与 Qwen 视觉 token 风格对齐，统一采用 <|...|> 形式
DEFAULT_PC_PATCH_TOKEN = "<|point_pad|>"
DEFAULT_PC_START_TOKEN = "<|pc_start|>"
DEFAULT_PC_END_TOKEN = "<|pc_end|>"

# 用于下游分割任务的功能性 token。
# 结构：
# {
#   "img": {token_name -> token_id(运行后) / token_str(初始化)},
#   "pc":  {token_name -> token_id(运行后) / token_str(初始化)},
# }
# 运行时会补全反向映射：token_id -> token_name。
FUNCTIONAL_TOKENS = {
    "img": {
        "img_aff_token": "<img_aff>",  # default fallback
    },
    "pc": {
        "pc_aff_token": "<pc_aff>",    # default fallback
    },
}


# ====================== 分布式日志工具 ======================

class _RankFilter(logging.Filter):
    """为 log record 注入 rank 属性，同时可限制只放行指定 rank。"""

    def __init__(self, rank: int, max_rank: int = -1):
        super().__init__()
        self.rank = rank
        self.max_rank = max_rank          # -1 表示不限制

    def filter(self, record):
        record.rank = self.rank           # 注入 rank，供 %(rank)s 使用
        if self.max_rank >= 0:
            return self.rank <= self.max_rank
        return True


def setup_logger(log_dir, local_rank=0, max_console_rank=1):
    """
    设置分布式训练日志系统。

    - 控制台: 仅 rank <= max_console_rank 输出（默认 rank 0/1）
    - 日志文件: 所有 rank 各自写独立文件，保存完整输出
      rank 0 → train_<timestamp>.log
      rank N → train_<timestamp>_rankN.log
    """
    train_logger = logging.getLogger("train")
    train_logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if train_logger.handlers:
        return train_logger

    # 先挂 logger 级 filter，保证所有 record 都有 rank 属性
    train_logger.addFilter(_RankFilter(local_rank))

    formatter = logging.Formatter(
        fmt="%(asctime)s [Rank %(rank)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台: 仅 rank <= max_console_rank 放行
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RankFilter(local_rank, max_rank=max_console_rank))
    train_logger.addHandler(console_handler)

    # 日志文件: 所有 rank 各自写独立文件
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"train_{timestamp}.log" if local_rank == 0 else f"train_{timestamp}_rank{local_rank}.log"
        log_file = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RankFilter(local_rank))   # 注入 rank 属性
        train_logger.addHandler(file_handler)
        train_logger.info(f"日志文件已创建: {log_file}")

    return train_logger


"""  ----------------------------------------------- utils functions ----------------------------------------------  """
def resolve_dtype(dtype_name):
    """解析 dtype（支持字符串或 torch.dtype）为 torch.dtype。"""
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if isinstance(dtype_name, str):
        key = dtype_name.strip().lower()
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32", "float"}:
            return torch.float32
    return None


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


def dict_to_cuda(input_dict, device=None):
    """将字典中的所有张量移动到指定设备，保持原有精度不变"""
    target_device = device or "cuda"
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device=target_device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            input_dict[k] = [item.to(device=target_device, non_blocking=True) for item in v]
    return input_dict

def get_model_device(model):
    """获取模型所在的设备"""
    return next(model.parameters()).device


def count_parameters(model, trainable_only=True):
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

# HACK
def save_checkpoints(model_engine, config, epoch, best_score, cur_ciou, save_dir):
    """
    修复后的Checkpoint保存函数：兼容DeepSpeed，保证分布式安全、文件完整性
    Args:
        model_engine: DeepSpeed封装的模型引擎
        config: 配置对象（含local_rank、log_dir等）
        epoch: 当前轮次
        best_score: 最佳giou
        cur_ciou: 当前ciou
        save_dir: 完整DeepSpeed Checkpoint保存目录
    """
    # ---------------- 方案1：保存完整DeepSpeed Checkpoint（断点续训） ----------------
    if config.local_rank == 0:
        # 安全删除目录：先检查存在性，加异常处理
        if os.path.exists(save_dir):
            try:
                shutil.rmtree(save_dir)
                logger.info(f"Deleted existing checkpoint dir: {save_dir}")
            except Exception as e:
                logger.error(f"Failed to delete dir {save_dir}: {e}", exc_info=True)
                raise  # 目录删除失败，终止保存（避免脏数据）
    
    # 分布式屏障：确保所有进程等待0号进程完成目录删除
    dist.barrier()
    
    # DeepSpeed保存完整Checkpoint（内置分布式兼容）
    model_engine.save_checkpoint(save_dir)
    if config.local_rank == 0:
        logger.info(f"DeepSpeed checkpoint saved to {save_dir}")

    # ---------------- 方案2：保存轻量级推理用Checkpoint ----------------
    if config.local_rank == 0:
        # 1. 检查日志目录是否存在，不存在则创建
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 2. 提取完整的模型state_dict（含parameters + buffers，兼容冻结层）
        # 从model_engine.module中获取原始模型，避免DeepSpeed封装层干扰
        model = model_engine.module
        model_state_dict = model.state_dict()
        
        # 3. 安全迁移到CPU（detach避免梯度关联，分步操作降低显存峰值）
        lightweight_state_dict = {}
        for name, tensor in model_state_dict.items():
            # 仅保存参数/缓冲区（过滤非张量值），detach后迁移到CPU
            if isinstance(tensor, torch.Tensor):
                lightweight_state_dict[name] = tensor.detach().cpu()
        
        # 4. 构造轻量级Checkpoint（补充epoch/指标，命名加epoch避免重复）
        ckpt_name = f"lightweight_epoch{epoch}_giou{best_score:.3f}_ciou{cur_ciou:.3f}.pth"
        lightweight_path = os.path.join(config.log_dir, ckpt_name)
        # 原子写入：先存临时文件，成功后重命名（避免写入失败损坏文件）
        temp_path = lightweight_path + ".tmp"
        
        lightweight_ckpt = {
            "epoch": epoch,
            "model_state_dict": lightweight_state_dict,
            "best_giou": best_score,
            "best_ciou": cur_ciou,
            "config": config  # 可选：保存配置，方便推理时复现
        }
        
        try:
            torch.save(lightweight_ckpt, temp_path)
            os.rename(temp_path, lightweight_path)  # 原子重命名
            logger.info(f"Saved lightweight checkpoint to {lightweight_path}")
        except Exception as e:
            logger.error(f"Failed to save lightweight checkpoint: {e}", exc_info=True)
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        
        # 5. 安全清理旧的轻量级Checkpoint（仅保留最新，过滤文件类型+异常处理）
        try:
            # 遍历log_dir下的文件，筛选轻量级ckpt
            ckpt_files = []
            for fname in os.listdir(config.log_dir):
                fpath = os.path.join(config.log_dir, fname)
                # 仅处理文件 + 以lightweight_开头 + 不是当前新文件
                if (os.path.isfile(fpath) and 
                    fname.startswith("lightweight_") and 
                    fname != ckpt_name):
                    ckpt_files.append(fpath)
            
            # 删除所有旧ckpt（若需保留多个，可按epoch排序后保留前N个）
            for old_fpath in ckpt_files:
                os.remove(old_fpath)
                logger.info(f"Deleted old lightweight checkpoint: {old_fpath}")
        except Exception as e:
            logger.warning(f"Failed to delete old checkpoints: {e}", exc_info=True)

# HACK
def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', 0)

