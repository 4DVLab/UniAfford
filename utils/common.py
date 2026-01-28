import os
import shutil
import torch
import torch.distributed as dist
import logging

# 建议初始化日志（全局配置）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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

