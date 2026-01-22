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
        # 防止除零错误
        self.avg = self.sum / self.count if self.count > 0 else 0

    def all_reduce(self):
        """分布式训练时同步所有进程的统计值"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # 检查 self.sum 是否是向量（numpy 数组或列表）
        import numpy as np
        if isinstance(self.sum, (np.ndarray, list)):
            # 向量情况：self.sum 是 shape(N,) 的数组
            sum_array = np.array(self.sum) if isinstance(self.sum, list) else self.sum
            sum_tensor = torch.tensor(sum_array, dtype=torch.float32, device=device)
            count_tensor = torch.tensor([self.count], dtype=torch.float32, device=device)
            
            # 分别对 sum 和 count 进行 all_reduce
            dist.all_reduce(sum_tensor, dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(count_tensor, dist.ReduceOp.SUM, async_op=False)
            
            self.sum = sum_tensor.cpu().numpy()
            self.count = count_tensor.item()
            # 防止除零错误
            self.avg = self.sum / self.count if self.count > 0 else self.sum
        else:
            # 标量情况：self.sum 是单个数值
            total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
            self.sum, self.count = total.tolist()
            # 防止除零错误
            self.avg = self.sum / self.count if self.count > 0 else 0.0

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


def debug_gradient_graph(loss, model, params_to_train):
    print("\n=== 开始梯度图连通性检查 ===")
    
    # 1. 获取所有应该被训练的参数的 ID 集合
    # 我们用 id() 来作为唯一标识
    trainable_param_ids = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    print(f"理论上应该更新的参数数量: {len(trainable_param_ids)}")

    # 2. 遍历 Loss 的梯度图，找到所有实际参与计算的参数
    # BFS 遍历
    queue = [loss.grad_fn]
    visited_nodes = set()
    connected_param_ids = set()
    
    # 计数器
    step = 0
    
    while queue:
        cur_fn = queue.pop(0)
        
        # 跳过空节点和已访问节点
        if cur_fn is None or cur_fn in visited_nodes:
            continue
            
        visited_nodes.add(cur_fn)
        step += 1

        # 检查当前节点是否是参数的累积梯度节点 (AccumulateGrad)
        # 在 PyTorch 中，叶子参数的 grad_fn 是 AccumulateGrad
        if hasattr(cur_fn, 'variable'):
            param = cur_fn.variable
            p_id = id(param)
            connected_param_ids.add(p_id)
        
        # 将父节点加入队列
        for next_fn, _ in cur_fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)

    print(f"梯度图遍历完成，共访问 {step} 个计算节点。")
    print(f"实际连接到 Loss 的参数数量: {len(connected_param_ids)}")

    # 3. 核心对比：找出断开的参数
    # 存在于 trainable_param_ids 但不存在于 connected_param_ids 的，就是“断链”的元凶
    disconnected_params = []
    
    # 注意：我们比较的是我们传给DeepSpeed的那些参数 (假设 params_to_train 里的都在 trainable_param_ids 里)
    # 为了严谨，我们直接检查所有 requires_grad=True 的参数
    for p_id, name in trainable_param_ids.items():
        if p_id not in connected_param_ids:
            disconnected_params.append(name)

    print("\n=== 诊断结果 ===")
    if not disconnected_params:
        print("✅ 通过检查：所有可训练参数都成功连接到了 Loss 上。")
    else:
        print(f"❌ 发现 {len(disconnected_params)} 个参数虽然 requires_grad=True，但没有连接到 Loss！")
        print("这会导致 DeepSpeed 报错 'NoneType' object has no attribute 'next_functions'。")
        print("断开连接的参数列表（前20个）：")
        for name in disconnected_params[:20]:
            print(f" - {name}")
            
    return disconnected_params
