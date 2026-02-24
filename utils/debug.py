from typing import Dict
import torch

def log_param_dtype_stats(model, logger, stage):
    """打印全部参数 dtype 分布（含冻结参数）。"""
    module_refs = {
        "mllm": getattr(model, "mllm", None),
        "image_decoder": getattr(model, "image_decoder", None),
        "point_decoder": getattr(model, "point_decoder", None),
    }

    def _count_dtypes(params_iter):
        counts = {}
        total = 0
        for p in params_iter:
            if not p.is_floating_point():
                continue
            total += 1
            key = str(p.dtype)
            counts[key] = counts.get(key, 0) + 1
        return total, counts

    total_all, counts_all = _count_dtypes(model.parameters())
    logger.info(f"[{stage}] param dtype(all): total={total_all}, dist={counts_all}")

    for name, module in module_refs.items():
        if module is None:
            continue
        total_sub, counts_sub = _count_dtypes(module.parameters())
        logger.info(f"[{stage}] param dtype({name}): total={total_sub}, dist={counts_sub}")

# discard
def align_mllm_trainable_dtypes(model, target_dtype, logger):
    """将 MLLM 中可训练浮点参数统一到目标 dtype（通常用于 LoRA 注入后对齐）。"""
    if target_dtype is None:
        return
    mllm_module = getattr(model, "mllm", None)
    if mllm_module is None:
        return

    converted = 0
    total_trainable = 0
    for name, param in mllm_module.named_parameters():
        if not param.requires_grad or not param.is_floating_point():
            continue
        total_trainable += 1
        if param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=target_dtype)
            converted += 1

    logger.info(
        f"MLLM 可训练参数 dtype 对齐 -> {target_dtype}: "
        f"trainable={total_trainable}, converted={converted}"
    )


def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _collect_batch_runtime_stats(input_dict: Dict, device: torch.device) -> Dict[str, float]:
    """收集用于排查显存峰值的轻量运行时统计信息。"""
    stats = {}
    if device.type != "cuda":
        return stats

    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    stats.update(
        mem_allocated_gb=allocated_gb,
        mem_reserved_gb=reserved_gb,
        mem_peak_allocated_gb=peak_allocated_gb,
    )

    attention_mask = input_dict.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() >= 2:
        seq_lens = attention_mask.sum(dim=1).float()
        stats["seq_len_max"] = float(seq_lens.max().item())
        stats["seq_len_mean"] = float(seq_lens.mean().item())

    grid_thw = input_dict.get("image_grid_thw")
    if isinstance(grid_thw, torch.Tensor) and grid_thw.dim() == 2 and grid_thw.shape[1] == 3:
        grid_tokens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).float()
        stats["vision_tokens_max"] = float(grid_tokens.max().item())
        stats["vision_tokens_mean"] = float(grid_tokens.mean().item())

    valid_lengths = input_dict.get("pc_valid_lengths")
    if isinstance(valid_lengths, torch.Tensor) and valid_lengths.numel() > 0:
        lengths = valid_lengths.float()
        stats["pc_points_max"] = float(lengths.max().item())
        stats["pc_points_mean"] = float(lengths.mean().item())

    return stats