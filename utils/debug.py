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
