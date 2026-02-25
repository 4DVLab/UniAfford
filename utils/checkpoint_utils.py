"""
Checkpoint 与 state_dict 工具：加载权重、提取/重命名 state_dict，为 FSDP/LoRA 等训练完的权重合并与键名适配提供支持。
"""

import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
from torch.nn import Module


def extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """从多种 checkpoint 结构中提取 state_dict。"""
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "module"):
            val = ckpt_obj.get(key, None)
            if isinstance(val, dict):
                return val
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"不支持的 checkpoint 结构: {type(ckpt_obj)}")


def replace_state_dict_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    new_prefix: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    对 state_dict 的 key 做前缀替换：若 key 以 prefix 开头，则用 new_prefix 替换该前缀。

    - new_prefix 为 None（默认）：等价于删除前缀。
    - new_prefix 为字符串：将 prefix 替换为 new_prefix（例如用于合并 FSDP/LoRA 时统一子模块前缀）。
    """
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = (new_prefix or "") + k[len(prefix):]
        else:
            new_key = k
        out[new_key] = v
    return out


def detect_best_state_dict(
    model: Module,
    raw_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    自动选择与当前模型最匹配的 state_dict 版本：
    尝试原始 key 及去除常见并行训练前缀（module / _orig_mod / _fsdp_wrapped_module 等）。
    """
    candidates = [
        ("raw", raw_state_dict),
        ("strip:module.", replace_state_dict_prefix(raw_state_dict, "module.", None)),
        # ("strip:module.", replace_state_dict_prefix(raw_state_dict, "mllm.model.base_model.model.model.", "mllm.model.model.")),
        # ("strip:_orig_mod.", replace_state_dict_prefix(raw_state_dict, "_orig_mod.", None)),
        # ("strip:_fsdp_wrapped_module.", replace_state_dict_prefix(raw_state_dict, "_fsdp_wrapped_module.", None)),
        # ("strip:module._orig_mod.", replace_state_dict_prefix(raw_state_dict, "module._orig_mod.", None)),
        # ("strip:module._fsdp_wrapped_module.", replace_state_dict_prefix(raw_state_dict, "module._fsdp_wrapped_module.", None)),
    ]

    model_keys = set(model.state_dict().keys())
    best_name = "raw"
    best_state = raw_state_dict
    best_score = (-1, float("inf"))
    for name, sd in candidates:
        keys = set(sd.keys())
        matched = len(keys & model_keys)
        mismatch = len(keys - model_keys) + len(model_keys - keys)
        score = (matched, -mismatch)
        if score > best_score:
            best_score = score
            best_name = name
            best_state = sd
    return best_state, best_name


def load_checkpoint_to_model(
    model: Module,
    ckpt_path: str,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> None:
    """
    通用 checkpoint 加载：从文件读取，自动提取 state_dict 并适配键名前缀（DS/FSDP 等），再加载到模型。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if os.path.isdir(ckpt_path):
        raise ValueError(
            f"给定路径是目录：{ckpt_path}。\n"
            "本函数仅支持 .pth 等单文件权重；若是 DeepSpeed ZeRO 目录，请先导出为 .pth。"
        )

    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = extract_state_dict(ckpt)
    state_dict, adapt_rule = detect_best_state_dict(model, state_dict)

    all_keys = list(state_dict.keys())
    if (
        "fsdp" in adapt_rule
        or any("fsdp_wrapped_module" in k for k in all_keys)
        or "fsdp" in os.path.basename(ckpt_path).lower()
    ):
        ckpt_type = "fsdp"
    else:
        ckpt_type = "deepspeed_or_plain"

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"已加载 checkpoint: {ckpt_path}")
    print(f"自动识别格式: {ckpt_type}，键名适配规则: {adapt_rule}")
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}，例如：{missing[:10]}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}，例如：{unexpected[:10]}")
