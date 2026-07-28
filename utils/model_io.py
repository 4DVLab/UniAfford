import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from configs import TrainingConfig
from utils.checkpoint_utils import detect_best_state_dict, extract_state_dict


PORTABLE_CHECKPOINT_VERSION = 1
HF_SAFETENSORS_INDEX = "model.safetensors.index.json"
HF_PYTORCH_INDEX = "pytorch_model.bin.index.json"
HF_META_FILE = "checkpoint_meta.json"
HF_TORCH_META_FILE = "checkpoint_meta.pt"


def _collect_directory_files(root_dir: str) -> Dict[str, bytes]:
    file_map: Dict[str, bytes] = {}
    for path in Path(root_dir).rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(root_dir).as_posix()
            file_map[rel_path] = path.read_bytes()
    return file_map


def _unwrap_peft_base_model(model) -> Any:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def _serialize_processor_assets(processor) -> Dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="ja_processor_") as tmpdir:
        processor.save_pretrained(tmpdir)
        return _collect_directory_files(tmpdir)


def _serialize_model_config_assets(model) -> Tuple[Dict[str, bytes], str]:
    base_model = _unwrap_peft_base_model(model)
    class_name = type(base_model).__name__
    with tempfile.TemporaryDirectory(prefix="ja_qwen_cfg_") as tmpdir:
        base_model.config.save_pretrained(tmpdir)
        generation_config = getattr(base_model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "save_pretrained"):
            generation_config.save_pretrained(tmpdir)
        return _collect_directory_files(tmpdir), class_name


def build_portable_assets(model: Any) -> Dict[str, Any]:
    processor_files = _serialize_processor_assets(model.mllm.processor)
    model_config_files, model_class_name = _serialize_model_config_assets(model.mllm.model)
    return {
        "portable_checkpoint_version": PORTABLE_CHECKPOINT_VERSION,
        "hf_assets": {
            "processor_files": processor_files,
            "model_config_files": model_config_files,
            "model_class_name": model_class_name,
        },
    }


def build_portable_checkpoint_payload(
    model_state_dict: Dict[str, torch.Tensor],
    meta: Optional[Dict[str, Any]] = None,
    training_cfg: Optional[TrainingConfig] = None,
    asset_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if meta:
        payload.update(meta)
    payload["model_state_dict"] = model_state_dict
    if asset_bundle:
        payload.update(asset_bundle)
    if training_cfg is not None:
        try:
            payload["training_config"] = training_cfg.to_json_dict(include_deepspeed=True)
        except TypeError:
            payload["training_config"] = training_cfg.to_json_dict()
    return payload


def save_portable_checkpoint(
    save_path: Union[str, os.PathLike],
    model_state_dict: Dict[str, torch.Tensor],
    meta: Optional[Dict[str, Any]] = None,
    training_cfg: Optional[TrainingConfig] = None,
    asset_bundle: Optional[Dict[str, Any]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    lr_dict: Optional[Dict[str, float]] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """保存项目统一 portable checkpoint，兼容训练恢复、验证和 HF 分片导出。"""
    payload = build_portable_checkpoint_payload(
        model_state_dict=model_state_dict,
        meta=meta,
        training_cfg=training_cfg,
        asset_bundle=asset_bundle,
    )

    if optimizer_state_dict is None and optimizer is not None:
        try:
            optimizer_state_dict = optimizer.state_dict()
        except Exception as exc:
            if logger is not None:
                logger.warning(f"保存 optimizer_state_dict 失败，将仅保存模型权重: {exc}")
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict

    if scheduler_state_dict is None and scheduler is not None:
        try:
            scheduler_state_dict = scheduler.state_dict()
        except Exception as exc:
            if logger is not None:
                logger.warning(f"保存 scheduler_state_dict 失败，将仅保存模型权重: {exc}")
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict

    if lr_dict is not None:
        payload["lr_dict"] = lr_dict

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    return payload


def _torch_load(path: Union[str, os.PathLike], map_location: Union[str, torch.device] = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_hf_index_path(ckpt_dir: Union[str, os.PathLike]) -> Path:
    ckpt_dir = Path(ckpt_dir)
    for name in (HF_SAFETENSORS_INDEX, HF_PYTORCH_INDEX):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"目录中未找到 Hugging Face 分片索引文件: {ckpt_dir} "
        f"(需要 {HF_SAFETENSORS_INDEX} 或 {HF_PYTORCH_INDEX})"
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_hf_sharded_payload(
    ckpt_dir: Union[str, os.PathLike],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    ckpt_dir = Path(ckpt_dir)
    index_path = _resolve_hf_index_path(ckpt_dir)
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"分片索引缺少有效 weight_map: {index_path}")

    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    shard_names = list(OrderedDict.fromkeys(weight_map.values()))
    for shard_name in shard_names:
        shard_path = ckpt_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"分片文件不存在: {shard_path}")

        if shard_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            shard_state = load_file(str(shard_path), device=str(map_location))
        else:
            shard_state = _torch_load(shard_path, map_location=map_location)
        if not isinstance(shard_state, dict):
            raise TypeError(f"分片不是 state_dict 结构: {shard_path}")

        for tensor_name, mapped_shard in weight_map.items():
            if mapped_shard == shard_name:
                if tensor_name not in shard_state:
                    raise KeyError(f"分片 {shard_path} 缺少索引中的 tensor: {tensor_name}")
                state_dict[tensor_name] = shard_state[tensor_name]

    torch_meta_path = ckpt_dir / HF_TORCH_META_FILE
    if torch_meta_path.exists():
        payload = _torch_load(torch_meta_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint 元信息不是 dict 结构: {torch_meta_path}")
    else:
        payload = _load_json_if_exists(ckpt_dir / HF_META_FILE)
    payload["model_state_dict"] = state_dict
    payload.setdefault("hf_sharded_checkpoint", True)
    payload.setdefault("hf_sharded_index", index_path.name)
    return payload


def load_checkpoint_payload(
    ckpt_path: str,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if os.path.isdir(ckpt_path):
        return _load_hf_sharded_payload(ckpt_path, map_location=map_location)
    payload = _torch_load(ckpt_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"不支持的 checkpoint 结构: {type(payload)}")
    return payload


def _resolve_training_config(
    ckpt_payload: Dict[str, Any],
    config_json_path: Optional[str] = None,
) -> TrainingConfig:
    if config_json_path:
        return TrainingConfig.from_json(config_json_path)
    embedded_cfg = ckpt_payload.get("training_config")
    if isinstance(embedded_cfg, dict):
        return TrainingConfig.from_json_dict(embedded_cfg)
    return TrainingConfig()


def resolve_training_config_from_payload(
    ckpt_payload: Dict[str, Any],
    config_json_path: Optional[str] = None,
) -> TrainingConfig:
    return _resolve_training_config(ckpt_payload, config_json_path=config_json_path)


def _inject_portable_assets(model_cfg, ckpt_payload: Dict[str, Any]) -> None:
    hf_assets = ckpt_payload.get("hf_assets") or {}
    if not isinstance(hf_assets, dict):
        return
    model_cfg.mllm.serialized_processor_files = hf_assets.get("processor_files")
    model_cfg.mllm.serialized_model_config_files = hf_assets.get("model_config_files")
    model_cfg.mllm.serialized_model_class_name = hf_assets.get("model_class_name")
    model_cfg.mllm.restore_from_checkpoint = True


def load_portable_model(
    ckpt_path: str,
    config_json_path: Optional[str] = None,
    training_cfg: Optional[TrainingConfig] = None,
    map_location: Union[str, torch.device] = "cpu",
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
    ckpt_payload: Optional[Dict[str, Any]] = None,
):
    ckpt_payload = ckpt_payload or load_checkpoint_payload(ckpt_path, map_location=map_location)
    training_cfg = training_cfg or _resolve_training_config(ckpt_payload, config_json_path=config_json_path)
    model_cfg = training_cfg.model_config
    _inject_portable_assets(model_cfg, ckpt_payload)

    # training_config 里常残留训练机绝对/相对 SAM 路径；便携加载时以 UniAfford ckpt 为准。
    vision_ckpt = getattr(model_cfg.image_decoder, "vision_pretrained", None)
    if vision_ckpt and not os.path.exists(vision_ckpt):
        print(
            f"[Warning] 本地 SAM 权重不存在: {vision_ckpt}；"
            "构建时跳过，改由 UniAfford checkpoint 中的 image_decoder 权重填充。"
        )
        model_cfg.image_decoder.vision_pretrained = None

    from model.UniAfford import UniAffordModel

    model = UniAffordModel(model_cfg)
    if training_cfg.lora.lora_r > 0:
        from peft import get_peft_model
        model.mllm.model = get_peft_model(model.mllm.model, training_cfg.lora.to_peft_config())

    state_dict = extract_state_dict(ckpt_payload)
    state_dict, adapt_rule = detect_best_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"已加载 checkpoint: {ckpt_path}，键名适配规则: {adapt_rule}")
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}，例如：{missing[:10]}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}，例如：{unexpected[:10]}")

    if device is not None:
        model.to(device)
    return model, training_cfg, ckpt_payload
