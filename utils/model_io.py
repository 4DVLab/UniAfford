import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from utils.checkpoint_utils import extract_state_dict, load_checkpoint_to_model


PORTABLE_CHECKPOINT_VERSION = 1


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


def build_portable_assets(model: JointAffordanceModel) -> Dict[str, Any]:
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


def load_checkpoint_payload(
    ckpt_path: str,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
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
):
    ckpt_payload = load_checkpoint_payload(ckpt_path, map_location=map_location)
    training_cfg = training_cfg or _resolve_training_config(ckpt_payload, config_json_path=config_json_path)
    model_cfg = training_cfg.model_config
    _inject_portable_assets(model_cfg, ckpt_payload)

    model = JointAffordanceModel(model_cfg)
    if training_cfg.lora.lora_r > 0:
        from peft import get_peft_model
        model.mllm.model = get_peft_model(model.mllm.model, training_cfg.lora.to_peft_config())

    if ckpt_payload.get("portable_checkpoint_version") is not None:
        state_dict = extract_state_dict(ckpt_payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[Warning] Missing keys: {len(missing)}，例如：{missing[:10]}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {len(unexpected)}，例如：{unexpected[:10]}")
    else:
        load_checkpoint_to_model(model, ckpt_path, map_location=map_location, strict=strict)

    if device is not None:
        model.to(device)
    return model, training_cfg, ckpt_payload
