import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_STATE_KEYS = ("model_state_dict", "state_dict", "module")
EXCLUDED_META_KEYS = DEFAULT_STATE_KEYS + ("optimizer_state_dict", "scheduler_state_dict")
DEFAULT_WRAPPER_PREFIXES = (
    "module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "module._orig_mod.",
    "module._fsdp_wrapped_module.",
)
SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|b)?\s*$", re.IGNORECASE)


def parse_size_to_bytes(text: str) -> int:
    match = SIZE_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid size: {text!r}. Examples: 500MB, 2GB, 5GiB")

    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported size unit: {unit}")
    size = int(value * multipliers[unit])
    if size <= 0:
        raise ValueError("Shard size must be greater than 0")
    return size


def load_checkpoint(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any, tensor_key: Optional[str]) -> Dict[str, torch.Tensor]:
    if tensor_key:
        if not isinstance(payload, dict) or tensor_key not in payload:
            raise KeyError(f"Checkpoint does not contain key {tensor_key!r}")
        state_dict = payload[tensor_key]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint key {tensor_key!r} is not a state_dict")
        return state_dict

    if isinstance(payload, dict):
        for key in DEFAULT_STATE_KEYS:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(k, str) for k in payload) and all(torch.is_tensor(v) for v in payload.values()):
            return payload

    raise TypeError(
        "Could not find a tensor state_dict. Pass --tensor_key if the checkpoint uses a custom key."
    )


def strip_wrapper_prefix(key: str, prefixes: Iterable[str]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def normalize_state_dict(
    state_dict: Dict[str, Any],
    strip_wrappers: bool,
    prefixes: Iterable[str],
) -> Tuple["OrderedDict[str, torch.Tensor]", List[str]]:
    tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    skipped: List[str] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            skipped.append(key)
            continue

        out_key = strip_wrapper_prefix(key, prefixes) if strip_wrappers else key
        if out_key in tensors:
            raise ValueError(
                f"Duplicate tensor key after prefix stripping: {out_key!r}. "
                "Use --no-strip_wrappers or adjust --strip_prefixes."
            )
        tensors[out_key] = value.detach().cpu()

    if not tensors:
        raise ValueError("No tensors found in the selected state_dict")
    return tensors, skipped


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def build_shards(
    state_dict: "OrderedDict[str, torch.Tensor]",
    max_shard_size: int,
) -> List[List[Tuple[str, torch.Tensor, int]]]:
    shards: List[List[Tuple[str, torch.Tensor, int]]] = []
    current: List[Tuple[str, torch.Tensor, int]] = []
    current_size = 0

    for name, tensor in state_dict.items():
        size = tensor_nbytes(tensor)
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append((name, tensor, size))
        current_size += size

    if current:
        shards.append(current)
    return shards


def save_shard_file(path: Path, shard_tensors: Dict[str, torch.Tensor], fmt: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "safetensors":
        from safetensors.torch import save_file

        contiguous = {name: tensor.contiguous() for name, tensor in shard_tensors.items()}
        save_file(contiguous, str(path), metadata={"format": "pt"})
    elif fmt == "bin":
        torch.save(shard_tensors, path)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    return path.stat().st_size


def copy_sidecar_metadata(payload: Any, output_dir: Path) -> Optional[Path]:
    if not isinstance(payload, dict):
        return None

    torch_sidecar: Dict[str, Any] = {}
    json_sidecar: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in EXCLUDED_META_KEYS:
            continue
        if isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values()):
            continue
        torch_sidecar[key] = value
        try:
            json.dumps(value)
        except TypeError:
            continue
        json_sidecar[key] = value

    if not torch_sidecar:
        return None

    path = output_dir / "checkpoint_meta.pt"
    torch.save(torch_sidecar, path)

    if json_sidecar:
        json_path = output_dir / "checkpoint_meta.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_sidecar, f, ensure_ascii=False, indent=2)
    return path


def write_readme(output_dir: Path, index_name: str, fmt: str, source: Path) -> Path:
    weight_pattern = "model-*.safetensors" if fmt == "safetensors" else "pytorch_model-*.bin"
    readme_path = output_dir / "README_SHARDS.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Sharded checkpoint",
                "",
                f"- Source checkpoint: `{source}`",
                f"- Index file: `{index_name}`",
                f"- Weight files: `{weight_pattern}`",
                "- Project metadata: `checkpoint_meta.pt`",
                "",
                "Upload the index file and all shard files to the Hugging Face model repository.",
                "For this project, load the tensors as a normal state_dict and pass them to `JointAffordanceModel.load_state_dict`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme_path


def shard_checkpoint_for_hf(
    input_ckpt: Path,
    output_dir: Path,
    max_shard_size: int,
    fmt: str,
    tensor_key: Optional[str],
    strip_wrappers: bool,
    strip_prefixes: Iterable[str],
    write_meta: bool,
) -> Dict[str, Any]:
    payload = load_checkpoint(input_ckpt)
    raw_state_dict = extract_state_dict(payload, tensor_key=tensor_key)
    state_dict, skipped = normalize_state_dict(raw_state_dict, strip_wrappers, strip_prefixes)
    shards = build_shards(state_dict, max_shard_size=max_shard_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_shards = len(shards)
    ext = "safetensors" if fmt == "safetensors" else "bin"
    prefix = "model" if fmt == "safetensors" else "pytorch_model"
    index_name = "model.safetensors.index.json" if fmt == "safetensors" else "pytorch_model.bin.index.json"

    weight_map: Dict[str, str] = {}
    shard_records: List[Dict[str, Any]] = []
    total_tensor_size = 0

    for shard_idx, shard in enumerate(shards, start=1):
        shard_name = f"{prefix}-{shard_idx:05d}-of-{total_shards:05d}.{ext}"
        shard_path = output_dir / shard_name
        shard_tensors = OrderedDict((name, tensor) for name, tensor, _ in shard)
        file_size = save_shard_file(shard_path, shard_tensors, fmt=fmt)
        tensor_size = sum(size for _, _, size in shard)
        total_tensor_size += tensor_size

        for name, _, _ in shard:
            weight_map[name] = shard_name
        shard_records.append(
            {
                "file": shard_name,
                "num_tensors": len(shard),
                "tensor_size_bytes": tensor_size,
                "file_size_bytes": file_size,
            }
        )

    index = {
        "metadata": {
            "total_size": total_tensor_size,
            "format": fmt,
            "source_checkpoint": str(input_ckpt),
        },
        "weight_map": weight_map,
    }
    index_path = output_dir / index_name
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_checkpoint": str(input_ckpt),
        "index_file": index_name,
        "format": fmt,
        "max_shard_size_bytes": max_shard_size,
        "num_shards": total_shards,
        "num_tensors": len(state_dict),
        "total_tensor_size_bytes": total_tensor_size,
        "skipped_non_tensor_keys": skipped,
        "shards": shard_records,
    }
    manifest_path = output_dir / "shard_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    meta_path = copy_sidecar_metadata(payload, output_dir) if write_meta else None
    readme_path = write_readme(output_dir, index_name=index_name, fmt=fmt, source=input_ckpt)

    result = dict(manifest)
    result["index_path"] = str(index_path)
    result["manifest_path"] = str(manifest_path)
    result["metadata_path"] = str(meta_path) if meta_path is not None else None
    result["readme_path"] = str(readme_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a wrapped/FSDP-style checkpoint into Hugging Face sharded weight files "
            "with a model index JSON."
        )
    )
    parser.add_argument("--input_ckpt", type=Path, required=True, help="Input .pth/.pt checkpoint")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for HF shard files")
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum tensor payload per shard, e.g. 500MB, 2GB, 5GiB. A single tensor is never split.",
    )
    parser.add_argument(
        "--format",
        choices=("safetensors", "bin"),
        default="safetensors",
        help="Output shard format. safetensors is recommended for Hugging Face uploads.",
    )
    parser.add_argument(
        "--tensor_key",
        type=str,
        default=None,
        help="Checkpoint key containing the state_dict. Auto-detects model_state_dict/state_dict/module by default.",
    )
    parser.add_argument(
        "--strip_wrappers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip common wrapper prefixes such as module. and _fsdp_wrapped_module.",
    )
    parser.add_argument(
        "--strip_prefixes",
        type=str,
        default=",".join(DEFAULT_WRAPPER_PREFIXES),
        help="Comma-separated prefixes to strip when --strip_wrappers is enabled.",
    )
    parser.add_argument(
        "--no_meta",
        action="store_true",
        help="Do not export JSON-serializable checkpoint metadata to checkpoint_meta.json.",
    )
    args = parser.parse_args()

    strip_prefixes = [item.strip() for item in args.strip_prefixes.split(",") if item.strip()]
    result = shard_checkpoint_for_hf(
        input_ckpt=args.input_ckpt.resolve(),
        output_dir=args.output_dir.resolve(),
        max_shard_size=parse_size_to_bytes(args.max_shard_size),
        fmt=args.format,
        tensor_key=args.tensor_key,
        strip_wrappers=args.strip_wrappers,
        strip_prefixes=strip_prefixes,
        write_meta=not args.no_meta,
    )

    print(f"[OK] Wrote {result['num_shards']} shard(s) to {args.output_dir}")
    print(f"     index: {result['index_path']}")
    print(f"     manifest: {result['manifest_path']}")
    if result["metadata_path"]:
        print(f"     metadata: {result['metadata_path']}")
    if result["skipped_non_tensor_keys"]:
        print(f"[WARN] Skipped {len(result['skipped_non_tensor_keys'])} non-tensor state_dict entries")


if __name__ == "__main__":
    main()
