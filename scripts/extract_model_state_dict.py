import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

import torch


DEFAULT_META_KEYS = [
    "epoch",
    "global_step",
    "best_epoch",
    "best_val_loss",
    "val_loss",
    "val_metrics",
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def split_file_for_hf(input_file: str, chunk_size_mb: int = 4096) -> Dict[str, Any]:
    """按固定大小分片并生成 manifest，便于 Hugging Face 上传。"""
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb 必须 > 0")
    chunk_bytes = int(chunk_size_mb) * 1024 * 1024
    total_size = os.path.getsize(input_file)
    base = os.path.basename(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))

    parts: List[Dict[str, Any]] = []
    idx = 0
    with open(input_file, "rb") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            idx += 1
            part_name = f"{base}.part{idx:05d}"
            part_path = os.path.join(out_dir, part_name)
            with open(part_path, "wb") as pf:
                pf.write(data)
            parts.append(
                {
                    "name": part_name,
                    "size_bytes": len(data),
                    "sha256": _sha256_file(part_path),
                }
            )

    manifest = {
        "original_file": base,
        "original_size_bytes": total_size,
        "original_sha256": _sha256_file(input_file),
        "chunk_size_mb": chunk_size_mb,
        "num_parts": len(parts),
        "parts": parts,
    }
    manifest_path = os.path.join(out_dir, f"{base}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _parse_meta_keys(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_checkpoint(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 不是 dict 结构: {path}")
    return payload


def _extract_model_state(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # 常见训练 checkpoint 格式
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]

    # 兼容“纯 state_dict 文件”：顶层本身就是参数字典
    if payload and all(isinstance(k, str) for k in payload.keys()) and all(
        torch.is_tensor(v) for v in payload.values()
    ):
        return payload

    raise ValueError("无法识别模型参数字典；请确认输入文件包含 model_state_dict 或本身就是 state_dict。")


def main():
    parser = argparse.ArgumentParser(
        description="从训练 checkpoint 提取仅用于推理/评估的模型参数，去除 optimizer/scheduler 等冗余内容。"
    )
    parser.add_argument("--input_ckpt", type=str, required=True, help="输入训练 checkpoint 路径")
    parser.add_argument("--output_ckpt", type=str, required=True, help="输出轻量 checkpoint 路径")
    parser.add_argument(
        "--keep_meta",
        action="store_true",
        help="保留少量训练元信息（epoch/global_step/best/val_metrics 等）",
    )
    parser.add_argument(
        "--meta_keys",
        type=str,
        default=",".join(DEFAULT_META_KEYS),
        help="需要保留的 meta 键，逗号分隔；仅在 --keep_meta 时生效",
    )
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=0,
        help=">0 时按该大小（MB）对输出权重分片，并生成 manifest，便于上传 Hugging Face",
    )
    args = parser.parse_args()

    input_ckpt = os.path.abspath(args.input_ckpt)
    output_ckpt = os.path.abspath(args.output_ckpt)
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)

    payload = _load_checkpoint(input_ckpt)
    model_state = _extract_model_state(payload)

    save_obj: Dict[str, Any] = {"model_state_dict": model_state}

    if args.keep_meta:
        meta_keys = _parse_meta_keys(args.meta_keys)
        for k in meta_keys:
            if k in payload:
                save_obj[k] = payload[k]

    torch.save(save_obj, output_ckpt)

    in_size_mb = os.path.getsize(input_ckpt) / (1024 * 1024)
    out_size_mb = os.path.getsize(output_ckpt) / (1024 * 1024)
    ratio = (out_size_mb / in_size_mb) if in_size_mb > 0 else 0.0
    print(f"[OK] 已提取模型参数: {output_ckpt}")
    print(f"     输入大小: {in_size_mb:.2f} MB")
    print(f"     输出大小: {out_size_mb:.2f} MB")
    print(f"     压缩比例: {ratio:.3f}")
    if args.chunk_size_mb and args.chunk_size_mb > 0:
        manifest = split_file_for_hf(output_ckpt, chunk_size_mb=args.chunk_size_mb)
        print(
            f"[OK] 已分片: {manifest['num_parts']} 个, "
            f"manifest={os.path.basename(output_ckpt)}.manifest.json"
        )


if __name__ == "__main__":
    main()

