import argparse
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist


def _sorted_rank_files(sharded_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(sharded_dir):
        if fn.startswith("rank") and fn.endswith(".pth"):
            files.append(os.path.join(sharded_dir, fn))
    files.sort()
    if not files:
        raise FileNotFoundError(f"未找到 rank*.pth: {sharded_dir}")
    return files


def _init_default_process_group_for_cpu() -> Tuple[bool, str]:
    """
    初始化默认 process group（单进程 CPU/gloo）。
    某些 FSDP 分片对象（如 ShardedTensor）反序列化/访问时要求默认 PG 已初始化。
    Returns:
        (created, init_file_path)
    """
    if dist.is_available() and not dist.is_initialized():
        tmp = tempfile.NamedTemporaryFile(prefix="fsdp_merge_pg_", suffix=".tmp", delete=False)
        init_file = tmp.name
        tmp.close()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        return True, init_file
    return False, ""


def _extract_full_from_sharded_tensor(values: Sequence[Any]) -> Tuple[bool, torch.Tensor]:
    """
    从多个 rank 的 ShardedTensor 对象重建完整 tensor（CPU）。
    """
    first = values[0]
    if not hasattr(first, "local_shards") or not hasattr(first, "metadata"):
        return False, torch.empty(0)
    try:
        meta = first.metadata()
        full_shape = tuple(meta.size)
        shards_meta = list(meta.shards_metadata)
    except Exception:
        return False, torch.empty(0)

    # 找到一个本地分片作为 dtype/device 参考
    ref_tensor = None
    for v in values:
        try:
            ls = v.local_shards()
            if ls:
                ref_tensor = ls[0].tensor
                break
        except Exception:
            continue
    if ref_tensor is None:
        return False, torch.empty(0)

    full = torch.zeros(full_shape, dtype=ref_tensor.dtype, device="cpu")
    # 汇总所有 rank 的本地 shard（world_size=1 时每个对象都在当前进程内）
    all_local = []
    for v in values:
        try:
            all_local.extend(list(v.local_shards()))
        except Exception:
            continue

    # local_shards 的顺序通常对应 shards_metadata，若不一致再做 shape/offset 回退匹配
    used = [False] * len(all_local)
    for sm in shards_meta:
        offs = list(sm.shard_offsets)
        sizes = list(sm.shard_sizes)
        placed = False
        for i, local in enumerate(all_local):
            if used[i]:
                continue
            t = local.tensor.detach().to("cpu")
            if list(t.shape) == sizes:
                slices = tuple(slice(o, o + s) for o, s in zip(offs, sizes))
                full[slices] = t
                used[i] = True
                placed = True
                break
        if not placed:
            # 无法匹配某个 shard，则认为重建失败
            return False, torch.empty(0)
    return True, full


def _try_concat_tensors(values: Sequence[torch.Tensor]) -> Tuple[bool, torch.Tensor]:
    """
    按 rank 顺序尝试将分片张量拼接为完整张量。
    规则：
    1) 若形状完全一致，直接返回 rank0（通常是 replicated/broadcast 参数）
    2) 若仅有一个维度大小不同、其他维度一致，则沿该维度 concat
    """
    if not values:
        raise ValueError("空张量列表")
    first = values[0]
    if all(tuple(v.shape) == tuple(first.shape) for v in values):
        return True, first

    ndims = first.dim()
    varying_dims = []
    for d in range(ndims):
        sizes = [int(v.shape[d]) for v in values]
        if len(set(sizes)) > 1:
            varying_dims.append(d)

    # 允许某些 rank 分片大小一致但非均分，仍可能只有一个变化维
    if len(varying_dims) == 1:
        dim = varying_dims[0]
        base_shape = list(first.shape)
        for v in values[1:]:
            if v.dim() != ndims:
                return False, first
            for d in range(ndims):
                if d == dim:
                    continue
                if int(v.shape[d]) != int(base_shape[d]):
                    return False, first
        return True, torch.cat(list(values), dim=dim)

    return False, first


def _merge_value(values: Sequence[Any]) -> Any:
    first = values[0]
    # 优先处理 ShardedTensor（FSDP SHARDED_STATE_DICT 常见）
    ok_sharded, merged_sharded = _extract_full_from_sharded_tensor(values)
    if ok_sharded:
        return merged_sharded
    if isinstance(first, torch.Tensor):
        ok, merged = _try_concat_tensors(values)  # type: ignore[arg-type]
        if ok:
            return merged
        return first
    if isinstance(first, dict):
        out: Dict[str, Any] = {}
        for k in first.keys():
            sub_vals = [v[k] for v in values if isinstance(v, dict) and k in v]
            if not sub_vals:
                continue
            out[k] = _merge_value(sub_vals)
        return out
    # 其他类型默认取 rank0
    return first


def merge_sharded_checkpoints(sharded_dir: str, output_pth: str) -> str:
    created_pg, init_file = _init_default_process_group_for_cpu()
    try:
        rank_files = _sorted_rank_files(sharded_dir)
        payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in rank_files]

        # 兼容两种结构：{..., model_state_dict=...} 或直接 state_dict
        state_dicts: List[Dict[str, Any]] = []
        for p in payloads:
            if isinstance(p, dict) and "model_state_dict" in p and isinstance(p["model_state_dict"], dict):
                state_dicts.append(p["model_state_dict"])
            elif isinstance(p, dict):
                state_dicts.append(p)
            else:
                raise TypeError(f"不支持的 rank payload 类型: {type(p)}")

        keys = list(state_dicts[0].keys())
        merged_state: Dict[str, Any] = {}
        for k in keys:
            vals = [sd[k] for sd in state_dicts if k in sd]
            if not vals:
                continue
            merged_state[k] = _merge_value(vals)

        rank0 = payloads[0]
        if isinstance(rank0, dict) and "model_state_dict" in rank0:
            merged_payload = dict(rank0)
            merged_payload["model_state_dict"] = merged_state
            merged_payload["merged_from_shards"] = True
            merged_payload["num_shards"] = len(rank_files)
        else:
            merged_payload = merged_state

        os.makedirs(os.path.dirname(os.path.abspath(output_pth)), exist_ok=True)
        torch.save(merged_payload, output_pth)
        return output_pth
    finally:
        if created_pg:
            try:
                dist.destroy_process_group()
            finally:
                if init_file and os.path.exists(init_file):
                    os.remove(init_file)


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


def main():
    parser = argparse.ArgumentParser(
        description="合并 FSDP 分片 checkpoint，并可按大小切块（用于 Hugging Face 上传）"
    )
    parser.add_argument("--sharded_dir", type=str, required=True, help="分片目录（包含 rank*.pth）")
    parser.add_argument("--output_pth", type=str, required=True, help="合并后的 pth 输出路径")
    parser.add_argument("--chunk_size_mb", type=int, default=0, help=">0 时对合并后的 pth 进行切块，单位 MB")
    args = parser.parse_args()

    merged_path = merge_sharded_checkpoints(args.sharded_dir, args.output_pth)
    print(f"[OK] 已合并: {merged_path}")

    if args.chunk_size_mb and args.chunk_size_mb > 0:
        manifest = split_file_for_hf(merged_path, chunk_size_mb=args.chunk_size_mb)
        print(
            f"[OK] 已切块: {manifest['num_parts']} 个, "
            f"manifest={os.path.basename(merged_path)}.manifest.json"
        )


if __name__ == "__main__":
    main()

