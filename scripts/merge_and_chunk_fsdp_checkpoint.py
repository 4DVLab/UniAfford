"""
Discard
之前为了快速保存模型权重，使用了特殊的权重保存方式，专门需要权重合成脚本才能把权重合成单一的文件
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)

# 允许从仓库根目录导入
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs import TrainingConfig
from model.joint_affordance import JointAffordanceModel
from peft import get_peft_model


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


def _infer_config_json(sharded_dir: str, config_json: str | None) -> str:
    if config_json is not None:
        return os.path.abspath(config_json)
    # 默认同 train_fsdp：checkpoints_fsdp/training_config.json
    cand = os.path.join(os.path.dirname(os.path.abspath(sharded_dir)), "training_config.json")
    return cand


def _load_rank_shard(sharded_dir: str, rank: int) -> Dict[str, Any]:
    shard_path = os.path.join(sharded_dir, f"rank{rank:05d}.pth")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"缺少当前 rank 分片文件: {shard_path}")
    payload = torch.load(shard_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"分片格式不合法（需包含 model_state_dict）: {shard_path}")
    return payload


def _build_model_for_merge(cfg: TrainingConfig, local_rank: int) -> FSDP:
    model_cfg = cfg.model_config
    model = JointAffordanceModel(model_cfg)

    if cfg.lora.lora_r > 0:
        model.mllm.model = get_peft_model(model.mllm.model, cfg.lora.to_peft_config())

    device = torch.device("cuda", local_rank)
    model.to(device=device)
    model_fsdp = FSDP(model, device_id=device, use_orig_params=True)
    model_fsdp.eval()
    return model_fsdp


def main():
    parser = argparse.ArgumentParser(
        description=(
            "分布式合并 FSDP 分片 checkpoint（每卡加载各自 rank shard，rank0 聚合 full 权重保存），"
            "并可选切块用于 Hugging Face 上传。"
        )
    )
    parser.add_argument("--sharded_dir", type=str, required=True, help="分片目录（包含 rankxxxxx.pth）")
    parser.add_argument("--output_pth", type=str, required=True, help="合并后的完整权重输出路径")
    parser.add_argument("--config_json", type=str, default=None, help="training_config.json 路径；默认从分片目录父目录推断")
    parser.add_argument("--chunk_size_mb", type=int, default=0, help=">0 时对合并后的 pth 进行切块，单位 MB")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    sharded_dir = os.path.abspath(args.sharded_dir)
    output_pth = os.path.abspath(args.output_pth)
    config_json_path = _infer_config_json(sharded_dir, args.config_json)

    if rank == 0:
        if not os.path.isdir(sharded_dir):
            raise FileNotFoundError(f"分片目录不存在: {sharded_dir}")
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"training_config.json 不存在: {config_json_path}")
        meta_path = os.path.join(sharded_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_ws = int(meta.get("world_size", world_size))
            if expected_ws != world_size:
                raise RuntimeError(
                    f"当前 world_size={world_size} 与分片记录 world_size={expected_ws} 不一致。"
                    "请使用与保存时相同卡数的 torchrun。"
                )
    dist.barrier()

    cfg = TrainingConfig.from_json(config_json_path)
    model_fsdp = _build_model_for_merge(cfg, local_rank)

    # 每张卡仅加载自己的 shard，避免单卡 OOM 和伪多卡环境问题
    rank_payload = _load_rank_shard(sharded_dir, rank)
    shard_state = rank_payload["model_state_dict"]

    with FSDP.state_dict_type(
        model_fsdp, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)
    ):
        model_fsdp.load_state_dict(shard_state, strict=True)

    dist.barrier()

    # rank0 聚合完整权重保存
    with FSDP.state_dict_type(
        model_fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        full_state = model_fsdp.state_dict()

    if rank == 0:
        os.makedirs(os.path.dirname(output_pth), exist_ok=True)
        save_payload = {
            "merged_from_shards": True,
            "source_sharded_dir": sharded_dir,
            "world_size": world_size,
            "epoch": rank_payload.get("epoch"),
            "global_step": rank_payload.get("global_step"),
            "best_epoch": rank_payload.get("best_epoch"),
            "best_val_loss": rank_payload.get("best_val_loss"),
            "val_loss": rank_payload.get("val_loss"),
            "val_metrics": rank_payload.get("val_metrics"),
            "model_state_dict": full_state,
        }
        torch.save(save_payload, output_pth)
        print(f"[OK] 合并完成: {output_pth}")

        if args.chunk_size_mb and args.chunk_size_mb > 0:
            manifest = split_file_for_hf(output_pth, chunk_size_mb=args.chunk_size_mb)
            print(
                f"[OK] 已切块: {manifest['num_parts']} 个, "
                f"manifest={os.path.basename(output_pth)}.manifest.json"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

