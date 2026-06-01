import os
import csv
from typing import List, Dict, Tuple
import numpy as np
import torch


def _append_task_vocab_embeddings(
    tsne_vectors: List[np.ndarray],
    tsne_records: List[Dict],
    model,
    tokenizer,
    max_points: int,
):
    """记录静态 task placeholder 在词表 embedding 空间中的位置。"""
    if len(tsne_vectors) >= max_points:
        return
    emb = model.mllm.model.get_input_embeddings()
    task_ids = getattr(model.router, "task_placeholder_ids", {})
    weight = emb.weight.detach()
    for task_name, token_id in task_ids.items():
        if len(tsne_vectors) >= max_points:
            break
        token_id = int(token_id)
        tsne_vectors.append(weight[token_id].float().cpu().numpy())
        tsne_records.append({
            "source": "vocab_task_token",
            "task": task_name,
            "route_task": "",
            "gt_task": task_name,
            "route_conf": "",
            "sample_id": "",
            "obj_type": "",
            "aff_type": "",
            "seq_pos": "",
            "token_id": token_id,
            "token_text": tokenizer.convert_ids_to_tokens(token_id),
        })


def _collect_tsne_batch(
    tsne_vectors: List[np.ndarray],
    tsne_records: List[Dict],
    output_dict: Dict,
    input_dict: Dict,
    model,
    tokenizer,
    max_points: int,
    ignore_index: int = -100,
):
    """从当前 batch 采样 MLLM hidden states，用于观察 task token 空间分布。"""
    if len(tsne_vectors) >= max_points:
        return
    hidden_states = output_dict.get("hidden_states")
    if not isinstance(hidden_states, torch.Tensor):
        return

    attention_mask = output_dict.get("attention_mask")
    labels = output_dict.get("labels")
    route_probs = output_dict.get("route_probs")
    route_ids = route_probs.argmax(dim=-1) if isinstance(route_probs, torch.Tensor) else None
    route_confs = route_probs.max(dim=-1).values if isinstance(route_probs, torch.Tensor) else None

    task_name_by_id = getattr(model.router, "task_name_by_id", {})
    placeholder_id_to_task_id = getattr(model.router, "placeholder_id_to_task_id", {})

    def _route_task(batch_i: int, pos: int) -> str:
        if route_ids is None:
            return ""
        route_idx = int(route_ids[batch_i, pos].item())
        return str(task_name_by_id.get(route_idx, route_idx))

    def _gt_task_and_token(batch_i: int, pos: int) -> Tuple[str, int, str, bool]:
        if not isinstance(labels, torch.Tensor) or pos + 1 >= labels.shape[1]:
            return "", -1, "", True
        token_id = int(labels[batch_i, pos + 1].item())
        if token_id == ignore_index:
            return "", token_id, "", False
        task_id = placeholder_id_to_task_id.get(token_id)
        task_name = str(task_name_by_id.get(int(task_id), "text")) if task_id is not None else "text"
        token_text = tokenizer.convert_ids_to_tokens(token_id) if token_id >= 0 else ""
        return task_name, token_id, token_text, True

    bsz, seq_len, _ = hidden_states.shape
    for i in range(bsz):
        if len(tsne_vectors) >= max_points:
            break
        sample_id = input_dict.get("sample_id")[i]
        obj_type = input_dict.get("obj_type")[i]
        aff_type = input_dict.get("aff_type")[i]
        valid_positions = []
        for pos in range(seq_len):
            if isinstance(attention_mask, torch.Tensor) and not bool(attention_mask[i, pos].item()):
                continue
            gt_task, token_id, token_text, valid_label = _gt_task_and_token(i, pos)
            if not valid_label:
                continue
            route_task = _route_task(i, pos)
            is_task_related = (gt_task not in {"", "text"}) or (route_task not in {"", "text"})
            valid_positions.append((0 if is_task_related else 1, pos, gt_task, token_id, token_text, route_task))

        # 优先保留 img/pc/latent 等 task 相关点，再用少量 text 点作背景。
        valid_positions.sort(key=lambda x: (x[0], x[1]))
        for _, pos, gt_task, token_id, token_text, route_task in valid_positions:
            if len(tsne_vectors) >= max_points:
                break
            route_conf = float(route_confs[i, pos].item()) if route_confs is not None else ""
            display_task = gt_task if gt_task not in {"", "text"} else (route_task or gt_task)
            tsne_vectors.append(hidden_states[i, pos].detach().float().cpu().numpy())
            tsne_records.append({
                "source": "mllm_hidden",
                "task": display_task,
                "route_task": route_task,
                "gt_task": gt_task,
                "route_conf": route_conf,
                "sample_id": sample_id,
                "obj_type": obj_type,
                "aff_type": aff_type,
                "seq_pos": int(pos),
                "token_id": int(token_id),
                "token_text": token_text,
            })


def _save_tsne_artifacts(tsne_vectors: List[np.ndarray], tsne_records: List[Dict], out_dir: str):
    if not tsne_vectors:
        print("未收集到 t-SNE embedding 点，跳过可视化。")
        return

    os.makedirs(out_dir, exist_ok=True)
    vectors = np.stack(tsne_vectors, axis=0).astype(np.float32)
    npz_path = os.path.join(out_dir, "tsne_embeddings.npz")
    csv_path = os.path.join(out_dir, "tsne_metadata.csv")
    np.savez_compressed(npz_path, embeddings=vectors)

    fields = [
        "source", "task", "route_task", "gt_task", "route_conf",
        "sample_id", "obj_type", "aff_type", "seq_pos", "token_id", "token_text",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tsne_records)
    print(f"t-SNE embedding 已保存到: {npz_path}")
    print(f"t-SNE metadata 已保存到: {csv_path}")

    if vectors.shape[0] < 5:
        print("t-SNE 点数少于 5，已跳过图片生成。")
        return

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"缺少 sklearn 或 matplotlib，已跳过 t-SNE 图片生成: {exc}")
        return

    reduced_input = vectors
    if vectors.shape[1] > 50 and vectors.shape[0] > 50:
        reduced_input = PCA(n_components=50, random_state=0).fit_transform(vectors)
    perplexity = min(30, max(2, (vectors.shape[0] - 1) // 3))
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        metric="cosine",
        random_state=0,
    ).fit_transform(reduced_input)

    tasks = [str(r.get("task") or r.get("route_task") or "unknown") for r in tsne_records]
    unique_tasks = sorted(set(tasks))
    plt.figure(figsize=(8, 6), dpi=160)
    for task in unique_tasks:
        idx = np.array([t == task for t in tasks])
        plt.scatter(coords[idx, 0], coords[idx, 1], s=10, alpha=0.75, label=task)
    plt.legend(markerscale=2, fontsize=8)
    plt.title("Task Token Embeddings t-SNE")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "tsne_task_tokens.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"t-SNE 可视化已保存到: {fig_path}")

