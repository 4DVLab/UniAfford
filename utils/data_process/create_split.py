import os
import csv
import json
import re
import random
from collections import defaultdict
import argparse



def _extract_id_from_name(name: str):
    """从文件名中提取样本 id，如 Mug_12_grasp.png / Mug_12.csv -> 12。"""
    m = re.search(r'_(\d+)(?:_|$)', name)
    return int(m.group(1)) if m else None


def save_split_from_disk(
    dataset_root: str,
    train_ratio: float = 1.0,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
):
    """
    轻量生成分割文件（train/val/test + metadata.json）。
    直接扫描已保存到磁盘的数据，不依赖内存中的 Modality 对象，适配 load_and_save 流式处理。
    """
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(r < 0 for r in ratios):
        raise ValueError("train/val/test 比例不能为负数")
    s = sum(ratios)
    if s <= 0:
        raise ValueError("train/val/test 比例之和必须大于 0")
    train_ratio, val_ratio, test_ratio = [r / s for r in ratios]

    dataset_root = os.path.abspath(dataset_root)
    all_ids = {
        "Instruction": defaultdict(lambda: defaultdict(set)),
        "Image": defaultdict(lambda: defaultdict(set)),
        "PointCloud": defaultdict(lambda: defaultdict(set)),
    }

    for obj_type in sorted(os.listdir(dataset_root)):
        obj_dir = os.path.join(dataset_root, obj_type)
        if not os.path.isdir(obj_dir):
            continue

        # 1) Instruction.csv -> (obj, aff) -> [ins_id]
        ins_csv = os.path.join(obj_dir, "Instruction.csv")
        if os.path.exists(ins_csv):
            with open(ins_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    aff = (row.get("aff_type") or "").strip()
                    sid = row.get("id")
                    if not aff or sid is None:
                        continue
                    try:
                        sid = int(sid)
                    except ValueError:
                        continue
                    all_ids["Instruction"][obj_type][aff].add(sid)

        # 2) Image/mask/<aff>/*.png -> (obj, aff) -> [img_id]
        mask_root = os.path.join(obj_dir, "Image", "mask")
        if os.path.isdir(mask_root):
            for aff in os.listdir(mask_root):
                aff_dir = os.path.join(mask_root, aff)
                if not os.path.isdir(aff_dir):
                    continue
                for fn in os.listdir(aff_dir):
                    if not fn.lower().endswith(".png"):
                        continue
                    sid = _extract_id_from_name(os.path.splitext(fn)[0])
                    if sid is not None:
                        all_ids["Image"][obj_type][aff].add(sid)

        # 3) PointCloud/*.csv 表头的 aff 列 -> (obj, aff) -> [pc_id]
        pc_root = os.path.join(obj_dir, "PointCloud")
        if os.path.isdir(pc_root):
            for fn in os.listdir(pc_root):
                if not fn.lower().endswith(".csv"):
                    continue
                sid = _extract_id_from_name(os.path.splitext(fn)[0])
                if sid is None:
                    continue
                file_path = os.path.join(pc_root, fn)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        first = f.readline().strip()
                except OSError:
                    continue
                if first.startswith("#"):
                    first = first[1:].strip()
                header = [h.strip() for h in first.split(",") if h.strip()]
                for aff in header[3:]:
                    all_ids["PointCloud"][obj_type][aff].add(sid)

    # set -> sorted list, 并过滤空项
    all_json = {}
    for modality in ("Instruction", "Image", "PointCloud"):
        all_json[modality] = {}
        for obj, aff_map in all_ids[modality].items():
            cleaned = {aff: sorted(list(ids)) for aff, ids in aff_map.items() if ids}
            if cleaned:
                all_json[modality][obj] = cleaned

    split_json = {
        "train": {"Instruction": {}, "Image": {}, "PointCloud": {}},
        "val": {"Instruction": {}, "Image": {}, "PointCloud": {}},
        "test": {"Instruction": {}, "Image": {}, "PointCloud": {}},
    }

    def _split_ids(ids: list[int]) -> tuple[list[int], list[int], list[int]]:
        if not ids:
            return [], [], []
        n = len(ids)
        shuffled = list(ids)
        random.shuffle(shuffled)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
        # 修正舍入误差，保证总数不变
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 0:
                n_train -= 1
            elif n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        while n_train + n_val + n_test < n:
            if train_ratio >= max(val_ratio, test_ratio):
                n_train += 1
            elif val_ratio >= test_ratio:
                n_val += 1
            else:
                n_test += 1

        train_lst = sorted(shuffled[:n_train])
        val_lst = sorted(shuffled[n_train:n_train + n_val])
        test_lst = sorted(shuffled[n_train + n_val:n_train + n_val + n_test])
        return train_lst, val_lst, test_lst

    for modality in ("Instruction", "Image", "PointCloud"):
        for obj, aff_map in all_json.get(modality, {}).items():
            for aff, ids in aff_map.items():
                tr, va, te = _split_ids(ids)
                if tr:
                    split_json["train"].setdefault(modality, {}).setdefault(obj, {})[aff] = tr
                if va:
                    split_json["val"].setdefault(modality, {}).setdefault(obj, {})[aff] = va
                if te:
                    split_json["test"].setdefault(modality, {}).setdefault(obj, {})[aff] = te

    def _counts(split_ids):
        out = {}
        for modality in ("Instruction", "Image", "PointCloud"):
            mod_counts = {"total": 0}
            for obj, aff_map in split_ids.get(modality, {}).items():
                obj_counts = {"total": 0}
                for aff, ids in aff_map.items():
                    obj_counts[aff] = len(ids)
                    obj_counts["total"] += len(ids)
                mod_counts[obj] = obj_counts
                mod_counts["total"] += obj_counts["total"]
            out[modality] = mod_counts
        return out

    def _count_paired_samples(split_ids):
        ins_ids = split_ids.get("Instruction", {})
        img_ids = split_ids.get("Image", {})
        pc_ids = split_ids.get("PointCloud", {})
        total = 0
        all_obj = set(ins_ids.keys()) | set(img_ids.keys()) | set(pc_ids.keys())
        for obj in all_obj:
            all_aff = set(ins_ids.get(obj, {}).keys()) | set(img_ids.get(obj, {}).keys()) | set(pc_ids.get(obj, {}).keys())
            for aff in all_aff:
                total += max(
                    len(ins_ids.get(obj, {}).get(aff, [])),
                    len(img_ids.get(obj, {}).get(aff, [])),
                    len(pc_ids.get(obj, {}).get(aff, [])),
                )
        return total

    n_train = _count_paired_samples(split_json["train"])
    n_val = _count_paired_samples(split_json["val"])
    n_test = _count_paired_samples(split_json["test"])
    n_total = n_train + n_val + n_test
    metadata = {
        "total_sample": n_total,
        "train_sample": n_train,
        "val_sample": n_val,
        "test_sample": n_test,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "obj_aff_count_by_split": {
            "train": _counts(split_json["train"]),
            "val": _counts(split_json["val"]),
            "test": _counts(split_json["test"]),
        },
    }

    def _dump_split_part(part: dict, f) -> None:
        """对齐 base_dataset：每个 aff 的 ids 列表保持在同一行。"""
        lines = ['{']
        mod_items = [(k, v) for k, v in part.items() if v]
        for mod_i, (mod_name, mod_data) in enumerate(mod_items):
            obj_items = [(k, {a: v for a, v in aff_map.items() if v}) for k, aff_map in mod_data.items()]
            obj_items = [(k, v) for k, v in obj_items if v]
            if not obj_items:
                continue
            lines.append(f'  "{mod_name}": {{')
            for obj_i, (obj_name, obj_data) in enumerate(obj_items):
                lines.append(f'    "{obj_name}": {{')
                aff_items = [(a, lst) for a, lst in obj_data.items() if lst]
                for aff_i, (aff_name, aff_list) in enumerate(aff_items):
                    ids_str = json.dumps(aff_list, ensure_ascii=False)
                    comma = ',' if aff_i < len(aff_items) - 1 else ''
                    lines.append(f'      "{aff_name}": {ids_str}{comma}')
                obj_comma = ',' if obj_i < len(obj_items) - 1 else ''
                lines.append(f'    }}{obj_comma}')
            mod_comma = ',' if mod_i < len(mod_items) - 1 else ''
            lines.append(f'  }}{mod_comma}')
        lines.append('}')
        f.write('\n'.join(lines))

    split_names = [n for n, r in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)) if r > 0]
    if not split_names:
        split_names = ["train"]
    for name in split_names:
        with open(os.path.join(dataset_root, f"{name}.json"), "w", encoding="utf-8") as f:
            _dump_split_part(split_json[name], f)
    with open(os.path.join(dataset_root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    saved = ", ".join([f"{n}.json" for n in split_names])
    print(f"分割文件已保存: {dataset_root}/{saved}, {dataset_root}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将数据集分割为训练集、验证集和测试集，并保存分割文件")
    parser.add_argument('-d', '--dataset_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='训练集比例，默认 1.0')
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='验证集比例，默认 0.0')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                        help='测试集比例，默认 0.0')
    args = parser.parse_args()
    save_split_from_disk(
        args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )