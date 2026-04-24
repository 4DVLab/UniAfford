import os
import csv
import json
import re
import random
from collections import defaultdict
import argparse
from typing import Dict, Optional, Literal


def _extract_id_from_name(name: str):
    """从文件名中提取样本 id，如 Mug_12_grasp.png / Mug_12.csv -> 12。"""
    m = re.search(r'_(\d+)(?:_|$)', name)
    return int(m.group(1)) if m else None


def _new_ids_struct():
    return {
        "Instruction": defaultdict(lambda: defaultdict(set)),
        "Image": defaultdict(lambda: defaultdict(set)),
        "PointCloud": defaultdict(lambda: defaultdict(set)),
    }


def _entry_sort_key(entry):
    if isinstance(entry, dict):
        return (
            int(entry.get("id", -1)) if entry.get("id") not in (None, "", "None", "none") else -1,
            int(entry.get("img_id", -1)) if entry.get("img_id") not in (None, "", "None", "none") else -1,
            int(entry.get("pc_id", -1)) if entry.get("pc_id") not in (None, "", "None", "none") else -1,
        )
    try:
        return (int(entry), -1, -1)
    except (TypeError, ValueError):
        return (-1, -1, -1)


def _normalize_optional_int(value):
    if value in (None, "", "None", "none"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_instruction_entry(entry):
    if isinstance(entry, str):
        stripped = entry.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(entry, dict):
        ins_id = _normalize_optional_int(entry.get("id", entry.get("ins_id")))
        img_id = _normalize_optional_int(entry.get("img_id"))
        pc_id = _normalize_optional_int(entry.get("pc_id"))
        if ins_id is None and img_id is None and pc_id is None:
            return None
        if img_id is None and pc_id is None:
            return ins_id
        return {"id": ins_id, "img_id": img_id, "pc_id": pc_id}
    try:
        return int(entry)
    except (TypeError, ValueError):
        return None


def _normalize_ids(sample_ids: Dict) -> Dict[str, Dict[str, Dict[str, list]]]:
    """归一化为 {Instruction/Image/PointCloud: {obj: {aff: [entries...]}}}。"""
    out = {"Instruction": {}, "Image": {}, "PointCloud": {}}
    alias = {"Instruction": ("Instruction", "ins"), "Image": ("Image", "img"), "PointCloud": ("PointCloud", "pc")}
    for mod, keys in alias.items():
        src = {}
        for k in keys:
            if k in sample_ids:
                src = sample_ids[k] or {}
                break
        out[mod] = {}
        for obj, aff_map in src.items():
            out[mod][obj] = {}
            for aff, ids in aff_map.items():
                cleaned = []
                for x in ids:
                    if mod == "Instruction":
                        normalized = _normalize_instruction_entry(x)
                        if normalized is not None:
                            cleaned.append(normalized)
                    else:
                        try:
                            cleaned.append(int(x))
                        except (TypeError, ValueError):
                            continue
                if cleaned:
                    deduped = []
                    seen = set()
                    for entry in cleaned:
                        if isinstance(entry, dict):
                            key = (entry.get("id"), entry.get("img_id"), entry.get("pc_id"))
                        else:
                            key = entry
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append(entry)
                    out[mod][obj][aff] = sorted(deduped, key=_entry_sort_key)
    return out


def _dump_split_part(part: dict, f) -> None:
    """每个 aff 的 ids 列表保持在同一行。"""
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


class SplitManager:
    """
    统一分割管理器（迁移自 base_dataset）：
    - 支持从磁盘扫描 IDs（默认）
    - 支持从内存（类注册表或手动 sample_ids）获取 IDs
    - 两种来源共用同一套采样/切分/保存逻辑
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = os.path.abspath(dataset_root)

    def collect_ids_from_disk(self) -> Dict:
        all_ids = _new_ids_struct()
        for obj_type in sorted(os.listdir(self.dataset_root)):
            obj_dir = os.path.join(self.dataset_root, obj_type)
            if not os.path.isdir(obj_dir):
                continue

            ins_csv = os.path.join(obj_dir, "Instruction.csv")
            if os.path.exists(ins_csv):
                with open(ins_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        aff = (row.get("aff_type") or "").strip()
                        sid = row.get("id")
                        if not aff or sid is None:
                            continue
                        ins_id = _normalize_optional_int(sid)
                        if ins_id is None:
                            continue
                        img_id = _normalize_optional_int(row.get("img_id"))
                        pc_id = _normalize_optional_int(row.get("pc_id"))
                        if img_id is None and pc_id is None:
                            all_ids["Instruction"][obj_type][aff].add(ins_id)
                        else:
                            all_ids["Instruction"][obj_type][aff].add(
                                json.dumps(
                                    {"id": ins_id, "img_id": img_id, "pc_id": pc_id},
                                    sort_keys=True,
                                )
                            )

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
        return _normalize_ids(all_ids)

    def collect_ids_from_memory(self, sample_ids: Optional[Dict] = None) -> Dict:
        """
        从内存获取 IDs：
        1) 若传入 sample_ids，直接归一化使用
        2) 否则从 base_dataset 的类注册表收集
        """
        if sample_ids is not None:
            return _normalize_ids(sample_ids)

        from utils.base_dataset import Instruction, Image, PointCloud  # lazy import, 避免循环依赖
        all_ids = _new_ids_struct()

        for obj_type, items in Instruction.all.items():
            for ins in items:
                if ins is None:
                    continue
                if getattr(ins, "img_id", None) is None and getattr(ins, "pc_id", None) is None:
                    all_ids["Instruction"][obj_type][str(ins.aff_type)].add(int(ins.id))
                else:
                    all_ids["Instruction"][obj_type][str(ins.aff_type)].add(
                        json.dumps(
                            {"id": int(ins.id), "img_id": ins.img_id, "pc_id": ins.pc_id},
                            sort_keys=True,
                        )
                    )

        for obj_type, items in Image.all.items():
            for img in items:
                if img is None:
                    continue
                for aff in img.aff_mask_dict.keys():
                    all_ids["Image"][obj_type][str(aff)].add(int(img.id))

        for obj_type, items in PointCloud.all.items():
            for pc in items:
                if pc is None:
                    continue
                for aff in pc.get_aff_types():
                    all_ids["PointCloud"][obj_type][str(aff)].add(int(pc.id))

        return _normalize_ids(all_ids)

    @staticmethod
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

    @staticmethod
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

    def split(
        self,
        train_ratio: float = 1.0,
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
        random_seed: int = 114514,
        sample_rate: Optional[float] = None,
        max_sample_per_group: Optional[int] = None,
        min_sample_per_group: int = 10,
        min_holdout_per_group: int = 5,
        balance_modalities: bool = False,
        id_source: Literal["disk", "memory"] = "disk",
        sample_ids: Optional[Dict] = None,
    ):
        ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
        if any(r < 0 for r in ratios):
            raise ValueError("train/val/test 比例不能为负数")
        s = sum(ratios)
        if s <= 0:
            raise ValueError("train/val/test 比例之和必须大于 0")
        train_ratio, val_ratio, test_ratio = [r / s for r in ratios]

        if sample_rate is not None and not (0 < float(sample_rate) <= 1.0):
            raise ValueError("sample_rate 需在 (0, 1] 内")

        random.seed(random_seed)
        if id_source == "disk":
            all_json = self.collect_ids_from_disk()
        elif id_source == "memory":
            all_json = self.collect_ids_from_memory(sample_ids=sample_ids)
        else:
            raise ValueError(f"id_source 必须为 'disk' 或 'memory'，当前: {id_source}")

        def _balance_one_split_part(part: Dict[str, Dict[str, Dict[str, list[int]]]]) -> None:
            """
            可选平衡（默认关闭，不推荐启用）：
            在同一 split 内，对每个 (obj_type, aff_type) 将图文分支(Instruction/Image)
            与 PointCloud 分支数量做对齐。通过有放回复制较小分支实现，会引入重复样本。
            """
            ins_mod = part.get("Instruction", {})
            img_mod = part.get("Image", {})
            pc_mod = part.get("PointCloud", {})
            all_obj = set(ins_mod.keys()) | set(pc_mod.keys())
            for obj in all_obj:
                common_aff = set(ins_mod.get(obj, {}).keys()) & set(pc_mod.get(obj, {}).keys())
                for aff in common_aff:
                    ins_ids = list(ins_mod.get(obj, {}).get(aff, []))
                    img_ids = list(img_mod.get(obj, {}).get(aff, []))
                    pc_ids = list(pc_mod.get(obj, {}).get(aff, []))
                    if not ins_ids or not pc_ids:
                        continue
                    target = max(len(ins_ids), len(pc_ids))
                    if target <= 0:
                        continue
                    # 同步扩展 ins/img，保持图文配对关系
                    if len(ins_ids) < target:
                        pair_idx = list(range(len(ins_ids)))
                        ext_idx = pair_idx + [random.choice(pair_idx) for _ in range(target - len(pair_idx))]
                        ins_mod.setdefault(obj, {})[aff] = [ins_ids[i] for i in ext_idx]
                        if img_ids:
                            img_mod.setdefault(obj, {})[aff] = [img_ids[i] for i in ext_idx]
                    # 扩展 pc
                    if len(pc_ids) < target:
                        ext_pc = pc_ids + [random.choice(pc_ids) for _ in range(target - len(pc_ids))]
                        pc_mod.setdefault(obj, {})[aff] = ext_pc

        split_json = {
            "train": {"Instruction": {}, "Image": {}, "PointCloud": {}},
            "val": {"Instruction": {}, "Image": {}, "PointCloud": {}},
            "test": {"Instruction": {}, "Image": {}, "PointCloud": {}},
        }
        needs_holdout = (val_ratio > 0 or test_ratio > 0)

        def _apply_sampling(ids: list, obj: str, aff: str) -> list:
            if sample_rate is None:
                return ids
            group_seed = hash((obj, aff, random_seed)) % (2 ** 32)
            rng = random.Random(group_seed)
            n_total = len(ids)
            n_target = max(min_sample_per_group, int(round(n_total * float(sample_rate))))
            if max_sample_per_group is not None:
                n_target = min(n_target, int(max_sample_per_group))
            n_target = min(max(1, n_target), n_total)
            return sorted(rng.sample(ids, n_target), key=_entry_sort_key)

        def _split_one_group(ids: list, obj: str, aff: str):
            ids = _apply_sampling(ids, obj, aff)
            n = len(ids)
            if n == 0:
                return [], [], []
            shuffled = list(ids)
            random.shuffle(shuffled)

            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            if needs_holdout:
                if test_ratio > 0:
                    n_test = max(min_holdout_per_group, n_test)
                if val_ratio > 0:
                    n_val = max(min_holdout_per_group, n_val)
                if n_test + n_val >= n:
                    overflow = n_test + n_val - (n - 1)
                    while overflow > 0 and (n_val > 0 or n_test > 0):
                        if n_val >= n_test and n_val > 0:
                            n_val -= 1
                        elif n_test > 0:
                            n_test -= 1
                        overflow -= 1
            else:
                n_test = 0
                n_val = 0
            train = sorted(shuffled[n_test + n_val:], key=_entry_sort_key)
            val = sorted(shuffled[n_test:n_test + n_val], key=_entry_sort_key)
            test = sorted(shuffled[:n_test], key=_entry_sort_key)
            return train, val, test

        for modality in ("Instruction", "Image", "PointCloud"):
            for obj, aff_map in all_json.get(modality, {}).items():
                for aff, ids in aff_map.items():
                    tr, va, te = _split_one_group(list(ids), obj, aff)
                    if tr:
                        split_json["train"].setdefault(modality, {}).setdefault(obj, {})[aff] = tr
                    if va:
                        split_json["val"].setdefault(modality, {}).setdefault(obj, {})[aff] = va
                    if te:
                        split_json["test"].setdefault(modality, {}).setdefault(obj, {})[aff] = te

        if balance_modalities:
            _balance_one_split_part(split_json["train"])
            _balance_one_split_part(split_json["val"])
            _balance_one_split_part(split_json["test"])

        n_train = self._count_paired_samples(split_json["train"])
        n_val = self._count_paired_samples(split_json["val"])
        n_test = self._count_paired_samples(split_json["test"])
        metadata = {
            "total_sample": n_train + n_val + n_test,
            "train_sample": n_train,
            "val_sample": n_val,
            "test_sample": n_test,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "id_source": id_source,
            "balance_modalities": bool(balance_modalities),
            "obj_aff_count_by_split": {
                "train": self._counts(split_json["train"]),
                "val": self._counts(split_json["val"]),
                "test": self._counts(split_json["test"]),
            },
        }
        if sample_rate is not None:
            metadata["sample_rate"] = float(sample_rate)
            metadata["max_sample_per_group"] = max_sample_per_group
            metadata["min_sample_per_group"] = min_sample_per_group

        split_names = [n for n, r in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)) if r > 0]
        if not split_names:
            split_names = ["train"]
        for name in split_names:
            with open(os.path.join(self.dataset_root, f"{name}.json"), "w", encoding="utf-8") as f:
                _dump_split_part(split_json[name], f)
        with open(os.path.join(self.dataset_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        saved = ", ".join([f"{n}.json" for n in split_names])
        print(f"分割文件已保存: {self.dataset_root}/{saved}, {self.dataset_root}/metadata.json")
        return split_json, metadata


def save_split_from_disk(
    dataset_root: str,
    train_ratio: float = 1.0,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    **kwargs,
):
    sm = SplitManager(dataset_root)
    return sm.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        id_source="disk",
        **kwargs,
    )


def save_split_from_memory(
    dataset_root: str,
    train_ratio: float = 1.0,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    sample_ids: Optional[Dict] = None,
    **kwargs,
):
    sm = SplitManager(dataset_root)
    return sm.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        id_source="memory",
        sample_ids=sample_ids,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一数据集分割工具（支持 disk / memory 两种 ID 来源）")
    parser.add_argument("-d", "--dataset_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="训练集比例，默认 0.95")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例，默认 0.05")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="测试集比例，默认 0.0")
    parser.add_argument("--id_source", type=str, choices=["disk", "memory"], default="disk", help="ID 来源，默认 disk")
    parser.add_argument("--seed", type=int, default=114514, help="随机种子")
    parser.add_argument("--sample_rate", type=float, default=None, help="组内采样比例 (0,1]，默认不采样")
    parser.add_argument(
        "--max_sample_per_group", type=int, default=None,
        help="每个 group 最大保留样本数；group 定义为 (modality, obj_type, aff_type) 的一个子集",
    )
    parser.add_argument(
        "--min_sample_per_group", type=int, default=10,
        help="按 sample_rate 采样时，每个 group 至少保留的样本数；group 定义同上",
    )
    parser.add_argument(
        "--min_holdout_per_group", type=int, default=5,
        help="启用 val/test 时，每个 group 在 holdout 集（val+test）中尽量预留的最小样本数；holdout 指从 train 划出的验证/测试样本",
    )
    parser.add_argument(
        "--balance_modalities",
        action="store_true",
        help="可选：在 split 内按 (obj_type, aff_type) 对齐图文与点云样本数（通过重复采样）。默认关闭，且不推荐启用。",
    )
    args = parser.parse_args()

    SplitManager(args.dataset_root).split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        sample_rate=args.sample_rate,
        max_sample_per_group=args.max_sample_per_group,
        min_sample_per_group=args.min_sample_per_group,
        min_holdout_per_group=args.min_holdout_per_group,
        balance_modalities=args.balance_modalities,
        id_source=args.id_source,
    )