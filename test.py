import os
from collections import Counter

import numpy as np
from datasets import load_from_disk

from utils.data_process.external_datasets_processing import ReasonAff_IMG


REASONAFF_ROOT = r"x:\Yiqian\Downloads\reasonaff"
SIZE_SAMPLE_LIMIT = 200
LOADER_SAMPLE_LIMIT = 50


def row_signature(row):
    """用于判断 train/test 是否有完全相同样本；不用 id，因为两个 split 的 id 都可能从 0 开始。"""
    return (
        str(row["problem"]),
        str(row["solution"]),
        str(row["aff_name"]),
        str(row["part_name"]),
        int(row["img_height"]),
        int(row["img_width"]),
    )


def inspect_split(root, split_name, root_dir, hf_split):
    ds_path = os.path.join(root, root_dir)
    ds = load_from_disk(ds_path)[hf_split]
    ds_np = ds.with_format("numpy", columns=["image", "mask"], output_all_columns=True)
    meta_ds = ds.remove_columns(["image", "mask"])

    image_sizes = Counter()
    meta_sizes = Counter()
    mask_sizes = Counter()
    square_images = 0
    image_meta_mismatch = 0
    mask_meta_mismatch = 0
    signatures = set()
    examples = []

    for row in meta_ds:
        signatures.add(row_signature(row))

    for idx, row in enumerate(ds_np):
        if idx >= SIZE_SAMPLE_LIMIT:
            break
        image = np.asarray(row["image"])
        mask = np.asarray(row["mask"])
        image_hw = tuple(image.shape[:2])
        mask_hw = tuple(mask.shape[:2])
        meta_hw = (int(row["img_height"]), int(row["img_width"]))

        image_sizes[image_hw] += 1
        mask_sizes[mask_hw] += 1
        meta_sizes[meta_hw] += 1
        square_images += int(image_hw[0] == image_hw[1])
        image_meta_mismatch += int(image_hw != meta_hw)
        mask_meta_mismatch += int(mask_hw != meta_hw)
        signatures.add(row_signature(row))

        if len(examples) < 5 and (image_hw[0] == image_hw[1] or image_hw != meta_hw):
            examples.append({
                "idx": idx,
                "id": row["id"],
                "image_hw": image_hw,
                "mask_hw": mask_hw,
                "meta_hw": meta_hw,
                "aff": row["aff_name"],
                "part": row["part_name"],
            })

    inspected = min(len(ds), SIZE_SAMPLE_LIMIT)
    print(f"\n[{split_name}] rows={len(ds)}, inspected image/mask samples={inspected}", flush=True)
    print(f"  image size top5: {image_sizes.most_common(5)}")
    print(f"  mask size top5:  {mask_sizes.most_common(5)}")
    print(f"  meta size top5:  {meta_sizes.most_common(5)}")
    print(f"  square images: {square_images}/{inspected}")
    print(f"  image != metadata size: {image_meta_mismatch}/{inspected}")
    print(f"  mask != metadata size:  {mask_meta_mismatch}/{inspected}")
    print(f"  examples: {examples}")
    return signatures


def inspect_reasonaff_loader(root):
    ReasonAff_IMG.split_names = ("train",)
    ReasonAff_IMG.progress_interval = 0
    loaded = 0
    first = None
    for img in ReasonAff_IMG.load_all(root):
        loaded += 1
        if first is None:
            first = {
                "obj_type": img.obj_type,
                "image_shape": img.img.shape,
                "mask_shapes": {k: v.shape for k, v in img.aff_mask_dict.items()},
            }
        if loaded >= LOADER_SAMPLE_LIMIT:
            break
    print(f"\n[ReasonAff_IMG loader default train] sampled={loaded}, first={first}", flush=True)


def main():
    train_sig = inspect_split(REASONAFF_ROOT, "train", "train_new", "train")
    test_sig = inspect_split(REASONAFF_ROOT, "test", "test_new", "test")
    overlap = train_sig & test_sig
    print(f"\n[train/test exact content overlap] {len(overlap)} samples", flush=True)
    if overlap:
        print(f"  first overlap signature: {next(iter(overlap))}")
    inspect_reasonaff_loader(REASONAFF_ROOT)


if __name__ == "__main__":
    main()
