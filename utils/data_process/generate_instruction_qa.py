""" testing, not in used!!
为统一格式数据集生成 Instruction QA。

用途：
- 扫描 {obj_type}/Image 与 {obj_type}/PointCloud；
- 调用 VLM / 3D-VLM 生成面向分割与泛化推理的 QA；
- 将 QA 以 JSON 字符串写入 Instruction.csv 的 ins 字段；
- 使用 img_id / pc_id 绑定到原始图像或点云样本。

生成的 ins 字段形如：
{"question": "...", "answer": "... <img_aff> ...", "kind": "segmentation", ...}

训练侧 utils/dataset.py 会识别该 JSON，并直接使用 question/answer 作为 MLLM 文本监督。
"""

import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


CSV_FIELDS = ["ins", "obj_type", "aff_type", "id", "img_id", "pc_id"]
IMG_TOKEN = "<img_aff>"
PC_TOKEN = "<pc_aff>"


def _normalize_label(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip().lower()
    return text if text else fallback


def _extract_id_from_name(name: str) -> Optional[int]:
    match = re.search(r"_(\d+)(?:_|$)", name)
    return int(match.group(1)) if match else None


def _read_existing_instruction_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _next_instruction_id(rows: List[Dict[str, str]]) -> int:
    max_id = 0
    for row in rows:
        try:
            max_id = max(max_id, int(row.get("id") or 0))
        except ValueError:
            continue
    return max_id + 1


def _write_instruction_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        _ensure_instruction_csv_schema(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _ensure_instruction_csv_schema(csv_path: str) -> None:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        if all(field in old_fields for field in CSV_FIELDS):
            return
        rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _encode_image_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(image_path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{payload}"


def _find_image_path(rgb_dir: str, obj_type: str, img_id: int) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        path = os.path.join(rgb_dir, f"{obj_type}_{img_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def _read_pointcloud_header(pc_path: str) -> List[str]:
    with open(pc_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("#"):
        first = first[1:].strip()
    return [x.strip() for x in first.split(",") if x.strip()]


def _read_pointcloud_preview(pc_path: str, max_lines: int = 24) -> str:
    lines = []
    with open(pc_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            lines.append(line.strip())
    return "\n".join(lines)


def iter_image_targets(dataset_root: str) -> Iterable[Dict[str, Any]]:
    for obj_type in sorted(os.listdir(dataset_root)):
        obj_dir = os.path.join(dataset_root, obj_type)
        rgb_dir = os.path.join(obj_dir, "Image", "rgb")
        mask_root = os.path.join(obj_dir, "Image", "mask")
        if not os.path.isdir(rgb_dir) or not os.path.isdir(mask_root):
            continue
        for aff_type in sorted(os.listdir(mask_root)):
            aff_dir = os.path.join(mask_root, aff_type)
            if not os.path.isdir(aff_dir):
                continue
            for mask_name in sorted(os.listdir(aff_dir)):
                if not mask_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    continue
                img_id = _extract_id_from_name(os.path.splitext(mask_name)[0])
                if img_id is None:
                    continue
                image_path = _find_image_path(rgb_dir, obj_type, img_id)
                if image_path is None:
                    continue
                yield {
                    "modality": "image",
                    "obj_type": _normalize_label(obj_type),
                    "aff_type": _normalize_label(aff_type),
                    "img_id": img_id,
                    "pc_id": None,
                    "image_path": image_path,
                    "mask_path": os.path.join(aff_dir, mask_name),
                }


def iter_pointcloud_targets(dataset_root: str) -> Iterable[Dict[str, Any]]:
    for obj_type in sorted(os.listdir(dataset_root)):
        pc_dir = os.path.join(dataset_root, obj_type, "PointCloud")
        if not os.path.isdir(pc_dir):
            continue
        for pc_name in sorted(os.listdir(pc_dir)):
            if not pc_name.lower().endswith(".csv"):
                continue
            pc_id = _extract_id_from_name(os.path.splitext(pc_name)[0])
            if pc_id is None:
                continue
            pc_path = os.path.join(pc_dir, pc_name)
            labels = _read_pointcloud_header(pc_path)[3:]
            for aff_type in labels:
                yield {
                    "modality": "pointcloud",
                    "obj_type": _normalize_label(obj_type),
                    "aff_type": _normalize_label(aff_type),
                    "img_id": None,
                    "pc_id": pc_id,
                    "pc_path": pc_path,
                    "pc_preview": _read_pointcloud_preview(pc_path),
                }


def build_generation_prompt(target: Dict[str, Any], num_seg: int, num_reasoning: int) -> str:
    marker = IMG_TOKEN if target["modality"] == "image" else PC_TOKEN
    modality_text = "image" if target["modality"] == "image" else "point cloud"
    pc_extra = ""
    if target["modality"] == "pointcloud":
        pc_extra = (
            "\nPoint cloud CSV preview is provided below. Columns after x,y,z are affordance labels.\n"
            f"{target.get('pc_preview', '')}\n"
        )

    return f"""
You are generating training QA pairs for a multimodal affordance segmentation model.

Target modality: {modality_text}
Object category: {target['obj_type']}
Affordance label: {target['aff_type']}
Segmentation marker that MUST appear in segmentation answers: {marker}
{pc_extra}

Generate QA pairs in JSON only:
{{
  "qa_pairs": [
    {{"kind": "segmentation", "question": "...", "answer": "..."}},
    {{"kind": "reasoning", "question": "...", "answer": "..."}}
  ]
}}

Requirements:
- Generate exactly {num_seg} segmentation QA pairs and {num_reasoning} reasoning QA pairs.
- Segmentation questions must clearly ask to segment, locate, highlight, mark, identify, or ground the {target['aff_type']} affordance region.
- Segmentation answers must explicitly include {marker}, and must mention that this marker corresponds to the {target['aff_type']} region.
- Reasoning questions must NOT explicitly request segmentation or grounding.
- Reasoning answers must NOT include {IMG_TOKEN} or {PC_TOKEN}.
- Cover diverse user wording: direct command, natural question, task-oriented request, affordance reasoning request.
- Keep answers concise and suitable as supervised assistant responses.
""".strip()


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Optional[str] = None,
    timeout: int = 120,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    content: Any
    if image_path:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _encode_image_data_url(image_path)}},
        ]
    else:
        content = prompt

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Do not wrap it in Markdown."},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": 1200,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"VLM request failed: HTTP {exc.code}: {detail}") from exc

    text = data["choices"][0]["message"]["content"]
    return parse_json_response(text)


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def template_qa(target: Dict[str, Any], num_seg: int, num_reasoning: int) -> Dict[str, Any]:
    marker = IMG_TOKEN if target["modality"] == "image" else PC_TOKEN
    obj = target["obj_type"]
    aff = target["aff_type"]
    seg_templates = [
        (
            f"Please segment the {aff} affordance region of the {obj}.",
            f"The {aff} affordance region of the {obj} is marked as {marker}.",
        ),
        (
            f"Which area of the {obj} should be highlighted for {aff} interaction?",
            f"For {aff} interaction, the corresponding region is {marker}.",
        ),
        (
            f"Locate and mark the part of the {obj} that supports {aff}.",
            f"The part supporting {aff} is represented by {marker}.",
        ),
    ]
    reasoning_templates = [
        (
            f"What visual or geometric clues suggest that the {obj} can support {aff}?",
            f"Look for the functional part whose shape, placement, or accessibility makes {aff} possible.",
        ),
        (
            f"Why is {aff} a plausible affordance for this {obj}?",
            f"It is plausible when the object contains a part designed or positioned for that interaction.",
        ),
    ]
    qa_pairs = []
    for question, answer in seg_templates[:num_seg]:
        qa_pairs.append({"kind": "segmentation", "question": question, "answer": answer})
    for question, answer in reasoning_templates[:num_reasoning]:
        qa_pairs.append({"kind": "reasoning", "question": question, "answer": answer})
    return {"qa_pairs": qa_pairs}


def validate_and_repair_qa(payload: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, str]]:
    marker = IMG_TOKEN if target["modality"] == "image" else PC_TOKEN
    qa_pairs = payload.get("qa_pairs", []) if isinstance(payload, dict) else []
    valid = []
    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        kind = str(qa.get("kind") or "").strip().lower()
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question or not answer:
            continue
        if kind == "segmentation":
            if marker not in answer:
                answer = f"{answer.rstrip()} The {target['aff_type']} region is {marker}."
        else:
            kind = "reasoning"
            answer = answer.replace(IMG_TOKEN, "").replace(PC_TOKEN, "").strip()
        valid.append({"kind": kind, "question": question, "answer": answer})
    return valid


def make_instruction_payload(qa: Dict[str, str], target: Dict[str, Any], source: str) -> str:
    payload = {
        "question": qa["question"],
        "answer": qa["answer"],
        "kind": qa["kind"],
        "source": source,
        "modality": target["modality"],
        "obj_type": target["obj_type"],
        "aff_type": target["aff_type"],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_existing_generated_keys(rows: List[Dict[str, str]]) -> set:
    keys = set()
    for row in rows:
        try:
            payload = json.loads(row.get("ins") or "")
        except json.JSONDecodeError:
            continue
        if payload.get("source") != "generated_vlm_qa":
            continue
        keys.add(
            (
                row.get("img_id") or "",
                row.get("pc_id") or "",
                row.get("aff_type") or "",
                payload.get("kind") or "",
                payload.get("question") or "",
            )
        )
    return keys


def generate_for_target(args, target: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = build_generation_prompt(target, args.num_segmentation, args.num_reasoning)
    if args.backend == "template":
        payload = template_qa(target, args.num_segmentation, args.num_reasoning)
    else:
        image_path = target.get("image_path") if target["modality"] == "image" else None
        payload = call_openai_compatible(
            base_url=args.base_url,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=args.model,
            prompt=prompt,
            image_path=image_path,
            timeout=args.timeout,
            temperature=args.temperature,
        )
    return validate_and_repair_qa(payload, target)


def write_generated_rows(args, obj_type: str, rows: List[Dict[str, Any]]) -> int:
    csv_path = os.path.join(args.dataset_root, obj_type, "Instruction.csv")
    existing_rows = _read_existing_instruction_rows(csv_path)
    existing_keys = build_existing_generated_keys(existing_rows)
    next_id = _next_instruction_id(existing_rows)
    to_write = []

    for row in rows:
        key = (
            str(row.get("img_id") or ""),
            str(row.get("pc_id") or ""),
            row.get("aff_type") or "",
            row.get("_kind") or "",
            row.get("_question") or "",
        )
        if args.skip_existing and key in existing_keys:
            continue
        row["id"] = next_id
        next_id += 1
        row.pop("_kind", None)
        row.pop("_question", None)
        to_write.append(row)

    _write_instruction_rows(csv_path, to_write)
    return len(to_write)


def main():
    parser = argparse.ArgumentParser(description="调用 VLM/3D-VLM 为统一格式数据集生成 Instruction QA。")
    parser.add_argument("-d", "--dataset_root", required=True, help="统一格式数据集根目录")
    parser.add_argument("--backend", choices=["openai", "template"], default="openai")
    parser.add_argument("--base_url", default=os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api_key", default=os.environ.get("VLM_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen3-vl"))
    parser.add_argument("--modalities", nargs="+", choices=["image", "pointcloud"], default=["image", "pointcloud"])
    parser.add_argument("--num_segmentation", type=int, default=3)
    parser.add_argument("--num_reasoning", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个 obj/aff/id 目标")
    parser.add_argument("--sleep", type=float, default=0.0, help="每次调用后的等待秒数")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    args.dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(args.dataset_root):
        raise ValueError(f"dataset_root 不存在: {args.dataset_root}")

    targets = []
    if "image" in args.modalities:
        targets.extend(iter_image_targets(args.dataset_root))
    if "pointcloud" in args.modalities:
        targets.extend(iter_pointcloud_targets(args.dataset_root))
    if args.limit is not None:
        targets = targets[: args.limit]

    rows_by_obj = defaultdict(list)
    for idx, target in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] generating {target['modality']} {target['obj_type']} / {target['aff_type']}")
        qa_pairs = generate_for_target(args, target)
        for qa in qa_pairs:
            rows_by_obj[target["obj_type"]].append(
                {
                    "ins": make_instruction_payload(qa, target, source="generated_vlm_qa"),
                    "obj_type": target["obj_type"],
                    "aff_type": target["aff_type"],
                    "img_id": "" if target.get("img_id") is None else target["img_id"],
                    "pc_id": "" if target.get("pc_id") is None else target["pc_id"],
                    "_kind": qa["kind"],
                    "_question": qa["question"],
                }
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    total = 0
    for obj_type, rows in rows_by_obj.items():
        total += write_generated_rows(args, obj_type, rows)
    print(f"已写入 {total} 条生成 QA 到 Instruction.csv")


if __name__ == "__main__":
    main()
