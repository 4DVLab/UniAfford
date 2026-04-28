#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint Affordance 在线演示应用。

用法示例：
python app.py --checkpoint_path path/to/checkpoint.pt --device cuda --port 7860
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from configs import TrainingConfig
from utils.common import DEFAULT_PC_TOKEN, dict_to_cuda, resolve_dtype
from utils.dataset import JointAffordanceTorchDataset, joint_affordance_collate_fn
from utils.model_io import load_portable_model


SUPPORTED_POINT_CLOUD_EXTS = (".ply", ".obj", ".stl", ".off", ".npy", ".npz", ".txt", ".csv", ".pcd")


def _resolve_config_json_path(checkpoint_path: str, config_json_arg: Optional[str]) -> Optional[str]:
    """与 validate.py 保持一致：优先显式配置，否则自动查找 checkpoint 同目录配置。"""
    if config_json_arg:
        if not os.path.exists(config_json_arg):
            raise FileNotFoundError(f"指定的配置 JSON 不存在: {config_json_arg}")
        return config_json_arg
    ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    candidate = os.path.join(ckpt_dir, "training_config.json")
    return candidate if os.path.exists(candidate) else None


def _normalize_image_size(image_size: Any) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        return int(image_size[0]), int(image_size[1])
    return 1024, 1024


def _file_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, os.PathLike)):
        return os.fspath(file_obj)
    return getattr(file_obj, "name", None) or getattr(file_obj, "path", None)


def _normalize_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().float()
    if mask.dim() > 2:
        mask = mask.squeeze()
    if mask.max() > 1.0 or mask.min() < 0.0:
        mask = mask.sigmoid()
    return mask.clamp(0.0, 1.0).cpu().numpy()


def _load_ascii_pcd(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start = None
    fields: List[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FIELDS"):
            fields = stripped.split()[1:]
        if upper.startswith("DATA"):
            if "ascii" not in upper.lower():
                raise ValueError("当前仅支持 ASCII PCD 文件")
            data_start = idx + 1
            break
    if data_start is None:
        raise ValueError("PCD 文件缺少 DATA ascii 头")

    data = np.loadtxt(lines[data_start:], dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if fields and all(axis in fields for axis in ("x", "y", "z")):
        cols = [fields.index(axis) for axis in ("x", "y", "z")]
        return data[:, cols]
    return data[:, :3]


def load_point_cloud(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        points = np.load(path)
    elif ext == ".npz":
        payload = np.load(path)
        key = "points" if "points" in payload else payload.files[0]
        points = payload[key]
    elif ext in {".txt", ".csv"}:
        delimiter = "," if ext == ".csv" else None
        points = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    elif ext == ".pcd":
        points = _load_ascii_pcd(path)
    elif ext in {".ply", ".obj", ".stl", ".off"}:
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("读取 mesh/ply 点云需要安装 trimesh") from exc
        mesh = trimesh.load(path, process=False)
        if hasattr(mesh, "vertices"):
            points = np.asarray(mesh.vertices)
        elif hasattr(mesh, "points"):
            points = np.asarray(mesh.points)
        else:
            raise ValueError(f"无法从文件中解析点云: {path}")
    else:
        raise ValueError(f"不支持的点云格式: {ext}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"点云应为 [N, 3+]，实际形状为 {points.shape}")
    points = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("点云中没有有效点")
    return points


class JointAffordanceDemoEngine:
    """复用 validate.py 的模型加载方式和数据集侧输入构造逻辑。"""

    def __init__(
        self,
        checkpoint_path: str,
        config_json: Optional[str] = None,
        qwen_model: Optional[str] = None,
        vision_pretrained: Optional[str] = None,
        device: str = "cuda",
    ):
        cfg_json_path = _resolve_config_json_path(checkpoint_path, config_json)
        if cfg_json_path is not None and os.path.exists(cfg_json_path):
            training_cfg = TrainingConfig.from_json(cfg_json_path)
            print(f"已加载训练配置: {cfg_json_path}")
        else:
            training_cfg = TrainingConfig()
            print("未找到训练配置 JSON，使用 TrainingConfig 默认值。")

        if qwen_model:
            training_cfg.model_config.mllm.qwen_model_name_or_path = qwen_model
        if vision_pretrained:
            training_cfg.model_config.image_decoder.vision_pretrained = vision_pretrained

        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.model, self.training_cfg, _ = load_portable_model(
            checkpoint_path,
            config_json_path=cfg_json_path,
            training_cfg=training_cfg,
            map_location="cpu",
            device=self.device,
            strict=False,
        )
        self.model.eval()

        self.model_cfg = self.training_cfg.model_config
        self.processor = self.model.mllm.processor
        self.tokenizer = self.processor.tokenizer
        self.image_size = _normalize_image_size(self.training_cfg.image_size)
        self.num_points = int(self.training_cfg.num_points)
        self.mask_threshold_2d = float(max(getattr(self.training_cfg, "mask_threshold_2d", 0.5), 0.5))
        self.mask_threshold_3d = float(getattr(self.training_cfg, "mask_threshold_3d", 0.5))
        self.mllm_precision = resolve_dtype(self.model_cfg.mllm.compute_dtype) or torch.bfloat16
        self.image_precision = resolve_dtype(self.model_cfg.image_decoder.compute_dtype) or torch.float32
        self.point_precision = resolve_dtype(self.model_cfg.point_decoder.compute_dtype) or torch.float32

        self._dataset_helper = JointAffordanceTorchDataset(
            [],
            processor=self.processor,
            image_size=self.image_size,
            num_points=self.num_points,
            mllm_precision=self.mllm_precision,
            image_precision=self.image_precision,
            point_precision=self.point_precision,
            use_sample_cache=False,
        )
        print(f"模型加载完成，设备: {self.device}")

    def _build_text(self, prompt: str, obj_type: str, aff_type: str, has_image: bool, has_pc: bool) -> Tuple[str, str]:
        obj_type = (obj_type or "object").strip()
        aff_type = (aff_type or "affordance").strip()
        question = (prompt or "").strip() or f"Please identify the {aff_type} affordance region of the {obj_type}."
        if has_pc:
            question = f"{question}\nPoint cloud input: {DEFAULT_PC_TOKEN}"

        answer_parts = []
        if has_image:
            answer_parts.append(f"The {aff_type} affordance region of the {obj_type} is <img_aff>.")
        if has_pc:
            answer_parts.append(f"The 3D {aff_type} affordance region of the {obj_type} is <pc_aff>.")
        if not answer_parts:
            answer_parts.append(f"I need visual input to identify the {aff_type} affordance region of the {obj_type}.")
        return question, " ".join(answer_parts)

    def _prepare_image(self, image: Optional[Image.Image]) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]], Optional[Image.Image], Image.Image]:
        if image is None:
            dummy = Image.new("RGB", (28, 28), color=(0, 0, 0))
            return None, None, None, dummy

        pil_img = image.convert("RGB")
        orig_w, orig_h = pil_img.size
        resized = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=self.image_precision).contiguous()
        return img_tensor.cpu(), (orig_h, orig_w), pil_img, resized

    def _prepare_point_cloud(self, file_obj: Any) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        path = _file_path(file_obj)
        if not path:
            return None, None

        points = load_point_cloud(path)
        replace = points.shape[0] < self.num_points
        indices = np.random.choice(points.shape[0], self.num_points, replace=replace)
        sampled_original = points[indices].astype(np.float32, copy=True)

        sampled = sampled_original.copy()
        sampled -= sampled.mean(axis=0, keepdims=True)
        max_dist = np.linalg.norm(sampled, axis=1).max()
        if max_dist > 0:
            sampled /= max_dist
        pc_tensor = torch.from_numpy(sampled).to(dtype=self.point_precision).contiguous()
        return pc_tensor.cpu(), sampled_original

    def _build_batch(
        self,
        image: Optional[Image.Image],
        point_cloud_file: Any,
        prompt: str,
        obj_type: str,
        aff_type: str,
    ) -> Tuple[Dict[str, Any], Optional[Image.Image], Optional[np.ndarray]]:
        image_tensor, original_size, original_image, qwen_image = self._prepare_image(image)
        pc_tensor, sampled_points = self._prepare_point_cloud(point_cloud_file)
        has_image = image_tensor is not None
        has_pc = pc_tensor is not None
        if not has_image and not has_pc:
            raise ValueError("请至少上传一张图片或一个点云文件")

        question, answer = self._build_text(prompt, obj_type, aff_type, has_image, has_pc)
        sample = self._dataset_helper._build_qwen_inputs(question, answer, qwen_image)
        sample.update(
            {
                "sample_id": "web",
                "obj_type": (obj_type or "object").strip(),
                "aff_type": (aff_type or "affordance").strip(),
                "data_source_id": {},
                "images": image_tensor,
                "img_gt": None,
                "original_size": original_size,
                "point_clouds": pc_tensor,
                "pc_gt": None,
            }
        )

        batch = joint_affordance_collate_fn(
            [sample],
            tokenizer=self.tokenizer,
            output_image_size=self.image_size,
            output_point_nums=self.num_points,
            mllm_precision=self.mllm_precision,
            image_precision=self.image_precision,
            point_precision=self.point_precision,
        )
        return batch, original_image, sampled_points

    @torch.no_grad()
    def infer(
        self,
        image: Optional[Image.Image],
        point_cloud_file: Any,
        prompt: str,
        obj_type: str,
        aff_type: str,
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
        batch, original_image, sampled_points = self._build_batch(image, point_cloud_file, prompt, obj_type, aff_type)
        batch = dict_to_cuda(batch, device=self.device)

        with torch.inference_mode():
            output = self.model(**batch)

        image_vis = None
        pc_vis = None
        status_parts = []

        image_logits = output.get("image_logits")
        has_image = bool(batch["img_valid_mask"][0].item())
        if has_image and isinstance(image_logits, torch.Tensor):
            mask = _normalize_mask(image_logits[0])
            orig_h, orig_w = batch["original_size_list"][0]
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            image_vis = self.visualize_2d(original_image, mask)
            status_parts.append(f"2D 高亮完成，mask 均值 {float(mask.mean()):.4f}，阈值 {self.mask_threshold_2d:.2f}")

        point_logits = output.get("point_logits")
        has_pc = int(batch["pc_valid_lengths"][0].item()) > 0
        if has_pc and isinstance(point_logits, torch.Tensor) and sampled_points is not None:
            valid_len = min(int(batch["pc_valid_lengths"][0].item()), sampled_points.shape[0], point_logits.shape[-1])
            pc_mask = _normalize_mask(point_logits[0])[:valid_len].reshape(-1)
            pc_vis = self.visualize_3d(sampled_points[:valid_len], pc_mask)
            status_parts.append(f"3D 高亮完成，采样点数 {valid_len}，mask 均值 {float(pc_mask.mean()):.4f}，阈值 {self.mask_threshold_3d:.2f}")

        if not status_parts:
            status_parts.append("模型没有返回可视化结果，请检查输入模态和 checkpoint 是否匹配。")
        return image_vis, pc_vis, "\n".join(status_parts)

    def visualize_2d(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if mask.shape[:2] != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0)

        binary = (mask >= self.mask_threshold_2d).astype(np.float32)
        heat = np.zeros_like(image_np)
        heat[..., 0] = 1.0
        heat[..., 1] = 0.15
        result = image_np * (1.0 - alpha * binary[..., None]) + heat * (alpha * binary[..., None])
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def visualize_3d(self, points: np.ndarray, mask: np.ndarray) -> Image.Image:
        mask = np.clip(mask.reshape(-1), 0.0, 1.0)
        binary = mask >= self.mask_threshold_3d

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
        colors[:, :] = np.array([0.55, 0.55, 0.55], dtype=np.float32)
        colors[binary] = np.array([1.0, 0.05, 0.02], dtype=np.float32)
        sizes = np.where(binary, 7.0, 2.0)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, alpha=0.85, linewidths=0)
        ax.set_title("Point Cloud Affordance Highlight")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        self._set_axes_equal(ax, points)
        ax.view_init(elev=22, azim=45)
        fig.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(rgba[..., :3].copy())
        plt.close(fig)
        return image

    @staticmethod
    def _set_axes_equal(ax, points: np.ndarray) -> None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centers = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)


def create_gradio_interface(engine: JointAffordanceDemoEngine):
    def run_inference(image, point_cloud_file, prompt, obj_type, aff_type):
        prompt = (prompt or "").strip()
        obj_type = (obj_type or "object").strip()
        aff_type = (aff_type or "affordance").strip()
        if not prompt:
            prompt = f"Please identify the {aff_type} affordance region of the {obj_type}."
        try:
            return engine.infer(image, point_cloud_file, prompt, obj_type, aff_type)
        except Exception as exc:
            return None, None, f"推理失败: {exc}"

    def create_image_input():
        image_kwargs = dict(type="pil", label="图片输入（可上传、拖拽或粘贴）", height=360)
        try:
            return gr.Image(sources=["upload", "clipboard"], **image_kwargs)
        except TypeError:
            return gr.Image(**image_kwargs)

    with gr.Blocks(title="2D-3D Joint Affordance Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 2D-3D Joint Affordance Demo
            上传或粘贴图片、拖入点云文件，并输入文本提示。模型会对可用模态执行推理，并在网页中显示高亮结果。
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = create_image_input()
                pc_input = gr.File(
                    label="点云输入（拖拽或选择文件）",
                    file_types=list(SUPPORTED_POINT_CLOUD_EXTS),
                )
                prompt = gr.Textbox(
                    label="文本提示",
                    placeholder="例如：Please identify the handle affordance region of the mug.",
                    lines=3,
                )
                with gr.Row():
                    obj_type = gr.Textbox(label="物体类别（可选）", value="object")
                    aff_type = gr.Textbox(label="可供性类别（可选）", value="affordance")
                run_button = gr.Button("运行模型推理", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="图片高亮结果", height=360)
                pc_output = gr.Image(label="点云高亮结果", height=360)
                status = gr.Textbox(label="状态", lines=5)

        run_button.click(
            fn=run_inference,
            inputs=[image_input, pc_input, prompt, obj_type, aff_type],
            outputs=[image_output, pc_output, status],
        )
        prompt.submit(
            fn=run_inference,
            inputs=[image_input, pc_input, prompt, obj_type, aff_type],
            outputs=[image_output, pc_output, status],
        )
        gr.Markdown(
            """
            支持的点云格式：`.ply`, `.obj`, `.stl`, `.off`, `.npy`, `.npz`, `.txt`, `.csv`, `.pcd`。
            `.pcd` 当前仅支持 ASCII 格式。点云会按训练配置采样到固定点数后推理。
            """
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Joint Affordance Gradio Demo")
    parser.add_argument("--checkpoint_path", "--model_path", dest="checkpoint_path", required=True, help="训练好的 checkpoint 路径")
    parser.add_argument("--config_json", type=str, default=None, help="训练配置 JSON；默认在 checkpoint 同目录查找")
    parser.add_argument("--qwen_model", type=str, default=None, help="Qwen 模型路径或名称，覆盖训练配置")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="SAM 权重路径，覆盖训练配置")
    parser.add_argument("--device", type=str, default="cuda", help="设备，例如 cuda、cuda:0、cpu")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio 监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--share", action="store_true", help="创建 Gradio 公共链接")
    args = parser.parse_args()

    engine = JointAffordanceDemoEngine(
        checkpoint_path=args.checkpoint_path,
        config_json=args.config_json,
        qwen_model=args.qwen_model,
        vision_pretrained=args.vision_pretrained,
        device=args.device,
    )
    demo = create_gradio_interface(engine)
    print(f"启动 Gradio 服务: http://localhost:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
