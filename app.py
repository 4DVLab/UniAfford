#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LISA 2D-3D Joint Affordance 在线演示应用
支持图像和点云的语言引导分割
"""

import os
import sys
import argparse
import numpy as np
import torch
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import tempfile
import trimesh

import transformers
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


class LISAInferenceEngine:
    """LISA 推理引擎"""
    
    def __init__(
        self,
        model_path: str,
        vision_pretrained: str = "PATH_TO_SAM_ViT-H",
        device: str = "cuda:0",
        precision: str = "bf16",
        image_size: int = 1024,
        num_points: int = 2048,
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型权重路径
            vision_pretrained: SAM 预训练权重路径
            device: 设备
            precision: 精度 (fp32, fp16, bf16)
            image_size: 图像尺寸
            num_points: 点云点数
        """
        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.num_points = num_points
        
        print(f"🚀 正在加载模型从 {model_path}...")
        
        # 初始化 tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # 添加特殊标记
        self.tokenizer.add_tokens("[SEG]")
        self.tokenizer.add_tokens("[AFF]")
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.aff_token_idx = self.tokenizer("[AFF]", add_special_tokens=False).input_ids[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_tokens(DEFAULT_IM_START_TOKEN, special_tokens=True)
            self.tokenizer.add_tokens(DEFAULT_IM_END_TOKEN, special_tokens=True)
        
        # 设置精度
        torch_dtype = torch.float32
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.half
        
        # 加载模型
        model_args = {
            "train_mask_decoder": True,
            "out_dim": 256,
            "ce_loss_weight": 1.0,
            "dice_loss_weight": 0.5,
            "bce_loss_weight": 2.0,
            "seg_token_idx": self.seg_token_idx,
            "aff_token_idx": self.aff_token_idx,
            "vision_pretrained": vision_pretrained,
            "vision_tower": "openai/clip-vit-large-patch14",
            "use_mm_start_end": True,
        }
        
        self.model = LISAForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **model_args
        )
        
        self.model.to(device=self.device)
        self.model.eval()
        
        # 初始化对话模板
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_2"]
        
        print("✅ 模型加载完成！")
    
    def preprocess_image(self, image: Image.Image):
        """
        预处理图像
        
        Args:
            image: PIL Image
            
        Returns:
            images: SAM 输入图像 [1, 3, H, W]
            images_clip: CLIP 输入图像 [1, 3, H', W']
            original_size: 原始尺寸 (H, W)
            resize_size: 调整后尺寸 (H, W)
        """
        # 转换为 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        original_size = image.size[::-1]  # (H, W)
        
        # 调整图像大小
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        resize_size = (self.image_size, self.image_size)
        
        # 转换为张量
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # 移动到设备
        if self.precision == "fp16":
            image_tensor = image_tensor.half()
        elif self.precision == "bf16":
            image_tensor = image_tensor.bfloat16()
        
        images = image_tensor.to(self.device)
        images_clip = images.clone()
        
        return images, images_clip, original_size, resize_size
    
    def preprocess_point_cloud(self, point_cloud_file):
        """
        预处理点云
        
        Args:
            point_cloud_file: 点云文件路径 (.ply, .pcd, .txt, .npy)
            
        Returns:
            point_clouds: 点云张量 [1, 3, N]
            original_points: 原始点云 numpy 数组 [N, 3]
        """
        # 读取点云
        if point_cloud_file.endswith('.ply'):
            mesh = trimesh.load(point_cloud_file)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
            else:
                points = np.array(mesh.points)
        elif point_cloud_file.endswith('.npy'):
            points = np.load(point_cloud_file)
        elif point_cloud_file.endswith('.txt'):
            points = np.loadtxt(point_cloud_file)
        else:
            raise ValueError(f"不支持的点云格式: {point_cloud_file}")
        
        # 确保形状为 [N, 3]
        if points.shape[1] != 3:
            points = points[:, :3]
        
        original_points = points.copy()
        
        # 采样到固定点数
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            # 上采样
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]
        
        # 归一化到单位球
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / max_dist
        
        # 转换为张量 [1, 3, N]
        point_tensor = torch.from_numpy(points.T).unsqueeze(0).float()
        
        if self.precision == "fp16":
            point_tensor = point_tensor.half()
        elif self.precision == "bf16":
            point_tensor = point_tensor.bfloat16()
        
        point_clouds = point_tensor.to(self.device)
        
        return point_clouds, original_points
    
    def prepare_text_input(self, text_prompt: str, has_image: bool, has_point_cloud: bool):
        """
        准备文本输入
        
        Args:
            text_prompt: 用户输入的文本提示
            has_image: 是否有图像输入
            has_point_cloud: 是否有点云输入
            
        Returns:
            input_ids: token IDs
            attention_masks: 注意力掩码
        """
        # 构建对话
        conv = conversation_lib.conv_templates["llava_llama_2"].copy()
        conv.messages = []
        
        # 添加图像标记（如果有图像）
        if has_image:
            text_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text_prompt
        
        # 添加 [SEG] 标记
        text_prompt = text_prompt + " [SEG]."
        
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        
        # 创建 attention mask
        attention_masks = torch.ones_like(input_ids)
        
        return input_ids, attention_masks
    
    @torch.no_grad()
    def inference_image(self, image: Image.Image, text_prompt: str):
        """
        图像分割推理
        
        Args:
            image: PIL Image
            text_prompt: 文本提示
            
        Returns:
            result_image: 带有分割掩码的结果图像
            mask: 预测的掩码
        """
        # 预处理图像
        images, images_clip, original_size, resize_size = self.preprocess_image(image)
        
        # 准备文本输入
        input_ids, attention_masks = self.prepare_text_input(
            text_prompt, has_image=True, has_point_cloud=False
        )
        
        # 推理
        output_dict = self.model(
            images=images,
            images_clip=images_clip,
            input_ids=input_ids,
            attention_masks=attention_masks,
            original_size_list=[original_size],
            resize_list=[resize_size],
            inference=True,
        )
        
        # 获取预测掩码
        if output_dict["pred_masks"] is not None and len(output_dict["pred_masks"]) > 0:
            pred_mask = output_dict["pred_masks"][0]  # [num_masks, H, W]
            
            # 如果有多个掩码，取第一个
            if pred_mask.dim() == 3:
                pred_mask = pred_mask[0]  # [H, W]
            
            # 应用 sigmoid（如果还没有）
            if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
                pred_mask = pred_mask.sigmoid()
            
            pred_mask = pred_mask.cpu().numpy()
            
            # 二值化
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
            
            # 可视化
            result_image = self.visualize_2d_mask(image, pred_mask_binary)
            
            return result_image, pred_mask
        else:
            return image, None
    
    @torch.no_grad()
    def inference_point_cloud(self, point_cloud_file: str, text_prompt: str):
        """
        点云分割推理
        
        Args:
            point_cloud_file: 点云文件路径
            text_prompt: 文本提示
            
        Returns:
            result_image: 可视化的点云图像
            mask: 预测的掩码
        """
        # 预处理点云
        point_clouds, original_points = self.preprocess_point_cloud(point_cloud_file)
        
        # 准备文本输入
        input_ids, attention_masks = self.prepare_text_input(
            text_prompt, has_image=False, has_point_cloud=True
        )
        
        # 推理
        output_dict = self.model(
            point_clouds=point_clouds,
            input_ids=input_ids,
            attention_masks=attention_masks,
            inference=True,
        )
        
        # 获取预测掩码
        if output_dict["pred_3d_masks"] is not None and len(output_dict["pred_3d_masks"]) > 0:
            pred_mask = output_dict["pred_3d_masks"][0]  # [N]
            
            # 应用 sigmoid（如果还没有）
            if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
                pred_mask = pred_mask.sigmoid()
            
            pred_mask = pred_mask.cpu().numpy()
            
            # 二值化
            pred_mask_binary = (pred_mask > 0.5).astype(np.float32)
            
            # 可视化
            result_image = self.visualize_3d_mask(
                point_clouds[0].cpu().numpy().T,  # [N, 3]
                pred_mask_binary
            )
            
            return result_image, pred_mask
        else:
            # 返回原始点云可视化
            result_image = self.visualize_3d_mask(
                point_clouds[0].cpu().numpy().T,
                np.zeros(point_clouds.shape[2])
            )
            return result_image, None
    
    def visualize_2d_mask(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.5):
        """
        可视化 2D 分割掩码
        
        Args:
            image: 原始图像
            mask: 二值掩码 [H, W]
            alpha: 透明度
            
        Returns:
            result_image: PIL Image
        """
        # 调整掩码大小以匹配图像
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            image.size, Image.NEAREST
        )
        mask_resized = np.array(mask_resized) / 255.0
        
        # 创建彩色掩码（红色）
        image_np = np.array(image).astype(np.float32) / 255.0
        color_mask = np.zeros_like(image_np)
        color_mask[:, :, 0] = mask_resized  # 红色通道
        
        # 混合
        result = image_np * (1 - alpha * mask_resized[:, :, None]) + color_mask * alpha
        result = (result * 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def visualize_3d_mask(self, points: np.ndarray, mask: np.ndarray):
        """
        可视化 3D 点云掩码
        
        Args:
            points: 点云 [N, 3]
            mask: 掩码 [N]
            
        Returns:
            result_image: PIL Image
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 根据掩码着色
        colors = np.zeros((len(points), 3))
        colors[mask > 0.5] = [1, 0, 0]  # 红色表示前景
        colors[mask <= 0.5] = [0.5, 0.5, 0.5]  # 灰色表示背景
        
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, s=1, alpha=0.8
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud Segmentation')
        
        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        
        return result_image


def create_gradio_interface(engine: LISAInferenceEngine):
    """创建 Gradio 界面"""
    
    def process_image(image, text_prompt):
        """处理图像输入"""
        if image is None:
            return None, "❌ 请上传图像"
        if not text_prompt or text_prompt.strip() == "":
            return None, "❌ 请输入文本提示"
        
        try:
            result_image, mask = engine.inference_image(image, text_prompt)
            if mask is not None:
                return result_image, f"✅ 分割完成！掩码覆盖率: {(mask > 0.5).mean():.2%}"
            else:
                return result_image, "⚠️ 未检测到目标"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"
    
    def process_point_cloud(point_cloud_file, text_prompt):
        """处理点云输入"""
        if point_cloud_file is None:
            return None, "❌ 请上传点云文件"
        if not text_prompt or text_prompt.strip() == "":
            return None, "❌ 请输入文本提示"
        
        try:
            result_image, mask = engine.inference_point_cloud(point_cloud_file, text_prompt)
            if mask is not None:
                return result_image, f"✅ 分割完成！掩码覆盖率: {(mask > 0.5).mean():.2%}"
            else:
                return result_image, "⚠️ 未检测到目标"
        except Exception as e:
            return None, f"❌ 错误: {str(e)}"
    
    # 创建界面
    with gr.Blocks(
        title="LISA 2D-3D Joint Affordance Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # 标题
        gr.HTML("""
        <div class="header">
            <h1>🎨 LISA 2D-3D Joint Affordance Demo</h1>
            <p>语言引导的图像和点云分割系统</p>
        </div>
        """)
        
        # 说明
        gr.Markdown("""
        ## 📖 使用说明
        
        1. **图像分割**：上传图像，输入描述目标的文本（如 "the red cup"），点击"分割图像"
        2. **点云分割**：上传点云文件（.ply, .npy, .txt），输入文本描述，点击"分割点云"
        3. 文本提示应该清晰描述要分割的目标物体
        
        ### 示例提示：
        - "the handle of the mug"（杯子的把手）
        - "the seat of the chair"（椅子的座位）
        - "the door of the cabinet"（柜子的门）
        """)
        
        # 创建两个标签页
        with gr.Tabs():
            # 图像分割标签页
            with gr.TabItem("🖼️ 图像分割"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="上传图像",
                            height=400
                        )
                        image_text = gr.Textbox(
                            label="文本提示",
                            placeholder="例如: the handle of the mug",
                            lines=2
                        )
                        image_button = gr.Button("🚀 分割图像", variant="primary", size="lg")
                    
                    with gr.Column():
                        image_output = gr.Image(
                            label="分割结果",
                            height=400
                        )
                        image_status = gr.Textbox(
                            label="状态",
                            lines=2
                        )
                
                # 示例
                gr.Examples(
                    examples=[
                        ["examples/mug.jpg", "the handle of the mug"],
                        ["examples/chair.jpg", "the seat of the chair"],
                    ],
                    inputs=[image_input, image_text],
                    label="示例"
                )
            
            # 点云分割标签页
            with gr.TabItem("☁️ 点云分割"):
                with gr.Row():
                    with gr.Column():
                        pc_input = gr.File(
                            label="上传点云文件 (.ply, .npy, .txt)",
                            file_types=[".ply", ".npy", ".txt", ".pcd"]
                        )
                        pc_text = gr.Textbox(
                            label="文本提示",
                            placeholder="例如: the handle of the mug",
                            lines=2
                        )
                        pc_button = gr.Button("🚀 分割点云", variant="primary", size="lg")
                    
                    with gr.Column():
                        pc_output = gr.Image(
                            label="分割结果（3D可视化）",
                            height=400
                        )
                        pc_status = gr.Textbox(
                            label="状态",
                            lines=2
                        )
                
                # 示例
                gr.Examples(
                    examples=[
                        ["examples/mug.ply", "the handle of the mug"],
                        ["examples/chair.ply", "the seat of the chair"],
                    ],
                    inputs=[pc_input, pc_text],
                    label="示例"
                )
        
        # 绑定事件
        image_button.click(
            fn=process_image,
            inputs=[image_input, image_text],
            outputs=[image_output, image_status]
        )
        
        pc_button.click(
            fn=process_point_cloud,
            inputs=[pc_input, pc_text],
            outputs=[pc_output, pc_status]
        )
        
        # 页脚
        gr.Markdown("""
        ---
        ### 💡 提示
        - 确保文本描述清晰且具体
        - 图像分辨率建议在 512x512 到 2048x2048 之间
        - 点云文件大小建议不超过 10MB
        - 首次推理可能需要较长时间进行模型预热
        
        ### 🔧 技术栈
        - **模型**: LISA (Language Instructed Segmentation Assistant)
        - **2D 分割**: SAM (Segment Anything Model)
        - **3D 分割**: PointNet++
        - **语言模型**: LLaVA + LLaMA
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="LISA 2D-3D Joint Affordance Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重路径"
    )
    parser.add_argument(
        "--vision_pretrained",
        type=str,
        default="PATH_TO_SAM_ViT-H",
        help="SAM 预训练权重路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备 (cuda:0, cpu)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="精度"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="图像尺寸"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="点云点数"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共链接"
    )
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    engine = LISAInferenceEngine(
        model_path=args.model_path,
        vision_pretrained=args.vision_pretrained,
        device=args.device,
        precision=args.precision,
        image_size=args.image_size,
        num_points=args.num_points,
    )
    
    # 创建界面
    demo = create_gradio_interface(engine)
    
    # 启动服务器
    print(f"\n🌐 启动 Gradio 服务器...")
    print(f"📍 本地地址: http://localhost:{args.port}")
    if args.share:
        print(f"🌍 公共地址将在启动后显示")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
