import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import TrainingConfig
from utils.trainability_summary import build_tree_from_model, serialize_tree, summarize_optimizer_groups_data


def apply_trainability_policy(model, training_cfg: TrainingConfig) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if training_cfg.lora.lora_r > 0:
        try:
            from peft import get_peft_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "当前环境未安装 peft，无法按 LoRA 配置检查可训练层。"
            ) from exc
        model.mllm.model = get_peft_model(model.mllm.model, training_cfg.lora.to_peft_config())

    for name, param in model.named_parameters():
        if any(key in name for key in training_cfg.name_of_params_to_train):
            param.requires_grad = True

def load_training_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config_json:
        cfg = TrainingConfig.from_json(args.config_json)
    else:
        cfg = TrainingConfig()

    if args.qwen_model:
        cfg.model_config.mllm.qwen_model_name_or_path = args.qwen_model
    if args.vision_pretrained:
        cfg.model_config.image_decoder.sam_pretrained_path = args.vision_pretrained
    if args.point_backbone_pretrained:
        cfg.model_config.mllm.point_encoder_pretrained = args.point_backbone_pretrained
    if args.point_backbone_pretrained_config:
        cfg.model_config.mllm.point_encoder_pretrained_config = args.point_backbone_pretrained_config
    if args.lora_r is not None:
        cfg.lora.lora_r = int(args.lora_r)
    if args.lora_target_modules:
        cfg.lora.lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if args.name_of_params_to_train:
        cfg.name_of_params_to_train = [x.strip() for x in args.name_of_params_to_train.split(",") if x.strip()]
    return cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看当前配置下模型哪些层被冻结/可训练，并导出为可读 JSON。"
    )
    parser.add_argument("--config_json", type=str, default=None, help="训练配置 JSON 路径；不传则使用默认 TrainingConfig")
    parser.add_argument("--qwen_model", type=str, default=None, help="覆写 qwen_model_name_or_path")
    parser.add_argument("--vision_pretrained", type=str, default=None, help="覆写视觉预训练路径")
    parser.add_argument("--point_backbone_pretrained", type=str, default=None, help="覆写点云 backbone 预训练权重")
    parser.add_argument("--point_backbone_pretrained_config", type=str, default=None, help="覆写点云 backbone 配置路径")
    parser.add_argument("--lora_r", type=int, default=None, help="覆写 LoRA rank")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="覆写 LoRA target_modules，逗号分隔")
    parser.add_argument("--name_of_params_to_train", type=str, default=None, help="覆写白名单，逗号分隔")
    parser.add_argument(
        "--state",
        type=str,
        default="all",
        choices=["all", "trainable", "frozen"],
        help="只看全部、只看可训练、或只看冻结层",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选，将 JSON 结果写入文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_cfg = load_training_config(args)
    from model.joint_affordance import JointAffordanceModel

    model = JointAffordanceModel(training_cfg.model_config)
    apply_trainability_policy(model, training_cfg)
    tree = build_tree_from_model(model)
    payload = {
        "config": {
            "config_source": os.path.abspath(args.config_json) if args.config_json else "TrainingConfig() defaults",
            "qwen_model": training_cfg.model_config.mllm.qwen_model_name_or_path,
            "lora_enabled": training_cfg.lora.lora_r > 0,
            "lora_r": training_cfg.lora.lora_r,
            "lora_target_modules": training_cfg.lora.lora_target_modules,
            "trainable_filters": training_cfg.name_of_params_to_train,
            "requested_state": args.state,
        },
        "optimizer_groups": summarize_optimizer_groups_data(model),
    }
    if args.state == "all":
        payload["module_trees"] = {
            "all": serialize_tree(tree, state="all"),
            "trainable": serialize_tree(tree, state="trainable"),
            "frozen": serialize_tree(tree, state="frozen"),
        }
    else:
        payload["module_tree"] = serialize_tree(tree, state=args.state)

    content = json.dumps(payload, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")
        print(f"\n[OK] Saved JSON report to: {output_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
