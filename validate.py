"""
验证脚本：加载训练好的模型并进行推理评估

使用方法：
    python validate.py --checkpoint_path <checkpoint路径> [其他参数]
"""
import os
import argparse
from functools import partial

import torch
from tqdm import tqdm
import transformers
from peft import LoraConfig, get_peft_model

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import DatasetManager, collate_fn
from utils.common import (
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_IM_START_TOKEN,
    dict_to_cuda
)
from utils.metrics import (
    MetricsTracker,
    evaluate_segmentation_batch,
    print_validation_summary,
)
from configs import TrainingConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证训练好的模型')
    
    # 必需参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='训练好的模型checkpoint路径（.pth文件）')
    
    # 可选参数
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='数据集目录（默认使用config中的设置）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='验证批次大小（默认：1）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='要评估的数据集分割（默认：test）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备（默认：cuda:0）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的worker数量（默认：4）')
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存预测结果')
    parser.add_argument('--output_dir', type=str, default='./validation_output',
                        help='预测结果保存目录（默认：./validation_output）')
    
    return parser.parse_args()


def load_model(checkpoint_path, config, device):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        config: 训练配置
        device: 设备
        
    Returns:
        model: 加载好的模型
        tokenizer: 分词器
    """
    print(f"正在加载模型从: {checkpoint_path}")
    
    # 创建tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.version,
        cache_dir=None,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 添加特殊标记
    tokenizer.add_tokens("[SEG]")  # 2D分割标记
    tokenizer.add_tokens("[AFF]")  # 3D affordance标记
    config.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    config.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    if config.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # 模型参数配置
    model_args = {
        "train_mask_decoder": config.train_mask_decoder,
        "out_dim": config.out_dim,
        "ce_loss_weight": config.ce_loss_weight,
        "dice_loss_weight": config.dice_loss_weight,
        "bce_loss_weight": config.bce_loss_weight,
        "seg_token_idx": config.seg_token_idx,
        "aff_token_idx": config.aff_token_idx,
        "vision_pretrained": config.vision_pretrained,
        "vision_tower": config.vision_tower,
        "use_mm_start_end": config.use_mm_start_end,
    }
    
    # 初始化模型
    print("正在初始化模型...")
    model = LISAForCausalLM.from_pretrained(
        config.version, dtype=config.precision, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 初始化视觉模块
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=config.precision, device=device)
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        config.conv_type
    ]

    model.resize_token_embeddings(len(tokenizer))
    
    # 初始化 LISA 模块（包括 SAM 和 3D 点云分割器）
    model.get_model().initialize_lisa_modules(model.get_model().config)

    # 先把所有参数冻结
    for p in model.parameters():
        p.requires_grad = False

    # LoRA 配置（如果使用）
    if config.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all([x not in name for x in config.name_of_params_to_train])
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(model, config.lora_target_modules)
        lora_config = config.get_lora_config()
        lora_config.target_modules = lora_target_modules
        model = get_peft_model(model, lora_config)
        print("已应用 LoRA 配置")

    # 加载checkpoint
    print(f"正在加载checkpoint权重...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同格式的checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        best_giou = checkpoint.get('best_giou', 'unknown')
        best_ciou = checkpoint.get('best_ciou', 'unknown')
        print(f"Checkpoint信息: Epoch={epoch}, Best gIoU={best_giou}, Best cIoU={best_ciou}")
    else:
        state_dict = checkpoint
    
    # 加载权重（只加载存在的参数）
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print(f"警告: checkpoint中的参数 {k} 在模型中不存在，跳过")
    
    # 检查是否有模型参数未被加载
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
    if missing_keys:
        print(f"\n警告: 以下模型参数未从checkpoint加载（将使用初始化值）:")
        for key in sorted(missing_keys):
            print(f"  - {key}")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    
    # 移动模型到指定设备
    model = model.to(device)
    model.eval()
    
    print("模型加载完成！\n")
    
    return model, tokenizer


def validate(model, val_loader, device, config, save_predictions=False, output_dir=None):
    """
    验证函数
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        config: 训练配置
        save_predictions: 是否保存预测结果
        output_dir: 预测结果保存目录
        
    Returns:
        giou: 2D Global IoU
        ciou: 2D Class IoU
        mae_3d: 3D MAE
        auc_3d: 3D AUC
        aiou_3d: 3D Average IoU
        sim_3d: 3D Similarity
    """
    # 使用 MetricsTracker 管理所有评估指标
    metrics_tracker = MetricsTracker()
    
    model.eval()
    
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"预测结果将保存到: {output_dir}")

    print("开始验证...")
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(tqdm(val_loader, desc='验证中')):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            output_dict = model(**input_dict, inference=True)

            # 使用统一的评估函数（支持批处理）
            evaluate_segmentation_batch(
                input_dict,
                output_dict,
                metrics_tracker,
                mask_threshold_2d=config.mask_threshold_2d
            )
            
            # 保存预测结果（可选）
            if save_predictions and output_dir:
                save_batch_predictions(
                    input_dict, 
                    output_dict, 
                    batch_idx, 
                    output_dir
                )

    # 计算最终结果
    giou, ciou = metrics_tracker.compute_2d_seg_results()
    mae_3d, auc_3d, aiou_3d, sim_3d = metrics_tracker.compute_3d_seg_results()

    # 打印验证摘要
    print("\n" + "="*80)
    print("验证结果摘要")
    print("="*80)
    print_validation_summary(
        epoch=0,  # 验证时不需要epoch信息
        metrics_tracker=metrics_tracker,
        giou=giou,
        ciou=ciou,
        mae_3d=mae_3d,
        auc_3d=auc_3d,
        aiou_3d=aiou_3d,
        sim_3d=sim_3d
    )
    print("="*80 + "\n")

    return giou, ciou, mae_3d, auc_3d, aiou_3d, sim_3d


def save_batch_predictions(input_dict, output_dict, batch_idx, output_dir):
    """
    保存批次预测结果
    
    Args:
        input_dict: 输入字典
        output_dict: 输出字典
        batch_idx: 批次索引
        output_dir: 输出目录
    """
    batch_size = len(input_dict.get('obj_type', []))
    
    for i in range(batch_size):
        sample_id = batch_idx * batch_size + i
        sample_dir = os.path.join(output_dir, f"sample_{sample_id:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存2D预测掩码
        if output_dict.get('pred_masks') is not None and i < len(output_dict['pred_masks']):
            pred_mask_2d = output_dict['pred_masks'][i].cpu().numpy()
            torch.save(pred_mask_2d, os.path.join(sample_dir, 'pred_mask_2d.pt'))
        
        # 保存3D预测掩码
        if output_dict.get('pred_masks_3d') is not None and i < len(output_dict['pred_masks_3d']):
            pred_mask_3d = output_dict['pred_masks_3d'][i].cpu().numpy()
            torch.save(pred_mask_3d, os.path.join(sample_dir, 'pred_mask_3d.pt'))
        
        # 保存元信息
        meta_info = {
            'obj_type': input_dict['obj_type'][i] if 'obj_type' in input_dict else None,
            'aff_type': input_dict['aff_type'][i] if 'aff_type' in input_dict else None,
        }
        torch.save(meta_info, os.path.join(sample_dir, 'meta_info.pt'))


def main():
    args = parse_args()
    
    # 加载配置
    config = TrainingConfig()
    
    # 覆盖配置（如果提供了命令行参数）
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    
    config.val_batch_size = args.batch_size
    config.workers = args.num_workers
    config.distributed = False  # 验证时不使用分布式
    config.local_rank = 0
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model, tokenizer = load_model(args.checkpoint_path, config, device)
    
    # 创建数据集
    print("正在加载数据集...")
    dataset_manager = DatasetManager(
        dataset_dir=config.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=config.vision_tower,
        precision=config.precision,
        image_size=config.image_size,
        num_points=config.num_points,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        use_mm_start_end=config.use_mm_start_end,
        conv_type=config.conv_type,
        use_sample_cache=config.use_sample_cache
    )
    
    # 选择数据集分割
    if args.split == 'train':
        eval_dataset = dataset_manager.get_train_dataset()
    elif args.split == 'val':
        eval_dataset = dataset_manager.get_val_dataset()
    else:  # test
        eval_dataset = dataset_manager.get_test_dataset()
    
    print(f"评估数据集: {args.split}, 样本数量: {len(eval_dataset)}\n")
    
    # 创建数据加载器
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            output_image_size=config.image_size,
            output_point_nums=config.num_points,
            precision=config.precision,
        ),
    )
    
    # 执行验证
    giou, ciou, mae_3d, auc_3d, aiou_3d, sim_3d = validate(
        model=model,
        val_loader=eval_loader,
        device=device,
        config=config,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # 保存评估结果到文件
    results = {
        'checkpoint_path': args.checkpoint_path,
        'dataset_split': args.split,
        'num_samples': len(eval_dataset),
        '2d_metrics': {
            'giou': float(giou),
            'ciou': float(ciou),
        },
        '3d_metrics': {
            'mae': float(mae_3d),
            'auc': float(auc_3d),
            'aiou': float(aiou_3d),
            'sim': float(sim_3d),
        }
    }
    
    results_file = os.path.join(args.output_dir if args.save_predictions else '.', 'validation_results.pt')
    torch.save(results, results_file)
    print(f"\n评估结果已保存到: {results_file}")


if __name__ == "__main__":
    main()

