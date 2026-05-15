# 进阶参考

本文档保留复现与二次开发需要的细节：配置约定、训练样本结构、模型输入输出、路由机制、调优建议与指标定义。安装与最短运行入口见根目录 [README](../README.md)；数据集格式见 [DATASET.md](DATASET.md)；批量渲染与 IAGNet/Mitsuba 出图见 [rendering.md](rendering.md)。

## 配置约定

所有继承自 `configs.base_config.Configs` 的配置类都以类变量 `defaults` 作为初始化默认值来源。修改训练、模型或推理配置默认值时，优先改对应配置类的 `defaults`，例如 `TrainingConfig.defaults["lr"]`、`MLLMConfigs.defaults["compute_dtype"]` 或 `PointDecoderConfigs.defaults["decode_mode"]`。

配置初始化顺序固定为：先深拷贝并合并 `defaults`，再叠加 `config_dict` 与显式关键字参数，最后执行 dtype 解析、字符串转列表、学习率派生、路径拼接等派生逻辑。因此派生字段也会读取修改后的 `defaults`。

## 数据集与 Split

完整数据集 README 已抽取到 [DATASET.md](DATASET.md)。该文件可以直接随数据集一起发布，包含目录结构、命名规范、`Instruction.csv`、2D mask、3D point-cloud CSV、split JSON、加载方式与检查清单。

本代码侧只依赖两个约定：

- `JointDataset(dataset_root=..., split_file=...)` 能按 split 找到对应模态 ID。
- 取样时用语义 `aff_type` 定位 2D mask 子目录和 3D CSV 中的 affordance 列，而不是依赖固定 mask index。

## 数据处理入口

- `utils/data_process/external_datasets_processing.py`：外部数据集读取、规范化保存，并默认生成 train-only split。
- `utils/data_process/merge_datasets.py`：合并多个数据集，重新编号、排序，并可选生成 split 文件。
- `utils/data_process/create_split.py`：独立生成 split 文件。

## 单样本与 Batch 输出

`JointDataSample.get_data()` 返回原始数据，`utils/dataset.py` 中的 `JointAffordanceTorchDataset._build_sample()` 将其转为模型可用的单样本字典：

```python
{
    "sample_id": int,
    "obj_type": str,
    "aff_type": str,
    "input_ids": Tensor[1, L],
    "labels": Tensor[1, L],
    "pixel_values": Tensor[N, C, H, W],
    "image_grid_thw": Tensor[N, 3],
    "images": Tensor[3, H, W],
    "img_gt": Tensor[H, W],
    "point_clouds": Tensor[N, 3],
    "pc_gt": Tensor[N],
}
```

`joint_affordance_collate_fn()` 会进一步组织为 batch：

```python
{
    "input_ids": Tensor[B, L],
    "labels": Tensor[B, L],
    "attention_mask": Tensor[B, L],
    "pixel_values": Tensor[B, 3, H, W],
    "image_grid_thw": Tensor[B, 3],
    "images": Tensor[B, 3, H, W],
    "img_gt_tensor": Tensor[B, H, W],
    "img_valid_mask": Tensor[B],
    "point_clouds": Tensor[B, N, 3],
    "pc_gt_tensor": Tensor[B, N],
    "pc_valid_lengths": Tensor[B],
}
```

字段会随样本可用模态变化而省略或用有效位标记。

## 模型输入输出

`JointAffordanceModel.forward` 接收与 collate 输出一致的字典，并返回：

```python
{
    "image_logits": Tensor[B, H, W] | None,
    "point_logits": Tensor[B, N] | None,
    "token_ids": Tensor[B, L] | None,
    "ce_loss": Tensor | None,
    "route_logits": Tensor[B, L, 3] | None,
    "route_probs": Tensor[B, L, 3] | None,
    "hidden_states": Tensor | None,
    "output": MLLMOutput | None,
    "aff_token_pairs": List[List[Tuple[str, Tensor]]] | None,
}
```

`image_logits` 与 `point_logits` 都是 logits；计算指标前会先做 `sigmoid()`。

## Router 分支策略

当前模型采用 **text / img / pc 三路软路由**：

1. 每个 token 的 hidden state 经 `route_head` 得到 `route_logits[b, l, 3]`。
2. `softmax` 后得到 text / img / pc 三类路由概率。
3. 所有 token 同时经过图像与点云分支头，再用对应路由概率做加权平均，得到 `img_emb` 与 `pc_emb`。
4. `img_emb` 输入 2D decoder，`pc_emb` 输入 3D decoder。

训练文本中显式保留 `<img_aff>` / `<pc_aff>` 占位，模型通过 `_build_route_mask_from_labels()` 按 next-token 对齐规则构造路由监督位置。语言 CE 会忽略这些占位标签位，主要由 `L_route`、`L_bal` 与下游分割损失学习路由。

## 3D 分支

当前 3D 分支采用“一次点云主干前向，产出两路特征”的设计：

- `PointTransformerV3(return_dual=True)` 返回 `enc_point` 与 `dec_point`。
- `enc_point.feat` 投影为 `mllm_point_tokens`，注入 MLLM。
- `dec_point.feat` 整理为 `per_point_features`，供 3D decoder 逐点计算 affordance 响应。
- `PointCloudHiddenStateDecoder` 将 MLLM 输出的 pc affordance token hidden 投影为 query，并与逐点特征计算相似度，输出 `point_logits`。

`mllm_point_tokens` 是 token 级语义特征，`per_point_features` 是逐点特征，两者职责不同。

## 调优指南

### 分层学习率

默认训练配置启用 `use_layerwise_lr=True`，参数组大致分为：

- `llm_lr`：LLM / LoRA / MLLM 相关参数。默认由 `lr * 0.01` 推导，适合保守微调语言语义。
- `vision_2d_lr`：2D decoder 任务头相关参数，默认 `5e-6`。
- `vision_3d_lr`：3D decoder 任务头相关参数，默认 `5e-4`。
- `lr`：router、投影层、非上述分组的其他新参数，默认 `1e-3`。

推荐从“下游任务头先学好，LLM 小步微调”的原则开始：

- 2D 或 3D mask 明显欠拟合：优先提高对应任务头学习率，而不是直接提高 `llm_lr`。
- 文本语义、路由或跨对象泛化差：小幅提高 `llm_lr` 或扩大 LoRA rank，但要观察语言 CE 与路由统计，避免破坏已有 MLLM 语义。
- 训练 early stage 抖动大：降低 `lr` 和 `vision_3d_lr`，或增加 `warmup_num_steps`。
- 只有 2D 好、3D 差：通常先调 `vision_3d_lr`、点云采样数、`point_decoder.backbone_mode`，不要先动 SAM 或 Qwen。
- 只有 3D 好、2D 差：先检查 `vision_pretrained`、图像输入尺寸、mask 阈值与 `vision_2d_lr`。

### 冻结策略

默认冻结所有编码层：

```text
mllm.model.visual,
point_encoder.point_backbone,
image_decoder.visual_model.image_encoder
```

这意味着训练主要更新 router、投影层、LoRA / MLLM 可训练部分，以及 2D/3D decoder 任务头。建议优先保持这个策略，除非数据量足够大且目标域与预训练域差异很大。

解冻顺序建议：

1. 先调下游 decoder 任务头。
2. 再调 LoRA / MLLM 小学习率。
3. 最后才考虑解冻视觉或点云 backbone，并显著降低对应学习率。

### 2D decoder 任务头

2D 分支使用 routed image affordance query 生成空间相似度热图，再作为 mask prompt 交给 SAM decoder。调优重点：

- 保持 `image_decoder.visual_model.image_encoder` 冻结，先让 query 投影和 SAM mask decoder 适配 affordance 语义。
- `vision_2d_lr` 不宜过大。2D mask 出现大片噪声或边界不稳定时，先降低学习率或增加 warmup。
- 若 2D 预测过于保守，优先检查 GT mask 归一化与 `mask_threshold_2d`，再调损失权重。
- 图像尺寸变化会影响 SAM 特征与 mask 对齐，应与训练配置中的 `image_size` 保持一致。

### 3D decoder 任务头

`PointDecoderConfigs` 里最关键的是：

- `backbone_mode`: `independent` 或 `shared`。
- `decode_mode`: `similarity` 或 `prompt`。
- `hidden_size`、`num_heads`、`grid_size`、`backbone_out_channels`。

当前推荐组合是：

```text
point_decoder.backbone_mode = "independent"
point_decoder.decode_mode = "similarity"
```

原因：

- `independent` 让 3D decoder 有自己的逐点特征编码路径，不直接受注入 MLLM 的点云 token 压缩影响，通常更利于逐点 mask。
- `shared` 更省参数，但更依赖 point encoder 输出的 `per_point_features` 质量；当点云 backbone 已经非常贴近当前数据分布时再考虑。
- `similarity` 解码头直接将 projected affordance query 与逐点特征做归一化相似度，结构简单、稳定、对小数据更友好。
- `prompt` 解码头有更强交互能力，但参数更多、训练更敏感；除非数据量和调参预算充足，否则不作为首选。

若 3D 任务头性能不足，建议按顺序排查：

1. `pc_gt_tensor` 与 `point_clouds` 是否逐点对齐。
2. `pc_valid_lengths` / mask 是否正确，padding 点不应参与损失。
3. `vision_3d_lr` 是否过高导致相似度空间震荡。
4. `num_points` 是否足够覆盖 affordance 区域。
5. `backbone_out_channels` 是否与独立 point decoder 的实际输出一致。

### Similarity 对齐与锚定方式

3D similarity head 的计算逻辑是：

1. routed pc token hidden state 经过 `text_hidden_fcs` 投影到 decoder hidden size。
2. 每个点的 `per_point_features` 经过 `point_proj` 投影到同一空间。
3. 两边做 L2 normalize。
4. 用点特征与 affordance query 做 dot product，并乘以可学习 `logit_scale`。

因此“锚定”不是固定某个点或某个类别原型，而是由文本中的 `<pc_aff>` / `<img_aff>` 监督位置把 MLLM token hidden 锚定到对应下游任务头。要让这个锚定稳定，需要注意：

- 训练文本模板中必须稳定出现 `<img_aff>` / `<pc_aff>`，否则 router 没有明确的 token-level 监督位置。
- `_build_route_mask_from_labels()` 使用 next-token 对齐，修改 tokenizer 或模板时要确认标签位置仍与预测位置对齐。
- 3D 分支不要把 `mllm_point_tokens` 当作逐点特征使用；逐点 logits 应来自与原始点数对齐的 `per_point_features`。
- 相似度空间对尺度敏感度较低，因为两边都 normalize；真正敏感的是 query 是否代表正确 affordance、点特征是否逐点对齐，以及 `logit_scale` 是否被异常学习率推到极端。
- 如果某些 affordance 经常混淆，优先增加 instruction 表述多样性和正负样本覆盖，而不是仅加深 decoder。

## 指标与阈值

评估时：

- `pred_2d = image_logits.sigmoid()`
- `pred_3d = point_logits.sigmoid()`

因此 IoU 类指标中的 threshold 作用于 sigmoid 后概率，不是原始 logits。若想表达 `logit > 0`，应在概率图上使用 `threshold=0.5`。

### 2D 指标

| 指标 | 处理方式 | 含义 |
|---|---|---|
| `gIoU` | 按阈值二值化后逐样本 IoU 平均 | 越高越好 |
| `cIoU` | 全测试集累计交并比 | 越高越好 |
| `P50` / `P50-95` | 统计 IoU 命中率 | 越高越好 |
| `KLD` | `KL(gt || pred)` | 越低越好 |
| `SIM` | 直方图交集 | 越高越好 |
| `NSS` | GT 区域的标准化预测响应 | 越高越好 |

### 3D 指标

| 指标 | 处理方式 | 含义 |
|---|---|---|
| `AUC` | 连续概率作为 score，GT 按 `>=0.5` 二值化 | 越高越好 |
| `aIOU` / `iou_3d` | 默认 `aIOU-20` 多阈值均值 | 越高越好 |
| `IoU` | 单阈值 IoU，主要用于阈值搜索或可视化 | 越高越好 |
| `MAE` | 连续概率与 GT 的平均绝对误差 | 越低越好 |
| `SIM` | GREAT 风格 histogram intersection | 越高越好 |

`P50/P50-95` 中的 `0.50~0.95` 是 IoU 命中阈值，不是预测 mask 的二值化阈值。若 benchmark 未规定阈值，建议在验证集上选择阈值并固定用于测试集。

## Citation

正式发表后在此处添加 BibTeX。
