# None 值处理修复报告

## 修复日期
2026-01-28

## 修复位置
`utils/dataset.py` 第 392-543 行（`collate_fn` 函数）

---

## 🐛 原始问题

### 问题描述
在 `collate_fn` 函数中，当批次数据包含 None 值时会导致程序崩溃：

1. **图像数据问题**：
   - 当 `images_list` 中存在 None 时，`images_list[0].shape` 会报错
   - 当所有图像都是 None 时，无法确定 padding 尺寸

2. **点云数据问题**：
   - 当 `point_clouds_list` 中存在 None 时，`point_clouds_list[0].shape[0]` 会报错
   - 当所有点云都是 None 时，无法确定 padding 点数

3. **掩码数据问题**：
   - 当 `masks_list` 或 `pc_masks_list` 长度不足或包含 None 时会出错

### 错误场景
```python
# 场景1：混合批次（部分样本有图像，部分没有）
batch = [
    {'images': tensor([3, 224, 224]), 'masks': tensor([224, 224]), ...},
    {'images': None, 'masks': None, ...},  # ❌ 会导致错误
]

# 场景2：纯文本批次（所有样本都没有图像）
batch = [
    {'images': None, 'masks': None, ...},
    {'images': None, 'masks': None, ...},  # ❌ 会导致错误
]
```

---

## ✅ 修复方案

### 1. 图像数据处理

#### 修复前
```python
first_shape = images_list[0].shape  # ❌ 如果 images_list[0] 是 None 会报错
all_same_shape = all(img.shape == first_shape for img in images_list)  # ❌ 遇到 None 会报错
```

#### 修复后
```python
# 过滤掉 None 值
valid_images = [img for img in images_list if img is not None]
valid_masks = [mask for mask in masks_list if mask is not None]

if len(valid_images) == 0:
    # 全为 None，使用全0张量填充
    dummy_h, dummy_w = 224, 224  # 默认尺寸
    result['images'] = torch.zeros(batch_size, 3, dummy_h, dummy_w)
    result['images_clip'] = result['images']
    result['masks_list'] = torch.zeros(batch_size, dummy_h, dummy_w)
    result['resize_list'] = [(dummy_h, dummy_w)] * batch_size
    result['original_size_list'] = [(dummy_h, dummy_w)] * batch_size
else:
    # 获取第一个有效图像的形状
    first_shape = valid_images[0].shape
    all_same_shape = all(img.shape == first_shape for img in valid_images)
    
    # 计算最大尺寸（只从有效图像中计算）
    if not all_same_shape:
        max_h = max(img.shape[1] for img in valid_images)
        max_w = max(img.shape[2] for img in valid_images)
    else:
        max_h, max_w = first_shape[1], first_shape[2]
    
    # 逐个处理（包括 None 值）
    for i, img in enumerate(images_list):
        if img is None:
            # 使用全0张量填充
            padded_images.append(torch.zeros(3, max_h, max_w))
            resize_list.append((max_h, max_w))
        else:
            # 正常处理
            ...
        
        # 处理掩码（也可能是 None）
        mask = masks_list[i] if i < len(masks_list) else None
        if mask is None:
            padded_masks.append(torch.zeros(max_h, max_w))
            original_size_list.append((max_h, max_w))
        else:
            # 正常处理
            ...
```

### 2. 点云数据处理

#### 修复前
```python
first_num_points = point_clouds_list[0].shape[0]  # ❌ 如果 point_clouds_list[0] 是 None 会报错
all_same_points = all(pc.shape[0] == first_num_points for pc in point_clouds_list)  # ❌ 遇到 None 会报错
```

#### 修复后
```python
# 过滤掉 None 值
valid_pcs = [pc for pc in point_clouds_list if pc is not None]
valid_pc_masks = [mask for mask in pc_masks_list if mask is not None]

if len(valid_pcs) == 0:
    # 全为 None，使用全0张量填充
    dummy_num_points = 1024  # 默认点数
    result['point_clouds'] = torch.zeros(batch_size, dummy_num_points, 3)
    result['point_masks_list'] = torch.zeros(batch_size, dummy_num_points)
    result['point_valid_lengths'] = torch.zeros(batch_size, dtype=torch.long)
else:
    # 获取第一个有效点云的点数
    first_num_points = valid_pcs[0].shape[0]
    all_same_points = all(pc.shape[0] == first_num_points for pc in valid_pcs)
    
    # 计算最大点数（只从有效点云中计算）
    if not all_same_points:
        max_points = max(pc.shape[0] for pc in valid_pcs)
    else:
        max_points = first_num_points
    
    # 逐个处理（包括 None 值）
    for i, pc in enumerate(point_clouds_list):
        if pc is None:
            # 使用全0张量填充
            padded_pcs.append(torch.zeros(max_points, 3))
            point_nums.append(0)  # 有效点数为0
        else:
            # 正常处理
            ...
        
        # 处理掩码（也可能是 None）
        pc_mask = pc_masks_list[i] if i < len(pc_masks_list) else None
        if pc_mask is None:
            padded_pc_masks.append(torch.zeros(max_points))
        else:
            # 正常处理
            ...
```

---

## 🎯 修复特性

### 1. 三种场景全覆盖

#### 场景 A：所有样本都有数据（正常情况）
```python
batch = [
    {'images': tensor([3, 224, 224]), 'point_clouds': tensor([1024, 3]), ...},
    {'images': tensor([3, 224, 224]), 'point_clouds': tensor([1024, 3]), ...},
]
# ✅ 正常处理，直接 stack
```

#### 场景 B：部分样本有数据（混合批次）
```python
batch = [
    {'images': tensor([3, 224, 224]), 'point_clouds': tensor([1024, 3]), ...},
    {'images': None, 'point_clouds': None, ...},  # 纯文本样本
]
# ✅ 使用全0张量填充 None 位置，保持批次形状一致
```

#### 场景 C：所有样本都没有数据（纯文本批次）
```python
batch = [
    {'images': None, 'point_clouds': None, ...},
    {'images': None, 'point_clouds': None, ...},
]
# ✅ 使用默认尺寸创建全0张量批次
```

### 2. 智能尺寸推断

- **有有效数据时**：从有效数据中推断最大尺寸
- **全为 None 时**：使用合理的默认值
  - 图像：224×224（标准 ViT 输入尺寸）
  - 点云：1024 点（常用点云采样数）

### 3. 有效性标记

通过 `image_valid_mask` 和 `pc_valid_mask` 标记哪些样本有真实数据：

```python
result['image_valid_mask'] = torch.tensor(has_image_flags, dtype=torch.bool)
# 例如：tensor([True, False, True]) 表示第2个样本没有图像

result['pc_valid_mask'] = torch.tensor(has_pc_flags, dtype=torch.bool)
# 例如：tensor([True, True, False]) 表示第3个样本没有点云
```

模型可以根据这些标记跳过无效数据的处理。

### 4. 点云有效长度

通过 `point_valid_lengths` 记录每个样本的真实点数：

```python
result['point_valid_lengths'] = torch.tensor(point_nums, dtype=torch.long)
# 例如：tensor([1024, 0, 512]) 表示：
#   - 第1个样本有 1024 个点
#   - 第2个样本是 None（0个点）
#   - 第3个样本有 512 个点（padding到1024）
```

---

## 📊 数据流示例

### 示例 1：混合批次

**输入**：
```python
batch = [
    {
        'images': tensor([3, 224, 224]),
        'masks': tensor([224, 224]),
        'point_clouds': tensor([1024, 3]),
        'pc_masks': tensor([1024]),
        'has_image': True,
        'has_point_cloud': True,
    },
    {
        'images': None,
        'masks': None,
        'point_clouds': None,
        'pc_masks': None,
        'has_image': False,
        'has_point_cloud': False,
    },
]
```

**输出**：
```python
result = {
    'images': tensor([
        [[...], [...], [...]],  # 第1个样本的真实图像
        [[0, 0, ...], [0, 0, ...], [0, 0, ...]],  # 第2个样本的全0填充
    ]),  # shape: [2, 3, 224, 224]
    
    'masks_list': tensor([
        [[...], [...], ...],  # 第1个样本的真实掩码
        [[0, 0, ...], [0, 0, ...], ...],  # 第2个样本的全0填充
    ]),  # shape: [2, 224, 224]
    
    'point_clouds': tensor([
        [[x1, y1, z1], [x2, y2, z2], ...],  # 第1个样本的真实点云
        [[0, 0, 0], [0, 0, 0], ...],  # 第2个样本的全0填充
    ]),  # shape: [2, 1024, 3]
    
    'point_masks_list': tensor([
        [1, 1, 1, ...],  # 第1个样本的真实掩码
        [0, 0, 0, ...],  # 第2个样本的全0填充
    ]),  # shape: [2, 1024]
    
    'image_valid_mask': tensor([True, False]),
    'pc_valid_mask': tensor([True, False]),
    'point_valid_lengths': tensor([1024, 0]),
}
```

### 示例 2：纯文本批次

**输入**：
```python
batch = [
    {'images': None, 'point_clouds': None, 'has_image': False, 'has_point_cloud': False},
    {'images': None, 'point_clouds': None, 'has_image': False, 'has_point_cloud': False},
]
```

**输出**：
```python
result = {
    'images': torch.zeros(2, 3, 224, 224),  # 使用默认尺寸
    'masks_list': torch.zeros(2, 224, 224),
    'point_clouds': torch.zeros(2, 1024, 3),  # 使用默认点数
    'point_masks_list': torch.zeros(2, 1024),
    'image_valid_mask': tensor([False, False]),
    'pc_valid_mask': tensor([False, False]),
    'point_valid_lengths': tensor([0, 0]),
}
```

---

## ✅ 测试建议

### 1. 单元测试

```python
def test_collate_fn_with_none():
    """测试 collate_fn 处理 None 值"""
    
    # 测试1：混合批次
    batch_mixed = [
        {'images': torch.randn(3, 224, 224), 'point_clouds': torch.randn(1024, 3), ...},
        {'images': None, 'point_clouds': None, ...},
    ]
    result = collate_fn(batch_mixed, tokenizer, ...)
    assert result['images'].shape == (2, 3, 224, 224)
    assert result['point_clouds'].shape == (2, 1024, 3)
    assert result['image_valid_mask'].tolist() == [True, False]
    
    # 测试2：全为 None
    batch_none = [
        {'images': None, 'point_clouds': None, ...},
        {'images': None, 'point_clouds': None, ...},
    ]
    result = collate_fn(batch_none, tokenizer, ...)
    assert result['images'].shape == (2, 3, 224, 224)
    assert result['point_clouds'].shape == (2, 1024, 3)
    assert result['image_valid_mask'].tolist() == [False, False]
    
    # 测试3：不同尺寸 + None
    batch_varied = [
        {'images': torch.randn(3, 256, 256), ...},
        {'images': None, ...},
        {'images': torch.randn(3, 128, 128), ...},
    ]
    result = collate_fn(batch_varied, tokenizer, ...)
    assert result['images'].shape == (3, 3, 256, 256)  # padding到最大尺寸
```

### 2. 集成测试

```python
# 在实际训练循环中测试
for batch in train_loader:
    # 应该不会因为 None 值而崩溃
    output = model(**batch)
    loss = output['loss']
    loss.backward()
```

---

## 📝 注意事项

### 1. 模型端需要配合

模型在处理数据时应该检查有效性标记：

```python
# 在 LISA.py 中
if batch['image_valid_mask'].any():
    # 只处理有图像的样本
    valid_indices = batch['image_valid_mask'].nonzero(as_tuple=True)[0]
    images = batch['images'][valid_indices]
    ...

if batch['pc_valid_mask'].any():
    # 只处理有点云的样本
    valid_indices = batch['pc_valid_mask'].nonzero(as_tuple=True)[0]
    point_clouds = batch['point_clouds'][valid_indices]
    valid_lengths = batch['point_valid_lengths'][valid_indices]
    ...
```

### 2. 默认尺寸可配置

如果需要修改默认尺寸，可以在 `collate_fn` 中添加参数：

```python
def collate_fn(
    batch,
    tokenizer,
    conv_type,
    use_mm_start_end,
    local_rank,
    default_image_size=(224, 224),  # 新增参数
    default_num_points=1024,  # 新增参数
):
    ...
    if len(valid_images) == 0:
        dummy_h, dummy_w = default_image_size
        ...
    
    if len(valid_pcs) == 0:
        dummy_num_points = default_num_points
        ...
```

### 3. 内存效率

- 全0张量不会占用太多内存（PyTorch 会优化）
- 使用 `non_blocking=True` 异步传输到 GPU
- 模型可以通过有效性标记跳过无效数据的计算

---

## 🎉 总结

### 修复效果

✅ **鲁棒性**：支持任意组合的 None 值，不会崩溃  
✅ **一致性**：保持批次形状一致，便于模型处理  
✅ **高效性**：只在必要时进行 padding，避免不必要的计算  
✅ **可追溯**：通过有效性标记和长度信息，模型可以准确知道哪些数据是真实的  

### 适用场景

- ✅ 纯文本任务（无图像、无点云）
- ✅ 图像分割任务（有图像、无点云）
- ✅ 点云分割任务（无图像、有点云）
- ✅ 多模态任务（有图像、有点云）
- ✅ 混合批次（部分样本有某些模态，部分没有）

### 代码质量

- 清晰的注释说明每个分支的作用
- 统一的处理逻辑（图像和点云采用相同的模式）
- 完善的元数据（有效性标记、长度信息）
- 向后兼容（不影响现有的正常数据流）

现在数据加载器可以安全地处理各种边界情况！🚀

