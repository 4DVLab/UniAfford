"""推理阶段的耗时、FLOPs、显存与参数规模统计工具。"""

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def format_count(value: int) -> str:
    """以便于阅读的单位格式化参数量。"""
    value = int(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} M"
    if value >= 1_000:
        return f"{value / 1_000:.3f} K"
    return str(value)


def format_bytes(value: int) -> str:
    """以二进制单位格式化内存字节数。"""
    value = int(value)
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.3f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.3f} MiB"
    return f"{value / 1024:.3f} KiB"


def _parameter_stats(module: nn.Module) -> Tuple[int, int, int]:
    """返回参数量、可训练参数量和参数存储字节数。"""
    params = list(module.parameters())
    total = sum(int(param.numel()) for param in params)
    trainable = sum(int(param.numel()) for param in params if param.requires_grad)
    storage_bytes = sum(int(param.numel() * param.element_size()) for param in params)
    return total, trainable, storage_bytes


def print_model_parameter_stats(model: nn.Module) -> None:
    """打印整模及主要推理模块的参数规模。"""
    modules = [
        ("MLLM", getattr(getattr(model, "mllm", None), "model", None)),
        ("Router", getattr(model, "router", None)),
        ("2D decoder（含 SAM）", getattr(model, "image_decoder", None)),
        ("3D encoder（共享时已包含于 MLLM wrapper）", getattr(model, "point_encoder", None)),
        ("3D decoder", getattr(model, "point_decoder", None)),
    ]
    total, trainable, storage_bytes = _parameter_stats(model)
    print("\n[Computational Cost] 模型参数规模")
    print(
        f"  {'完整模型':<34} 参数={format_count(total):>10}  "
        f"可训练={format_count(trainable):>10}  参数存储={format_bytes(storage_bytes):>10}"
    )
    seen = set()
    for name, module in modules:
        if not isinstance(module, nn.Module) or id(module) in seen:
            continue
        seen.add(id(module))
        module_total, module_trainable, module_bytes = _parameter_stats(module)
        print(
            f"  {name:<34} 参数={format_count(module_total):>10}  "
            f"可训练={format_count(module_trainable):>10}  参数存储={format_bytes(module_bytes):>10}"
        )
    print("  注：共享模块会同时出现在父模块统计中，各分项不应直接相加。")


def _profiling_targets(model: nn.Module):
    """返回主要计算模块及其日志分组。"""
    targets = []

    def add(group: str, module: Optional[nn.Module]) -> None:
        if isinstance(module, nn.Module):
            targets.append((group, module))

    mllm = getattr(model, "mllm", None)
    add("MLLM", getattr(mllm, "model", mllm))

    router = getattr(model, "router", None)
    for attr in ("route_head", "img_branch_head", "pc_branch_head"):
        add("Router", getattr(router, attr, None))

    image_decoder = getattr(model, "image_decoder", None)
    # get_visual_embs() 直接调用 image_encoder，不经过 image_decoder.forward。
    add("2D decoder", getattr(image_decoder, "image_encoder", None))
    add("2D decoder", image_decoder)

    point_encoder = getattr(model, "point_encoder", None)
    add("3D encoder", getattr(point_encoder, "point_backbone", point_encoder))
    add("3D decoder", getattr(model, "point_decoder", None))

    unique_targets = []
    registered = set()
    for group, module in targets:
        if id(module) in registered:
            continue
        registered.add(id(module))
        unique_targets.append((group, module))
    return unique_targets


class InferenceTimingProfiler:
    """利用 CUDA Event/CPU 高精度时钟统计端到端及主要模块的真实推理耗时。"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.active = False
        self.handles = []
        self.pending = defaultdict(list)
        self.cpu_starts = defaultdict(list)
        self.total_ms = defaultdict(float)
        self.e2e_batch_ms: List[float] = []
        self.total_samples = 0
        self._e2e_start = None
        self._register_targets(model)

    def _register_targets(self, model: nn.Module) -> None:
        for group, module in _profiling_targets(model):
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(group)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(group)))

    def _make_pre_hook(self, group: str):
        def hook(_module, _inputs):
            if not self.active:
                return
            if self.use_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.pending[group].append([event, None])
            else:
                self.cpu_starts[group].append(time.perf_counter())

        return hook

    def _make_post_hook(self, group: str):
        def hook(_module, _inputs, _output):
            if not self.active:
                return
            if self.use_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.pending[group][-1][1] = event
            else:
                start = self.cpu_starts[group].pop()
                self.total_ms[group] += (time.perf_counter() - start) * 1000.0

        return hook

    def start_batch(self, measured: bool) -> None:
        self.active = bool(measured)
        if not self.active:
            return
        if self.use_cuda:
            self._e2e_start = torch.cuda.Event(enable_timing=True)
            self._e2e_start.record()
        else:
            self._e2e_start = time.perf_counter()

    def finish_batch(self, batch_size: int) -> None:
        if not self.active:
            return
        if self.use_cuda:
            e2e_end = torch.cuda.Event(enable_timing=True)
            e2e_end.record()
            torch.cuda.synchronize(self.device)
            e2e_ms = float(self._e2e_start.elapsed_time(e2e_end))
            for group, event_pairs in self.pending.items():
                for start, end in event_pairs:
                    if end is not None:
                        self.total_ms[group] += float(start.elapsed_time(end))
            self.pending.clear()
        else:
            e2e_ms = (time.perf_counter() - self._e2e_start) * 1000.0
        self.e2e_batch_ms.append(e2e_ms)
        self.total_samples += int(batch_size)
        self.active = False

    def print_summary(self, warmup_batches: int) -> None:
        print("\n[Computational Cost] 推理速度")
        if not self.e2e_batch_ms or self.total_samples <= 0:
            print(f"  没有可统计 batch（warm-up batches={warmup_batches}）。")
            return
        values = np.asarray(self.e2e_batch_ms, dtype=np.float64)
        total_e2e_ms = float(values.sum())
        print(
            f"  端到端: mean={values.mean():.3f} ms/batch, "
            f"median={np.median(values):.3f} ms/batch, "
            f"P95={np.percentile(values, 95):.3f} ms/batch"
        )
        print(
            f"  端到端: {total_e2e_ms / self.total_samples:.3f} ms/sample, "
            f"吞吐量={self.total_samples / (total_e2e_ms / 1000.0):.3f} samples/s"
        )
        for group in ("MLLM", "Router", "2D decoder", "3D encoder", "3D decoder"):
            module_ms = float(self.total_ms.get(group, 0.0))
            ratio = 100.0 * module_ms / total_e2e_ms if total_e2e_ms > 0 else 0.0
            print(
                f"  {group:<12}: {module_ms / len(values):.3f} ms/batch, "
                f"{module_ms / self.total_samples:.3f} ms/sample, 占端到端 {ratio:.2f}%"
            )
        print(f"  统计 batch={len(values)}，跳过 warm-up batch={warmup_batches}。")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ModuleFlopAttributor:
    """用模块前后全局 FLOPs 差值归因，避免依赖 FlopCounter 的类名层级键。"""

    def __init__(self, model: nn.Module, counter):
        self.counter = counter
        self.starts = defaultdict(list)
        self.flops = defaultdict(int)
        self.handles = []
        for group, module in _profiling_targets(model):
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(group)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(group)))

    def _make_pre_hook(self, group: str):
        def hook(_module, _inputs):
            self.starts[group].append(int(self.counter.get_total_flops()))

        return hook

    def _make_post_hook(self, group: str):
        def hook(_module, _inputs, _output):
            start = self.starts[group].pop()
            self.flops[group] += max(0, int(self.counter.get_total_flops()) - start)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def profile_flops_once(
    model: nn.Module,
    inference_fn: Callable[[], Any],
) -> Optional[Dict[str, int]]:
    """在一次真实推理上统计 PyTorch FlopCounter 支持的算子 FLOPs。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[Computational Cost] FLOPs 统计不可用：{exc}")
        return None

    attributor = None
    try:
        counter = FlopCounterMode(display=False, depth=None)
        attributor = ModuleFlopAttributor(model, counter)
        with counter:
            profile_output = inference_fn()
        del profile_output
        module_flops = {"Total": int(counter.get_total_flops())}
        module_flops.update({name: int(value) for name, value in attributor.flops.items()})
        return module_flops
    except Exception as exc:
        print(f"[Computational Cost] FLOPs 统计失败，将继续正常验证：{type(exc).__name__}: {exc}")
        return None
    finally:
        if attributor is not None:
            attributor.close()


def print_flops_summary(module_flops: Optional[Dict[str, int]], batch_size: int) -> None:
    """打印一次代表性推理的总 FLOPs、模块 FLOPs 及占比。"""
    if not module_flops:
        return
    total = int(module_flops.get("Total", 0))
    print("\n[Computational Cost] FLOPs（单个代表性 batch）")
    if total <= 0:
        print("  未统计到受支持算子的 FLOPs。")
        return
    batch_size = max(1, int(batch_size))
    print(f"  {'Total':<12}: {total / 1e9:.3f} GFLOPs/batch, {total / batch_size / 1e9:.3f} GFLOPs/sample")
    groups = ("MLLM", "Router", "2D decoder", "3D encoder", "3D decoder")
    for group in groups:
        value = int(module_flops.get(group, 0))
        print(
            f"  {group:<12}: {value / 1e9:.3f} GFLOPs/batch, "
            f"{value / batch_size / 1e9:.3f} GFLOPs/sample, 占总量 {100.0 * value / total:.2f}%"
        )
    assigned = sum(int(module_flops.get(group, 0)) for group in groups)
    other = max(0, total - assigned)
    print(f"  {'其他/未归因':<12}: {other / 1e9:.3f} GFLOPs/batch, 占总量 {100.0 * other / total:.2f}%")
    print("  注：仅统计 PyTorch FlopCounter 已支持的算子；自定义 CUDA、稀疏算子和部分插值操作可能被低估。")


def print_gpu_memory_summary(device: torch.device, baseline_allocated: int) -> None:
    """打印当前 CUDA 设备的推理峰值显存。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    print("\n[Computational Cost] GPU 显存")
    print(f"  模型加载后基线 allocated : {format_bytes(baseline_allocated)}")
    print(f"  推理峰值 allocated       : {format_bytes(peak_allocated)}")
    print(f"  推理增量峰值             : {format_bytes(max(0, peak_allocated - baseline_allocated))}")
    print(f"  推理峰值 reserved        : {format_bytes(peak_reserved)}")
