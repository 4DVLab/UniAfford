import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ParamInfo:
    name: str
    numel: int
    trainable: bool


@dataclass
class TreeNode:
    name: str
    children: Dict[str, "TreeNode"] = field(default_factory=dict)
    direct_params: List[ParamInfo] = field(default_factory=list)

    def add_param(self, module_parts: List[str], info: ParamInfo) -> None:
        if not module_parts:
            self.direct_params.append(info)
            return
        head = module_parts[0]
        if head not in self.children:
            self.children[head] = TreeNode(name=head)
        self.children[head].add_param(module_parts[1:], info)


def _is_index(part: str) -> bool:
    return part.isdigit()


def _state_matches(trainable_params: int, total_params: int, state: str) -> bool:
    if total_params <= 0:
        return False
    if state == "all":
        return True
    if state == "trainable":
        return trainable_params > 0
    if state == "frozen":
        return trainable_params < total_params
    raise ValueError(f"Unsupported state: {state}")


def _aggregate_node(node: TreeNode) -> Tuple[int, int, int, int]:
    total_tensors = len(node.direct_params)
    trainable_tensors = sum(1 for p in node.direct_params if p.trainable)
    total_params = sum(p.numel for p in node.direct_params)
    trainable_params = sum(p.numel for p in node.direct_params if p.trainable)
    for child in node.children.values():
        child_total_tensors, child_trainable_tensors, child_total_params, child_trainable_params = _aggregate_node(child)
        total_tensors += child_total_tensors
        trainable_tensors += child_trainable_tensors
        total_params += child_total_params
        trainable_params += child_trainable_params
    return total_tensors, trainable_tensors, total_params, trainable_params


def _merge_nodes(name: str, nodes: Iterable[TreeNode]) -> TreeNode:
    merged = TreeNode(name=name)
    for node in nodes:
        merged.direct_params.extend(node.direct_params)
        for child_name, child in node.children.items():
            if child_name not in merged.children:
                merged.children[child_name] = TreeNode(name=child_name)
            merged.children[child_name] = _merge_nodes(
                child_name,
                [merged.children[child_name], child],
            )
    return merged


def render_tree_lines(
    node: TreeNode,
    state: str,
    indent: int = 0,
    parent_name: Optional[str] = None,
) -> List[str]:
    total_tensors, trainable_tensors, total_params, trainable_params = _aggregate_node(node)
    if not _state_matches(trainable_params, total_params, state):
        return []

    label = node.name
    if parent_name is not None and label == parent_name:
        label = f"{label} (merged)"
    prefix = "  " * indent
    state_label = (
        "trainable"
        if trainable_params == total_params
        else ("frozen" if trainable_params == 0 else "mixed")
    )
    lines = [
        (
            f"{prefix}- {label} [{state_label}] "
            f"tensors={trainable_tensors}/{total_tensors}, "
            f"params={trainable_params:,}/{total_params:,}"
        )
    ]

    direct_groups: Dict[str, List[ParamInfo]] = defaultdict(list)
    for param in node.direct_params:
        param_name = param.name.rsplit(".", 1)[-1]
        direct_groups[param_name].append(param)
    for param_name in sorted(direct_groups.keys()):
        group = direct_groups[param_name]
        group_total_params = sum(x.numel for x in group)
        group_trainable_params = sum(x.numel for x in group if x.trainable)
        if not _state_matches(group_trainable_params, group_total_params, state):
            continue
        suffix = f" x{len(group)}" if len(group) > 1 else ""
        group_state = (
            "trainable"
            if group_trainable_params == group_total_params
            else ("frozen" if group_trainable_params == 0 else "mixed")
        )
        lines.append(
            (
                f"{prefix}  - {param_name}{suffix} [{group_state}] "
                f"params={group_trainable_params:,}/{group_total_params:,}"
            )
        )

    named_children = {k: v for k, v in node.children.items() if not _is_index(k)}
    indexed_children = [v for k, v in node.children.items() if _is_index(k)]

    for child_name in sorted(named_children.keys()):
        lines.extend(render_tree_lines(named_children[child_name], state=state, indent=indent + 1))

    if indexed_children:
        merged_indexed = _merge_nodes(node.name, indexed_children)
        merged_name = f"{node.name} x{len(indexed_children)}"
        merged_indexed.name = merged_name
        lines.extend(render_tree_lines(merged_indexed, state=state, indent=indent + 1, parent_name=node.name))

    return lines


def build_tree_from_model(model) -> TreeNode:
    root = TreeNode(name="model")
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module_parts = [part for part in module_name.split(".") if part]
        root.add_param(
            module_parts,
            ParamInfo(name=name, numel=param.numel(), trainable=bool(param.requires_grad)),
        )
    return root


def summarize_optimizer_groups(model) -> List[str]:
    groups = {
        "llm": [],
        "vision_2d": [],
        "vision_3d": [],
        "other": [],
    }
    used_ids = set()

    def collect(module, bucket_name: str) -> None:
        if module is None:
            return
        for name, param in module.named_parameters():
            if param.requires_grad and param.is_floating_point():
                groups[bucket_name].append((name, param))
                used_ids.add(id(param))

    collect(getattr(model, "mllm", None), "llm")
    collect(getattr(model, "image_decoder", None), "vision_2d")
    collect(getattr(model, "point_decoder", None), "vision_3d")

    for name, param in model.named_parameters():
        if param.requires_grad and param.is_floating_point() and id(param) not in used_ids:
            groups["other"].append((name, param))

    lines = ["Optimizer groups:"]
    for group_name in ("llm", "vision_2d", "vision_3d", "other"):
        params = groups[group_name]
        total_numel = sum(p.numel() for _, p in params)
        example_names = ", ".join(name for name, _ in params[:5]) if params else "-"
        lines.append(
            f"  - {group_name}: tensors={len(params)}, params={total_numel:,}, examples={example_names}"
        )
    return lines


def format_trainability_summary(
    model,
    states: Sequence[str] = ("trainable", "frozen"),
    max_lines_per_state: Optional[int] = 120,
    include_optimizer_groups: bool = True,
) -> str:
    tree = build_tree_from_model(model)
    sections: List[str] = []
    if include_optimizer_groups:
        sections.extend(summarize_optimizer_groups(model))
        sections.append("")

    for state in states:
        title = f"{state.capitalize()} module tree:"
        lines = render_tree_lines(tree, state=state)
        if max_lines_per_state is not None and len(lines) > max_lines_per_state:
            hidden = len(lines) - max_lines_per_state
            lines = lines[:max_lines_per_state] + [f"... ({hidden} more lines omitted)"]
        sections.append(title)
        sections.extend(lines or ["- <empty>"])
        sections.append("")

    while sections and sections[-1] == "":
        sections.pop()
    return "\n".join(sections)


def log_trainability_summary(
    model,
    logger,
    output_path: str,
    states: Sequence[str] = ("trainable", "frozen"),
    max_lines_per_state: Optional[int] = 120,
) -> None:
    summary = format_trainability_summary(
        model,
        states=states,
        max_lines_per_state=max_lines_per_state,
        include_optimizer_groups=True,
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n")
    logger.info(f"Trainability summary saved to: {output_path}")
