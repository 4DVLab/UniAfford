from __future__ import annotations

"""
Qwen generate() 路径的 router feedback 局部 patch。

设计目标：
- 不改 Transformers 全局 GenerationMixin，只替换当前 Qwen 实例的 _sample 方法；
- 不沿用 Mirage 的 latent_start/latent_pad/latent_end 协议；
- 每一步都用外层 HeadRouter 判断 last hidden state 的任务归属；
- text 路由保持原始 token lookup，非 text 路由把 hidden state 写回下一步 inputs_embeds。

这个文件只处理 generate() 的 sample/greedy 循环。训练 forward 仍使用普通
teacher-forcing 前向；验证/推理通过这里记录 task token hidden 供下游 decoder 使用。
"""

import types
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.generation.utils import (
    ALL_CACHE_NAMES,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)


def _last_hidden_state(outputs: Any) -> Optional[torch.Tensor]:
    """
    兼容不同 Qwen/Transformers 输出格式，统一取最后一层 hidden states。

    本 patch 每步都要用 last hidden 做 router 决策；标准输出通常是 hidden_states tuple，
    但本地改造或 wrapper 也可能直接返回 tensor，因此集中在这里做格式兜底。
    """
    # Transformers 标准输出在 outputs.hidden_states，通常是每层 hidden 的 tuple。
    # Qwen 不同版本或本地 wrapper 可能返回 tensor，因此这里先处理最常见的两种情况。
    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    if hidden_states is not None and len(hidden_states) > 0:
        candidate = hidden_states[-1]
        if isinstance(candidate, torch.Tensor):
            return candidate

    # 兜底兼容 last_hidden_state 或 tuple/list 第 0 项，避免具体模型输出结构变化导致 patch 失效。
    for candidate in (
        getattr(outputs, "last_hidden_state", None),
        outputs[0] if isinstance(outputs, (tuple, list)) and len(outputs) > 0 else None,
    ):
        if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
            return candidate
    return None


def _embed_token_ids(model: Any, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """text 分支保持原始 LLM 行为：token id 通过 embedding table 查表得到下一步输入。"""
    embeddings = model.get_input_embeddings()(token_ids)
    return embeddings.to(dtype=dtype)


def _get_generation_router(model: Any) -> Optional[nn.Module]:
    """
    取运行时挂载的 HeadRouter。

    router 不注册为 Qwen 子模块，只作为运行时引用使用，避免污染 Qwen 的 state_dict。
    """
    router = getattr(model, "_ja_generation_router", None)
    if router is None:
        return None
    return router


def _route_next_tokens_and_embeds(
    model: Any,
    router: nn.Module,
    step_hidden: torch.Tensor,
    do_sample: bool,
    next_token_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    根据 router 同时决定“记录的 token id”和“下一步输入 embedding”。

    这里是和 Mirage 最大的差异：不是遇到某个 latent token 才进入 latent mode，
    而是每一步都让 router 判断当前 hidden 属于 text 还是下游任务 token。
    """
    # HeadRouter 的接口按序列处理，这里把单步 hidden 扩成 [B, 1, C]。
    # routed[:, 0] 是当前 decode step 的 hard route 结果。
    route_logits, route_probs, routed = router.route_hidden_states(
        step_hidden.unsqueeze(1),
        img_available=getattr(model, "_ja_generation_img_available", None),
        pc_available=getattr(model, "_ja_generation_pc_available", None),
    )
    hard_route = routed[:, 0]
    text_route_idx = int(getattr(router, "route_text_idx", 0))
    text_mask = hard_route.eq(text_route_idx)

    # text 路由保持 Transformers 原始逻辑：sample/greedy 得到词表 token。
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        text_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        text_token_ids = torch.argmax(next_token_scores, dim=-1)

    next_ids = text_token_ids.clone()
    # 非 text 路由不使用采样出来的 token 语义，只把对应任务 placeholder 写回 input_ids。
    # 这样日志、对齐、停止条件仍能看到一个离散 token 序列。
    # 注意：placeholder 只承担“记录/占位”职责，不参与下一步 embedding lookup。
    for task_name, route_idx in getattr(router, "task_id_by_name", {}).items():
        if task_name == "text":
            continue
        placeholder_id = getattr(router, "task_placeholder_ids", {}).get(task_name)
        if placeholder_id is None:
            continue
        task_mask = hard_route.eq(int(route_idx))
        next_ids = torch.where(task_mask, torch.full_like(next_ids, int(placeholder_id)), next_ids)

    # 真正进入下一步模型的是 embedding：
    # - text：使用采样/greedy 得到的 token id 查 embedding table；
    # - 非 text：直接使用当前 step_hidden 作为下一步输入，避免退回普通词表语义。
    # detach 是推理 generate() 路径的安全处理，防止 hidden feedback 意外保留历史计算图。
    text_next_embeds = _embed_token_ids(model, text_token_ids, dtype=step_hidden.dtype)
    hidden_next_embeds = step_hidden.detach().to(dtype=text_next_embeds.dtype)
    next_embeds = torch.where(text_mask.view(-1, 1), text_next_embeds, hidden_next_embeds)
    return next_ids, next_embeds, hard_route, route_logits[:, 0, :], route_probs[:, 0, :]


def _prepare_latent_model_inputs(
    model: Any,
    input_ids: torch.LongTensor,
    next_inputs_embeds: Optional[torch.Tensor],
    model_kwargs: dict,
) -> dict:
    """
    封装 prepare_inputs_for_generation，并在需要时强制使用 inputs_embeds。

    使用 KV cache 时，decode 阶段每轮只需要喂一个新位置。对于 router 判为非 text 的位置，
    这个新位置没有普通词表 embedding，需要把上一轮 hidden state 作为 next_inputs_embeds 传入。
    """
    # use_cache=True 时 prepare_inputs_for_generation 只需要准备最新一个 token 位置；
    # use_cache=False 时仍允许模型按完整序列逻辑处理。
    next_sequence_length = 1 if model_kwargs.get("use_cache", False) else None
    # 防止外部 model_kwargs 已携带 inputs_embeds 时与显式参数重复。
    clean_model_kwargs = {k: v for k, v in model_kwargs.items() if k != "inputs_embeds"}
    if next_inputs_embeds is None:
        return model.prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            **clean_model_kwargs,
        )

    # 这里仍调用模型自己的 prepare_inputs_for_generation，保留 Qwen 对 position_ids、
    # cache_position、attention_mask、视觉输入等字段的原生处理。
    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        next_sequence_length=next_sequence_length,
        inputs_embeds=next_inputs_embeds,
        **clean_model_kwargs,
    )
    # input_ids 已经在外层序列中追加；本轮 forward 只喂 embedding，避免模型重复查表。
    model_inputs["input_ids"] = None
    model_inputs["inputs_embeds"] = next_inputs_embeds
    return model_inputs


def router_feedback_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    """
    transformers.generation.utils.GenerationMixin._sample 的实例级替换版本。

    这里修改 generation loop 的位置，并修改下一步输入的写回方式。
    每一步都用外部 HeadRouter 对 last hidden 做 hard route，决定是继续使用词表采样还是使用 hidden 写回：
    - text: 维持原始 logits -> token id -> embedding lookup；
    - 非 text: input_ids 写成对应任务 placeholder，下一步 inputs_embeds 直接写回当前 hidden state。
    """
    router = _get_generation_router(self)
    if router is None:
        # patch 可以先于 JointAffordanceModel.router 创建；未挂 router 时保持原始 generate 行为。
        original_sample = getattr(self, "_ja_original_sample", None)
        if original_sample is None:
            raise RuntimeError("router_feedback_sample requires a router or an original _sample fallback.")
        return original_sample(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    trace_hidden_states = []
    trace_routes = []
    trace_route_logits = []
    trace_route_probs = []
    trace_token_ids = []
    trace_logits = []

    # 以下变量基本复制 Transformers _sample 的控制面，保证外部 generate 参数语义不变。
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    user_output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # 以下输出收集逻辑保持 Transformers 原始 _sample 的结构，避免破坏 return_dict_in_generate。
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and user_output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if user_output_hidden_states else None
        )

    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    # 保留 Transformers 的 compiled call 分支；当前 patch 只改变“下一步输入写回方式”，
    # 不改变模型 forward 调用入口选择。
    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else self.__call__
    )

    # 下一轮 decode 要喂入的单 token embedding。None 表示仍使用原始 token id 路径。
    next_inputs_embeds = None

    prefill_consumed = False
    original_output_hidden_states = generation_config.output_hidden_states
    # router 决策依赖 hidden state，即使用户 generate 时没要求返回 hidden，也必须临时打开。
    # finally 中会恢复原值，因此不会污染后续普通 generate 调用。
    generation_config.output_hidden_states = True
    try:
        # prefill 消化完整 prompt，并建立 KV cache；后续循环只做单步增量 decode。
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                # 增量 decode 阶段统一通过 next_inputs_embeds 传入上一轮选出的输入向量。
                # text 的向量来自词表 lookup，非 text 的向量来自 hidden-state 写回。
                model_inputs = _prepare_latent_model_inputs(
                    self,
                    input_ids=input_ids,
                    next_inputs_embeds=next_inputs_embeds,
                    model_kwargs=model_kwargs,
                )
                model_inputs["output_hidden_states"] = True
                with self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True

            # 更新 KV cache、attention_mask、cache_position 等状态，保持和原始 _sample 一致。
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # 仅在用户请求时收集调试/返回信息；router 内部需要的 hidden 不走这里。
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if user_output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # 取当前 step 的最后一层 hidden。这个 hidden 同时服务两个目的：
            # 1) 交给 router 做 hard route；
            # 2) 当 route 为非 text 时，作为下一步 inputs_embeds 写回。
            last_hidden = _last_hidden_state(outputs)
            if last_hidden is None:
                raise RuntimeError("Router generation patch requires hidden_states from the model output.")
            step_hidden = last_hidden[:, -1, :]
            # 核心分岔：router 决定本步记录的 token id，以及下一步使用 lookup 还是 hidden 写回。
            next_tokens, routed_next_embeds, hard_route, step_route_logits, step_route_probs = _route_next_tokens_and_embeds(
                model=self,
                router=router,
                step_hidden=step_hidden,
                do_sample=do_sample,
                next_token_scores=next_token_scores,
            )

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            trace_hidden_states.append(step_hidden.detach())
            trace_routes.append(hard_route.detach())
            trace_route_logits.append(step_route_logits.detach())
            trace_route_probs.append(step_route_probs.detach())
            trace_token_ids.append(next_tokens.detach())
            trace_logits.append(next_token_logits.detach())

            # 已结束样本的下一步输入使用 pad embedding，避免 hidden feedback 污染结束后的 padding token。
            # 这里要在 next_tokens 被 pad 化之后再查表，确保 padding 分支和 Transformers 原始行为一致。
            if has_eos_stopping_criteria:
                pad_embeds = _embed_token_ids(self, next_tokens, dtype=routed_next_embeds.dtype)
                alive_mask = unfinished_sequences.bool().view(-1, 1)
                routed_next_embeds = torch.where(alive_mask, routed_next_embeds, pad_embeds)
            next_inputs_embeds = routed_next_embeds.unsqueeze(1)

            # input_ids 始终记录可读的离散序列：text 为真实 token，非 text 为任务 placeholder。
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # stopping_criteria 仍只看离散 input_ids 序列；非 text step 写入 placeholder，
            # 因此 max_length/eos 等停止逻辑不会直接感知 hidden feedback 的连续向量。
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            del outputs
    finally:
        # 恢复用户原始配置，避免一次 generate 改变后续调用的输出行为。
        generation_config.output_hidden_states = original_output_hidden_states

    # 将生成过程的路由轨迹挂在当前 Qwen 实例上，供外层 JointAffordanceModel 取 task hidden 解码。
    # 不塞进 Transformers Generate*Output，避免绑定具体 transformers dataclass 字段。
    object.__setattr__(
        self,
        "_ja_generation_feedback_trace",
        {
            "step_hidden_states": torch.stack(trace_hidden_states, dim=1) if trace_hidden_states else None,
            "step_routes": torch.stack(trace_routes, dim=1) if trace_routes else None,
            "step_route_logits": torch.stack(trace_route_logits, dim=1) if trace_route_logits else None,
            "step_route_probs": torch.stack(trace_route_probs, dim=1) if trace_route_probs else None,
            "step_token_ids": torch.stack(trace_token_ids, dim=1) if trace_token_ids else None,
            "step_logits": torch.stack(trace_logits, dim=1) if trace_logits else None,
        },
    )

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        # 返回结构沿用 Transformers 原始类型；cache 从 model_kwargs 中取最新状态。
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=cache,
        )
    return input_ids


def patch_router_generation_feedback(model: Any, *, router: Optional[nn.Module] = None) -> Any:
    """
    对单个 Qwen 实例做实例级 monkey patch。

    不修改 transformers.GenerationMixin 全局类，避免影响同进程里的其他模型。
    router 可以稍后再挂载，因为 MLLMBackbone 初始化时 JointAffordanceModel.router 尚未创建。
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("patch_router_generation_feedback expects a transformers model with a config.")

    if not hasattr(model, "_ja_original_sample"):
        # 保存原始 _sample；未挂 router 或需要回退时仍可走原始 generate。
        model._ja_original_sample = model._sample
    if router is not None:
        # 避免 nn.Module.__setattr__ 将 router 注册成 Qwen 子模块。
        object.__setattr__(model, "_ja_generation_router", router)
    # MethodType 只绑定当前模型实例，不会修改 Qwen 类或 GenerationMixin 类本身。
    # 因此同一进程里加载的其他 Transformers 模型不会受到影响。
    model._sample = types.MethodType(router_feedback_sample, model)
    model._ja_router_generation_patch_enabled = True
    return model


def attach_generation_router(model: Any, router: nn.Module) -> Any:
    """在 HeadRouter 创建完成后，把 router 作为运行时引用挂到 Qwen 实例上。"""
    object.__setattr__(model, "_ja_generation_router", router)
    return model
