"""Unsloth SFT entry point for Phase 2 fine-tuning.

Unified for all 3 categories (A, B, C). Category-specific behavior
(data paths, chat templates) is driven by config YAML.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_GLM_OOM_FALLBACK_RANK = 32

_LOSS_MASK_ALL_TOKENS = "all_tokens"
_LOSS_MASK_RESPONSE_ONLY = "response_only"
_VALID_LOSS_MASKS = (_LOSS_MASK_ALL_TOKENS, _LOSS_MASK_RESPONSE_ONLY)


def render_response_only_sample(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize one conversation, masking non-assistant spans to -100.

    Walks `messages` incrementally: for each turn, tokenizes
    `apply_chat_template(prefix)` and `apply_chat_template(prefix + turn)` and
    takes the delta. Tokens from non-assistant turns get label -100; assistant
    tokens carry through unmasked. Truncates from the left so the final
    assistant turn always survives.

    Chat-template-agnostic — relies only on `apply_chat_template`'s
    deterministic prefix-extension behavior. Used by Recipe B (response_only)
    in docs/fine_tuning_recipes.md.

    Returns {"input_ids", "attention_mask", "labels"} for a single sample.
    """
    ids: list[int] = []
    labels: list[int] = []

    def _encode(msgs: list[dict[str, str]]) -> list[int]:
        if not msgs:
            return []
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
        # Some processors / wrappers return BatchEncoding; normalize to list.
        if hasattr(out, "tolist"):
            out = out.tolist()
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    prev_encoded: list[int] = []
    for i, msg in enumerate(messages):
        extended = _encode(messages[: i + 1])
        # Guard against tokenizers that don't deterministically extend
        # (rare; e.g. some templates re-render BOS). Fall back to encoding
        # this turn alone if the prefix doesn't match the head of `extended`.
        if (
            len(extended) >= len(prev_encoded)
            and extended[: len(prev_encoded)] == prev_encoded
        ):
            new = extended[len(prev_encoded) :]
        else:
            new = _encode([msg])
            extended = prev_encoded + new
        ids.extend(new)
        if msg.get("role") == "assistant":
            labels.extend(new)
        else:
            labels.extend([-100] * len(new))
        prev_encoded = extended

    # Left-truncate to preserve the final assistant turn.
    if len(ids) > max_seq_length:
        cut = len(ids) - max_seq_length
        ids = ids[cut:]
        labels = labels[cut:]

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


@dataclass(frozen=True)
class SFTResult:
    """Result of an SFT training run."""

    checkpoint_path: Path | None = None
    best_eval_loss: float | None = None
    total_steps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    param_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _load_sft_config(config_path: Path) -> dict[str, Any]:
    """Load SFT config YAML (v3 format with stage/framework fields)."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if config.get("stage") != "sft":
        raise ValueError(f"Expected stage='sft', got '{config.get('stage')}'")
    return config


def _lora_spec_for(config: dict[str, Any], model_name: str) -> Any:
    """Return the LoRATargetSpec for `model_name`, falling back to config name."""
    from llm_workflow_agents.training.lora_targets import get_lora_target_spec

    if not model_name:
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "") or model_cfg.get("config_path", "")
    return get_lora_target_spec(model_name)


def _resolve_lora_targets(config: dict[str, Any], model_name: str = "") -> list[str]:
    """Resolve LoRA targets from config, falling back to registry.

    `model_name` should be the HF model ID (e.g. "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic").
    Passing the YAML config_path won't work because the registry's substring
    patterns (e.g. "gemma-4-26b") only match HF-style hyphenated names.
    """
    from llm_workflow_agents.training.lora_targets import get_lora_target_spec

    targets = config.get("lora", {}).get("target_modules", "auto")
    if targets == "auto":
        # Fall back to fields on `config.model` if no explicit name was passed.
        if not model_name:
            model_cfg = config.get("model", {})
            model_name = model_cfg.get("name", "") or model_cfg.get("config_path", "")
        spec = get_lora_target_spec(model_name)
        return list(spec.target_modules)
    return targets if isinstance(targets, list) else [targets]


def _patch_gemma4_rope_stride(model: Any) -> int:
    """Force `inv_freq_expanded` contiguous in Gemma-4 RoPE — instance-level.

    Why instance-level: Unsloth pre-generates a copy of the Gemma-4 module
    in `unsloth_compiled_cache/unsloth_compiled_module_gemma4.py` and
    routes calls through *that* module's own `patched_forward`, bypassing
    any class-level monkey-patch on transformers' `Gemma4TextRotaryEmbedding`.
    Patching the bound `forward` on each loaded module instance catches
    both code paths.

    Root cause: the original forward builds `inv_freq_expanded` via
    `.expand(B, -1, 1)`, which is a stride-0 view on the batch dim.
    cuBLAS in CUDA 13 rejects batched sgemm with 0-stride batch
    (CUBLAS_STATUS_INVALID_VALUE); CUDA 12 was more permissive.
    `.contiguous()` materializes a real tensor before the matmul.

    Returns the number of modules patched.
    """
    import torch
    from types import MethodType

    try:
        from transformers.modeling_rope_utils import dynamic_rope_update
    except ImportError:
        def dynamic_rope_update(fn):  # type: ignore[no-redef]
            return fn

    @torch.no_grad()
    @dynamic_rope_update
    def patched_forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # Build (B, N) and (B, S) tensors and use broadcast multiply,
        # mathematically identical to upstream's bmm of shape
        # (B, N, 1) @ (B, 1, S) since K=1 makes the sum trivial. This
        # avoids cuBLAS sgemm entirely — torch 2.10 + CUDA 13 cuBLAS
        # rejects K=1 batched sgemm with CUBLAS_STATUS_INVALID_VALUE.
        inv_freq_b = inv_freq[None, :].float().expand(position_ids.shape[0], -1).to(x.device)
        pos_f = position_ids.float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # (B, N, 1) * (B, 1, S) -> (B, N, S), then transpose to (B, S, N)
            freqs = (inv_freq_b.unsqueeze(-1) * pos_f.unsqueeze(1)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    patched = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in {"Gemma4TextRotaryEmbedding", "Gemma4RotaryEmbedding"}:
            continue
        if getattr(module, "_unsloth_workflow_patched", False):
            continue
        module.forward = MethodType(patched_forward, module)
        module._unsloth_workflow_patched = True
        patched += 1
    if patched:
        logger.info("gemma4_rope_stride_patch_applied", instances=patched)
    else:
        logger.debug("gemma4_rope_stride_patch_skipped", reason="no Gemma-4 RoPE modules in model")
    return patched


def train_sft(
    config_path: Path,
    resume_from_checkpoint: bool | str | Path | None = None,
) -> SFTResult:
    """Run Unsloth SFT pipeline.

    Pipeline:
      1. Load base model via FastLanguageModel.from_pretrained()
      2. Apply LoRA via FastLanguageModel.get_peft_model()
      3. Configure SFTTrainer with packing + per-model chat template
      4. Train for num_epochs, checkpoint every 500 steps
      5. Select best checkpoint by validation loss
      6. Return SFTResult

    Args:
        config_path: Path to SFT YAML config.
        resume_from_checkpoint: Resume training from a checkpoint.
            - True: auto-detect the most recent checkpoint under output_dir.
            - str / Path: explicit checkpoint directory.
            - None / False (default): start fresh.
    """
    config = _load_sft_config(config_path)
    lora_cfg = config.get("lora", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    logging_cfg = config.get("logging", {})

    loss_mask = training_cfg.get("loss_mask", _LOSS_MASK_ALL_TOKENS)
    if loss_mask not in _VALID_LOSS_MASKS:
        return SFTResult(
            error=(
                f"Unknown training.loss_mask={loss_mask!r}. "
                f"Valid values: {_VALID_LOSS_MASKS}."
            )
        )
    if loss_mask == _LOSS_MASK_RESPONSE_ONLY:
        # eval_loss averages over a different denominator under response_only
        # (only assistant tokens) vs all_tokens (every non-pad token), so the
        # two recipes' eval_loss numbers are not directly comparable. Within a
        # single run, metric_for_best_model: eval_loss still selects correctly.
        logger.warning(
            "sft_loss_mask_response_only",
            note=(
                "eval_loss is NOT comparable across loss_mask recipes; "
                "rank recipes via eval/agent_benchmark.py weighted_workflow_score."
            ),
        )

    model_config_path = config.get("model", {}).get("config_path")
    if not model_config_path:
        return SFTResult(error="model.config_path not set in SFT config")

    # Load model config to get HF model name
    import yaml

    with open(model_config_path) as f:
        model_yaml = yaml.safe_load(f) or {}
    model_section = model_yaml.get("model", model_yaml)
    model_name = model_section["name"]
    is_4bit = training_cfg.get("precision") == "qlora_4bit"

    # Framework selector. `unsloth` (default) uses FastLanguageModel +
    # Unsloth's compiled training kernels. `trl` falls back to vanilla
    # transformers + PEFT — slower, but bypasses Unsloth's MoE-expert
    # `_grouped_mm` LoRA path that crashes on Gemma-4 26B-A4B.
    #
    # Resolution order (first non-empty wins). Per-model settings beat
    # per-category defaults so individual models can pin a framework
    # without requiring a separate SFT config:
    #   1. SFT_FRAMEWORK env var (one-off override)
    #   2. Model YAML: training.framework or top-level framework
    #   3. SFT config: training.framework or top-level framework
    #   4. Default "unsloth"
    framework = (
        os.environ.get("SFT_FRAMEWORK")
        or model_yaml.get("training", {}).get("framework")
        or model_yaml.get("framework")
        or training_cfg.get("framework")
        or config.get("framework")
        or "unsloth"
    ).lower()
    if framework not in ("unsloth", "trl"):
        return SFTResult(error=f"Unknown training framework: {framework!r}")
    logger.info("sft_framework_selected", framework=framework, model=model_name)

    if framework == "unsloth":
        from unsloth import FastLanguageModel

    logger.info(
        "sft_starting",
        model=model_name,
        precision=training_cfg.get("precision", "bf16"),
        lora_rank=lora_cfg.get("rank", 64),
    )

    lora_rank = lora_cfg.get("rank", 64)
    if framework == "unsloth":
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=training_cfg.get("max_seq_length", 8192),
                dtype=None,
                load_in_4bit=is_4bit,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and lora_rank > _GLM_OOM_FALLBACK_RANK:
                logger.warning(
                    "sft_oom_fallback",
                    original_rank=lora_rank,
                    fallback_rank=_GLM_OOM_FALLBACK_RANK,
                )
                import torch

                torch.cuda.empty_cache()
                lora_rank = _GLM_OOM_FALLBACK_RANK
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=training_cfg.get("max_seq_length", 8192),
                    dtype=None,
                    load_in_4bit=is_4bit,
                )
            else:
                return SFTResult(error=str(e))
    else:
        # TRL + PEFT path. Use AutoProcessor when the checkpoint is multimodal
        # (Gemma-4) so chat-template + tokenization keep working; otherwise
        # AutoTokenizer. Model goes through standard transformers; LoRA is
        # injected via peft.get_peft_model below.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import AutoProcessor
            tokenizer = AutoProcessor.from_pretrained(model_name)
        except (ValueError, OSError, KeyError):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        if is_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            return SFTResult(error=str(e))

    # Patch Gemma-4 RoPE forward on every loaded RoPE module instance.
    # Must happen before LoRA injection so that downstream wrappers (PEFT,
    # gradient checkpointing) see the patched bound forward.
    _patch_gemma4_rope_stride(model)

    target_modules = _resolve_lora_targets(config, model_name=model_name)
    if not target_modules:
        return SFTResult(
            error=(
                f"No LoRA target_modules resolved for model {model_name!r}. "
                "Add a registry entry in training/lora_targets.py or set "
                "lora.target_modules explicitly in the SFT config."
            )
        )

    # MoE models route LoRA through PEFT's ParamWrapper for the expert
    # tensors (experts.gate_up_proj / experts.down_proj — nn.Parameter, not
    # nn.Linear). ParamWrapper rejects lora_dropout != 0, so force it to 0
    # for any MoE config and log the override.
    #
    # Multimodal configs (e.g. Gemma-4) nest the LM under `text_config`, so
    # walk top-level + known sub-configs and check the registry warnings.
    lora_dropout = lora_cfg.get("dropout", 0.05)

    def _find_num_experts(model_obj: Any) -> int:
        cfg = getattr(model_obj, "config", None)
        if cfg is None:
            return 0
        candidates = [cfg]
        for attr in ("text_config", "language_config", "decoder_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                candidates.append(sub)
        for c in candidates:
            for field_name in ("num_local_experts", "num_experts", "num_routed_experts"):
                n = getattr(c, field_name, 0) or 0
                if n:
                    return int(n)
        return 0

    num_experts = _find_num_experts(model)
    is_moe_by_registry = any(
        "moe" in w.lower()
        for w in getattr(_lora_spec_for(config, model_name), "warnings", ())
    )
    if (num_experts or is_moe_by_registry) and lora_dropout != 0:
        logger.warning(
            "moe_dropout_override",
            model=model_name,
            num_experts=num_experts,
            registry_flagged_moe=is_moe_by_registry,
            requested_dropout=lora_dropout,
            applied_dropout=0.0,
            reason="PEFT ParamWrapper (used for MoE experts) requires dropout=0",
        )
        lora_dropout = 0.0

    # Per Unsloth's Gemma-4 docs (https://unsloth.ai/docs/models/gemma-4/train),
    # use the toggle-flag form for Gemma-4 — Unsloth picks the correct module
    # set (including MoE experts) internally. Fall back to the explicit
    # target_modules registry list for non-Gemma-4 models.
    model_family = (model_yaml.get("model", model_yaml).get("family") or "").lower()
    use_unsloth_toggles = model_family == "gemma" and "gemma-4" in model_name.lower()

    if framework == "trl":
        # Vanilla PEFT LoRA injection. No Unsloth fast paths, no MoE
        # grouped_mm — adapters land on every nn.Linear matched by
        # target_modules, including per-expert FFN modules in MoE blocks.
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if is_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Without this, the frozen input-embeddings produce activations
            # with requires_grad=False; the gradient-checkpoint boundary then
            # detaches the autograd graph and LoRA params receive zero grad.
            model.enable_input_require_grads()

        # PEFT only LoRA-wraps a fixed set of module types (nn.Linear,
        # nn.Embedding, etc.). Some Gemma-4 projection layers are
        # `Gemma4ClippableLinear` wrappers around an inner `nn.Linear`,
        # so when *all* projections in the model are wrappers, the
        # registry's short names (`q_proj`, `k_proj`, ...) match nothing
        # PEFT can wrap and we have to redirect to `*.linear`. But the
        # text decoder of Gemma-4 conditional-generation models uses
        # bare `nn.Linear` while only the vision/audio towers use the
        # wrapper. A blanket "any wrapper present anywhere" rewrite
        # attaches LoRA to the vision tower only — text-only SFT never
        # activates those params and grad_norm stays 0.
        #
        # Positive test: only rewrite if no bare nn.Linear with the
        # target leaf name already exists. If at least one does, PEFT
        # will attach LoRA to those (the language decoder's projections),
        # which is what we want. Robust to any model wrapping.
        target_modules_resolved = list(target_modules)
        target_leaf_names = set(target_modules)
        bare_target_count = sum(
            1
            for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in target_leaf_names
        )
        wrapper_target_count = sum(
            1
            for name, m in model.named_modules()
            if type(m).__name__ == "Gemma4ClippableLinear"
            and name.rsplit(".", 1)[-1] in target_leaf_names
        )
        logger.info(
            "trl_lora_target_module_scan",
            bare_linear_matches=bare_target_count,
            clippable_wrapper_matches=wrapper_target_count,
        )
        exclude_modules: str | None = None
        if bare_target_count > 0 and wrapper_target_count > 0:
            # Mixed population — Gemma-4 conditional-generation has bare
            # nn.Linear in the language decoder AND Gemma4ClippableLinear
            # in the vision/audio towers. PEFT's suffix matcher catches
            # both and raises on the wrapper. Exclude the tower paths so
            # LoRA lands only on the decoder.
            exclude_modules = r".*(vision_tower|audio_tower)\..*"
            logger.info(
                "trl_lora_excluding_multimodal_towers",
                exclude_pattern=exclude_modules,
            )
        elif bare_target_count == 0 and wrapper_target_count > 0:
            target_modules_resolved = [
                f"{t}.linear" for t in target_modules_resolved
            ]
            logger.info(
                "trl_lora_targets_rewritten_for_wrapper",
                wrapper="Gemma4ClippableLinear",
                original=list(target_modules),
                rewritten=target_modules_resolved,
            )

        lora_config_kwargs: dict[str, Any] = {
            "r": lora_rank,
            "lora_alpha": lora_cfg.get("alpha", lora_rank * 2),
            "lora_dropout": lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": target_modules_resolved,
        }
        if exclude_modules is not None:
            lora_config_kwargs["exclude_modules"] = exclude_modules
        peft_config = LoraConfig(**lora_config_kwargs)
        model = get_peft_model(model, peft_config)
    elif use_unsloth_toggles:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_cfg.get("alpha", lora_rank),
            lora_dropout=lora_dropout,
            bias="none",
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            finetune_vision_layers=False,
            use_gradient_checkpointing="unsloth",
            random_state=lora_cfg.get("seed", 3407),
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_cfg.get("alpha", 128),
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
        )

    # Freeze router weights if configured
    if lora_cfg.get("freeze_router", False):
        from llm_workflow_agents.training.train_specialist import _freeze_modules

        _freeze_modules(model, ["mlp.gate"])

    # Load dataset.
    # We bypass `load_dataset("json", ...)` because PyArrow's JSON reader
    # tries to infer a single schema for nested fields across all rows, and
    # `messages[].annotations.tool_calls[].arguments` is intentionally
    # heterogeneous (each tool defines its own argument shape — numbers in
    # one row, strings in another). SFT only needs role + content, so we
    # drop annotations and feed Arrow uniform string columns.
    import json as _json

    from datasets import Dataset

    data_source = data_cfg.get("source", "")

    def _coerce_content(c: Any) -> str:
        # Some rows (tool replies, multimodal stubs) carry content as a dict
        # or list. Arrow can't mix struct and string within one column, so
        # JSON-encode anything non-str. Empty/None becomes "".
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        return _json.dumps(c, ensure_ascii=False)

    # Re-enrich the system prompt at load time so training sees the same
    # prompt the benchmark sees. Existing SFT JSONL has stale enrichment
    # baked in (broken workflow_script + old FORMAT_RULES); force_rebuild=True
    # strips it and rebuilds from current code using the row's upstream
    # fields (workflow_graph, tool_schemas, messages, language).
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    def _load_split(name: str) -> Dataset:
        path = Path(data_source) / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"SFT split missing: {path}")
        rows: list[dict[str, Any]] = []
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                raw = _json.loads(line)
                raw_msgs = raw.get("messages", []) or []
                # Rebuild messages[0] (system) if upstream fields are present.
                if raw_msgs and raw_msgs[0].get("role") == "system" and raw.get("workflow_graph"):
                    raw_msgs = list(raw_msgs)
                    raw_msgs[0] = {
                        "role": "system",
                        "content": build_enriched_system_prompt(
                            raw, raw_msgs[0].get("content") or "", force_rebuild=True
                        ),
                    }
                msgs = [
                    {
                        "role": m.get("role", "") or "",
                        "content": _coerce_content(m.get("content")),
                    }
                    for m in raw_msgs
                ]
                rows.append({"messages": msgs})
        if not rows:
            raise ValueError(f"SFT split is empty: {path}")
        return Dataset.from_list(rows)

    train_ds = _load_split("train")
    eval_ds = _load_split("validation")

    # Render chat template AND pre-tokenize. TRL 0.23 + transformers 5.7's
    # SFTTrainer no longer auto-tokenizes a `text` column reliably under
    # Unsloth's wrapper — the default collator ends up trying to pad raw
    # strings. Producing `input_ids` directly sidesteps the dispatch.
    max_seq_length_for_tokenize = training_cfg.get("max_seq_length", 8192)

    def _render_chat(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in batch["messages"]
        ]
        # Gemma-4's `tokenizer` from FastLanguageModel is actually a
        # multimodal Processor whose first positional arg is `images`.
        # Pass `text=` explicitly so we route through the text path on
        # both plain tokenizers and processors.
        # Force padding=False — Gemma4Processor's text path defaults to
        # padding="max_length", which would silently pad every sample to
        # max_seq_length and make subsequent packing impossible.
        encodings = tokenizer(
            text=texts,
            truncation=True,
            max_length=max_seq_length_for_tokenize,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    def _render_chat_response_only(batch: dict[str, Any]) -> dict[str, Any]:
        # Per-turn re-tokenization. Emits a `labels` column with -100 over
        # system/user/tool spans; TRL's default collator preserves any
        # pre-existing `labels` column and only pads with -100.
        out_ids: list[list[int]] = []
        out_attn: list[list[int]] = []
        out_labels: list[list[int]] = []
        for msgs in batch["messages"]:
            sample = render_response_only_sample(
                msgs, tokenizer, max_seq_length_for_tokenize
            )
            out_ids.append(sample["input_ids"])
            out_attn.append(sample["attention_mask"])
            out_labels.append(sample["labels"])
        return {
            "input_ids": out_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
        }

    _render_fn = (
        _render_chat_response_only
        if loss_mask == _LOSS_MASK_RESPONSE_ONLY
        else _render_chat
    )
    logger.info("sft_dataset_render", loss_mask=loss_mask)
    train_ds = train_ds.map(_render_fn, batched=True, remove_columns=["messages"])
    eval_ds = eval_ds.map(_render_fn, batched=True, remove_columns=["messages"])

    # Manual sample packing. Unsloth disables its native packing for
    # vision-language model classes (Gemma-4 here), even though our
    # training is text-only and we already produced clean input_ids.
    # Use First-Fit Decreasing: sort samples by length descending, then
    # place each into the first existing window with room. FFD is the
    # standard SFT-data packing recipe and is significantly tighter than
    # streaming greedy when long and short samples are interleaved.
    if training_cfg.get("packing", False):
        eos_id = (
            getattr(tokenizer, "eos_token_id", None)
            or getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
            or 0
        )

        has_labels = loss_mask == _LOSS_MASK_RESPONSE_ONLY

        def _pack(ds: Dataset, name: str) -> Dataset:
            # Materialize lengths and rows. Iterate via `for ex in ds` so
            # the per-row dict comes through Datasets' standard accessor.
            # Carry an optional `labels` channel for response_only mode; the
            # EOS separator inserted between concatenated samples is masked
            # to -100 so the model never receives gradient on filler tokens.
            samples: list[tuple[list[int], list[int], list[int] | None]] = []
            for ex in ds:
                ids = list(ex["input_ids"])[:max_seq_length_for_tokenize]
                attn = list(ex["attention_mask"])[:max_seq_length_for_tokenize]
                labs = (
                    list(ex["labels"])[:max_seq_length_for_tokenize]
                    if has_labels
                    else None
                )
                samples.append((ids, attn, labs))

            # FFD: longest first, then first-fit into existing bins.
            samples.sort(key=lambda s: -len(s[0]))
            # each bin = [ids_list, attn_list, labels_list_or_None]
            bins: list[list[list[int] | None]] = []
            for ids_l, attn_l, labs_l in samples:
                placed = False
                for b in bins:
                    sep = 1 if b[0] else 0
                    if len(b[0]) + sep + len(ids_l) <= max_seq_length_for_tokenize:
                        if sep:
                            b[0].append(eos_id)
                            b[1].append(1)
                            if has_labels and b[2] is not None:
                                b[2].append(-100)
                        b[0].extend(ids_l)
                        b[1].extend(attn_l)
                        if has_labels and b[2] is not None and labs_l is not None:
                            b[2].extend(labs_l)
                        placed = True
                        break
                if not placed:
                    bins.append(
                        [
                            list(ids_l),
                            list(attn_l),
                            list(labs_l) if (has_labels and labs_l is not None) else None,
                        ]
                    )

            packed_dict: dict[str, list[list[int]]] = {
                "input_ids": [b[0] for b in bins],
                "attention_mask": [b[1] for b in bins],
            }
            if has_labels:
                packed_dict["labels"] = [b[2] for b in bins]  # type: ignore[misc]
            packed = Dataset.from_dict(packed_dict)
            total_in_tokens = sum(len(s[0]) for s in samples)
            total_out_tokens = sum(len(b[0]) for b in bins)
            logger.info(
                "sft_dataset_packed",
                split=name,
                pre=len(ds),
                post=len(packed),
                window=max_seq_length_for_tokenize,
                density=round(len(ds) / max(1, len(packed)), 2),
                util=round(total_in_tokens / max(1, total_out_tokens), 3),
                has_labels=has_labels,
            )
            return packed

        train_ds = _pack(train_ds, "train")
        eval_ds = _pack(eval_ds, "validation")

    # Configure trainer.
    # Per Unsloth's Gemma-4 docs: per_device_train_batch_size=1 with grad
    # accumulation, adamw_8bit, packing intentionally off (Unsloth's example
    # does not enable it). Effective batch size in YAML maps to
    # gradient_accumulation_steps when per_device is fixed at 1.
    from trl import SFTConfig, SFTTrainer

    output_dir = Path("checkpoints") / Path(config_path).stem / Path(model_name).name
    effective_bs = training_cfg.get("effective_batch_size", 8)
    per_device_bs = training_cfg.get("per_device_train_batch_size", 1)
    grad_accum = max(1, effective_bs // max(1, per_device_bs))

    # On the TRL+PEFT path we already enabled gradient checkpointing on the
    # model (Unsloth's path uses its own "unsloth" checkpointing strategy via
    # FastLanguageModel.get_peft_model, so let SFTConfig handle it there).
    gc_in_trainer = (
        training_cfg.get("gradient_checkpointing", True)
        if framework == "unsloth"
        else False
    )

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=training_cfg.get("num_epochs", 3),
        bf16=training_cfg.get("precision", "bf16") != "fp16",
        gradient_checkpointing=gc_in_trainer,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training_cfg.get("optim", "adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.001),
        # max_seq_length was a SFTConfig kwarg through TRL 0.22 and removed
        # from SFTConfig in 0.23+ (it's now controlled by dataset preparation).
        # We already truncate in _render_chat and pre-pack to max_seq_length,
        # so the trainer doesn't need it. Pass via SFTConfig only when
        # supported, otherwise rely on the pre-tokenized lengths.
        **(
            {"max_seq_length": training_cfg.get("max_seq_length", 8192)}
            if "max_seq_length" in inspect.signature(SFTConfig).parameters
            else {}
        ),
        # We pre-pack the dataset ourselves above when training_cfg.packing
        # is True, so the trainer-level packing must be off — otherwise it
        # would either re-pack (under TRL) or be silently ignored (under
        # Unsloth's VL-model guard) while masking what's actually happening.
        packing=False,
        save_strategy="steps",
        save_steps=logging_cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=logging_cfg.get("eval_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model=logging_cfg.get("metric_for_best_model", "eval_loss"),
        report_to="wandb" if logging_cfg.get("wandb_project") else "none",
        seed=training_cfg.get("seed", 3407),
    )

    # Unsloth's SFTTrainer wrapper silently skips sample packing when the
    # passed tokenizer looks like a multimodal Processor ("Sample packing
    # skipped (processor-based model detected)"). For Gemma-4 we did all
    # tokenization ourselves via _render_chat — input_ids already exist
    # in the dataset, so the trainer only uses the tokenizer for padding
    # metadata. Hand it the inner text tokenizer to bypass the guard.
    trainer_tokenizer = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        inner = tokenizer.tokenizer
        if inner is not None:
            logger.info(
                "sft_using_inner_tokenizer",
                outer=type(tokenizer).__name__,
                inner=type(inner).__name__,
                reason="bypass Unsloth packing-skip guard for processor-based models",
            )
            trainer_tokenizer = inner

    # TRL 0.23 renamed `tokenizer` to `processing_class` on SFTTrainer.
    # Detect which kwarg is supported and pass the tokenizer under that name.
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    sft_trainer_params = inspect.signature(SFTTrainer).parameters
    if "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = trainer_tokenizer
    elif "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = trainer_tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # SDPA backend diagnostic. Gemma-4's head_dim=512 forces SDPA (FA2's
    # max is 256). PyTorch picks the best available backend among
    # {flash, mem_efficient, math}. Math is the slow fallback. Logging
    # which backends are *enabled* tells us which SDPA can pick from; the
    # actual selection per-call is opaque, so we also dispatch a single
    # representative forward through `sdpa_kernel([EFFICIENT_ATTENTION])`
    # to verify the mem-efficient path doesn't error out for this model.
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        logger.info(
            "sdpa_backend_availability",
            flash=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),
        )

        # Probe: try a 1-token dummy forward forced to mem_efficient. If this
        # raises, mem_efficient is unsupported for this model's head_dim and
        # SDPA must use math (the slow path).
        try:
            # Multimodal processors (e.g. Gemma4Processor) don't expose
            # eos_token_id directly — fall back to the wrapped tokenizer.
            probe_eos = (
                getattr(tokenizer, "eos_token_id", None)
                or getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
                or 1
            )
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                dummy = torch.tensor([[probe_eos]], device=model.device)
                model(input_ids=dummy)
            logger.info("sdpa_mem_efficient_probe", supported=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "sdpa_mem_efficient_probe",
                supported=False,
                error=str(e)[:200],
                hint="SDPA will fall back to the math kernel — attention is the bottleneck.",
            )
    except ImportError:
        # torch.nn.attention.sdpa_kernel was added in PyTorch 2.5; older
        # versions just skip the probe.
        pass

    # Resolve `resume_from_checkpoint`. `True` → auto-detect latest checkpoint
    # under output_dir; explicit Path/str → pass through; otherwise start fresh.
    resume_arg: bool | str | None
    if resume_from_checkpoint in (None, False):
        resume_arg = None
    elif resume_from_checkpoint is True:
        ckpts = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if ckpts:
            resume_arg = str(ckpts[-1])
            logger.info("sft_resuming", from_checkpoint=resume_arg)
        else:
            logger.warning(
                "sft_resume_requested_but_no_checkpoint", output_dir=str(output_dir)
            )
            resume_arg = None
    else:
        resume_arg = str(resume_from_checkpoint)
        logger.info("sft_resuming", from_checkpoint=resume_arg)

    result = trainer.train(resume_from_checkpoint=resume_arg)
    eval_metrics = trainer.evaluate()

    from llm_workflow_agents.training.lora_targets import get_trainable_param_summary

    return SFTResult(
        checkpoint_path=output_dir,
        best_eval_loss=eval_metrics.get("eval_loss"),
        total_steps=result.global_step,
        metrics={**result.metrics, **eval_metrics},
        param_summary=get_trainable_param_summary(model),
    )
