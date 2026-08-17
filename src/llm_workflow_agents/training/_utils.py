"""Internal shared utilities for training modules.

Extracted here to avoid importing private functions across sibling modules.
All heavy imports (torch, transformers) are deferred to function bodies.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_workflow_agents.config.schema import TrainingModelConfig


def unwrap_unsloth_gemma4_kv_zero_proxy() -> None:
    """Disarm the Unsloth-Zoo Gemma-4 ``_Gemma4KVSharedSafeProxy`` wrapper.

    Why: unsloth_zoo 2026.5.4's ``patch_Gemma4{,Text}Config_kv_shared_zero``
    wraps ``get_text_config`` so it returns a proxy whose ``__getattr__``
    raises ``AttributeError`` for ``num_kv_shared_layers`` (to trick
    ``hasattr`` checks in transformers' ``cache_utils`` into skipping a
    ``layer_types[:-0] == []`` slice). transformers 5.9.0's
    ``PreTrainedConfig.validate_token_ids`` iterates the text config and
    calls raw ``getattr`` on every attribute — the proxy's raise escapes
    and breaks ``AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")``
    entirely. Both GRPO's ``_detect_model_family`` and Unsloth's own
    ``get_transformers_model_type`` then fail to resolve the base model,
    surfacing as ``TypeError: Unsloth: Cannot determine model type for
    config file: None``.

    Fix: replace the wrapper with one that strips the proxy off the result
    before returning. The companion ``_make_kv_shared_zero_safe_init``
    wrappers on ``DynamicCache.__init__`` / ``StaticCache.__init__`` (same
    unsloth_zoo module) already handle the original ``layer_types[:-0]``
    bug via transient del/restore of the attribute, so dropping the proxy
    does not regress cache construction.

    Shared here (rather than defined per-caller) because it must run before
    *every* Gemma-4 ``FastLanguageModel.from_pretrained`` call in every
    training entry point — GRPO and DPO/ORPO both load Gemma-4 checkpoints
    through it, and this is exactly the "fixed in place instead of shared"
    trap this module's docstring warns about.

    Safe to remove once unsloth_zoo > 2026.5.4 ships a proxy compatible
    with transformers 5.9.0's strict-dataclass validators.
    """
    try:
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4Config,
            Gemma4TextConfig,
        )
    except ImportError:
        return  # transformers without Gemma-4 — nothing to unwrap.

    _sentinel = "_unsloth_gemma4_proxy_unwrapped"

    def _install(cls: type) -> None:
        wrapped = cls.get_text_config
        if getattr(wrapped, _sentinel, False):
            return

        def get_text_config(self, decoder=None, encoder=None):  # noqa: ANN001
            result = wrapped(self, decoder=decoder, encoder=encoder)
            if type(result).__name__ == "_Gemma4KVSharedSafeProxy":
                return object.__getattribute__(result, "_real")
            return result

        setattr(get_text_config, _sentinel, True)
        get_text_config.__qualname__ = wrapped.__qualname__
        get_text_config.__doc__ = wrapped.__doc__
        cls.get_text_config = get_text_config

    _install(Gemma4Config)
    _install(Gemma4TextConfig)


def normalize_chat_template_ids(out: Any) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to ``list[int]``.

    transformers 5.x returns a ``BatchEncoding`` where 4.x returned a plain
    list, and processors may return a batched tensor. This has broken two call
    sites in two different ways — silently in
    ``sft.py::render_response_only_sample`` (``list(mapping)`` yields the KEYS,
    so samples rendered as 2 tokens with 0 unmasked labels and training signal
    vanished without a crash; bac1d98) and loudly in
    ``trajectory_rollout.py::_derive_turn_end_id`` (``int(ids[-1])`` raising on
    a ``tokenizers.Encoding``, which disabled trajectory rollouts entirely).

    The second happened because the first was fixed in place instead of shared.
    Both now call this.

    Order is load-bearing: unwrap the mapping **before** any list coercion,
    because ``list(BatchEncoding)`` succeeds and returns key names rather than
    raising. A mapping without ``input_ids`` raises ``KeyError`` — failing
    loudly beats silently returning something list-shaped and wrong.
    """
    if isinstance(out, Mapping):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(x) for x in out]


def _build_training_arguments(config: TrainingModelConfig, output_dir: Path) -> dict[str, Any]:
    """Build HuggingFace TrainingArguments kwargs from config."""
    micro_batch_size = config.training.effective_batch_size // config.training.gradient_accumulation_steps

    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler_type": config.training.lr_scheduler,
        "warmup_ratio": config.training.warmup_ratio,
        "num_train_epochs": config.training.num_epochs,
        "max_seq_length": config.training.max_seq_length,
        "logging_steps": 10,
        "save_strategy": config.training.save_strategy,
        "save_steps": config.training.save_steps,
        "eval_strategy": "steps",
        "eval_steps": config.training.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.training.metric_for_best_model,
        "greater_is_better": False,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "gradient_checkpointing": config.training.gradient_checkpointing,
    }

    # Precision
    if config.training.mixed_precision == "bf16":
        args_kwargs["bf16"] = True
    elif config.training.mixed_precision == "fp16":
        args_kwargs["fp16"] = True

    # Gradient checkpointing kwargs
    if config.training.gradient_checkpointing:
        args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return args_kwargs
