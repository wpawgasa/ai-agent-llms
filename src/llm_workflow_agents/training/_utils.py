"""Internal shared utilities for training modules.

Extracted here to avoid importing private functions across sibling modules.
All heavy imports (torch, transformers) are deferred to function bodies.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_workflow_agents.config.schema import TrainingModelConfig


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
