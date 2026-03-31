"""Internal shared utilities for training modules.

Extracted here to avoid importing private functions across sibling modules.
All heavy imports (torch, transformers) are deferred to function bodies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_workflow_agents.config.schema import TrainingModelConfig


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
