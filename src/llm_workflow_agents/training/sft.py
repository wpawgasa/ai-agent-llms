"""Unsloth SFT entry point for Phase 2 fine-tuning.

Unified for all 3 categories (A, B, C). Category-specific behavior
(data paths, chat templates) is driven by config YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_GLM_OOM_FALLBACK_RANK = 32


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


def _resolve_lora_targets(config: dict[str, Any]) -> list[str]:
    """Resolve LoRA targets from config, falling back to registry."""
    from llm_workflow_agents.training.lora_targets import get_lora_target_spec

    targets = config.get("lora", {}).get("target_modules", "auto")
    if targets == "auto":
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("config_path", "") or model_cfg.get("name", "")
        spec = get_lora_target_spec(model_name)
        return list(spec.target_modules)
    return targets if isinstance(targets, list) else [targets]


def train_sft(config_path: Path) -> SFTResult:
    """Run Unsloth SFT pipeline.

    Pipeline:
      1. Load base model via FastLanguageModel.from_pretrained()
      2. Apply LoRA via FastLanguageModel.get_peft_model()
      3. Configure SFTTrainer with packing + per-model chat template
      4. Train for num_epochs, checkpoint every 500 steps
      5. Select best checkpoint by validation loss
      6. Return SFTResult
    """
    from unsloth import FastLanguageModel

    config = _load_sft_config(config_path)
    lora_cfg = config.get("lora", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    logging_cfg = config.get("logging", {})

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

    logger.info(
        "sft_starting",
        model=model_name,
        precision=training_cfg.get("precision", "bf16"),
        lora_rank=lora_cfg.get("rank", 64),
    )

    lora_rank = lora_cfg.get("rank", 64)
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

    target_modules = _resolve_lora_targets(config)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
    )

    # Freeze router weights if configured
    if lora_cfg.get("freeze_router", False):
        from llm_workflow_agents.training.train_specialist import _freeze_modules

        _freeze_modules(model, ["mlp.gate"])

    # Load dataset
    from datasets import load_dataset

    data_source = data_cfg.get("source", "")
    train_ds = load_dataset("json", data_dir=data_source, split="train")
    eval_ds = load_dataset("json", data_dir=data_source, split="validation")

    # Configure trainer
    from transformers import TrainingArguments
    from trl import SFTTrainer

    output_dir = Path("checkpoints") / Path(config_path).stem
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        per_device_train_batch_size=training_cfg.get("effective_batch_size", 8),
        num_train_epochs=training_cfg.get("num_epochs", 3),
        bf16=training_cfg.get("precision", "bf16") != "fp16",
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=logging_cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=logging_cfg.get("eval_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model=logging_cfg.get("metric_for_best_model", "eval_loss"),
        report_to="wandb" if logging_cfg.get("wandb_project") else "none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        packing=training_cfg.get("packing", True),
        max_seq_length=training_cfg.get("max_seq_length", 8192),
    )

    result = trainer.train()
    eval_metrics = trainer.evaluate()

    from llm_workflow_agents.training.lora_targets import get_trainable_param_summary

    return SFTResult(
        checkpoint_path=output_dir,
        best_eval_loss=eval_metrics.get("eval_loss"),
        total_steps=result.global_step,
        metrics={**result.metrics, **eval_metrics},
        param_summary=get_trainable_param_summary(model),
    )
