"""Unified SFTTrainer entry point for Experiment B specialist fine-tuning.

Pipeline:
  1. Load base model in BF16
  2. Apply LoRA via PEFT to target modules (from config)
  3. Enable gradient checkpointing
  4. Configure SFTTrainer with packing + chat template
  5. Train with W&B logging
  6. Save best checkpoint by val loss (every 500 steps)
  7. Return TrainingResult with metrics and checkpoint path

All heavy imports (torch, transformers, peft, trl) are deferred to
function bodies so that this module can be imported without GPU access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.config.loader import load_training_model_config
from llm_workflow_agents.config.schema import TrainingModelConfig
from llm_workflow_agents.training._utils import _build_training_arguments
from llm_workflow_agents.training.lora_targets import get_lora_target_spec, get_trainable_param_summary

logger = structlog.get_logger(__name__)

# GLM VRAM fallback rank (R7 risk mitigation) — used when configured rank > this
_GLM_OOM_FALLBACK_RANK = 32


@dataclass
class TrainingResult:
    """Result of a training run."""

    checkpoint_path: Path | None = None
    best_eval_loss: float | None = None
    total_steps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    param_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _build_peft_config(config: TrainingModelConfig) -> dict[str, Any]:
    """Build PEFT LoRA config kwargs from TrainingModelConfig.

    Returns a dict so that the caller handles the actual LoraConfig import.
    """
    lora_spec = get_lora_target_spec(
        model_name=config.model.name,
        model_family=config.model.family,
        explicit_targets=config.lora.target_modules or None,
    )

    peft_kwargs: dict[str, Any] = {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    if lora_spec.target_modules:
        peft_kwargs["target_modules"] = lora_spec.target_modules

    if config.lora.modules_to_save:
        peft_kwargs["modules_to_save"] = config.lora.modules_to_save

    return peft_kwargs


def _freeze_modules(model: Any, patterns: list[str]) -> int:
    """Freeze parameters matching given name patterns. Returns count frozen."""
    frozen = 0
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = False
                frozen += 1
                break
    return frozen


def _load_and_prepare_model(
    config: TrainingModelConfig,
    lora_rank: int | None = None,
) -> tuple[Any, Any]:
    """Load base model and apply LoRA adapter.

    Args:
        config: Training model configuration.
        lora_rank: Override LoRA rank (for VRAM fallback).

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading_base_model", model=config.model.name, precision=config.model.precision)

    torch_dtype = torch.bfloat16 if config.model.precision == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build and apply LoRA
    peft_kwargs = _build_peft_config(config)
    if lora_rank is not None:
        peft_kwargs["r"] = lora_rank
        peft_kwargs["lora_alpha"] = lora_rank * 2  # Maintain alpha/rank ratio

    lora_config = LoraConfig(**peft_kwargs)
    model = get_peft_model(model, lora_config)

    # Freeze specific modules if needed (e.g., MoE router)
    lora_spec = get_lora_target_spec(
        model_name=config.model.name,
        model_family=config.model.family,
    )
    if lora_spec.modules_to_freeze:
        frozen = _freeze_modules(model, lora_spec.modules_to_freeze)
        logger.info("frozen_modules", count=frozen, patterns=lora_spec.modules_to_freeze)

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    param_summary = get_trainable_param_summary(model)
    logger.info("model_prepared", **param_summary)

    return model, tokenizer


def _load_datasets(config: TrainingModelConfig) -> tuple[Any, Any]:
    """Load training and validation datasets.

    Returns (train_dataset, eval_dataset).
    """
    from datasets import load_dataset

    data_config = config.training.training_data
    splits = data_config.splits

    # Try loading from first source
    for source in data_config.sources:
        if source == "custom_synthetic":
            continue
        try:
            ds = load_dataset(source)
            if "train" in ds and "test" in ds:
                return ds["train"], ds["test"]
            if "train" in ds:
                split_ds = ds["train"].train_test_split(
                    test_size=splits.get("val", 0.1),
                    seed=42,
                )
                return split_ds["train"], split_ds["test"]
        except Exception:
            logger.warning("dataset_load_failed", source=source, exc_info=True)
            continue

    # Fallback: try loading from local JSONL
    train_path = Path("data/output/exp_b/train.jsonl")
    val_path = Path("data/output/exp_b/val.jsonl")
    if train_path.exists() and val_path.exists():
        train_ds = load_dataset("json", data_files=str(train_path), split="train")
        val_ds = load_dataset("json", data_files=str(val_path), split="train")
        return train_ds, val_ds

    raise FileNotFoundError(
        "No training data found. Run data generation first or provide valid dataset sources."
    )


def train(config_path: Path) -> TrainingResult:
    """Run the full specialist training pipeline.

    Args:
        config_path: Path to a training model config YAML.

    Returns:
        TrainingResult with checkpoint path, metrics, and parameter summary.
    """
    config = load_training_model_config(config_path)
    return train_from_config(config)


def train_from_config(
    config: TrainingModelConfig,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Run training from an already-loaded config.

    Args:
        config: Loaded TrainingModelConfig.
        output_dir: Override output directory.

    Returns:
        TrainingResult with checkpoint path and metrics.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    if output_dir is None:
        model_short = config.model.name.split("/")[-1].lower().replace("-", "_")
        output_dir = Path(f"checkpoints/{model_short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = TrainingResult()

    # Determine LoRA ranks to try (VRAM fallback for GLM — Risk R7).
    # Always start from the user-configured rank; only append fallback if GLM and rank > fallback.
    is_glm = "glm" in config.model.name.lower()
    if is_glm and config.lora.rank > _GLM_OOM_FALLBACK_RANK:
        logger.warning(
            "glm_oom_fallback_enabled",
            configured_rank=config.lora.rank,
            fallback_rank=_GLM_OOM_FALLBACK_RANK,
            reason="GLM VRAM constraint (Risk R7) — will retry at fallback rank on OOM",
        )
        ranks_to_try = [config.lora.rank, _GLM_OOM_FALLBACK_RANK]
    else:
        ranks_to_try = [config.lora.rank]

    model = None
    tokenizer = None

    for rank in ranks_to_try:
        try:
            logger.info("attempting_training", model=config.model.name, lora_rank=rank)
            model, tokenizer = _load_and_prepare_model(config, lora_rank=rank)
            result.param_summary = get_trainable_param_summary(model)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and rank != ranks_to_try[-1]:
                logger.warning("oom_fallback", rank=rank, next_rank=ranks_to_try[-1])
                import torch
                torch.cuda.empty_cache()
                continue
            result.error = f"OOM even at rank {rank}: {exc}"
            logger.error("training_failed_oom", error=result.error)
            return result

    if model is None:
        result.error = "Failed to load model"
        return result

    # Load datasets
    try:
        train_dataset, eval_dataset = _load_datasets(config)
    except FileNotFoundError as exc:
        result.error = str(exc)
        logger.error("dataset_not_found", error=result.error)
        return result

    # Build training arguments
    training_args_kwargs = _build_training_arguments(config, output_dir)
    training_args = TrainingArguments(**training_args_kwargs)

    # Configure SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        packing=config.training.packing,
    )

    # Train
    logger.info("starting_training", output_dir=str(output_dir))
    train_output = trainer.train()

    # Save best checkpoint
    trainer.save_model(str(output_dir / "best"))

    # Collect results
    result.checkpoint_path = output_dir / "best"
    result.total_steps = train_output.global_step
    result.metrics = train_output.metrics

    # Run final evaluation
    eval_metrics = trainer.evaluate()
    result.metrics.update(eval_metrics)
    result.best_eval_loss = eval_metrics.get("eval_loss")

    logger.info(
        "training_complete",
        steps=result.total_steps,
        eval_loss=result.best_eval_loss,
        checkpoint=str(result.checkpoint_path),
    )

    return result
