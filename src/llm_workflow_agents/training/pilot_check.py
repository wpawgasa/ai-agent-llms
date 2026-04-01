"""100-step pilot SFT on top-2 Phase 1 candidates (Risk R3).

If the #1 candidate shows degradation after 100 steps,
auto-fallback to #2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_DEGRADATION_THRESHOLD = 1.1  # final_loss > initial_loss * 1.1 = degraded


@dataclass(frozen=True)
class PilotResult:
    """Result of a pilot SFT run."""

    model_name: str = ""
    initial_loss: float | None = None
    final_loss: float | None = None
    loss_curve: list[float] = field(default_factory=list)
    degraded: bool = False
    error: str | None = None


def _run_single_pilot(
    model_name: str,
    task_data: Path,
    pilot_steps: int,
) -> PilotResult:
    """Run pilot SFT for a single model.

    Uses Unsloth FastLanguageModel for fast loading + 100-step SFT
    to check if the model responds to fine-tuning.
    """
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="auto",
            use_gradient_checkpointing="unsloth",
        )

        from datasets import load_dataset

        train_ds = load_dataset("json", data_dir=str(task_data), split="train")

        from transformers import TrainingArguments
        from trl import SFTTrainer

        output_dir = Path("checkpoints") / f"pilot_{model_name.replace('/', '_')}"
        args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=pilot_steps,
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_ds,
            max_seq_length=4096,
        )

        result = trainer.train()

        # Extract loss curve from training log
        loss_curve = [
            entry["loss"]
            for entry in trainer.state.log_history
            if "loss" in entry
        ]

        initial_loss = loss_curve[0] if loss_curve else None
        final_loss = loss_curve[-1] if loss_curve else None

        degraded = False
        if initial_loss is not None and final_loss is not None:
            degraded = final_loss > initial_loss * _DEGRADATION_THRESHOLD

        return PilotResult(
            model_name=model_name,
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_curve=loss_curve,
            degraded=degraded,
        )

    except Exception as e:
        logger.error("pilot_failed", model=model_name, error=str(e))
        return PilotResult(model_name=model_name, error=str(e))


def run_pilot_sft(
    top_2_models: list[str],
    task_data: Path,
    pilot_steps: int = 100,
) -> dict[str, PilotResult]:
    """Run 100-step pilot SFT on top-2 candidates.

    Args:
        top_2_models: List of 2 HF model IDs (ranked #1, #2).
        task_data: Path to task data directory.
        pilot_steps: Number of pilot training steps.

    Returns:
        Dict mapping model name to PilotResult.
        If #1 degraded, caller should use #2.
    """
    results: dict[str, PilotResult] = {}
    for model_name in top_2_models:
        logger.info("pilot_starting", model=model_name, steps=pilot_steps)
        result = _run_single_pilot(model_name, task_data, pilot_steps)
        results[model_name] = result
        logger.info(
            "pilot_complete",
            model=model_name,
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            degraded=result.degraded,
        )

    # Log recommendation
    if len(top_2_models) >= 2:
        first = results.get(top_2_models[0])
        if first and first.degraded:
            logger.warning(
                "pilot_degradation_detected",
                model=top_2_models[0],
                fallback=top_2_models[1],
            )

    return results
