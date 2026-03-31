"""Graph extraction fine-tuning for Experiment C.

Same LoRA pipeline as train_specialist.py, but with:
  - System prompt instructing JSON graph extraction
  - User = workflow prompt text
  - Assistant = JSON graph (WorkflowGraph schema)
  - At inference: apply Outlines/XGrammar constrained decoding
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.config.loader import load_training_model_config
from llm_workflow_agents.config.schema import TrainingModelConfig
from llm_workflow_agents.training._utils import _build_training_arguments
from llm_workflow_agents.training.train_specialist import (
    TrainingResult,
    _GLM_OOM_FALLBACK_RANK,
    _load_and_prepare_model,
)

logger = structlog.get_logger(__name__)

GRAPH_EXTRACTION_SYSTEM_PROMPT = (
    "You are a workflow graph extraction assistant. Given a natural-language "
    "workflow description, extract a structured JSON graph with the following schema:\n"
    "{\n"
    '  "nodes": [{"id": "S1", "name": "...", "tools": [...], "entry_actions": [...]}],\n'
    '  "edges": [{"from_state": "S1", "to_state": "S2", "condition": "...", "priority": 0}],\n'
    '  "initial_state": "S1",\n'
    '  "terminal_states": ["S3"]\n'
    "}\n"
    "Output ONLY valid JSON. Do not include any explanation."
)


def _load_graph_datasets(config: TrainingModelConfig) -> tuple[Any, Any]:
    """Load graph extraction training data from Exp C output.

    Returns (train_dataset, eval_dataset).
    """
    from datasets import load_dataset

    train_path = Path("data/output/exp_c/train.jsonl")
    val_path = Path("data/output/exp_c/val.jsonl")

    if train_path.exists() and val_path.exists():
        train_ds = load_dataset("json", data_files=str(train_path), split="train")
        val_ds = load_dataset("json", data_files=str(val_path), split="train")
        return train_ds, val_ds

    raise FileNotFoundError(
        "Graph pair data not found at data/output/exp_c/. "
        "Run generate_graph_pairs() first."
    )


def train_graph_extractor(config_path: Path) -> TrainingResult:
    """Fine-tune a model for graph extraction.

    Args:
        config_path: Path to a training model config YAML.

    Returns:
        TrainingResult with checkpoint path and metrics.
    """
    config = load_training_model_config(config_path)
    return train_graph_extractor_from_config(config)


def train_graph_extractor_from_config(
    config: TrainingModelConfig,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Run graph extraction training from an already-loaded config.

    Args:
        config: Loaded TrainingModelConfig.
        output_dir: Override output directory.

    Returns:
        TrainingResult with checkpoint path and metrics.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from llm_workflow_agents.training.lora_targets import get_trainable_param_summary

    if output_dir is None:
        model_short = config.model.name.split("/")[-1].lower().replace("-", "_")
        output_dir = Path(f"checkpoints/{model_short}_graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = TrainingResult()

    # Determine LoRA ranks to try (VRAM fallback for GLM — Risk R7)
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

    # Load graph datasets
    try:
        train_dataset, eval_dataset = _load_graph_datasets(config)
    except FileNotFoundError as exc:
        result.error = str(exc)
        logger.error("dataset_not_found", error=result.error)
        return result

    # Prepend system prompt to every training example so the model learns
    # to produce JSON graphs given the structured extraction instruction.
    def _prepend_system_prompt(example: dict) -> dict:
        if "messages" in example:
            example = dict(example)
            example["messages"] = [
                {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM_PROMPT},
                *example["messages"],
            ]
        return example

    train_dataset = train_dataset.map(_prepend_system_prompt)
    eval_dataset = eval_dataset.map(_prepend_system_prompt)

    # Build training arguments
    training_args_kwargs = _build_training_arguments(config, output_dir)
    training_args = TrainingArguments(**training_args_kwargs)

    # Configure SFTTrainer for graph extraction
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        packing=config.training.packing,
    )

    # Train
    logger.info("starting_graph_extraction_training", output_dir=str(output_dir))
    train_output = trainer.train()

    # Save best checkpoint
    trainer.save_model(str(output_dir / "best"))

    result.checkpoint_path = output_dir / "best"
    result.total_steps = train_output.global_step
    result.metrics = train_output.metrics

    # Final evaluation
    eval_metrics = trainer.evaluate()
    result.metrics.update(eval_metrics)
    result.best_eval_loss = eval_metrics.get("eval_loss")

    logger.info(
        "graph_extraction_training_complete",
        steps=result.total_steps,
        eval_loss=result.best_eval_loss,
        checkpoint=str(result.checkpoint_path),
    )

    return result
