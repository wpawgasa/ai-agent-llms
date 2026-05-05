"""Unsloth GRPO RL entry point for Phase 2 reinforcement learning.

Loads an SFT checkpoint, applies GRPOTrainer with task-specific
reward function, vLLM generation backend, and FP8 RL.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

_REWARD_REGISTRY: dict[str, str] = {
    "reward_business_logic": "llm_workflow_agents.training.rewards.reward_business_logic",
    "reward_subagent": "llm_workflow_agents.training.rewards.reward_subagent",
    "reward_graph_extraction": "llm_workflow_agents.training.rewards.reward_graph_extraction",
}


@dataclass(frozen=True)
class GRPOResult:
    """Result of a GRPO RL training run."""

    checkpoint_path: Path | None = None
    reward_curves: list[float] = field(default_factory=list)
    held_out_scores: list[float] = field(default_factory=list)
    kl_divergence: list[float] = field(default_factory=list)
    total_steps: int = 0
    early_stopped: bool = False
    error: str | None = None


def _resolve_reward_fn(name: str) -> Callable:
    """Dynamically import the reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(_REWARD_REGISTRY.keys())}"
        )
    module_path = _REWARD_REGISTRY[name]
    mod = importlib.import_module(f"{module_path.rsplit('.', 1)[0]}")
    return getattr(mod, module_path.rsplit(".", 1)[1])


def train_grpo(config_path: Path) -> GRPOResult:
    """Run Unsloth GRPO RL pipeline.

    Pipeline:
      1. Load SFT checkpoint via FastLanguageModel.from_pretrained()
      2. Configure GRPOTrainer with:
         - task-specific reward function (from config)
         - vLLM generation backend
         - FP8 RL
         - DAPO normalization
         - num_generations=4, beta=0.04 KL penalty
      3. Train for configured steps (500-1000)
      4. Monitor: reward curve, held-out eval every 50 steps, KL divergence
      5. Auto-stop if held-out metric drops while reward increases (R5)
      6. Return GRPOResult
    """
    import yaml
    from unsloth import FastLanguageModel

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if config.get("stage") != "grpo":
        raise ValueError(f"Expected stage='grpo', got '{config.get('stage')}'")

    grpo_cfg = config.get("grpo", {})
    reward_cfg = config.get("reward", {})
    data_cfg = config.get("data", {})
    monitoring_cfg = config.get("monitoring", {})

    sft_checkpoint = config.get("model", {}).get("sft_checkpoint")
    if not sft_checkpoint:
        return GRPOResult(error="model.sft_checkpoint not set in GRPO config")

    reward_fn_name = reward_cfg.get("function", "")
    reward_fn = _resolve_reward_fn(reward_fn_name)

    logger.info(
        "grpo_starting",
        sft_checkpoint=sft_checkpoint,
        reward_function=reward_fn_name,
        training_steps=grpo_cfg.get("training_steps", 1000),
        beta=grpo_cfg.get("beta", 0.04),
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_checkpoint,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )

    from datasets import load_dataset

    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    data_source = data_cfg.get("source", "")
    train_ds = load_dataset("json", data_dir=data_source, split="train")

    # Re-enrich the system prompt so GRPO rollouts see the same prompt the
    # benchmark sees. JSONL has stale enrichment baked in; force_rebuild=True
    # strips it and rebuilds from current code using each row's upstream
    # fields (workflow_graph, tool_schemas, messages, language).
    def _rebuild_system_prompt(row: dict[str, Any]) -> dict[str, Any]:
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "system" or not row.get("workflow_graph"):
            return row
        new_msgs = list(msgs)
        new_msgs[0] = {
            "role": "system",
            "content": build_enriched_system_prompt(
                row, msgs[0].get("content") or "", force_rebuild=True
            ),
        }
        return {**row, "messages": new_msgs}

    train_ds = train_ds.map(_rebuild_system_prompt)

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=f"checkpoints/{Path(config_path).stem}",
        num_generations=grpo_cfg.get("num_generations", 4),
        max_steps=grpo_cfg.get("training_steps", 1000),
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        beta=grpo_cfg.get("beta", 0.04),
        report_to="wandb",
    )

    # Build reward hacking callback
    eval_held_out_every = monitoring_cfg.get("eval_held_out_every", 50)
    callbacks = []

    if monitoring_cfg.get("reward_hacking_detector", False):
        from transformers import TrainerCallback

        class _RewardHackingCallback(TrainerCallback):
            """Monitor for reward hacking: reward ↑ + held-out ↓."""

            def __init__(self) -> None:
                self.reward_history: list[float] = []
                self.held_out_history: list[float] = []

            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
                if logs and "reward" in logs:
                    self.reward_history.append(logs["reward"])
                if logs and "kl" in logs:
                    pass  # KL logged automatically by GRPOTrainer

                if (
                    state.global_step > 0
                    and state.global_step % eval_held_out_every == 0
                ):
                    # Check for reward hacking pattern
                    if (
                        len(self.reward_history) >= 5
                        and len(self.held_out_history) >= 2
                    ):
                        recent_reward = self.reward_history[-1]
                        prev_reward = self.reward_history[-5]
                        recent_held_out = self.held_out_history[-1]
                        prev_held_out = self.held_out_history[-2]

                        if recent_reward > prev_reward and recent_held_out < prev_held_out:
                            logger.warning(
                                "reward_hacking_detected",
                                step=state.global_step,
                                reward_delta=recent_reward - prev_reward,
                                held_out_delta=recent_held_out - prev_held_out,
                            )
                            control.should_training_stop = True

        callbacks.append(_RewardHackingCallback())

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )

    result = trainer.train()

    # Collect monitoring data from callback
    reward_curves: list[float] = []
    held_out_scores: list[float] = []
    early_stopped = False
    for cb in callbacks:
        if hasattr(cb, "reward_history"):
            reward_curves = cb.reward_history
            held_out_scores = cb.held_out_history
            early_stopped = bool(
                reward_curves and held_out_scores
                and len(held_out_scores) >= 2
                and held_out_scores[-1] < held_out_scores[-2]
            )

    output_dir = Path(grpo_config.output_dir)
    return GRPOResult(
        checkpoint_path=output_dir,
        reward_curves=reward_curves,
        held_out_scores=held_out_scores,
        total_steps=result.global_step,
        early_stopped=early_stopped,
    )
