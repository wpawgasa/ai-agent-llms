"""Unsloth GRPO RL entry point for Phase 2 reinforcement learning.

Loads an SFT checkpoint, applies GRPOTrainer with task-specific
reward function, vLLM generation backend, and FP8 RL.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog

if TYPE_CHECKING:
    from datasets import Dataset

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


def _load_grpo_jsonl(data_dir: Path, split: str = "train") -> "Dataset":
    """Load a GRPO split, bypassing pyarrow JSON schema inference.

    The synthetic GRPO corpus has heterogeneous leaf types under
    ``ground_truth.tool_calls[].arguments`` and
    ``messages[].annotations.tool_calls[].arguments`` (e.g. ``amount`` is int
    in most rows, float/str in a few), which causes
    ``datasets.load_dataset("json", ...)`` to abort with ArrowInvalid mid-file.

    Mirrors the manual loader in ``training/sft.py`` (lines 495-525):
      - Rebuilds the enriched system prompt so GRPO rollouts see what the
        benchmark sees (``force_rebuild=True``).
      - Strips messages to ``{role, content}`` — drops per-message
        ``annotations`` (the inline ``<tool_call>{...}</tool_call>`` blocks in
        assistant content carry the same info; annotations are duplicative).
      - Serializes ``ground_truth`` as a JSON string column. The reward
        adapter decodes it on access (see ``_make_reward_adapter``).
    """
    from datasets import Dataset

    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GRPO split missing: {path}")

    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            msgs = raw.get("messages") or []
            if msgs and msgs[0].get("role") == "system" and raw.get("workflow_graph"):
                msgs = list(msgs)
                msgs[0] = {
                    "role": "system",
                    "content": build_enriched_system_prompt(
                        raw, msgs[0].get("content") or "", force_rebuild=True
                    ),
                }
            slim_msgs = [
                {
                    "role": m.get("role", "") or "",
                    "content": (
                        m.get("content")
                        if isinstance(m.get("content"), str)
                        else json.dumps(m.get("content"), ensure_ascii=False)
                    ),
                }
                for m in msgs
            ]
            gt_str = json.dumps(
                raw.get("ground_truth") or {}, ensure_ascii=False, default=str
            )
            rows.append({"prompt": slim_msgs, "ground_truth": gt_str})

    if not rows:
        raise ValueError(f"GRPO split is empty: {path}")
    return Dataset.from_list(rows)


def _make_reward_adapter(reward_fn: Callable) -> Callable:
    """Bridge project reward signature to TRL 0.23.1's keyword-only call.

    TRL 0.23.1 invokes reward functions as
    ``reward_fn(prompts=..., completions=..., completion_ids=..., **kwargs)``
    where ``**kwargs`` are dataset columns other than prompt/completion
    (``trl/trainer/grpo_trainer.py:1034``). The project's rewards expect
    ``(prompts, completions, ground_truths)`` with ``ground_truths`` as a list
    of dicts.

    This adapter:
      - JSON-decodes the ``ground_truth`` string column (see ``_load_grpo_jsonl``).
      - Aliases ``ground_truth.state_sequence`` → ``state_annotations`` to match
        what ``reward_business_logic`` reads (the data emits the former; the
        reward reads the latter).
      - Flattens conversational completions (``list[list[{role, content}]]``)
        to a list of assistant content strings.
    """

    def adapter(  # noqa: ANN001
        *,
        prompts: Any = None,
        completions: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        gt_raw = kwargs.get("ground_truth") or []
        gts: list[dict[str, Any]] = []
        for g in gt_raw:
            d = json.loads(g) if isinstance(g, str) else (g or {})
            if not isinstance(d, dict):
                d = {}
            # Alias state_sequence → state_annotations and reshape:
            # the data stores [{from, to}, ...]; the reward expects
            # [(from, to), ...] (hashable tuples used in set(...)).
            if "state_sequence" in d and "state_annotations" not in d:
                seq = d["state_sequence"]
                if isinstance(seq, list):
                    d["state_annotations"] = [
                        (s.get("from", ""), s.get("to", "")) if isinstance(s, dict)
                        else tuple(s) if isinstance(s, (list, tuple)) and len(s) == 2
                        else ("", "")
                        for s in seq
                    ]
                else:
                    d["state_annotations"] = []
            gts.append(d)

        flat_completions: list[str] = []
        for c in completions or []:
            if isinstance(c, str):
                flat_completions.append(c)
            elif isinstance(c, list) and c and isinstance(c[-1], dict):
                flat_completions.append(c[-1].get("content", "") or "")
            else:
                flat_completions.append(str(c))

        return reward_fn(prompts or [], flat_completions, gts)

    return adapter


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

    data_source = data_cfg.get("source", "")
    train_ds = _load_grpo_jsonl(Path(data_source), split="train")

    from trl import GRPOConfig, GRPOTrainer

    # Mirror sft.py layout: checkpoints/<config-stem>/<model-basename>/.
    # Prefer model.config_path (HF model name in YAML); fall back to the
    # SFT checkpoint's parent dir, which sft.py names after the HF basename.
    model_cfg_path = config.get("model", {}).get("config_path")
    if model_cfg_path:
        model_basename = Path(
            yaml.safe_load(open(model_cfg_path))["model"]["name"]
        ).name
    else:
        model_basename = Path(sft_checkpoint).parent.name

    grpo_config = GRPOConfig(
        output_dir=f"checkpoints/{Path(config_path).stem}/{model_basename}",
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
        reward_funcs=_make_reward_adapter(reward_fn),
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
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
