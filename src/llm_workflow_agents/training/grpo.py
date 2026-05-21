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

# Model families that Unsloth's `fast_inference=True` rejects with a
# RuntimeError from `unsloth/models/vision.py:610` because they are not in
# the hardcoded `VLLM_SUPPORTED_VLM` allowlist (currently qwen2_5_vl,
# gemma3, mistral3, qwen3_vl, qwen3_vl_moe — as of unsloth 2026.5.2).
# When the SFT checkpoint's `config.model_type` matches one of these,
# `train_grpo` auto-falls back to HF `model.generate()` rollouts even if
# the YAML requests `generation_backend: vllm`.
UNSLOTH_VLLM_INCOMPATIBLE_FAMILIES: frozenset[str] = frozenset({
    "gemma4",  # SigLIP + Gemma4 multimodal stack; not in VLLM_SUPPORTED_VLM.
})


def _detect_model_family(sft_checkpoint: str) -> str | None:
    """Return ``config.model_type`` for the SFT checkpoint, or None on failure.

    SFT checkpoints are PEFT adapters (``adapter_config.json`` only, no full
    model ``config.json``), so we first resolve the base model via
    ``base_model_name_or_path`` from the adapter config. Failures are
    non-fatal — we conservatively assume compatibility and let Unsloth
    raise its own error if it knows better.
    """
    try:
        from transformers import AutoConfig

        ckpt_path = Path(sft_checkpoint)
        adapter_cfg_path = ckpt_path / "adapter_config.json"
        if adapter_cfg_path.is_file():
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
            base_model = adapter_cfg.get("base_model_name_or_path", "")
            if not base_model:
                return None
            cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
        else:
            cfg = AutoConfig.from_pretrained(sft_checkpoint, trust_remote_code=False)
        family = (getattr(cfg, "model_type", "") or "").lower()
        return family or None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "vllm_compat_detect_failed",
            sft_checkpoint=sft_checkpoint,
            error=str(exc),
        )
        return None


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


def _slim_content(content: Any) -> str:
    """Coerce a message ``content`` value to a chat-template-renderable string."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _load_grpo_jsonl(data_dir: Path, split: str = "train") -> "Dataset":
    """Load a GRPO split as one (prompt, ground_truth) row per user→assistant turn.

    The synthetic corpus stores full multi-turn conversations (~49 messages
    each). TRL 0.23.1's ``apply_chat_template`` requires the ``prompt`` to
    end on ``user`` or ``assistant`` (``trl/data_utils.py:158``), so we slice
    each conversation at every ``user → assistant`` boundary and emit one
    GRPO row per boundary. Assistant turns preceded by ``tool`` responses
    are skipped (TRL rejects ``tool`` as the last role); this loses signal
    on tool-response continuations but unblocks training without forking TRL.

    Per emitted row:
      - ``prompt``: messages up to and including the user turn, stripped to
        ``{role, content}``. The leading system message is re-enriched via
        ``build_enriched_system_prompt`` so rollouts see the same prompt
        the benchmark sees.
      - ``ground_truth`` (JSON string column to bypass pyarrow schema
        inference; see ``_make_reward_adapter`` for the decode):
        * ``state_sequence`` — the single ``{from, to}`` transition from
          this assistant turn's ``annotations.state_transition``.
        * ``tool_calls`` — the tool calls from this assistant turn's
          ``annotations.tool_calls`` (per-turn, not the whole conversation).
        * ``messages`` — just this assistant message; ``chain_propagation``
          is neutralized for single-turn rows (its score is 1.0 when the
          chain has ≤1 link).
        * ``terminal_state`` / ``terminal_reached`` — propagated from the
          conversation's ground truth, but ``terminal_reached`` is True
          only on the FINAL emitted row from a conversation that originally
          reached its terminal state. Non-terminal rows have
          ``terminal_reached=False`` so the reward correctly skips the
          completion sub-reward (see ``reward_business_logic.py:72``).
    """
    from datasets import Dataset

    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GRPO split missing: {path}")

    rows: list[dict[str, Any]] = []
    n_convs = 0
    n_skipped_tool_preceded = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            n_convs += 1
            raw_msgs = raw.get("messages") or []
            if (
                raw_msgs
                and raw_msgs[0].get("role") == "system"
                and raw.get("workflow_graph")
            ):
                raw_msgs = [
                    {
                        "role": "system",
                        "content": build_enriched_system_prompt(
                            raw, raw_msgs[0].get("content") or "", force_rebuild=True
                        ),
                    },
                    *raw_msgs[1:],
                ]

            gt_full = raw.get("ground_truth") or {}
            terminal_state = gt_full.get("terminal_state", "") or ""
            terminal_reached_overall = bool(gt_full.get("terminal_reached", True))

            # Legal-edge set for the transition_legality reward component.
            # Sourced from ground truth (not the prompt) so it stays correct
            # even when a late-turn prompt is truncated at max_prompt_length.
            wf_graph = raw.get("workflow_graph") or {}
            if isinstance(wf_graph, str):
                try:
                    wf_graph = json.loads(wf_graph)
                except (ValueError, TypeError):
                    wf_graph = {}
            valid_transitions = [
                [t.get("from", ""), t.get("to", "")]
                for t in wf_graph.get("transitions", [])
                if isinstance(t, dict)
            ]

            asst_indices = [
                i for i, m in enumerate(raw_msgs) if m.get("role") == "assistant"
            ]
            valid_pairs = [
                i for i in asst_indices
                if i > 0 and raw_msgs[i - 1].get("role") == "user"
            ]
            n_skipped_tool_preceded += len(asst_indices) - len(valid_pairs)

            for j, asst_idx in enumerate(valid_pairs):
                prompt = [
                    {
                        "role": m.get("role", "") or "",
                        "content": _slim_content(m.get("content")),
                    }
                    for m in raw_msgs[:asst_idx]
                ]
                asst_msg = raw_msgs[asst_idx]
                ann = asst_msg.get("annotations") or {}
                state_trans = ann.get("state_transition") or {}
                state_seq = [state_trans] if state_trans else []
                tool_calls = ann.get("tool_calls") or []

                is_terminal_row = (
                    j == len(valid_pairs) - 1 and terminal_reached_overall
                )
                row_gt = {
                    "state_sequence": state_seq,
                    "tool_calls": tool_calls,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": _slim_content(asst_msg.get("content")),
                        }
                    ],
                    "terminal_state": terminal_state,
                    "terminal_reached": is_terminal_row,
                    "valid_transitions": valid_transitions,
                }
                rows.append(
                    {
                        "prompt": prompt,
                        "ground_truth": json.dumps(
                            row_gt, ensure_ascii=False, default=str
                        ),
                    }
                )

    if not rows:
        raise ValueError(f"GRPO split is empty after slicing: {path}")
    logger.info(
        "grpo_data_loaded",
        split=split,
        conversations=n_convs,
        rows=len(rows),
        skipped_tool_preceded_turns=n_skipped_tool_preceded,
    )
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

    # Generation backend — "vllm" enables Unsloth's colocate vLLM engine
    # (shares weights with the training model; no second copy of the 26B+
    # checkpoint). Any other value falls back to HF model.generate().
    gen_backend = str(grpo_cfg.get("generation_backend", "hf")).lower()
    vllm_requested = gen_backend == "vllm"
    vllm_gpu_util = float(grpo_cfg.get("vllm_gpu_memory_utilization", 0.55))
    max_lora_rank = int(config.get("lora", {}).get("rank", 64))

    # Some model families are rejected by Unsloth's `fast_inference` check
    # (unsloth/models/vision.py:610). For those, silently fall back to HF
    # rollouts with a warning so training still runs.
    family = _detect_model_family(sft_checkpoint) if vllm_requested else None
    use_vllm = vllm_requested
    if vllm_requested and family in UNSLOTH_VLLM_INCOMPATIBLE_FAMILIES:
        use_vllm = False
        logger.warning(
            "vllm_rollout_disabled_unsloth_incompat",
            model_family=family,
            yaml_setting=gen_backend,
            effective_backend="hf",
            note=(
                "Unsloth fast_inference does not support this model family "
                "(see unsloth.models.vision.VLLM_SUPPORTED_VLM allowlist). "
                "Falling back to HF model.generate() for rollouts. Step time "
                "will be significantly slower until Unsloth adds support or "
                "this run is switched to a supported family (qwen2_5_vl, "
                "gemma3, mistral3, qwen3_vl, qwen3_vl_moe)."
            ),
        )

    logger.info(
        "grpo_starting",
        sft_checkpoint=sft_checkpoint,
        reward_function=reward_fn_name,
        training_steps=grpo_cfg.get("training_steps", 1000),
        beta=grpo_cfg.get("beta", 0.04),
        generation_backend="vllm" if use_vllm else "hf",
        model_family=family,
        vllm_gpu_memory_utilization=vllm_gpu_util if use_vllm else None,
        max_lora_rank=max_lora_rank if use_vllm else None,
    )

    fast_inference_kwargs: dict[str, Any] = {}
    if use_vllm:
        fast_inference_kwargs = {
            "fast_inference": True,
            "gpu_memory_utilization": vllm_gpu_util,
            "max_lora_rank": max_lora_rank,
        }

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_checkpoint,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
        **fast_inference_kwargs,
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

    sampling_cfg = grpo_cfg.get("sampling", {}) or {}
    grpo_kwargs: dict[str, Any] = dict(
        output_dir=f"checkpoints/{Path(config_path).stem}/{model_basename}",
        num_generations=grpo_cfg.get("num_generations", 8),
        max_steps=grpo_cfg.get("training_steps", 1000),
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        beta=grpo_cfg.get("beta", 0.04),
        # Sampling diversity — higher temperature widens the group of N
        # generations so they don't collapse to identical completions
        # (the root cause of frac_reward_zero_std≈1 in the first run).
        temperature=float(sampling_cfg.get("temperature", 1.0)),
        top_p=float(sampling_cfg.get("top_p", 0.95)),
        # Short warmup — default behavior reached peak LR only at ~step 750
        # of 1000, leaving the policy almost untrained. 5% warmup hits peak
        # by ~step 50.
        warmup_ratio=float(grpo_cfg.get("warmup_ratio", 0.05)),
        # Checkpoint cadence — default `save_steps=500` was too sparse for
        # resumability (a killed run lost everything below optimizer step
        # 500). 100 gives a ~30-min safety net; cap retention to 3 to bound
        # disk usage at ~5 GB for the Gemma-4 26B QLoRA adapter sizes.
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        save_total_limit=int(grpo_cfg.get("save_total_limit", 3)),
        report_to="wandb",
    )
    # Optional GRPOConfig kwargs — only set when present in YAML so existing
    # configs that don't specify them keep TRL's defaults. The diagnosis
    # doc (docs/grpo_diagnosis_gemma4_26b.md) recommends:
    #   loss_type=grpo (over TRL's dapo default; BNPO was too sensitive in
    #     the killed run with a low-entropy ref policy)
    #   max_completion_length=512 (TRL default 256 caused 16% truncation rate)
    #   log_completions=true / num_completions_to_print=4 (sample groups land
    #     in W&B alongside frac_reward_zero_std — load-bearing for the
    #     50-step diagnostic).
    for key in (
        "loss_type",
        "max_completion_length",
        "max_prompt_length",
        "log_completions",
        "num_completions_to_print",
    ):
        if key in grpo_cfg:
            grpo_kwargs[key] = grpo_cfg[key]
    if use_vllm:
        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=vllm_gpu_util,
            vllm_tensor_parallel_size=1,
            vllm_importance_sampling_correction=True,
        )
    grpo_config = GRPOConfig(**grpo_kwargs)

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

    # Auto-resume from the highest-numbered checkpoint in output_dir if one
    # exists. GRPOTrainer inherits transformers.Trainer.train(), which loads
    # optimizer.pt + scheduler.pt + trainer_state.json from the checkpoint
    # — warmup picks up where it left off, LR resumes mid-curve. Set
    # WANDB_RESUME=allow and WANDB_RUN_ID=<previous-id> in the environment
    # to continue the same W&B run; otherwise a fresh run is started.
    ckpt_dir = Path(grpo_config.output_dir)
    resume_from: str | None = None
    if ckpt_dir.is_dir():
        existing = sorted(
            (p for p in ckpt_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if existing:
            resume_from = str(existing[-1])
            logger.info("grpo_resuming", from_checkpoint=resume_from)
        else:
            logger.info("grpo_starting_fresh", output_dir=str(ckpt_dir))

    result = trainer.train(resume_from_checkpoint=resume_from)

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
