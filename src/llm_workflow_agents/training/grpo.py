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


def _unwrap_unsloth_gemma4_kv_zero_proxy() -> None:
    """Disarm the Unsloth-Zoo Gemma-4 ``_Gemma4KVSharedSafeProxy`` wrapper.

    Why: unsloth_zoo 2026.5.4's ``patch_Gemma4{,Text}Config_kv_shared_zero``
    wraps ``get_text_config`` so it returns a proxy whose ``__getattr__``
    raises ``AttributeError`` for ``num_kv_shared_layers`` (to trick
    ``hasattr`` checks in transformers' ``cache_utils`` into skipping a
    ``layer_types[:-0] == []`` slice). transformers 5.9.0's
    ``PreTrainedConfig.validate_token_ids`` iterates the text config and
    calls raw ``getattr`` on every attribute — the proxy's raise escapes
    and breaks ``AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")``
    entirely. Both ``_detect_model_family`` (this file) and Unsloth's own
    ``get_transformers_model_type`` then fail to resolve the base model,
    surfacing as ``TypeError: Unsloth: Cannot determine model type for
    config file: None``.

    Fix: replace the wrapper with one that strips the proxy off the result
    before returning. The companion ``_make_kv_shared_zero_safe_init``
    wrappers on ``DynamicCache.__init__`` / ``StaticCache.__init__`` (same
    unsloth_zoo module) already handle the original ``layer_types[:-0]``
    bug via transient del/restore of the attribute, so dropping the proxy
    does not regress cache construction.

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
                if i > 0 and raw_msgs[i - 1].get("role") in ("user", "system")
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


# Latest-step instrumentation stashed by _make_reward_adapter; consumed by
# _UniqueCompletionsCallback on the next on_log. TRL 0.23.1 logs `entropy`,
# `reward_std`, `frac_reward_zero_std`, `kl`, `completions/mean_length`, etc.
# natively (trl/trainer/grpo_trainer.py around line 1500–1730), so this
# module only adds `unique_completions_per_group` — the metric that would
# have surfaced the 2026-05-25 5a5w4jqr stub-attractor drift at step ~10
# instead of step 50. See docs/grpo_diagnosis_gemma4_26b.md.
_LATEST_INSTRUMENTATION: dict[str, float] = {}


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

        # Stash unique-completions-per-group for the next on_log call.
        # GRPO batches K rollouts per prompt; we group by the prompt content
        # (str-coerce to handle list[dict] chat-formatted prompts) and count
        # unique completion text per group, then average.
        if flat_completions and prompts is not None:
            groups: dict[str, list[str]] = {}
            for p, c in zip(prompts, flat_completions):
                key = str(p)[:512]
                groups.setdefault(key, []).append(c.strip())
            if groups:
                uniques = [len(set(cs)) for cs in groups.values()]
                sizes = [len(cs) for cs in groups.values()]
                _LATEST_INSTRUMENTATION["unique_completions_per_group"] = (
                    sum(uniques) / len(uniques)
                )
                _LATEST_INSTRUMENTATION["group_size"] = sum(sizes) / len(sizes)

        return reward_fn(prompts or [], flat_completions, gts)

    return adapter


def _heldout_composite_score(
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> float:
    """Deployment-aligned held-out quality score, computed with STRICT metrics.

    Mirrors ``eval.composite_score.compute_weighted_workflow_score``:
    ``0.4 * state_transition_acc + 0.4 * strict_tool_f1 + 0.2 * task_completion``,
    averaged over the held-out rows.

    Deliberately uses the *strict* scorers (``state_sequence_match``,
    ``tool_call_f1`` = ``compute_ast_f1``, ``reached_terminal``) rather than the
    graded components the training reward optimizes (``graded_tool_call_f1``,
    ``_graded_state_match``, ``transition_legality``). Keeping the held-out metric
    numerically independent of the training reward is what makes a reward-vs-
    quality divergence (reward hacking, Risk R5) detectable — see
    docs/grpo_diagnosis_gemma4_26b.md. Pure/CPU-only and unit-tested.
    """
    from llm_workflow_agents.training.reward_utils import (
        extract_state_annotations,
        extract_tool_calls,
        reached_terminal,
        state_sequence_match,
        tool_call_f1,
    )

    if not completions:
        return 0.0

    scores: list[float] = []
    for comp, gt in zip(completions, ground_truths):
        gt = gt or {}
        gt_seq = gt.get("state_sequence") or []
        gt_trans = [
            (s.get("from", ""), s.get("to", ""))
            if isinstance(s, dict)
            else tuple(s)
            if isinstance(s, (list, tuple)) and len(s) == 2
            else ("", "")
            for s in gt_seq
        ]
        pred_trans = extract_state_annotations(comp)
        if gt_trans:
            state_acc = state_sequence_match(pred_trans, gt_trans)
        else:
            state_acc = 1.0 if not pred_trans else 0.0

        tool_f1 = tool_call_f1(extract_tool_calls(comp), gt.get("tool_calls") or [])

        terminal = gt.get("terminal_state") or ""
        task = 1.0 if terminal and reached_terminal(comp, terminal) else 0.0

        scores.append(0.4 * state_acc + 0.4 * tool_f1 + 0.2 * task)

    return sum(scores) / len(scores)


def _is_reward_hacking(
    reward_history: list[float],
    held_out_history: list[float],
    lookback: int = 5,
) -> bool:
    """Reward-hacking test: training reward rising while held-out quality falls.

    Returns True only when there is enough history AND the latest training
    reward is above the reward ``lookback`` logs ago AND the latest held-out
    composite is below the previous one. Pure/unit-tested — the callback wires
    it to ``control.should_training_stop``.
    """
    if len(reward_history) < lookback or len(held_out_history) < 2:
        return False
    return (
        reward_history[-1] > reward_history[-lookback]
        and held_out_history[-1] < held_out_history[-2]
    )


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

    # Importing unsloth installs unsloth_zoo's Gemma-4 KV-zero proxy on
    # ``Gemma4{,Text}Config.get_text_config``. That proxy breaks
    # ``AutoConfig.from_pretrained`` for Gemma-4 26B-A4B / 31B under
    # transformers 5.9.0 — both ``_detect_model_family`` below and Unsloth's
    # own loader rely on it. Disarm immediately.
    _unwrap_unsloth_gemma4_kv_zero_proxy()

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

    # Held-out subset for the R5 reward-hacking guardrail. Loaded once; the
    # callback generates greedy completions on these prompts every
    # ``eval_held_out_every`` steps and scores them with an independent
    # composite metric (see _HeldOutEvalCallback / _heldout_composite_score).
    held_out_rows: list[dict[str, Any]] = []
    if monitoring_cfg.get("reward_hacking_detector", False):
        n_held_out = int(monitoring_cfg.get("eval_held_out_num_prompts", 50))
        try:
            val_ds = _load_grpo_jsonl(Path(data_source), split="validation")
            held_out_rows = [val_ds[i] for i in range(min(n_held_out, len(val_ds)))]
            logger.info("grpo_heldout_loaded", n_prompts=len(held_out_rows))
        except FileNotFoundError:
            logger.warning(
                "grpo_heldout_split_missing",
                note="validation split not found; held-out eval disabled",
                data_source=data_source,
            )

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
    #   loss_type=dr_grpo (TRL's "grpo" carries a documented short-completion
    #     length bias that drove the df4dot2d 211→29-token collapse; "dr_grpo"
    #     is length-bias-free — see the 2026-05-29 re-audit)
    #   max_completion_length=512 (TRL default 256 caused 16% truncation rate)
    #   log_completions=true / num_completions_to_print=4 (sample groups land
    #     in W&B alongside frac_reward_zero_std — load-bearing for the
    #     50-step diagnostic).
    #   scale_rewards=none — STOP dividing advantages by the per-group reward
    #     std. TRL's default "group" divided by std≈0.003–0.04 on near-constant
    #     groups, amplifying advantages 60–1060× → grad-norm 1126, KL 40 in
    #     df4dot2d. This is the primary instability fix (2026-05-29 re-audit).
    #   max_grad_norm=0.2 — explicit tight gradient clip (TRL default 1.0 left
    #     the clipped direction dominated by exploding components).
    #   generation_batch_size — set >num_generations to get >1 unique prompt
    #     per step (df4dot2d ran 8/8 = 1 prompt/step → high prompt-draw noise).
    for key in (
        "loss_type",
        "max_completion_length",
        "max_prompt_length",
        "log_completions",
        "num_completions_to_print",
        "scale_rewards",
        "max_grad_norm",
        "generation_batch_size",
        "epsilon",
        # Batch geometry — explicit so the TRL divisibility constraints are
        # satisfied deterministically rather than relying on defaults. TRL
        # 0.23.1 requires generation_batch_size % (per_device_train_batch_size
        # * num_processes) == 0 and generation_batch_size % num_generations
        # == 0; steps_per_generation is then derived. See grpo_config.py
        # __post_init__ (lines 882-918).
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
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

    # Always-on: surface unique_completions_per_group in W&B / TRL logs.
    # TRL 0.23.1 already logs `entropy`, `reward_std`, `frac_reward_zero_std`
    # natively; this adds the one metric that would have flagged the
    # 2026-05-25 stub-attractor drift well before step 50.
    from transformers import TrainerCallback

    class _UniqueCompletionsCallback(TrainerCallback):
        """Inject unique_completions_per_group into the standard log dict.

        The reward adapter stashes the latest batch's value on
        ``_LATEST_INSTRUMENTATION``; this callback copies it onto every
        ``on_log`` event so the W&B integration picks it up alongside TRL's
        native metrics. No direct ``wandb.log`` calls — transformers'
        logger forwarder handles fan-out.
        """

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if logs is None:
                return
            for key in ("unique_completions_per_group", "group_size"):
                if key in _LATEST_INSTRUMENTATION:
                    logs[f"train/{key}"] = _LATEST_INSTRUMENTATION[key]

    callbacks.append(_UniqueCompletionsCallback())

    if monitoring_cfg.get("reward_hacking_detector", False):
        held_out_max_new = int(grpo_cfg.get("max_completion_length", 512))

        class _HeldOutEvalCallback(TrainerCallback):
            """Real held-out quality guardrail (Risk R5).

            Every ``eval_held_out_every`` steps, greedily generates completions
            on a fixed held-out prompt subset, scores them with the independent
            ``_heldout_composite_score`` (strict metrics, distinct from the
            graded training reward), logs ``eval/held_out_composite``, and stops
            training when ``_is_reward_hacking`` fires (train reward ↑ while
            held-out quality ↓). Replaces the previous stub whose
            ``held_out_history`` was never populated, so the auto-stop could
            never trigger.
            """

            def __init__(self, model, tokenizer, rows) -> None:  # noqa: ANN001
                self.model = model
                self.tokenizer = tokenizer
                self.rows = rows
                self.reward_history: list[float] = []
                self.held_out_history: list[float] = []

            def _evaluate(self) -> float | None:
                if not self.rows:
                    return None
                import torch

                tok = self.tokenizer
                model = self.model
                was_training = model.training
                model.eval()
                completions: list[str] = []
                gts: list[dict[str, Any]] = []
                try:
                    with torch.no_grad():
                        for row in self.rows:
                            text = tok.apply_chat_template(
                                row["prompt"],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            enc = tok(
                                text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=7680,
                            ).to(model.device)
                            out = model.generate(
                                **enc,
                                max_new_tokens=held_out_max_new,
                                do_sample=False,
                            )
                            gen = tok.decode(
                                out[0][enc["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                            )
                            completions.append(gen)
                            gt_raw = row.get("ground_truth")
                            gts.append(
                                json.loads(gt_raw)
                                if isinstance(gt_raw, str)
                                else (gt_raw or {})
                            )
                except Exception as exc:  # noqa: BLE001 — never kill training on eval
                    logger.warning("grpo_heldout_eval_failed", error=str(exc))
                    return None
                finally:
                    if was_training:
                        model.train()
                return _heldout_composite_score(completions, gts)

            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
                if logs and "reward" in logs:
                    self.reward_history.append(logs["reward"])
                    if self.held_out_history:
                        logs["eval/held_out_composite"] = self.held_out_history[-1]

            def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
                if (
                    state.global_step > 0
                    and state.global_step % eval_held_out_every == 0
                ):
                    score = self._evaluate()
                    if score is None:
                        return
                    self.held_out_history.append(score)
                    logger.info(
                        "grpo_heldout_eval",
                        step=state.global_step,
                        held_out_composite=score,
                    )
                    if _is_reward_hacking(self.reward_history, self.held_out_history):
                        logger.warning(
                            "reward_hacking_detected",
                            step=state.global_step,
                            reward_recent=self.reward_history[-1],
                            reward_prev=self.reward_history[-5],
                            held_out_recent=self.held_out_history[-1],
                            held_out_prev=self.held_out_history[-2],
                        )
                        control.should_training_stop = True

        callbacks.append(
            _HeldOutEvalCallback(model, tokenizer, held_out_rows)
        )

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
