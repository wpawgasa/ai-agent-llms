"""Unsloth GRPO RL entry point for Phase 2 reinforcement learning.

Loads an SFT checkpoint, applies GRPOTrainer with task-specific
reward function, vLLM generation backend, and FP8 RL.
"""

from __future__ import annotations

import importlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog

# Shared with dpo.py, not redefined per entry point — see the imported names'
# own modules for why duplicating either previously caused a regression to go
# unnoticed. `_utils.py` holds the Gemma-4 KV-zero-proxy workaround every
# Unsloth `FastLanguageModel.from_pretrained` call needs first; `reward_utils.py`
# holds the held-out composite scorer and reward-hacking test the guardrail
# callback uses, aliased back to this module's existing private names.
from llm_workflow_agents.training._utils import (
    unwrap_unsloth_gemma4_kv_zero_proxy as _unwrap_unsloth_gemma4_kv_zero_proxy,
)
from llm_workflow_agents.training.reward_utils import (
    heldout_composite_score as _heldout_composite_score,
)
from llm_workflow_agents.training.reward_utils import (
    is_reward_hacking as _is_reward_hacking,
)

if TYPE_CHECKING:
    from datasets import Dataset

logger = structlog.get_logger(__name__)

_REWARD_REGISTRY: dict[str, str] = {
    "reward_business_logic": "llm_workflow_agents.training.rewards.reward_business_logic",
    "reward_business_logic_trajectory": "llm_workflow_agents.training.rewards.reward_business_logic_trajectory",
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


# GT sanitization: drop tool calls whose REQUIRED args carry a fabricated
# placeholder (null-sentinel) or an out-of-range score. The synthetic corpus's
# invalid_tool_inputs behavior (~15% of turns) produced GT tool calls that fire
# an action with a required identifier the conversation never established
# (e.g. apply_for_loan(customer_id="UNKNOWN"), dispute_bill(account_id="000000"),
# collect_nps(score=11)). Rewarding those trains the policy to fabricate-and-fire
# instead of asking for the missing value. Sourced from the ckpt-1000 audit
# (runs/preflight/gt_overeager_review_ckpt1000.csv). Flip to False to restore the
# raw GT (e.g. to reproduce a pre-sanitization run).
_SANITIZE_INVALID_TOOL_GT = True

_NULL_SENTINEL_RE = re.compile(
    r"^(unknown|n/?a|none|null|tbd|placeholder|pending|0{3,}|0000+)$", re.IGNORECASE
)
_SCORE_ARG_KEYS = frozenset({"score", "rating", "nps", "satisfaction", "csat"})


def _required_args_by_tool(tool_schemas: Any) -> dict[str, set[str]]:
    """Map tool name -> set of required parameter names from a conversation's schemas."""
    out: dict[str, set[str]] = {}
    for ts in tool_schemas or []:
        fn = ts.get("function", ts) if isinstance(ts, dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            params = fn.get("parameters", {}) or {}
            out[name] = set(params.get("required", []) or [])
    return out


def _gt_tool_call_is_invalid(tool_call: dict, required: set[str]) -> bool:
    """True iff a REQUIRED arg holds a null-sentinel string or an out-of-range score.

    Only REQUIRED args count: a placeholder on an OPTIONAL field (e.g.
    ``log_complaint_trend(region="unknown")``) is a harmless default, not a
    fabricated identifier, so it is left intact. Pure/unit-tested.
    """
    args = tool_call.get("arguments") if isinstance(tool_call, dict) else None
    if not isinstance(args, dict):
        return False
    for key, val in args.items():
        if (
            key.lower() in _SCORE_ARG_KEYS
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
            and not (0 <= val <= 10)
        ):
            return True
        if (
            key in required
            and isinstance(val, str)
            and _NULL_SENTINEL_RE.match(val.strip())
        ):
            return True
    return False


def _sanitize_gt_tool_calls(
    tool_calls: list, required_by_tool: dict[str, set[str]]
) -> tuple[list, int]:
    """Drop invalid (fabricated-required-arg) tool calls from a turn's GT.

    Returns ``(kept_tool_calls, n_removed)``. Pure/unit-tested.
    """
    if not _SANITIZE_INVALID_TOOL_GT or not tool_calls:
        return tool_calls, 0
    kept = [
        tc
        for tc in tool_calls
        if not _gt_tool_call_is_invalid(tc, required_by_tool.get(tc.get("name"), set()))
    ]
    return kept, len(tool_calls) - len(kept)


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
    n_sanitized_tool_calls = 0
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

            required_by_tool = _required_args_by_tool(raw.get("tool_schemas"))

            asst_indices = [
                i for i, m in enumerate(raw_msgs) if m.get("role") == "assistant"
            ]
            valid_pairs = [
                i for i in asst_indices
                if i > 0
                and raw_msgs[i - 1].get("role") in ("user", "system")
                # A loss:false turn stays in the prompt prefix but never
                # becomes a training row. See sft.py for why.
                and raw_msgs[i].get("loss", True) is not False
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
                tool_calls, n_removed = _sanitize_gt_tool_calls(
                    tool_calls, required_by_tool
                )
                n_sanitized_tool_calls += n_removed

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
        sanitized_invalid_tool_calls=n_sanitized_tool_calls,
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


def _load_grpo_trajectory_dataset(
    data_dir: Path, split: str = "train"
) -> tuple["Dataset", dict[str, Any]]:
    """Load a split as ONE row per conversation, plus its gold-script index.

    The per-turn loader (:func:`_load_grpo_jsonl`) emits a row per assistant
    turn, which is what leaves the reward on its ceiling: measured on 206 real
    C2 completions the per-turn reward takes 11 distinct values and is exactly
    1.0 on 81.1% of them, so a GRPO group ties and the advantage is zero.
    Trajectory mode scores a whole replayed conversation instead, which is what
    ``reward_business_logic_trajectory`` aggregates into a near-continuous
    distribution.

    Returns ``(dataset, script_index)`` where ``script_index`` maps
    :func:`prompt_key` to the conversation's :class:`GoldScript`. The rollout
    looks scripts up by that key and treats a miss as a hard ``KeyError``, so
    the key must be computed from the exact prompt stored on the row — hence
    both are built here, together, rather than by two functions that could
    drift.

    Conversations that violate ``build_gold_script``'s one-transition-per-turn
    invariant are skipped and counted, not fatal — one bad row must not take
    down a load of thousands. An entirely empty result *is* fatal, because a
    silently empty training set would start a run that cannot learn.
    """
    from datasets import Dataset

    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
    from llm_workflow_agents.training.trajectory_rollout import (
        build_gold_script,
        prompt_key,
    )

    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GRPO split missing: {path}")

    rows: list[dict[str, Any]] = []
    script_index: dict[str, Any] = {}
    n_convs = n_skipped = n_dup = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            n_convs += 1
            msgs = raw.get("messages") or []
            enriched = ""
            if msgs and msgs[0].get("role") == "system":
                enriched = (
                    build_enriched_system_prompt(
                        raw, msgs[0].get("content") or "", force_rebuild=True
                    )
                    if raw.get("workflow_graph")
                    else (msgs[0].get("content") or "")
                )
            try:
                script = build_gold_script(raw, enriched)
            except ValueError as exc:
                n_skipped += 1
                logger.debug(
                    "grpo_trajectory_conv_skipped",
                    conversation_id=raw.get("conversation_id"),
                    error=str(exc),
                )
                continue

            key = prompt_key(script.prompt_messages)
            if key in script_index:
                # Two conversations share a prompt prefix. The rollout resolves
                # scripts by this key, so a dict would keep only the last one and
                # every earlier colliding row would replay a DIFFERENT
                # conversation's gold segments and be scored against its
                # transitions and tool calls — silent per-row reward corruption.
                # Real corpus: 2,558 conversations -> 2,420 keys, so 138 rows
                # (5.4%) were affected, one key colliding 8 ways. Keep the first
                # and drop the rest so rows and index agree exactly.
                n_dup += 1
                continue

            script_index[key] = script
            rows.append(
                {
                    "prompt": script.prompt_messages,
                    "ground_truth": json.dumps(
                        {
                            "state_sequence": [
                                {"from": f, "to": t} for f, t in script.gold_transitions
                            ],
                            "tool_calls": script.gold_tool_calls,
                            "terminal_state": script.terminal_state,
                            "terminal_reached": script.terminal_reached,
                            "valid_transitions": script.valid_transitions,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    if not rows:
        raise ValueError(
            f"No usable conversations in {path} ({n_convs} read, {n_skipped} "
            f"skipped on the gold-transition/assistant-turn invariant, {n_dup} "
            "on prompt collisions). Refusing to start a run on an empty "
            "training set."
        )

    logger.info(
        "grpo_trajectory_data_loaded",
        split=split,
        conversations=n_convs,
        rows=len(rows),
        skipped_invariant=n_skipped,
        skipped_prompt_collision=n_dup,
        indexed_scripts=len(script_index),
    )
    return Dataset.from_list(rows), script_index


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


def _make_trajectory_reward_adapter(reward_fn: Callable) -> Callable:
    """Bridge TRL's keyword call to a trajectory reward's 4-arg signature.

    A trajectory rollout (``trajectory_rollout.make_replay_rollout_func``) returns
    per-completion ``trajectory`` (JSON list of model turn texts) and
    ``rollout_meta`` (JSON dict) extra fields, which TRL 1.0.0 forwards into the
    reward kwargs alongside the ``ground_truth`` dataset column. Unlike
    :func:`_make_reward_adapter`, this adapter does **not** collapse
    ``completions`` — TRL decodes the whole interleaved (model + injected gold)
    stream into ``completions``, which must not be scored. It reads the turn
    texts from the ``trajectory`` field instead and calls
    ``reward_fn(prompts, trajectories, metas, ground_truths)``.
    """

    def adapter(  # noqa: ANN001
        *,
        prompts: Any = None,
        completions: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        traj_raw = kwargs.get("trajectory")
        if traj_raw is None:
            raise ValueError(
                "trajectory reward adapter requires a 'trajectory' extra field "
                "from the rollout_func, but none was passed. Is rollout.mode "
                "'trajectory' wired to make_replay_rollout_func?"
            )
        trajectories: list[list[str]] = [
            json.loads(t) if isinstance(t, str) else (list(t) if t else [])
            for t in traj_raw
        ]

        meta_raw = kwargs.get("rollout_meta") or [None] * len(trajectories)
        metas: list[dict[str, Any]] = [
            json.loads(m) if isinstance(m, str) else (m or {}) for m in meta_raw
        ]

        gt_raw = kwargs.get("ground_truth") or []
        gts: list[dict[str, Any]] = []
        for g in gt_raw:
            d = json.loads(g) if isinstance(g, str) else (g or {})
            gts.append(d if isinstance(d, dict) else {})

        # Stash unique-trajectories-per-group (variance signal) for the next
        # on_log, keyed by prompt — mirrors _make_reward_adapter's instrumentation
        # but uses the joined turn texts as the completion identity.
        if trajectories and prompts is not None:
            groups: dict[str, list[str]] = {}
            for p, t in zip(prompts, trajectories):
                groups.setdefault(str(p)[:512], []).append("\n".join(t))
            if groups:
                uniques = [len(set(cs)) for cs in groups.values()]
                sizes = [len(cs) for cs in groups.values()]
                _LATEST_INSTRUMENTATION["unique_completions_per_group"] = sum(
                    uniques
                ) / len(uniques)
                _LATEST_INSTRUMENTATION["group_size"] = sum(sizes) / len(sizes)

        return reward_fn(prompts or [], trajectories, metas, gts)

    return adapter


def _tool_bearing_mix_indices(
    has_tool: "Sequence[bool]",
    ratio: float | None,
    seed: int = 42,
) -> list[int]:
    """Pick row indices so tool-bearing turns are ``ratio`` of the training set.

    GRPO learns from *within-group* reward variance. On Task A, 63.9% of turns
    carry no ground-truth tool call (9,771 of 27,056 are tool-bearing) and
    ``tool_call_f1([], []) == 1.0`` hands those rows 0.40 of the reward for
    free; combined with C2's 0.9369 state accuracy, ~59% of prompts score
    exactly 1.0 and the whole group ties. The 2026-08-16 diagnostic measured
    the consequence directly: ``reward_std 0, frac_reward_zero_std 1`` at every
    step, on an otherwise perfectly stable optimizer.

    **This is a partial mitigation, not a fix.** Scoring C2's real held-out
    completions through this same reward, a greedy score below 1.0 — the proxy
    for a prompt whose group can vary at all — occurs on 38.0% of tool-bearing
    rows but only 8.9% of no-tool rows. So the expected share of informative
    prompts moves 19.4% (natural) → 23.5% (ratio 0.5) → 29.3% (ratio 0.7) →
    38.0% (ratio 1.0). Even a pure tool-only slice leaves ~62% of groups tied,
    because the graded training reward also saturates on tool-bearing rows
    (mean 0.886 here versus a strict held-out ``tool_f1`` of 0.636). Reward
    resolution is the other half of the problem and is not addressed here.

    Rebalances, never eliminates. Keeps **all** tool-bearing rows (the scarce,
    informative ones) and downsamples non-tool rows to hit the target. A pure
    tool-only slice is available at ``ratio=1.0`` but is the R15-shaped
    setting: that risk analysis showed a structurally unconditional behaviour
    gets learned as an unconditional habit, so removing every no-tool turn
    invites "always call a tool" and would regress C2's 1.5% spurious-call
    rate on turns that need none.

    Returns sorted indices to keep. ``ratio=None`` is a passthrough, so runs
    that don't set the key are byte-identical to before. A ratio at or below
    the corpus's natural rate is also a passthrough — the goal is to raise the
    tool share, never to discard data to hit a number the corpus already beats.
    """
    if ratio is None:
        return list(range(len(has_tool)))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"tool_bearing_ratio must be in [0, 1], got {ratio}")

    tool_idx = [i for i, t in enumerate(has_tool) if t]
    other_idx = [i for i, t in enumerate(has_tool) if not t]
    if not tool_idx or not other_idx:
        return list(range(len(has_tool)))

    if ratio >= 1.0:
        return tool_idx
    # keep all tool rows; solve n_other for tool/(tool + n_other) == ratio
    n_other = int(round(len(tool_idx) * (1.0 - ratio) / ratio))
    if n_other >= len(other_idx):
        return list(range(len(has_tool)))  # already at or above target

    rng = random.Random(seed)
    return sorted(tool_idx + rng.sample(other_idx, n_other))


def _filter_grpo_config_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs the installed TRL ``GRPOConfig`` does not accept.

    TRL moves fields between releases — 1.0.0 removed ``max_prompt_length``,
    which turned a config written against 0.23.x into a ``TypeError`` at
    ``GRPOConfig(**kwargs)``, i.e. after the model and both dataset splits had
    already loaded. Filtering keeps a config launchable across versions.

    Returns ``(kept, dropped)``; the caller must log ``dropped``. Silence is the
    failure mode that matters here: R16 happened because a length parameter was
    dropped quietly (``max_seq_length`` → ``max_length`` in TRL 0.23+ turned a
    guarded branch into a no-op), and every Cat A SFT run trained on a
    1024-token window for months with no warning. Prefer a loud, harmless
    warning over a silent, invisible behaviour change.
    """
    import dataclasses

    from trl import GRPOConfig

    supported = {f.name for f in dataclasses.fields(GRPOConfig)}
    kept = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(kept))
    return kept, dropped


_RUN_STAMP_RE = re.compile(r"_\d{8}T\d{6}Z$")


def _resolve_output_dir(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Resolve the checkpoint output directory for a GRPO run.

    Precedence:
      1. Explicit ``output_dir`` in the config — the only way to give a run a
         distinct directory (otherwise a second run overwrites the first).
      2. The config filename stem, with any trailing run stamp removed.

    Mirrors ``sft.py::_resolve_output_dir``. The stamp strip is what keeps a
    run-stamped patched config (R13) from silently relocating checkpoints to a
    per-run path the DVC stage does not track — the 2026-07-22 fault on the SFT
    side, which left the declared output weightless. Provenance belongs in the
    config filename; the checkpoint path stays stable unless asked to change.
    """
    explicit = config.get("output_dir")
    run_name = str(explicit) if explicit else _RUN_STAMP_RE.sub("", config_path.stem)
    return Path("checkpoints") / run_name / Path(model_name).name


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

    # Re-arm the Gemma-4 proxy unwrap. It ran before this load, but
    # FastLanguageModel.from_pretrained re-applies unsloth_zoo's temporary
    # patches, which reinstalls `_Gemma4KVSharedSafeProxy` on top of ours — so
    # the pre-load call is undone exactly when it starts to matter. Everything
    # that later builds or validates a config hits the proxy's deliberate
    # `num_kv_shared_layers` AttributeError: it is what made every held-out eval
    # fail with `grpo_heldout_eval_failed` (silently disabling the R5
    # reward-hacking guardrail while training continued) and what killed the
    # first trajectory-rollout run outright.
    _unwrap_unsloth_gemma4_kv_zero_proxy()

    data_source = data_cfg.get("source", "")

    # Rollout mode. "turn" (default) scores one assistant turn per row.
    # "trajectory" replays a whole conversation, injecting the gold user/tool
    # segments between the model's own turns, and scores the aggregate — the
    # designed fix for the per-turn reward's ceiling (11 distinct values, 81.1%
    # exactly 1.0 on real completions => tied groups => zero GRPO advantage).
    rollout_cfg = config.get("rollout", {}) or {}
    rollout_mode = str(rollout_cfg.get("mode", "turn")).lower()
    if rollout_mode not in ("turn", "trajectory"):
        raise ValueError(
            f"rollout.mode must be 'turn' or 'trajectory', got {rollout_mode!r}"
        )
    use_trajectory = rollout_mode == "trajectory"

    script_index: dict[str, Any] = {}
    if use_trajectory:
        from llm_workflow_agents.training.trajectory_rollout import (
            TrajectoryRolloutConfig,
            assert_trajectory_rollout_support,
            make_replay_rollout_func,
        )

        # Fail before the (expensive) data load if this TRL can't do it.
        assert_trajectory_rollout_support()
        # The trajectory adapter calls reward_fn(prompts, trajectories, metas,
        # gts) — four args. A per-turn reward takes three, so a mismatched pair
        # would blow up mid-training with an arity TypeError instead of here.
        if not reward_fn_name.endswith("_trajectory"):
            raise ValueError(
                f"rollout.mode=trajectory requires a trajectory reward "
                f"(e.g. 'reward_business_logic_trajectory'), got "
                f"{reward_fn_name!r}. A per-turn reward has a 3-arg signature "
                "and cannot score replayed trajectories."
            )
        train_ds, script_index = _load_grpo_trajectory_dataset(
            Path(data_source), split="train"
        )
    else:
        train_ds = _load_grpo_jsonl(Path(data_source), split="train")

    # Optional tool-bearing rebalance. Applied to the TRAINING set only — the
    # held-out rows below must keep the corpus's natural distribution, because
    # they measure real capability for the R5 guardrail and would otherwise
    # report a metric on a distribution nothing else uses.
    tool_ratio = data_cfg.get("tool_bearing_ratio")
    if tool_ratio is not None and use_trajectory:
        # The mix is a per-TURN notion; trajectory rows are whole conversations,
        # nearly all of which contain at least one tool call, so applying it here
        # would be a no-op dressed up as a control. Refuse rather than mislead.
        logger.warning(
            "grpo_tool_bearing_mix_ignored",
            reason="rollout.mode=trajectory scores whole conversations, not turns",
            requested_ratio=float(tool_ratio),
        )
        tool_ratio = None
    if tool_ratio is not None:
        gts = train_ds["ground_truth"]
        has_tool = [bool(json.loads(g).get("tool_calls")) for g in gts]
        keep = _tool_bearing_mix_indices(
            has_tool, float(tool_ratio), seed=int(data_cfg.get("tool_bearing_seed", 42))
        )
        before_n, before_tool = len(has_tool), sum(has_tool)
        train_ds = train_ds.select(keep)
        after_tool = sum(has_tool[i] for i in keep)
        logger.info(
            "grpo_tool_bearing_mix",
            requested_ratio=float(tool_ratio),
            rows_before=before_n,
            rows_after=len(keep),
            tool_share_before=round(before_tool / before_n, 4) if before_n else 0.0,
            tool_share_after=round(after_tool / len(keep), 4) if keep else 0.0,
            note=(
                "Rebalances toward turns where the policy has headroom; a "
                "no-tool-heavy mix ties every group's reward and yields zero "
                "GRPO advantage. Held-out eval keeps the natural distribution."
            ),
        )

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
        output_dir=str(_resolve_output_dir(config, Path(config_path), model_basename)),
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
    # Drop anything the installed TRL doesn't accept, but name it loudly —
    # a silently-dropped length knob is exactly how R16 went unnoticed.
    grpo_kwargs, dropped_kwargs = _filter_grpo_config_kwargs(grpo_kwargs)
    if dropped_kwargs:
        logger.warning(
            "grpo_config_kwargs_unsupported",
            dropped=dropped_kwargs,
            trl_version=getattr(importlib.import_module("trl"), "__version__", "?"),
            note=(
                "These keys are set in the YAML but do not exist on this TRL's "
                "GRPOConfig, so they have NO effect. If a dropped key bounds a "
                "length or a gradient, verify the run is still within budget."
            ),
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
                                # Explicit config is required on Gemma-4 +
                                # Unsloth. Left None, transformers takes the
                                # `self.config._get_generation_parameters()`
                                # branch, which re-validates the model config,
                                # and unsloth_zoo's Gemma-4 proxy deliberately
                                # hides `num_kv_shared_layers` -> AttributeError.
                                # That is what made every held-out eval fail with
                                # `grpo_heldout_eval_failed`, silently disabling
                                # the R5 reward-hacking guardrail for the whole
                                # run while training carried on regardless.
                                generation_config=getattr(
                                    model, "generation_config", None
                                ),
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

    trainer_kwargs: dict[str, Any] = {}
    if use_trajectory:
        traj_cfg = TrajectoryRolloutConfig(
            max_turns=int(rollout_cfg.get("max_turns", 24)),
            per_turn_max_new_tokens=int(
                rollout_cfg.get("per_turn_max_new_tokens", 256)
            ),
            # Must equal GRPOConfig.max_completion_length or the rollout and the
            # trainer disagree about the budget; take it from the resolved
            # GRPOConfig rather than the YAML so there is one source of truth.
            max_completion_tokens=int(grpo_config.max_completion_length),
            stall_turn_limit=int(rollout_cfg.get("stall_turn_limit", 2)),
            temperature=float(grpo_config.temperature),
            top_p=float(grpo_config.top_p),
        )
        trainer_kwargs["rollout_func"] = make_replay_rollout_func(
            script_index, traj_cfg
        )
        reward_adapter = _make_trajectory_reward_adapter(reward_fn)
        logger.info(
            "grpo_trajectory_rollout_enabled",
            reward_function=reward_fn_name,
            scripts=len(script_index),
            max_turns=traj_cfg.max_turns,
            per_turn_max_new_tokens=traj_cfg.per_turn_max_new_tokens,
            max_completion_tokens=traj_cfg.max_completion_tokens,
        )
    else:
        reward_adapter = _make_reward_adapter(reward_fn)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_adapter,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
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
