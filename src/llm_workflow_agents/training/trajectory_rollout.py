"""Multi-turn trajectory rollout for GRPO (Cat A workflow conversations).

Replaces GRPO's per-turn single-completion rollout with a whole-conversation
*replay* rollout: the model free-runs its assistant turns while the gold
conversation's user turns and tool results are replayed as a fixed script, and
the rollout truncates when the model's emitted state transition leaves the gold
path. A trajectory-level reward then yields one scalar per rollout — restoring
the within-group reward variance that per-turn scoring structurally collapses
(see ``docs/superpowers/specs/2026-07-09-multiturn-trajectory-reward-design.md``).

This module holds the pure, unit-testable trajectory logic (gold-script
construction, turn alignment) plus the in-process ``model.generate`` rollout and
its TRL ``rollout_func`` adapter. It imports ``trl``/``torch`` lazily inside the
functions that need them so the module imports on a box without the training
stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
)

_SUPPORT_CHECKED = False


def assert_trajectory_rollout_support() -> None:
    """Fail fast if the installed TRL lacks the rollout_func/env_mask pathway.

    The trajectory rollout depends on TRL 1.0.0's ``GRPOTrainer`` accepting a
    custom ``rollout_func`` and honoring an ``env_mask`` extra field (which masks
    injected gold user/tool tokens out of the policy-gradient loss). TRL 0.24.0
    and earlier lack both. Unsloth also monkey-patches ``GRPOTrainer`` at import,
    so we inspect the *installed, possibly-patched* ``_generate`` source rather
    than trusting the version string alone.

    Raises:
        RuntimeError: if ``GRPOTrainer._generate`` does not reference both
            ``rollout_func`` and ``env_mask``.
    """
    import inspect

    import trl
    from trl.trainer.grpo_trainer import GRPOTrainer

    src = inspect.getsource(GRPOTrainer._generate)
    if "rollout_func" not in src or "env_mask" not in src:
        raise RuntimeError(
            f"TRL {trl.__version__}: GRPOTrainer._generate lacks the "
            "rollout_func/env_mask pathway required by trajectory GRPO "
            "(present in trl==1.0.0). Check the training env's trl pin and "
            "whether Unsloth's RL patch replaced _generate."
        )


# --------------------------------------------------------------------------- #
# Pure trajectory logic: gold-script construction + turn alignment.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GoldScript:
    """The replay script + gold references for one conversation.

    ``segments[t]`` is the list of gold NON-assistant messages (tool/user) that
    follow gold assistant turn ``t`` — replayed into the rollout context after
    the model produces its own turn ``t``. ``len(segments) ==
    len(gold_transitions) == n_gold_assistant_turns`` (enforced by
    :func:`build_gold_script`). The last segment is usually ``[]`` (conversations
    end on an assistant turn).
    """

    conversation_id: str
    prompt_messages: list[dict[str, str]]  # [enriched system (+ first user)]
    segments: list[list[dict[str, str]]]
    gold_transitions: list[tuple[str, str]]  # ground_truth.state_sequence
    gold_tool_calls: list[dict[str, Any]]  # flat ground_truth.tool_calls
    terminal_state: str
    terminal_reached: bool
    valid_transitions: list[list[str]]  # workflow_graph edges as [from, to]


def _slim_content(content: Any) -> str:
    """Coerce a message ``content`` value to a template-renderable string.

    Mirrors ``grpo._slim_content`` but is duplicated here (three trivial lines)
    to keep ``trajectory_rollout`` free of a ``grpo`` import — ``grpo`` imports
    *this* module for the rollout wiring, so importing back would be circular.
    """
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _slim_msg(m: dict[str, Any]) -> dict[str, str]:
    return {"role": m.get("role", ""), "content": _slim_content(m.get("content"))}


def build_gold_script(raw_row: dict[str, Any], enriched_system: str) -> GoldScript:
    """Build a :class:`GoldScript` from one Task A conversation JSONL row.

    ``enriched_system`` is precomputed by the loader (via
    ``build_enriched_system_prompt``) and swapped in for the row's original
    system-message content, so rollouts see the same system prompt as the
    benchmark. Raises ``ValueError`` if the conversation has no assistant turns
    or if the gold ``state_sequence`` length does not match the assistant-turn
    count (the per-turn one-transition invariant).
    """
    messages = raw_row.get("messages", []) or []
    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not asst_indices:
        raise ValueError(
            f"conversation {raw_row.get('conversation_id')!r} has no assistant turns"
        )

    prompt_messages = [_slim_msg(m) for m in messages[: asst_indices[0]]]
    if prompt_messages and prompt_messages[0]["role"] == "system":
        prompt_messages[0]["content"] = enriched_system

    segments: list[list[dict[str, str]]] = []
    for k, idx in enumerate(asst_indices):
        end = asst_indices[k + 1] if k + 1 < len(asst_indices) else len(messages)
        seg = [
            _slim_msg(m)
            for m in messages[idx + 1 : end]
            if m.get("role") != "assistant"
        ]
        segments.append(seg)

    gt = raw_row.get("ground_truth", {}) or {}
    gold_transitions = [
        (s.get("from", ""), s.get("to", ""))
        for s in gt.get("state_sequence", [])
        if isinstance(s, dict)
    ]
    if len(gold_transitions) != len(asst_indices):
        raise ValueError(
            f"gold_transitions ({len(gold_transitions)}) != assistant turns "
            f"({len(asst_indices)}) for {raw_row.get('conversation_id')!r}"
        )

    graph = raw_row.get("workflow_graph", {}) or {}
    valid_transitions = [
        [t.get("from", ""), t.get("to", "")]
        for t in graph.get("transitions", [])
        if isinstance(t, dict)
    ]

    return GoldScript(
        conversation_id=raw_row.get("conversation_id", ""),
        prompt_messages=prompt_messages,
        segments=segments,
        gold_transitions=gold_transitions,
        gold_tool_calls=list(gt.get("tool_calls", []) or []),
        terminal_state=gt.get("terminal_state", "") or "",
        terminal_reached=bool(gt.get("terminal_reached", False)),
        valid_transitions=valid_transitions,
    )


def prompt_key(prompt_messages: list[dict[str, str]]) -> str:
    """Stable content hash of a prompt message list (order-insensitive keys)."""
    blob = json.dumps(prompt_messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def classify_turn(
    pred_transitions: list[tuple[str, str]],
    cursor: int,
    gold_transitions: list[tuple[str, str]],
) -> tuple[str, int]:
    """Align a model turn's emitted transitions against the gold spine.

    Self-loops ``(X, X)`` are neutral "stay" markers and are dropped before
    matching. Returns one of:

    - ``("stall", cursor)`` — no effective (non-self-loop) transition emitted.
    - ``("advance", cursor + n)`` — the emitted transitions are exactly the next
      ``n`` consecutive gold transitions from ``cursor`` (order-sensitive; ``n``
      may be >1 only for an exact consecutive gold run).
    - ``("diverged", cursor)`` — anything else (wrong target, non-consecutive, or
      any transition once ``cursor`` has reached the end of the gold spine).
    """
    effective = [tuple(t) for t in pred_transitions if t[0] != t[1]]
    if not effective:
        return ("stall", cursor)
    window = [tuple(g) for g in gold_transitions[cursor : cursor + len(effective)]]
    if effective == window:  # window is non-empty here (equal to non-empty effective)
        return ("advance", cursor + len(effective))
    return ("diverged", cursor)


# --------------------------------------------------------------------------- #
# In-process replay rollout (HF model.generate, one assistant turn per round).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrajectoryRolloutConfig:
    """Knobs for the replay rollout (subset surfaced in the YAML config)."""

    max_turns: int = 24  # gold p90 length ~21
    per_turn_max_new_tokens: int = 256
    max_completion_tokens: int = 4096  # must equal GRPOConfig.max_completion_length
    stall_turn_limit: int = 2  # consecutive no-transition turns before stop
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class RolloutSample:
    """One completed trajectory: token ids + the env_mask + scored inputs."""

    prompt_ids: list[int]
    completion_ids: list[int]
    env_mask: list[int]  # len == len(completion_ids); 1=model token, 0=injected/forced
    turn_texts: list[str]  # decoded model turns, in order
    meta: dict[str, Any]


@dataclass
class _RolloutState:
    """Mutable per-conversation bookkeeping during the turn loop."""

    script: GoldScript
    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)
    turn_texts: list[str] = field(default_factory=list)
    pred_transitions: list[tuple[str, str]] = field(default_factory=list)
    cursor: int = 0
    n_stall: int = 0
    consec_stall: int = 0
    active: bool = True
    stop_reason: str = ""


def _derive_turn_end_id(tokenizer: Any) -> int:
    """Return the token id that terminates an assistant turn for this template.

    Rendered from a dummy ``[user, assistant]`` exchange without a generation
    prompt — the last token is the assistant turn terminator (``<|im_end|>`` for
    Qwen ChatML, ``<end_of_turn>`` for Gemma). Tokenizer-agnostic.
    """
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        add_generation_prompt=False,
        tokenize=True,
    )
    return int(ids[-1])


def _segment_suffix_ids(tokenizer: Any, segment_msgs: list[dict[str, str]]) -> list[int]:
    """Token ids for injecting a gold segment + the next assistant gen header.

    Uses the dummy-conversation diff (as TRL's ``_get_tool_suffix_ids``): render
    a throwaway ``[user, assistant]`` prefix, then the prefix + the segment with
    a generation prompt, and return the tokens beyond the shared prefix. These
    ids (gold tool/user text + the ``<assistant>`` header) are injected with
    ``env_mask=0`` so they are context-only, never trained.
    """
    dummy = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]
    prefix = tokenizer.apply_chat_template(
        dummy, add_generation_prompt=False, tokenize=True
    )
    full = tokenizer.apply_chat_template(
        dummy + list(segment_msgs), add_generation_prompt=True, tokenize=True
    )
    n = len(prefix)
    if full[:n] != prefix:  # template not exactly prefix-stable — trim the common run
        n = 0
        while n < len(prefix) and n < len(full) and prefix[n] == full[n]:
            n += 1
    return [int(t) for t in full[n:]]


def _left_pad(seqs: list[list[int]], pad_id: int):  # noqa: ANN201
    """Left-pad a batch of id lists into ``(input_ids, attention_mask)`` tensors."""
    import torch

    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            input_ids[i, maxlen - len(s) :] = torch.tensor(s, dtype=torch.long)
            attn[i, maxlen - len(s) :] = 1
    return input_ids, attn


def _truncate_at_turn_end(
    ids: list[int], turn_end_ids: set[int]
) -> tuple[list[int], bool]:
    """Cut generated ids at (and including) the first turn-end token."""
    for i, tok in enumerate(ids):
        if tok in turn_end_ids:
            return ids[: i + 1], True
    return ids, False


def _decide_stop(
    state: _RolloutState, label: str, turn_idx: int, cfg: TrajectoryRolloutConfig
) -> str:
    """Return a stop reason for this turn, or '' to continue to the next turn."""
    n_gold = len(state.script.gold_transitions)
    if label == "diverged":
        return "diverged"
    if state.cursor >= n_gold:
        return "gold_complete"
    if state.consec_stall >= cfg.stall_turn_limit:
        return "stall"
    if turn_idx + 1 >= len(state.script.segments):
        return "script_exhausted"
    if turn_idx + 1 >= cfg.max_turns:
        return "turn_cap"
    return ""


def run_replay_rollout(
    model: Any,
    tokenizer: Any,
    scripts: list[GoldScript],
    cfg: TrajectoryRolloutConfig,
) -> list[RolloutSample]:
    """Free-run each conversation's assistant turns against its replayed gold script.

    Per round, all still-active conversations generate one assistant turn in a
    single batched ``model.generate`` (left-padded), each turn is aligned via
    :func:`classify_turn`, and unless a stop condition fires the matching gold
    segment (user + tool turns) is injected with ``env_mask=0``. The trajectory
    is force-terminated on ``tokenizer.eos_token_id`` so a divergence-truncated
    completion is never zeroed by TRL's ``mask_truncated_completions``.
    """
    import torch

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    eos_id = int(tokenizer.eos_token_id)
    turn_end_ids = {_derive_turn_end_id(tokenizer), eos_id}

    states = [
        _RolloutState(
            script=sc,
            prompt_ids=[
                int(t)
                for t in tokenizer.apply_chat_template(
                    sc.prompt_messages, add_generation_prompt=True, tokenize=True
                )
            ],
        )
        for sc in scripts
    ]

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        for turn_idx in range(cfg.max_turns):
            active = [s for s in states if s.active]
            if not active:
                break
            seqs = [s.prompt_ids + s.completion_ids for s in active]
            input_ids, attn = _left_pad(seqs, pad_id)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": cfg.per_turn_max_new_tokens,
                "do_sample": cfg.do_sample,
                "eos_token_id": list(turn_end_ids),
                "pad_token_id": pad_id,
            }
            if cfg.do_sample:
                gen_kwargs["temperature"] = cfg.temperature
                gen_kwargs["top_p"] = cfg.top_p
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            in_len = input_ids.shape[1]
            for i, s in enumerate(active):
                new_ids = [int(t) for t in out[i, in_len:].tolist() if int(t) != pad_id]
                turn_ids, had_end = _truncate_at_turn_end(new_ids, turn_end_ids)
                s.completion_ids.extend(turn_ids)
                s.env_mask.extend([1] * len(turn_ids))
                if not had_end:  # ran out of budget mid-turn — force a masked turn-end
                    s.completion_ids.append(eos_id)
                    s.env_mask.append(0)

                turn_text = tokenizer.decode(turn_ids, skip_special_tokens=True)
                s.turn_texts.append(turn_text)
                transitions = extract_state_annotations(turn_text)
                s.pred_transitions.extend(transitions)
                label, s.cursor = classify_turn(
                    transitions, s.cursor, s.script.gold_transitions
                )
                if label == "stall":
                    s.n_stall += 1
                    s.consec_stall += 1
                else:
                    s.consec_stall = 0

                stop = _decide_stop(s, label, turn_idx, cfg)
                if stop:
                    s.active = False
                    s.stop_reason = stop
                    continue

                seg_ids = _segment_suffix_ids(
                    tokenizer, s.script.segments[turn_idx]
                )
                s.completion_ids.extend(seg_ids)
                s.env_mask.extend([0] * len(seg_ids))
                if (
                    len(s.completion_ids) + cfg.per_turn_max_new_tokens
                    > cfg.max_completion_tokens
                ):
                    s.active = False
                    s.stop_reason = "budget"
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    samples: list[RolloutSample] = []
    for s in states:
        if s.active:  # exhausted the turn loop without an explicit stop
            s.stop_reason = "turn_cap"
            s.active = False
        if not s.completion_ids or s.completion_ids[-1] != eos_id:
            s.completion_ids.append(eos_id)
            s.env_mask.append(0)
        samples.append(
            RolloutSample(
                prompt_ids=s.prompt_ids,
                completion_ids=s.completion_ids,
                env_mask=s.env_mask,
                turn_texts=s.turn_texts,
                meta={
                    "cursor": s.cursor,
                    "stop_reason": s.stop_reason,
                    "n_model_turns": len(s.turn_texts),
                    "n_stall_turns": s.n_stall,
                    "gold_len": len(s.script.gold_transitions),
                    "conversation_id": s.script.conversation_id,
                },
            )
        )
    return samples


def make_replay_rollout_func(
    script_index: dict[str, GoldScript],
    cfg: TrajectoryRolloutConfig,
) -> Callable[[list, Any], dict[str, Any]]:
    """Build the TRL ``rollout_func`` closure over a prompt→GoldScript index.

    The returned callable takes ``(prompts, trainer)``, looks up each prompt's
    gold script by :func:`prompt_key` (a missing key is a hard ``KeyError`` — we
    never silently fall back to single-turn), runs the in-process replay, and
    returns the TRL contract dict plus the ``env_mask``/``trajectory``/
    ``rollout_meta`` extra fields.
    """

    def rollout_func(prompts: list, trainer: Any) -> dict[str, Any]:
        global _SUPPORT_CHECKED
        if not _SUPPORT_CHECKED:
            assert_trajectory_rollout_support()
            _SUPPORT_CHECKED = True

        scripts = [script_index[prompt_key(p)] for p in prompts]

        args = getattr(trainer, "args", None)
        run_cfg = cfg
        if args is not None:
            run_cfg = TrajectoryRolloutConfig(
                max_turns=cfg.max_turns,
                per_turn_max_new_tokens=cfg.per_turn_max_new_tokens,
                max_completion_tokens=cfg.max_completion_tokens,
                stall_turn_limit=cfg.stall_turn_limit,
                temperature=getattr(args, "temperature", cfg.temperature),
                top_p=getattr(args, "top_p", cfg.top_p),
                do_sample=cfg.do_sample,
            )

        samples = run_replay_rollout(
            trainer.model, trainer.processing_class, scripts, run_cfg
        )
        return {
            "prompt_ids": [s.prompt_ids for s in samples],
            "completion_ids": [s.completion_ids for s in samples],
            "logprobs": None,
            "env_mask": [s.env_mask for s in samples],
            "trajectory": [json.dumps(s.turn_texts, ensure_ascii=False) for s in samples],
            "rollout_meta": [json.dumps(s.meta, ensure_ascii=False) for s in samples],
        }

    return rollout_func
