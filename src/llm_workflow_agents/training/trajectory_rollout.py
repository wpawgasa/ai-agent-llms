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
from dataclasses import dataclass
from typing import Any


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
