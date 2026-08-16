"""Contrastive preference pairs for Cat A workflow turns.

R18 (`docs/grpo_reward_resolution_investigation.md`) established that GRPO
cannot learn on this task: the reward takes 11 distinct values across 206 real
completions and lands on exactly 1.0 for 81.1% of them, so a GRPO group ties and
the advantage is identically zero. Richer prompt mix, trajectory aggregation and
higher sampling temperature each moved the needle and none produced a usable
signal.

A preference objective sidesteps the problem entirely — each pair carries a
guaranteed margin, so no reward variance is required.

The corruptions here mirror the failures C2 actually makes on held-out data
rather than invented ones (percentages over the 71 tool-bearing audit rows):

  ``drop_tool_calls``      announce-but-don't-call — narrates the action, emits
                           no ``<tool_call>``. 9/71.
  ``flip_state_transition``  advances where the convention wants a self-loop, or
                           self-loops where it should advance. Spurious
                           self-loops on advancing rows: 26.9% in v4, 2.8% in C2.
  ``corrupt_tool_args``    right tool name, wrong arguments. 18/71 — the current
                           bottleneck.

**Every function returns ``None`` when it cannot apply.** A corruption that
silently returns its input would produce a pair whose ``rejected`` equals its
``chosen``: zero margin, and a direct signal that the gold answer is also the
bad one. Callers must treat ``None`` as "skip this pair", never as "use the
original".

These are synthetic negatives. Per R18 they should be *complemented* by
negatives mined from the model's own generations, which are on-distribution;
synthetic-only negatives teach the model to discriminate against this module's
corruption function. R15 is the precedent — a structurally uniform edit was
learned as an unconditional habit.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

# `[STATE: X → Y]` — the corpus uses U+2192; tolerate ASCII `->` too.
_STATE_RE = re.compile(r"\[STATE:\s*([^\]\-→]+?)\s*(?:→|->)\s*([^\]]+?)\s*\]")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def drop_tool_calls(text: str) -> str | None:
    """Remove every ``<tool_call>`` block, keeping narration and annotation.

    Reproduces the announce-but-don't-call failure: the turn still *says* it is
    performing the action, and no call is emitted. Returns ``None`` when the
    turn has no tool call, since dropping nothing would yield a zero-margin
    pair.
    """
    if not _TOOL_CALL_RE.search(text):
        return None
    out = _TOOL_CALL_RE.sub("", text)
    # Collapse the blank lines the removal leaves behind.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out or None


def flip_state_transition(
    text: str,
    valid_transitions: list[list[str]] | None,
    seed: int = 0,
) -> str | None:
    """Invert the turn's state annotation between self-loop and advance.

    A gold self-loop ``X → X`` becomes ``X → Z`` for a **legal** successor Z
    (drawn from ``valid_transitions``); a gold advance ``X → Y`` becomes the
    spurious self-loop ``X → X``.

    The successor must be legal on purpose. An illegal one would be rejectable
    on `transition_legality` grounds alone, so the pair would teach "do not
    hallucinate states" — already largely solved — instead of "do not advance on
    a tool-calling turn", which is the behaviour in question.

    Returns ``None`` when there is no annotation, or when a self-loop has no
    legal successor to advance to.
    """
    match = _STATE_RE.search(text)
    if not match:
        return None
    src, dst = match.group(1).strip(), match.group(2).strip()

    if src == dst:
        successors = sorted(
            {
                t[1]
                for t in (valid_transitions or [])
                if len(t) == 2 and t[0] == src and t[1] != src
            }
        )
        if not successors:
            return None
        new_dst = random.Random(seed).choice(successors)
    else:
        new_dst = src

    replacement = f"[STATE: {src} → {new_dst}]"
    out = text[: match.start()] + replacement + text[match.end() :]
    return out if out != text else None


_DROP = object()  # sentinel: remove the argument entirely


def _corrupt_value(value: Any, rng: random.Random) -> Any:
    """Return a plausible-but-wrong variant of an argument value.

    Every branch must stay *in-distribution*. A corruption that leaves a
    synthetic marker (an ``_x`` suffix, a `"CORRUPTED"` literal) is trivially
    separable, so the model learns to reject that marker rather than to get
    arguments right — it would score well on the pair set and be no better at
    the task. This is the same trap R15 documents at corpus level.

    Returns :data:`_DROP` to signal "delete this argument", which models the
    missing-required-argument failure and needs no invented token.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + rng.choice([1, 2, 7, -1, -3])
    if isinstance(value, str):
        digits = [i for i, ch in enumerate(value) if ch.isdigit()]
        if digits:
            # Perturb one digit — keeps the identifier shape and changes
            # identity: the "right tool, wrong record" failure.
            i = rng.choice(digits)
            new_digit = str((int(value[i]) + rng.randint(1, 8)) % 10)
            return value[:i] + new_digit + value[i + 1 :]
        # No digits to perturb and nothing to swap with: drop it rather than
        # inventing a marker string.
        return _DROP
    if isinstance(value, list):
        return value[:-1] if len(value) > 1 else _DROP
    if isinstance(value, dict):
        return _DROP
    return _DROP


def corrupt_tool_args(text: str, seed: int = 0) -> str | None:
    """Keep the tool name, corrupt exactly one argument value.

    Targets the largest residual failure bucket: 18 of 71 tool-bearing held-out
    rows had the right tool name and wrong arguments. Only the arguments change
    — the state annotation and narration are preserved so the pair isolates
    argument fidelity rather than testing two behaviours at once.

    Returns ``None`` when there is no tool call, or when no call carries
    arguments to corrupt.
    """
    rng = random.Random(seed)
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None

    candidates: list[tuple[re.Match[str], dict[str, Any]]] = []
    for m in matches:
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("arguments"), dict):
            if payload["arguments"]:
                candidates.append((m, payload))
    if not candidates:
        return None

    match, payload = candidates[rng.randrange(len(candidates))]
    args = dict(payload["arguments"])

    # Prefer swapping two string arguments when the call has them: a
    # value-for-value mix-up (nationality <-> destination) is the most
    # in-distribution wrong-argument error there is, and leaves no synthetic
    # trace for the model to key on.
    str_keys = sorted(
        k for k, v in args.items() if isinstance(v, str) and v and v.strip()
    )
    if len(str_keys) >= 2:
        a, b = rng.sample(str_keys, 2)
        args[a], args[b] = args[b], args[a]
    else:
        key = sorted(args)[rng.randrange(len(args))]
        new_value = _corrupt_value(args[key], rng)
        if new_value is _DROP:
            del args[key]
        else:
            args[key] = new_value

    if args == payload["arguments"]:
        return None
    payload = {**payload, "arguments": args}

    body = json.dumps(payload, ensure_ascii=False)
    out = (
        text[: match.start()]
        + f"<tool_call>{body}</tool_call>"
        + text[match.end() :]
    )
    return out if out != text else None
