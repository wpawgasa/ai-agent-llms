"""Mechanical repairs for Task A conversations whose values cannot be traced.

Three repairs, each split into a *plan* (decide, once, what to change) and an
*apply* (make exactly that change), so a frozen benchmark is rebuilt from a
ledger of decisions rather than from code that may change later -- the pattern
R25 used to build the v2 text stratum. The checks that motivate each repair
live in :mod:`source_traceability`; see CLAUDE.md R28.

1. **Identifier remap.** Give every simple entity identifier (``CUST-882``) a
   fresh value of the same shape that appears nowhere in the training corpus,
   so memorising training values earns nothing on the benchmark. Identifiers
   with structure (``PLAN_50GB``, ``CARD-ENDING-4455``) are skipped: their
   digits carry meaning or are spoken elsewhere, and rewriting them would make
   the conversation contradict itself.
2. **Session context.** A confident unsourced tool argument -- one the model
   could not have known -- becomes a session-context fact the served prompt
   states (rendered by ``system_prompt.render_session_context``).
3. **Stay merges.** A run of self-loop turns in one state that ends in a tool
   call becomes a single turn, so the harness asks the model for the call
   instead of copying it from gold. Advance-then-stay pairs are never merged:
   that pair is how the stay convention enters a tool state.

Order matters when planning: remap, then merge, then plan session context.
Merging moves prose into the calling turn, and a calling turn cannot source its
own value, so session context must be planned on the merged conversation.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from llm_workflow_agents.data._workflow_script import _STATE_RE
from llm_workflow_agents.data.source_traceability import (
    _IDENTIFIER_RE,
    DEFAULT_FACT_ALLOWLIST,
    collect_identifiers,
    find_unsourced_argument_values,
)
from llm_workflow_agents.data.state_convention import parse_assistant_turns

# A prefix and one digit run, nothing else: CUST-882, POL998877, TX_101.
_SIMPLE_IDENTIFIER_RE = re.compile(r"([A-Z]{2,}[-_]?)(\d{2,})")
# Enumerate every candidate when the space is this small, so a seeded draw
# always finds a free value instead of giving up after unlucky tries.
_ENUMERATE_LIMIT = 200_000
_RANDOM_TRIES = 10_000


@dataclass(frozen=True)
class RemapDecision:
    """What happens to one identifier: remapped to ``new``, or skipped and why."""

    old: str
    new: str | None
    reason: str   # "remapped" | "general_knowledge" | "complex_shape" | "digits_referenced_elsewhere" | "no_free_value"


# --------------------------------------------------------------------------- helpers


def _body(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in sample.get("messages", []) if m.get("role") != "system"]


def _bounded(identifier: str) -> str:
    """Regex for ``identifier`` as a whole token, never part of a longer one."""
    return rf"(?<![A-Za-z0-9_-]){re.escape(identifier)}(?![A-Za-z0-9_-])"


def _map_strings(value: Any, fn) -> Any:
    """Apply ``fn`` to every string leaf of a JSON-like value."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    return value


def _has_call(message: dict[str, Any]) -> bool:
    return (
        "<tool_call>" in (message.get("content") or "")
        or bool((message.get("annotations") or {}).get("tool_calls"))
        or bool(message.get("tool_calls"))
    )


def _is_barge_in(message: dict[str, Any]) -> bool:
    return "loss" in message or "<unspoken>" in (message.get("content") or "")


# --------------------------------------------------------------------------- 1. identifiers


def _fresh_digits(length: int, rng: random.Random, prefix: str, unavailable: set[str]) -> str | None:
    low, high = 10 ** (length - 1), 10**length - 1
    if high - low + 1 <= _ENUMERATE_LIMIT:
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
    else:
        candidates = (rng.randint(low, high) for _ in range(_RANDOM_TRIES))
    for number in candidates:
        if f"{prefix}{number}" not in unavailable:
            return str(number)
    return None


def plan_identifier_remap(
    sample: dict[str, Any],
    vocabulary_text: str,
    forbidden: set[str],
    rng: random.Random,
    taken: set[str],
    keep: frozenset[str] = DEFAULT_FACT_ALLOWLIST,
) -> list[RemapDecision]:
    """Decide a fresh value for each simple entity identifier in ``sample``.

    ``vocabulary_text`` (the rendered prompt) excludes state names, tool names
    and enum values. A new value keeps the prefix and digit count, and is never
    in ``forbidden`` (every identifier in the training corpus and the benchmark)
    or ``taken`` (values already handed out). ``taken`` is updated in place so
    one set shared across a whole corpus keeps new values unique.

    Tokens in ``keep`` are general knowledge, not entity identifiers, and are
    never rewritten: ``AES-256`` is identifier-shaped, and remapping it once
    turned "AES-256 encryption" into "AES-616 encryption".
    """
    body = _body(sample)
    identifiers = sorted(collect_identifiers(body, vocabulary_text=vocabulary_text))
    text = " ".join(json.dumps(m, ensure_ascii=False) for m in body)
    text += " " + json.dumps(sample.get("ground_truth", {}), ensure_ascii=False)
    # Digits spoken on their own ("ending 882") are what a rewrite would miss.
    bare_text = _IDENTIFIER_RE.sub(" ", text)

    decisions: list[RemapDecision] = []
    for old in identifiers:
        if old in keep:
            decisions.append(RemapDecision(old, None, "general_knowledge"))
            continue
        match = _SIMPLE_IDENTIFIER_RE.fullmatch(old)
        if not match:
            decisions.append(RemapDecision(old, None, "complex_shape"))
            continue
        prefix, digits = match.groups()
        if re.search(rf"(?<!\d){digits}(?!\d)", bare_text):
            decisions.append(RemapDecision(old, None, "digits_referenced_elsewhere"))
            continue
        fresh = _fresh_digits(len(digits), rng, prefix, forbidden | taken | {old})
        if fresh is None:
            decisions.append(RemapDecision(old, None, "no_free_value"))
            continue
        new = f"{prefix}{fresh}"
        taken.add(new)
        decisions.append(RemapDecision(old, new, "remapped"))
    return decisions


def apply_identifier_remap(sample: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rewrite every occurrence of each ``old -> new`` pair; return a new sample.

    Rewrites conversation messages (content, annotations, structured calls),
    ``ground_truth`` and ``session_context``. The system message, tool schemas
    and workflow graph are instruction vocabulary and are left untouched.
    """
    out = copy.deepcopy(sample)
    if not mapping:
        return out
    pattern = re.compile("|".join(_bounded(old) for old in sorted(mapping, key=len, reverse=True)))

    def rewrite(text: str) -> str:
        return pattern.sub(lambda m: mapping[m.group(0)], text)

    out["messages"] = [
        m if m.get("role") == "system" else _map_strings(m, rewrite) for m in out.get("messages", [])
    ]
    for key in ("ground_truth", "session_context"):
        if key in out:
            out[key] = _map_strings(out[key], rewrite)
    return out


# --------------------------------------------------------------------------- 2. session context


def plan_session_context(sample: dict[str, Any], prompt_text: str) -> dict[str, str]:
    """Session-context facts for every confident unsourced tool argument.

    Keyed by argument name; a second, different value for the same argument
    gets ``<name>_2``. A value already stated is not stated again.
    """
    existing = dict(sample.get("session_context") or {})
    findings = find_unsourced_argument_values(
        _body(sample), sample.get("tool_schemas"), prompt_text, existing
    )
    planned: dict[str, str] = {}
    stated = set(map(str, existing.values()))
    for finding in findings:
        if finding.confidence != "confident" or finding.value in stated:
            continue
        key, n = finding.argument or "value", 1
        while key in existing or key in planned:
            n += 1
            key = f"{finding.argument}_{n}"
        planned[key] = finding.value
        stated.add(finding.value)
    return planned


# --------------------------------------------------------------------------- 3. stay merges


def plan_stay_merges(sample: dict[str, Any]) -> list[tuple[int, ...]]:
    """Runs of adjacent self-loop turns in one state that end in a tool call.

    Returns message indices (into the full list, system message included).
    Only a run whose last turn carries the call and whose earlier turns are
    prose is planned; a run touching a barge-in turn never is.
    """
    messages = sample.get("messages", [])
    labels = parse_assistant_turns(messages)
    runs: list[tuple[int, ...]] = []
    run: list[int] = []

    def close() -> None:
        if len(run) >= 2:
            first_state = labels[run[0]].from_state  # type: ignore[union-attr]
            *prose, last = run
            if (
                _has_call(messages[last])
                and not any(_has_call(messages[i]) for i in prose)
                and not any(_is_barge_in(messages[i]) for i in run)
                and all(labels[i].from_state == first_state for i in run)  # type: ignore[union-attr]
            ):
                runs.append(tuple(run))

    for index, label in enumerate(labels):
        is_stay = label is not None and label.from_state == label.to_state
        continues = (
            is_stay
            and run
            and index == run[-1] + 1
            and labels[run[-1]].from_state == label.from_state  # type: ignore[union-attr]
        )
        if continues:
            run.append(index)
            continue
        close()
        run = [index] if is_stay else []
    close()
    return runs


def _after_marker(content: str) -> str:
    match = _STATE_RE.search(content)
    return (content[match.end():] if match else content).strip()


def apply_stay_merges(sample: dict[str, Any], runs: list[tuple[int, ...]]) -> dict[str, Any]:
    """Merge each planned run into its final (calling) turn; return a new sample.

    ``ground_truth.state_sequence`` loses one self-loop entry per absorbed turn.
    It must be aligned one-to-one with the annotated assistant turns, or this
    raises rather than guess which entry to drop.
    """
    out = copy.deepcopy(sample)
    if not runs:
        return out
    messages = out["messages"]
    labels = parse_assistant_turns(messages)
    ordinal = {label.msg_index: n for n, label in enumerate(l for l in labels if l is not None)}
    sequence = out.get("ground_truth", {}).get("state_sequence")
    if sequence is None or len(sequence) != len(ordinal):
        raise ValueError(
            f"state_sequence has {0 if sequence is None else len(sequence)} entries for "
            f"{len(ordinal)} annotated assistant turns; refusing to merge"
        )

    drop_messages: list[int] = []
    drop_entries: list[int] = []
    for run in runs:
        *prose, last = run
        state = labels[last].from_state  # type: ignore[union-attr]
        final = messages[last]
        marker = _STATE_RE.search(final["content"]).group(0)
        pieces = [_after_marker(messages[i]["content"]) for i in prose] + [_after_marker(final["content"])]
        final["content"] = marker + "\n" + "\n".join(p for p in pieces if p)
        for i in prose:
            entry = sequence[ordinal[i]]
            if (entry.get("from"), entry.get("to")) != (state, state):
                raise ValueError(f"state_sequence entry {ordinal[i]} is {entry}, expected a {state} self-loop")
            drop_messages.append(i)
            drop_entries.append(ordinal[i])

    for i in sorted(drop_messages, reverse=True):
        del messages[i]
    for n in sorted(drop_entries, reverse=True):
        del sequence[n]
    return out
