"""Single source of truth for the tool-call state-annotation convention.

Convention: an assistant turn that emits <tool_call> must annotate a
self-loop [STATE: X -> X]. The advance to a new state happens on a LATER
turn, after the tool result has been seen. See
docs/cat_a_state_annotation_convention_review.md and
docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.

Pure stdlib, no heavy imports, so quality_profiler, data_validator,
generate_workflows, and the remediation scripts can all import this without
import cycles (mirrors the posture of _workflow_script.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_workflow_agents.data._workflow_script import _STATE_RE

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class TurnLabel:
    """One assistant turn's state annotation and tool-call shape."""

    msg_index: int
    turn_index: int          # 1-based ordinal among assistant turns
    from_state: str
    to_state: str
    arrow: str                # "→" or "->", verbatim from the row
    has_tool_call: bool
    prose_prefix: str         # text between the marker and the first <tool_call>; "" if none
    tail: str                 # from the first "<tool_call>" onward, verbatim; "" if none
    tool_names: tuple[str, ...]


def _extract_tool_name(raw_json: str) -> str | None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    name = parsed.get("name")
    return name if isinstance(name, str) else None


def parse_assistant_turns(messages: list[dict[str, Any]]) -> list[TurnLabel | None]:
    """Parse every assistant message into a TurnLabel; None if unparsable.

    Returns one entry per entry in ``messages`` (same length, same order),
    so ``labels[i]`` corresponds to ``messages[i]``. Non-assistant messages
    and assistant messages with no ``[STATE: X -> Y]`` marker are ``None``.
    """
    labels: list[TurnLabel | None] = []
    turn_index = 0
    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            labels.append(None)
            continue
        turn_index += 1
        content = msg.get("content") or ""
        m = _STATE_RE.search(content)
        if not m:
            labels.append(None)
            continue
        arrow = "->" if content[m.start():m.end()].find("->") != -1 else "→"
        after_marker = content[m.end():]
        tc_match = _TOOL_CALL_RE.search(after_marker)
        if tc_match:
            prose_prefix = after_marker[: tc_match.start()].strip()
            tail = after_marker[tc_match.start():].strip()
            tool_names = tuple(
                name
                for tc in _TOOL_CALL_RE.finditer(after_marker)
                if (name := _extract_tool_name(tc.group(1))) is not None
            )
        else:
            prose_prefix, tail, tool_names = "", "", ()
        labels.append(
            TurnLabel(
                msg_index=msg_index,
                turn_index=turn_index,
                from_state=m.group(1),
                to_state=m.group(2),
                arrow=arrow,
                has_tool_call=tc_match is not None,
                prose_prefix=prose_prefix,
                tail=tail,
                tool_names=tool_names,
            )
        )
    return labels


def find_tool_stay_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Return one message per tool-call turn that advances instead of staying.

    Reads inline content markers only (via parse_assistant_turns), consistent
    with _backfill_annotations' rule that content is authoritative over the
    structured annotations dict. An empty list means every tool-call turn in
    this conversation already conforms.
    """
    violations: list[str] = []
    for label in parse_assistant_turns(messages):
        if label is None or not label.has_tool_call:
            continue
        if label.from_state != label.to_state:
            violations.append(
                f"assistant turn {label.turn_index} issues a <tool_call> but "
                f"annotates an advancing transition [{label.from_state} -> "
                f"{label.to_state}]; a tool-execution turn must annotate "
                f"[{label.from_state} -> {label.from_state}] and advance on "
                f"a later turn"
            )
    return violations
