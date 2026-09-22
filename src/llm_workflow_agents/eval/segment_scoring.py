"""Score a replayed Task A conversation by segment, not by assistant turn.

A *segment* is a maximal run of ground-truth assistant turns with no user
message or tool result between them. The corpus splits one piece of agent
work across such runs whenever it wants speech and a tool call on separate
turns — ``[STATE: W → X] Let me check that.`` then ``[STATE: X → X]
<tool_call>...`` — but nothing in the served prompt tells a model to do
that, and a model that says both in one reply has done the same work.

Scoring turn by turn therefore paid a model for matching the corpus's
split, not for the work: a combined reply scored a spurious call on the
speech turn and a missing call on the call turn. The harness also could not
ask for the second turn of a pair at all (the context ended on the model's
own reply) and copied it from ground truth, where it scored as perfect.

Here each segment is scored as one turn on both sides:

- **Tool calls** — the segment's ground-truth calls against every call the
  model made while answering it, in any of its replies.
- **State** — one transition per segment, from the state the segment opens in
  to the state it ends in, and only if the model got there legally: its
  annotations must chain (each one starts where the previous one ended) and
  every tool call must sit under a self-loop annotation — the tool-call stay
  convention, which the served prompt states. A segment that breaks either
  rule gets the transition ``[STATE: <from> → ILLEGAL_TRANSITION]``, which
  matches no ground truth and ends no conversation.

The replay (``agent_benchmark._replay_conversation``) keeps predictions
aligned with the ground truth message by message: a segment's replies are
joined into its first slot and its other slots are left empty. A segment the
model could not be asked for at all — an opening turn on an API that refuses
a request with no user message — carries ``"unscored": True`` and is dropped
from BOTH views, so it neither helps nor hurts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llm_workflow_agents.eval.state_accuracy import STATE_ANNOTATION_PATTERN

ILLEGAL_STATE = "ILLEGAL_TRANSITION"

_STATE_RE = re.compile(STATE_ANNOTATION_PATTERN)
_TOOL_CALL_RE = re.compile(r"<tool_call>")


@dataclass
class SegmentStats:
    """Counts over every segment scored in one or more conversations."""

    segments: int = 0
    multi_turn_segments: int = 0
    unscored_segments: int = 0
    illegal_discontinuous: int = 0
    illegal_stay_rule: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "segments": self.segments,
            "multi_turn_segments": self.multi_turn_segments,
            "unscored_segments": self.unscored_segments,
            "illegal_discontinuous": self.illegal_discontinuous,
            "illegal_stay_rule": self.illegal_stay_rule,
        }


def segment_bounds(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """``[start, end)`` index pairs of every maximal run of assistant messages."""
    bounds: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") != "assistant":
            i += 1
            continue
        j = i
        while j < len(messages) and messages[j].get("role") == "assistant":
            j += 1
        bounds.append((i, j))
        i = j
    return bounds


def normalize_segment_states(content: str) -> tuple[str, str | None]:
    """Collapse a segment's state annotations into one legal transition.

    Returns ``(content, problem)``. With no annotation the content is returned
    unchanged. Otherwise every annotation is removed and one is written where
    the first stood: ``[STATE: first.from → last.to]`` when the chain is legal,
    ``[STATE: first.from → ILLEGAL_TRANSITION]`` when it is not. ``problem`` is
    ``None``, ``"discontinuous"`` or ``"stay_rule"``.
    """
    matches = list(_STATE_RE.finditer(content))
    if not matches:
        return content, None

    problem: str | None = None
    for prev, cur in zip(matches, matches[1:]):
        if cur.group(1) != prev.group(2):
            problem = "discontinuous"
            break
    if problem is None:
        for call in _TOOL_CALL_RE.finditer(content):
            governing = [m for m in matches if m.start() < call.start()]
            if governing and governing[-1].group(1) != governing[-1].group(2):
                problem = "stay_rule"
                break

    first, last = matches[0], matches[-1]
    target = last.group(2) if problem is None else ILLEGAL_STATE
    replacement = f"[STATE: {first.group(1)} → {target}]"

    pieces: list[str] = []
    cursor = 0
    for k, m in enumerate(matches):
        pieces.append(content[cursor:m.start()])
        if k == 0:
            pieces.append(replacement)
        cursor = m.end()
    pieces.append(content[cursor:])
    return "".join(pieces), problem


def _merge_ground_truth(turns: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[Any] = []
    transitions: list[dict[str, Any]] = []
    for turn in turns:
        annotations = turn.get("annotations") or {}
        calls.extend(annotations.get("tool_calls") or [])
        if annotations.get("state_transition"):
            transitions.append(annotations["state_transition"])
    merged_annotations: dict[str, Any] = {"tool_calls": calls}
    if transitions:
        merged_annotations["state_transition"] = {
            "from": transitions[0].get("from", ""),
            "to": transitions[-1].get("to", ""),
        }
    return {
        "role": "assistant",
        "content": "\n".join(t.get("content") or "" for t in turns),
        "annotations": merged_annotations,
    }


def segment_scoring_view(
    ground_truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    stats: SegmentStats | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(gt_view, pred_view)`` with one message per segment on each side.

    Non-assistant messages pass through unchanged. The two views have equal
    length, so every per-turn metric can keep pairing them by position.
    """
    if len(ground_truth) != len(predicted):
        raise ValueError(
            f"ground truth has {len(ground_truth)} messages, prediction {len(predicted)}"
        )
    gt_view: list[dict[str, Any]] = []
    pred_view: list[dict[str, Any]] = []
    bounds = {start: end for start, end in segment_bounds(ground_truth)}
    i = 0
    while i < len(ground_truth):
        if i not in bounds:
            gt_view.append(ground_truth[i])
            pred_view.append(predicted[i])
            i += 1
            continue
        end = bounds[i]
        gt_turns, pred_turns = ground_truth[i:end], predicted[i:end]
        if stats is not None:
            stats.segments += 1
            stats.multi_turn_segments += end - i > 1
        if any(p.get("unscored") for p in pred_turns):
            if stats is not None:
                stats.unscored_segments += 1
            i = end
            continue
        joined = "\n".join(p.get("content") or "" for p in pred_turns if p.get("content"))
        content, problem = normalize_segment_states(joined)
        if stats is not None and problem == "discontinuous":
            stats.illegal_discontinuous += 1
        if stats is not None and problem == "stay_rule":
            stats.illegal_stay_rule += 1
        gt_view.append(_merge_ground_truth(gt_turns))
        pred_view.append({"role": "assistant", "content": content})
        i = end
    return gt_view, pred_view


def reply_texts(predicted: list[dict[str, Any]]) -> list[str]:
    """Every reply the model generated, one string per request, in order.

    For guardrails that measure the shape of a single reply (voice chunk
    diagnostics), which a segment's joined content would distort.
    """
    out: list[str] = []
    for msg in predicted:
        if msg.get("role") != "assistant" or msg.get("unscored"):
            continue
        if "replies" in msg:
            out.extend(msg["replies"])
        elif msg.get("content"):
            out.append(msg["content"])
    return out
