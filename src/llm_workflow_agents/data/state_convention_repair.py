"""Deterministic repair of the tool-call stay convention (see
state_convention.py and docs/superpowers/specs/2026-07-31-task-a-tool-stay-
convention-design.md for the algorithm and invariant table).

plan_repair() is pure and never mutates its input. apply_plan() (Task 3)
consumes a RepairPlan and returns a new record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_workflow_agents.data.state_convention import parse_assistant_turns

Move = Literal[
    "none",
    "relabel",
    "insert_handoff_turn",
    "append_closing_pair",
    "drop",
]


@dataclass(frozen=True)
class TurnPlan:
    """One post-repair assistant turn, mapped back to its source (if any)."""

    source_turn_index: int | None   # index into parse_assistant_turns()'s output; None if inserted
    from_state: str
    to_state: str
    arrow: str
    content_op: Literal["relabel", "insert", "verbatim"]


@dataclass(frozen=True)
class InsertRequest:
    """One message the corpus-remediator agent must author (Task 6/13)."""

    insert_id: str                       # "<file_stem>:<line_index>:<ordinal>", filled by the CLI (Task 4)
    position_after_msg_index: int         # insert immediately after this index in messages[]
    role: Literal["user", "assistant"]
    required_marker: str                  # e.g. "[STATE: B -> B]"; "" for a user-role insert


@dataclass
class RepairPlan:
    key: tuple[str, int] = ("", -1)       # (file_stem, line_index); filled by the CLI, not here
    conversation_id: str = ""
    move: Move = "none"
    turns: list[TurnPlan] = field(default_factory=list)
    inserts: list[InsertRequest] = field(default_factory=list)
    drift_turns: list[int] = field(default_factory=list)
    infeasible_reason: str | None = None


def _marker(from_state: str, to_state: str, arrow: str) -> str:
    """The literal marker text an authored insert's content must start with.

    Must agree byte for byte with ``apply_plan``'s ``_new_marker`` (Task 3) and
    with the ledger gate's ``content.startswith(required_marker)`` check
    (Task 12), arrow glyph included.
    """
    return f"[STATE: {from_state} {arrow} {to_state}]"


def plan_repair(record: dict[str, Any]) -> RepairPlan:
    """Plan the whole-trajectory requeue repair for one conversation record.

    Pure: never mutates ``record``. See the design spec's "Core algorithm:
    whole-trajectory requeue" section.

    A single walk, no move search: every displaced tool turn — **fused or
    bare** — relabels in place to a self-loop at its own original
    ``from_state``, keeping prose and ``<tool_call>`` together in one message.
    That is what makes the tool-placement invariant hold *by construction*: a
    tool call is never moved to a different message at a different state, so
    ``infer_state_tools_from_messages``' attribution cannot change. An earlier
    design split fused turns (``split_fused_tool_turn``); it was removed after
    measurement showed it re-attributed ~66% of its tool calls to the
    destination state.

    Index conventions (two different spaces, deliberately):
      * ``TurnPlan.source_turn_index`` is a **message** index — an index into
        ``parse_assistant_turns(messages)``'s output, which Task 1 returns
        message-aligned (one entry per input message, ``None`` for
        non-assistant/unparsable ones). This is what ``apply_plan`` needs in
        order to rewrite ``messages[i]`` in place.
      * ``RepairPlan.drift_turns`` holds 1-based **assistant-turn** ordinals
        (``TurnLabel.turn_index``), because it is a human-review pointer
        ("assistant turn 4 reads oddly now"), not an addressing handle.
    """
    messages = record["messages"]
    wg = record["workflow_graph"]
    declared_edges = {(t["from"], t["to"]) for t in wg["transitions"]}
    terminals = set(wg["terminal"])
    conversation_id = record.get("conversation_id", "")

    labels = [label for label in parse_assistant_turns(messages) if label is not None]
    if not labels:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                          infeasible_reason="no parsable assistant turns")

    cur = labels[0].from_state
    pending: deque[str] = deque()
    turns: list[TurnPlan] = []
    drift_turns: list[int] = []
    any_relabelled = False

    for label in labels:
        if label.has_tool_call:
            if cur != label.from_state:
                # Stacked tool turn: no prose turn in between to carry the
                # requeued advance, so one authored assistant message must
                # bridge cur -> from_state. Covers bare-after-bare and
                # bare-after-relabelled-fused alike (a fused turn's
                # relabel-in-place does not advance cur).
                return RepairPlan(
                    conversation_id=conversation_id, move="insert_handoff_turn",
                    inserts=[InsertRequest(
                        insert_id="", position_after_msg_index=label.msg_index - 1,
                        role="assistant",
                        required_marker=_marker(cur, label.from_state, label.arrow),
                    )],
                )
            if label.to_state != label.from_state:
                # Relabel the WHOLE message -- prose and <tool_call>, if fused,
                # stay together unchanged -- to a self-loop at its ORIGINAL
                # from_state, and displace the advance onto the next prose
                # turn. Never split a fused turn: moving the tool call into a
                # different message at the destination state would change
                # infer_state_tools_from_messages' attribution for that tool,
                # breaking the "tool from-state never changes" invariant.
                any_relabelled = True
                pending.append(label.to_state)
                turns.append(TurnPlan(label.msg_index, cur, cur, label.arrow, "relabel"))
            else:
                turns.append(TurnPlan(label.msg_index, cur, cur, label.arrow, "verbatim"))
        else:
            if pending:
                nxt = pending.popleft()
                if nxt != label.to_state or label.from_state != cur:
                    drift_turns.append(label.turn_index)
                turns.append(TurnPlan(label.msg_index, cur, nxt, label.arrow, "relabel"))
                cur = nxt
                if label.to_state != label.from_state and label.to_state != nxt:
                    pending.append(label.to_state)
            else:
                turns.append(TurnPlan(label.msg_index, cur, label.to_state,
                                      label.arrow, "verbatim"))
                cur = label.to_state

    if len(pending) > 1:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                          infeasible_reason=f"deficit of {len(pending)} states at end (only 1 supported)")

    for tp in turns:
        if tp.from_state != tp.to_state and (tp.from_state, tp.to_state) not in declared_edges:
            return RepairPlan(
                conversation_id=conversation_id, move="drop",
                infeasible_reason=f"undeclared transition [{tp.from_state} -> {tp.to_state}]",
            )

    if pending:
        deficit = pending[0]
        # The appended closing turn emits (cur, deficit); it is subject to the
        # same "declared edge" and "last emitted .to is a terminal" rules as
        # every other emitted pair.
        if cur != deficit and (cur, deficit) not in declared_edges:
            return RepairPlan(
                conversation_id=conversation_id, move="drop",
                infeasible_reason=f"undeclared transition [{cur} -> {deficit}]",
            )
        if deficit not in terminals:
            return RepairPlan(conversation_id=conversation_id, move="drop",
                              infeasible_reason=f"final state '{deficit}' is not a terminal")
        return RepairPlan(
            conversation_id=conversation_id, move="append_closing_pair",
            inserts=[
                InsertRequest(insert_id="", position_after_msg_index=len(messages) - 1,
                              role="user", required_marker=""),
                InsertRequest(insert_id="", position_after_msg_index=len(messages),
                              role="assistant",
                              required_marker=_marker(cur, deficit, labels[-1].arrow)),
            ],
        )

    if cur not in terminals:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                          infeasible_reason=f"final state '{cur}' is not a terminal")

    if not any_relabelled:
        return RepairPlan(conversation_id=conversation_id, move="none")

    return RepairPlan(conversation_id=conversation_id, move="relabel", turns=turns,
                      drift_turns=drift_turns)
