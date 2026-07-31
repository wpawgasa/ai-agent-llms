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

from llm_workflow_agents.data.state_convention import TurnLabel, parse_assistant_turns

Move = Literal[
    "none",
    "relabel",
    "split_fused_tool_turn",
    "insert_handoff_turn",
    "append_closing_pair",
    "drop",
]

# The design spec's move ladder, "ranked by cost" (§Move ladder). Used to pick
# between the two walks in plan_repair: never spend an authored message on a
# conversation a cheaper structural move can fix, and never restructure a
# message when a pure relabel already works.
_MOVE_COST: dict[str, int] = {
    "none": 0,
    "relabel": 1,
    "split_fused_tool_turn": 2,
    "insert_handoff_turn": 3,
    "append_closing_pair": 4,
    "drop": 5,
}


@dataclass(frozen=True)
class TurnPlan:
    """One post-repair assistant turn, mapped back to its source (if any)."""

    source_turn_index: int | None   # index into parse_assistant_turns()'s output; None if inserted
    from_state: str
    to_state: str
    arrow: str
    content_op: Literal["relabel", "split_head", "split_tail", "insert", "verbatim"]


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


def _walk(
    labels: list[TurnLabel],
    messages: list[dict[str, Any]],
    declared_edges: set[tuple[str, str]],
    terminals: set[str],
    conversation_id: str,
    *,
    allow_split: bool,
) -> RepairPlan:
    """One pass of the whole-trajectory requeue walk.

    ``allow_split`` selects ladder move 2: when True, a fused advancing tool
    turn (prose before its ``<tool_call>``) is broken into a prose turn that
    carries the advance and a tool turn that self-loops in the new state, which
    resolves the advance immediately instead of queueing it. When False, fused
    turns are requeued exactly like bare ones.
    """
    cur = labels[0].from_state
    pending: deque[str] = deque()
    turns: list[TurnPlan] = []
    drift_turns: list[int] = []
    any_relabelled = False
    any_split = False

    for label in labels:
        if label.has_tool_call:
            if cur != label.from_state:
                # Stacked tool turn: no prose turn in between to carry the
                # requeued advance, and (in this pass) nothing splittable, so
                # one authored assistant message must bridge cur -> from_state.
                return RepairPlan(
                    conversation_id=conversation_id, move="insert_handoff_turn",
                    inserts=[InsertRequest(
                        insert_id="", position_after_msg_index=label.msg_index - 1,
                        role="assistant",
                        required_marker=_marker(cur, label.from_state, label.arrow),
                    )],
                )
            if label.to_state != label.from_state:
                any_relabelled = True
                if allow_split and label.prose_prefix:
                    any_split = True
                    turns.append(TurnPlan(label.msg_index, cur, label.to_state,
                                          label.arrow, "split_head"))
                    turns.append(TurnPlan(label.msg_index, label.to_state, label.to_state,
                                          label.arrow, "split_tail"))
                    cur = label.to_state
                else:
                    # Force the self-loop; displace the advance onto the next
                    # prose turn.
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

    move: Move = "split_fused_tool_turn" if any_split else "relabel"
    return RepairPlan(conversation_id=conversation_id, move=move, turns=turns,
                      drift_turns=drift_turns)


def plan_repair(record: dict[str, Any]) -> RepairPlan:
    """Plan the whole-trajectory requeue repair for one conversation record.

    Pure: never mutates ``record``. See the design spec's "Core algorithm:
    whole-trajectory requeue" section.

    Walks the trajectory twice at most and keeps the cheapest feasible move
    (``_MOVE_COST``, the spec's cost-ranked ladder): the first walk uses pure
    requeue, and only if that needs authored text is a second walk tried with
    fused-turn splitting enabled. So a conversation is never restructured when
    a plain relabel suffices, and never sent to a human/agent author when a
    zero-authored-text structural move suffices.

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

    plan = _walk(labels, messages, declared_edges, terminals, conversation_id,
                 allow_split=False)
    if _MOVE_COST[plan.move] >= _MOVE_COST["insert_handoff_turn"]:
        alternative = _walk(labels, messages, declared_edges, terminals, conversation_id,
                            allow_split=True)
        if _MOVE_COST[alternative.move] < _MOVE_COST[plan.move]:
            plan = alternative
    return plan
