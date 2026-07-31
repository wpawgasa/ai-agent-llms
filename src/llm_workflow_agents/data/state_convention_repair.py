"""Deterministic repair of the tool-call stay convention (see
state_convention.py and docs/superpowers/specs/2026-07-31-task-a-tool-stay-
convention-design.md for the algorithm and invariant table).

plan_repair() is pure and never mutates its input. apply_plan() (Task 3)
consumes a RepairPlan and returns a new record.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_workflow_agents.data._workflow_script import (
    _STATE_RE,
    find_continuity_violations,
    find_shape_violations,
    find_tool_placement_violations,
    infer_state_tools_from_messages,
)
from llm_workflow_agents.data.state_convention import (
    find_tool_stay_violations,
    parse_assistant_turns,
)

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


@dataclass
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

    The walk is single-pass but **not** single-fix: every stacked-tool
    infeasibility it meets contributes its own bridge ``InsertRequest`` and the
    walk continues, so one plan repairs a whole conversation. A conversation
    that needs mid-conversation bridges *and* a tail close carries all of them
    in ``inserts`` under the ``append_closing_pair`` label. Callers that gate on
    ``move in ("insert_handoff_turn", "append_closing_pair")`` and treat
    ``inserts`` as a flat list of independent authoring requests need no change
    to handle N inserts instead of 1.

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
    inserts: list[InsertRequest] = []
    bridge_edges: list[tuple[str, str]] = []
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
                #
                # The walk then CONTINUES as if the bridge had already been
                # emitted (``cur = label.from_state``) instead of returning
                # here, so a conversation with several stacked-tool
                # infeasibilities gets every bridge planned in one pass. An
                # earlier version returned at the first one, which left the
                # rest of the conversation unplanned: verify_repaired caught
                # the leftovers, but ~half of a 1,150-conversation bucket then
                # had to be discarded as "unrepairable" despite being
                # repairable.
                #
                # position_after_msg_index is a PRE-insert message index for
                # every bridge, including the second and later ones;
                # apply_plan inserts back-to-front, so earlier positions stay
                # valid as it goes.
                inserts.append(InsertRequest(
                    insert_id="", position_after_msg_index=label.msg_index - 1,
                    role="assistant",
                    required_marker=_marker(cur, label.from_state, label.arrow),
                ))
                bridge_edges.append((cur, label.from_state))
                any_relabelled = True
                # The bridge turn CARRIES the displaced advance, exactly like
                # the prose-turn branch below, so it must consume the queue
                # head -- otherwise the same advance is emitted twice: once by
                # the bridge and again by the next prose turn, which then
                # cascades into a phantom tail deficit. Measured on the real
                # corpus: all 1,630 bridge events have pending[0] ==
                # label.from_state, and NOT draining here turns 426 otherwise
                # repairable conversations into drops.
                #
                # The == guard is deliberate. If the head does not match, the
                # source conversation is discontinuous in a way this walk
                # cannot model; leaving the entry queued lets the existing
                # "deficit of N states at end" / declared-edge gates below
                # reject it, rather than silently consuming the wrong advance.
                if pending and pending[0] == label.from_state:
                    pending.popleft()
                cur = label.from_state
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

    emitted_edges = [(tp.from_state, tp.to_state) for tp in turns] + bridge_edges
    for src, dst in emitted_edges:
        if src != dst and (src, dst) not in declared_edges:
            return RepairPlan(
                conversation_id=conversation_id, move="drop",
                infeasible_reason=f"undeclared transition [{src} -> {dst}]",
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
        # The loop has finished, so ``turns`` is the COMPLETE relabel set for
        # the conversation; the appended pair is on top of it, not instead of
        # it. Likewise the closing pair is APPENDED to any mid-conversation
        # bridges already planned above rather than replacing them: a
        # conversation that needs both ships both, under the single
        # ``append_closing_pair`` label. Task 4/11/12 gate on
        # ``move in ("insert_handoff_turn", "append_closing_pair")`` and treat
        # ``inserts`` as a flat list of independent authoring requests, so a
        # combined plan needs no special handling there.
        return RepairPlan(
            conversation_id=conversation_id, move="append_closing_pair",
            turns=turns, drift_turns=drift_turns,
            inserts=inserts + [
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

    if inserts:
        # Bridges only, no tail deficit. Reached only after the terminal and
        # declared-edge gates above, which the pre-accumulation version
        # skipped entirely by returning from inside the loop.
        return RepairPlan(
            conversation_id=conversation_id, move="insert_handoff_turn",
            turns=turns, drift_turns=drift_turns, inserts=inserts,
        )

    if not any_relabelled:
        return RepairPlan(conversation_id=conversation_id, move="none")

    return RepairPlan(conversation_id=conversation_id, move="relabel", turns=turns,
                      drift_turns=drift_turns)


def _replace_marker(content: str, new_marker: str) -> str:
    """Replace the first ``[STATE: X -> Y]`` marker in ``content`` verbatim.

    The lambda replacement is deliberate: ``re.sub`` would otherwise interpret
    backslash escapes in ``new_marker`` as group references.
    """
    return _STATE_RE.sub(lambda _m: new_marker, content, count=1)


def apply_plan(
    record: dict[str, Any],
    plan: RepairPlan,
    ledger_entries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a NEW record with ``plan`` applied, or None if required ledger
    entries are missing (or the plan's move is ``drop``). Never mutates
    ``record``.
    """
    if plan.move == "none":
        return copy.deepcopy(record)
    if plan.move == "drop":
        return None

    record = copy.deepcopy(record)
    messages = record["messages"]

    # Relabels are applied for EVERY move that carries them, not just
    # move == "relabel": an insert_handoff_turn / append_closing_pair plan can
    # also carry relabels decided for earlier turns of the same conversation.
    # They must run BEFORE the inserts, because TurnPlan.source_turn_index was
    # computed by plan_repair against the PRE-insert message list.
    if plan.turns:
        for tp in plan.turns:
            if tp.content_op == "verbatim":
                continue
            # source_turn_index is a MESSAGE index (TurnLabel.msg_index), not
            # an assistant-only ordinal -- see plan_repair's docstring.
            msg_index = tp.source_turn_index
            new_marker = _marker(tp.from_state, tp.to_state, tp.arrow)
            messages[msg_index]["content"] = _replace_marker(
                messages[msg_index]["content"], new_marker
            )
            messages[msg_index]["annotations"] = {
                **(messages[msg_index].get("annotations") or {}),
                "state_transition": {"from": tp.from_state, "to": tp.to_state},
            }

    if plan.move in ("insert_handoff_turn", "append_closing_pair"):
        if ledger_entries is None:
            return None
        # inserts are applied back-to-front to keep earlier positions valid
        for req in sorted(plan.inserts, key=lambda r: -r.position_after_msg_index):
            entry = ledger_entries.get(req.insert_id)
            if entry is None:
                return None
            messages.insert(req.position_after_msg_index + 1, {
                "role": req.role, "content": entry["content"], "annotations": None,
            })

    record["ground_truth"] = rederive_ground_truth(record)
    return record


def rederive_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    """Re-derive ground_truth from (possibly repaired) messages, merging into
    the existing dict so fields like ``terminal_reached`` survive. A no-op on
    the ground_truth of an already-conformant record.

    Side effect: normalises ``record["messages"]`` annotations in place via
    ``_backfill_annotations`` (the same normalisation the generator applies),
    so on a corpus row whose messages carry no ``annotations`` key this DOES
    add one. ``apply_plan`` calls this on its own deep copy, so callers going
    through ``apply_plan`` never see their input touched.
    """
    from llm_workflow_agents.data.generate_workflows import (
        WorkflowGraph, WorkflowState, WorkflowTransition,
        _backfill_annotations, _extract_ground_truth,
    )

    messages = record["messages"]
    _backfill_annotations(messages)
    wg = record["workflow_graph"]
    states = [WorkflowState(id=name, name=name) for name in wg["states"]]
    transitions = [
        WorkflowTransition(from_state=t["from"], to_state=t["to"],
                           condition=t.get("condition", ""), priority=t.get("priority", 0))
        for t in wg["transitions"]
    ]
    workflow = WorkflowGraph(
        states=states, transitions=transitions,
        initial_state=wg["initial"], terminal_states=list(wg["terminal"]),
    )
    fresh = _extract_ground_truth(messages, workflow)
    gt = dict(record.get("ground_truth") or {})
    gt.update(fresh)
    return gt


def verify_repaired(record: dict[str, Any]) -> list[str]:
    """The post-repair gate. Empty list == accept.

    ``allowed_by_state`` is inferred from the *post-repair* messages
    themselves, so ``find_tool_placement_violations`` can never fire from a
    state/tool mismatch here (a tool is by construction "allowed" in the state
    the repair attributed it to); its job in this function is structural, and
    the schema_names arm still catches a tool absent from ``tool_schemas``.
    The real invariant -- "the repair did not change which state any tool is
    called from" -- needs both the before and after messages, so it lives in
    Task 4's CLI as an equality check on
    ``infer_state_tools_from_messages``, not here.
    """
    messages = record["messages"]
    wg = record["workflow_graph"]
    violations: list[str] = []
    violations += find_tool_stay_violations(messages)
    violations += find_continuity_violations(messages, wg["initial"], set(wg["terminal"]))
    violations += find_shape_violations(messages, record.get("conversation_initiator", "user"))

    schema_names = {t["function"]["name"] for t in record.get("tool_schemas", [])} or None
    allowed_by_state = infer_state_tools_from_messages(messages)
    violations += find_tool_placement_violations(allowed_by_state, messages, schema_names)

    declared = {(t["from"], t["to"]) for t in wg["transitions"]}
    labels = [label for label in parse_assistant_turns(messages) if label is not None]
    for label in labels:
        if label.from_state != label.to_state and (label.from_state, label.to_state) not in declared:
            violations.append(f"undeclared transition [{label.from_state} -> {label.to_state}]")

    expected_seq = [{"from": label.from_state, "to": label.to_state} for label in labels]
    actual_seq = (record.get("ground_truth") or {}).get("state_sequence")
    if actual_seq != expected_seq:
        violations.append(
            f"ground_truth.state_sequence {actual_seq} does not match message "
            f"markers {expected_seq}"
        )
    return violations
