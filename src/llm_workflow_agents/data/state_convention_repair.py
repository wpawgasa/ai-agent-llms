"""Deterministic repair of the tool-call stay convention (see
state_convention.py and docs/superpowers/specs/2026-07-31-task-a-tool-stay-
convention-design.md for the algorithm and invariant table).

plan_repair() is pure and never mutates its input. apply_plan() (Task 3)
consumes a RepairPlan and returns a new record.
"""

from __future__ import annotations

import copy
import json
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


def _errored(message: dict[str, Any] | None) -> bool:
    """True if ``message`` is a ``role: "tool"`` result reporting a failure.

    The corpus emits errors as ``{"error": "..."}`` (see
    ``generate_workflows.py``'s teacher prompt), but a tool result is untrusted
    text, so a payload that will not parse falls back to a substring test
    rather than being silently treated as a success.
    """
    if not message or message.get("role") != "tool":
        return False
    content = message.get("content") or ""
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return '"error"' in content
    return isinstance(payload, dict) and (
        "error" in payload or payload.get("status") == "error"
    )


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

    A bridge is an assistant *prose* message spliced between two existing
    messages, so it can create an illegal consecutive-assistant-prose pair on
    either side (see ``find_shape_violations``). Each side that would break is
    padded with a short user-role acknowledgement, so one stacked-tool
    infeasibility contributes one, two, or three ``InsertRequest``s. Inserts
    that share a ``position_after_msg_index`` appear in the repaired
    conversation in the order they were appended here -- ``apply_plan`` breaks
    position ties by reverse list index to guarantee that.

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
            if cur != label.from_state and _errored(
                messages[label.msg_index - 1] if label.msg_index >= 1 else None
            ):
                # RETRY AFTER A TOOL ERROR — do not bridge, do not advance.
                #
                # Convention rule 4: on a tool error the turn stays in its
                # state and may retry; the advance happens only after a
                # success. The generic stacked-tool branch below would instead
                # author a bridge asserting cur -> from_state directly after
                # `{"error": ...}`, which is (a) semantically backwards, and
                # (b) an impossible authoring request -- corpus-remediator.md
                # forbids narrating success after an error, so the agent could
                # only refuse or hallucinate. Measured on the real corpus: 107
                # of 912 advancing bridges landed after an errored result (104
                # same-tool retries, 3 a different tool; advancing is wrong in
                # both, so the rule is "never drain an advance across a tool
                # error" rather than "never drain across a same-tool retry").
                #
                # Leaving the already-queued advance in `pending` is the whole
                # fix: it survives the retry and drains at the first prose turn
                # after a SUCCESSFUL result, which is where the existing prose
                # already says the step completed.
                #
                # The retry turn's OWN advance still has to be queued behind
                # it, exactly as the ordinary tool-turn branch below does.
                # Dropping it on the floor would be silent corruption rather
                # than a caught failure: `verify_repaired` re-derives
                # ground_truth from the repaired markers, so a trajectory
                # missing a transition agrees with itself and passes every
                # gate. Caught on the real corpus -- a 4-deep stacked chain
                # with an error in the middle (`L4_058_2`) lost three states.
                any_relabelled = True
                if label.to_state != label.from_state:
                    pending.append(label.to_state)
                turns.append(TurnPlan(label.msg_index, cur, cur, label.arrow, "relabel"))
                continue
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
                # valid as it goes. Inserts sharing one position come out in
                # list order (apply_plan breaks position ties by reverse list
                # index), so the order they are appended here is the order they
                # appear in the repaired conversation.
                bridge_pos = label.msg_index - 1
                # A bridge is an assistant PROSE message spliced between two
                # existing ones, and find_shape_violations rejects two
                # assistant messages in a row unless the LATER one is a *pure*
                # tool-call turn. So the splice can break the shape on either
                # side, and each side is repaired by a short user
                # acknowledgement that makes the adjacency assistant->user /
                # user->assistant (both always legal).
                #
                # Leading edge: the message the bridge lands after is itself an
                # assistant turn, so (that turn, bridge) would be two prose
                # turns in a row. 15 such violations on the real corpus, all
                # with a BARE successor -- i.e. not reachable via the
                # prose_prefix test below, which is why both checks exist.
                if bridge_pos >= 0 and messages[bridge_pos].get("role") == "assistant":
                    inserts.append(InsertRequest(
                        insert_id="", position_after_msg_index=bridge_pos,
                        role="user", required_marker="",
                    ))
                inserts.append(InsertRequest(
                    insert_id="", position_after_msg_index=bridge_pos,
                    role="assistant",
                    required_marker=_marker(cur, label.from_state, label.arrow),
                ))
                # Trailing edge: the stacked turn is FUSED (prose before its
                # <tool_call>), so (bridge, stacked turn) would be two prose
                # turns in a row. A BARE stacked turn is a pure tool-call turn,
                # explicitly allowed after an assistant message, so it needs no
                # ack. ``prose_prefix`` is exactly find_shape_violations'
                # predicate: it is "" iff the content after the marker lstrips
                # to a leading "<tool_call>".
                #
                # The alternative -- relaxing find_shape_violations for a
                # bridge followed by a fused turn -- was rejected deliberately:
                # that function is shared with the teacher-facing generator, so
                # loosening it would widen what the TEACHER may emit, a much
                # bigger decision than a corpus-remediation bug fix.
                if label.prose_prefix:
                    inserts.append(InsertRequest(
                        insert_id="", position_after_msg_index=bridge_pos,
                        role="user", required_marker="",
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
        # Inserts are applied back-to-front so earlier positions stay valid.
        #
        # The ``-i`` tie-break is load-bearing, not cosmetic. Several inserts
        # can share one ``position_after_msg_index`` (a bridge plus its user
        # ack), and each insert at a given slot shoves the previously inserted
        # one to the right. Applying same-position inserts in REVERSE list
        # order therefore leaves them in LIST order in the output, which is the
        # contract plan_repair is written against. A plain stable sort on
        # ``-position`` alone silently reverses each such group -- measured:
        # ["first", "second", "third"] comes out ["third", "second", "first"]
        # (tests/unit/test_state_convention_repair.py::
        # test_apply_plan_inserts_at_the_same_position_keep_their_list_order).
        for _i, req in sorted(
            enumerate(plan.inserts), key=lambda p: (-p[1].position_after_msg_index, -p[0])
        ):
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
