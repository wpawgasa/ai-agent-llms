import copy
import json

from llm_workflow_agents.data.state_convention_repair import InsertRequest, plan_repair
from llm_workflow_agents.data.state_convention import find_tool_stay_violations


def _record(messages, transitions, terminal=("TERMINAL",), initial="A"):
    return {
        "conversation_id": "T_001",
        "workflow_graph": {
            "states": sorted({t for pair in transitions for t in pair} | {initial, *terminal}),
            "transitions": [{"from": a, "to": b, "condition": "", "priority": 0} for a, b in transitions],
            "initial": initial,
            "terminal": list(terminal),
        },
        "messages": messages,
        "ground_truth": {"terminal_reached": True},
    }


def _amsg(content):
    return {"role": "assistant", "content": content, "annotations": None}


def test_conformant_record_yields_move_none():
    msgs = [
        {"role": "user", "content": "hi", "annotations": None},
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ]
    plan = plan_repair(_record(msgs, [("A", "TERMINAL")]))
    assert plan.move == "none"
    assert plan.turns == []


def test_case_a_two_turn_relabel():
    # tool turn wrongly claims A->B; the *next* turn stays at B->B.
    # Correct: tool turn A->A, next turn A->B.
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "relabel"
    by_index = {t.source_turn_index: t for t in plan.turns}
    assert by_index[0].from_state == "A" and by_index[0].to_state == "A"
    assert by_index[2].from_state == "A" and by_index[2].to_state == "B"


def test_fused_tool_turn_relabels_in_place_keeping_prose_and_tool_together():
    # A fused turn (prose + <tool_call> in one message) that wrongly advances
    # A->B relabels to a SINGLE self-loop A->A, keeping the prose and the
    # tool call together in the same message -- exactly like a bare tool
    # turn. This is deliberate: physically splitting the tool call into its
    # own message at the destination state would change which state the
    # tool is attributed to (infer_state_tools_from_messages), breaking the
    # "a tool's from-state never changes" invariant. See
    # docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    # Same shape as test_case_a_two_turn_relabel, just with a fused (prose +
    # tool_call) first turn instead of a bare one -- proving plan_repair
    # treats them identically.
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "relabel"
    tp = plan.turns[0]
    assert tp.from_state == "A" and tp.to_state == "A"
    assert tp.content_op == "relabel"


def test_stacked_tool_turn_after_a_fused_turn_needs_insert_handoff():
    # Because the fused turn above relabels WITHOUT advancing cur, a second,
    # already-bare tool turn immediately following it (originally marked
    # B->B, i.e. looking self-consistent on its own) cannot fire -- cur is
    # still A, not B -- so this is a stacked-tool infeasibility needing a
    # hand-off, exactly as if the fused turn had carried no prose at all.
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → B]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → TERMINAL]\nDone."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B"), ("B", "TERMINAL")]))
    assert plan.move == "insert_handoff_turn"
    assert plan.inserts[0].role == "assistant"
    assert "B" in plan.inserts[0].required_marker


def test_tail_deficit_requires_append_closing_pair():
    msgs = [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "TERMINAL")]))
    assert plan.move == "append_closing_pair"
    assert len(plan.inserts) == 2  # a user ack + the assistant closing turn
    assert plan.inserts[0].role == "user"
    assert plan.inserts[1].role == "assistant"
    assert "TERMINAL" in plan.inserts[1].required_marker


def test_bare_stacked_tool_turns_need_insert_handoff():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t1", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → TERMINAL]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "B"), ("B", "TERMINAL")]))
    assert plan.move == "insert_handoff_turn"
    assert len(plan.inserts) == 1
    assert plan.inserts[0].role == "assistant"
    assert "B" in plan.inserts[0].required_marker


def test_undeclared_edge_makes_plan_infeasible_and_drops():
    msgs = [
        _amsg('[STATE: A → Z]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "drop"
    assert plan.infeasible_reason is not None


def test_tool_from_state_never_changes():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {"x": 1}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    tool_turn = next(t for t in plan.turns if t.source_turn_index == 0)
    assert tool_turn.from_state == "A"  # unchanged even though it now self-loops


def test_arrow_glyph_preserved():
    msgs = [
        _amsg('[STATE: A -> B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B -> B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert all(t.arrow == "->" for t in plan.turns)


# --- Task 3: apply_plan / verify_repaired / rederive_ground_truth ---------
#
# NOTE on fixtures below: every record passed to verify_repaired() opens with a
# user turn. find_shape_violations() requires the first non-system message to
# match conversation_initiator (defaulting to "user"), so the assistant-first
# fixtures used by the Task 2 plan_repair tests above -- which never call
# verify_repaired() -- would report a spurious shape violation here. Message
# indices in the assertions account for that leading turn.

from llm_workflow_agents.data.state_convention_repair import (  # noqa: E402
    apply_plan, verify_repaired, rederive_ground_truth,
)


def test_rederive_ground_truth_is_noop_on_conformant_record():
    msgs = [
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ]
    rec = _record(msgs, [("A", "TERMINAL")])
    rec["ground_truth"] = {
        "state_sequence": [{"from": "A", "to": "A"}, {"from": "A", "to": "TERMINAL"}],
        "tool_calls": [{"name": "t", "arguments": {}}],
        "tool_chain_dependencies": [[{"name": "t", "arguments": {}}]],
        "terminal_state": "TERMINAL",
        "terminal_reached": True,
    }
    before = copy.deepcopy(rec["ground_truth"])
    after = rederive_ground_truth(rec)
    assert after == before


def test_apply_plan_relabel_updates_marker_annotations_and_ground_truth():
    msgs = [
        {"role": "user", "content": "hi", "annotations": None},
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    plan = plan_repair(rec)
    repaired = apply_plan(rec, plan)
    assert repaired is not rec  # never mutates input
    assert "[STATE: A → A]" in repaired["messages"][1]["content"]
    assert repaired["messages"][1]["annotations"]["state_transition"] == {"from": "A", "to": "A"}
    assert "[STATE: A → B]" in repaired["messages"][3]["content"]
    assert repaired["ground_truth"]["state_sequence"] == [
        {"from": "A", "to": "A"}, {"from": "A", "to": "B"},
    ]
    assert verify_repaired(repaired) == []


def test_apply_plan_does_not_mutate_input():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    original = copy.deepcopy(rec)
    plan = plan_repair(rec)
    apply_plan(rec, plan)
    assert rec == original


def test_apply_plan_relabels_fused_turn_without_splitting_or_reattributing_tool():
    msgs = [
        {"role": "user", "content": "hi", "annotations": None},
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    plan = plan_repair(rec)
    repaired = apply_plan(rec, plan)
    assert verify_repaired(repaired) == []
    assistant_msgs = [m for m in repaired["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2  # unchanged from input -- no split, no inserted turn
    assert assistant_msgs[0]["content"].startswith("[STATE: A → A]")
    assert "Checking that now." in assistant_msgs[0]["content"]
    assert "<tool_call>" in assistant_msgs[0]["content"]
    assert assistant_msgs[1]["content"].startswith("[STATE: A → B]")
    from llm_workflow_agents.data._workflow_script import infer_state_tools_from_messages
    assert infer_state_tools_from_messages(msgs) == infer_state_tools_from_messages(repaired["messages"])


def test_apply_plan_returns_none_without_required_ledger_entry():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t1", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → TERMINAL]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
    ]
    rec = _record(msgs, [("A", "B"), ("B", "TERMINAL")])
    plan = plan_repair(rec)
    assert plan.move == "insert_handoff_turn"
    assert apply_plan(rec, plan, ledger_entries=None) is None


def test_verify_repaired_catches_state_sequence_mismatch():
    msgs = [
        {"role": "user", "content": "hi", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone."),
    ]
    rec = _record(msgs, [("A", "TERMINAL")])
    rec["ground_truth"] = {"state_sequence": [{"from": "A", "to": "WRONG"}],
                            "tool_calls": [], "tool_chain_dependencies": [],
                            "terminal_state": "TERMINAL", "terminal_reached": True}
    violations = verify_repaired(rec)
    assert any("state_sequence" in v for v in violations)


# --- Fix round 1: relabels accumulated before an insert must not be dropped ---


def _stub_ledger(plan, stem="f", line_index=0):
    """Emulate the Task 4 CLI: assign insert_ids, then author stub content.

    Also exercises the fact that InsertRequest is no longer frozen.
    """
    entries = {}
    for i, ins in enumerate(plan.inserts):
        ins.insert_id = f"{stem}:{line_index}:{i}"
        content = (f"{ins.required_marker}\nOk, let me follow up on that."
                   if ins.role == "assistant" else "Sure, thanks.")
        entries[ins.insert_id] = {"content": content}
    return entries


def test_insert_request_is_mutable_so_the_cli_can_assign_insert_id():
    req = InsertRequest(insert_id="", position_after_msg_index=0,
                        role="assistant", required_marker="[STATE: A → A]")
    req.insert_id = "stem:3:0"
    assert req.insert_id == "stem:3:0"


def test_insert_handoff_plan_keeps_and_applies_earlier_relabels():
    # Same shape as test_stacked_tool_turn_after_a_fused_turn_needs_insert_handoff:
    # the FUSED first turn needs an A->A relabel, and a later stacked tool turn
    # forces an insert_handoff_turn. The relabel must survive into the plan and
    # be written by apply_plan -- otherwise msg 0 keeps its wrong [A -> B]
    # marker on a tool turn and the repaired record is still non-conformant.
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → B]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → TERMINAL]\nDone."),
    ]
    rec = _record(msgs, [("A", "B"), ("B", "TERMINAL")])
    plan = plan_repair(rec)
    assert plan.move == "insert_handoff_turn"
    assert [(t.source_turn_index, t.from_state, t.to_state) for t in plan.turns
            if t.content_op == "relabel"] == [(0, "A", "A")]

    repaired = apply_plan(rec, plan, ledger_entries=_stub_ledger(plan))
    assert repaired is not None
    assert repaired["messages"][0]["content"].startswith("[STATE: A → A]")
    assert find_tool_stay_violations(repaired["messages"]) == []
    assert rec["messages"][0]["content"].startswith("[STATE: A → B]")  # input untouched


def test_append_closing_pair_plan_keeps_and_applies_earlier_relabels():
    # Same shape as test_tail_deficit_requires_append_closing_pair. The single
    # fused tool turn relabels to A->A and the displaced A->TERMINAL advance is
    # carried by the appended closing turn.
    msgs = [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ]
    rec = _record(msgs, [("A", "TERMINAL")])
    plan = plan_repair(rec)
    assert plan.move == "append_closing_pair"
    assert [(t.source_turn_index, t.from_state, t.to_state) for t in plan.turns
            if t.content_op == "relabel"] == [(0, "A", "A")]

    repaired = apply_plan(rec, plan, ledger_entries=_stub_ledger(plan))
    assert repaired is not None
    assert repaired["messages"][0]["content"].startswith("[STATE: A → A]")
    assert [m["role"] for m in repaired["messages"]] == ["assistant", "user", "assistant"]
    assert repaired["messages"][2]["content"].startswith("[STATE: A → TERMINAL]")
    assert find_tool_stay_violations(repaired["messages"]) == []
