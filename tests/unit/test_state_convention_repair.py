import copy
import json

from llm_workflow_agents.data.state_convention_repair import plan_repair
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


def test_stacked_tool_turns_split_fused_head():
    # first tool turn is fused (has prose before the <tool_call>) and wrongly
    # advances A->B; splitting its prose off as a separate A->B turn lets the
    # tail become a clean B->B tool turn, which absorbs the second tool call.
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
    assert plan.move == "split_fused_tool_turn"
    assert any(t.content_op == "split_head" for t in plan.turns)
    assert any(t.content_op == "split_tail" for t in plan.turns)


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
