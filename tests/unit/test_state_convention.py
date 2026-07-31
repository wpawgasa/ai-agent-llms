from llm_workflow_agents.data.state_convention import (
    TurnLabel,
    parse_assistant_turns,
    find_tool_stay_violations,
)


def _msgs(*contents_and_roles):
    return [{"role": r, "content": c, "annotations": None} for r, c in contents_and_roles]


def test_bare_tool_turn_has_empty_prose_prefix():
    msgs = _msgs(("user", "hi"), ("assistant",
        '[STATE: LOOKUP → LOOKUP]\n<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'))
    labels = parse_assistant_turns(msgs)
    assert labels[0] is None
    label = labels[1]
    assert label.from_state == "LOOKUP" and label.to_state == "LOOKUP"
    assert label.has_tool_call is True
    assert label.prose_prefix == ""
    assert label.tool_names == ("lookup",)


def test_fused_tool_turn_splits_prose_from_tail():
    msgs = _msgs(("assistant",
        '[STATE: A → B]\nLet me check that for you.\n'
        '<tool_call>{"name": "check", "arguments": {"x": 1}}</tool_call>'))
    label = parse_assistant_turns(msgs)[0]
    assert label.prose_prefix == "Let me check that for you."
    assert label.tail.startswith("<tool_call>")
    assert label.arrow == "→"


def test_parse_preserves_ascii_arrow():
    msgs = _msgs(("assistant", "[STATE: A -> A]\nok"))
    assert parse_assistant_turns(msgs)[0].arrow == "->"


def test_prose_turn_advancing_is_not_a_violation():
    msgs = _msgs(("assistant", "[STATE: A → B]\nAll set, thanks!"))
    assert find_tool_stay_violations(msgs) == []


def test_advancing_tool_turn_yields_one_violation():
    msgs = _msgs(("assistant",
        '[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    violations = find_tool_stay_violations(msgs)
    assert len(violations) == 1
    assert "turn 1" in violations[0]
    assert "[A -> B]" in violations[0]
    assert "[A -> A]" in violations[0]


def test_conformant_tool_turn_yields_no_violation():
    msgs = _msgs(("assistant",
        '[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    assert find_tool_stay_violations(msgs) == []


def test_ignores_non_assistant_roles():
    msgs = _msgs(("tool", '<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    assert find_tool_stay_violations(msgs) == []


def test_turn_with_no_marker_is_none_and_not_a_violation():
    msgs = _msgs(("assistant", "no marker here"))
    labels = parse_assistant_turns(msgs)
    assert labels == [None]
    assert find_tool_stay_violations(msgs) == []
