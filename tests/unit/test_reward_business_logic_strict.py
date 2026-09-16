"""The strict Cat A reward pays for exact arguments, not for the tool name.

``reward_business_logic`` saturates on well-trained models (CLAUDE.md R23): 0.4
for the tool name alone, and graded partial credit that barely moves when one
argument is wrong. These tests pin the behaviours that make the strict reward
separate samples that differ only in an argument.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.rewards.reward_business_logic import (
    reward_business_logic,
)
from llm_workflow_agents.training.rewards.reward_business_logic_strict import (
    W_STATE,
    W_TOOL,
    exact_state_score,
    reward_business_logic_strict,
    strict_tool_call_score,
)


def _completion(state, calls):
    import json

    text = f"[STATE: {state[0]} → {state[1]}]"
    for name, args in calls:
        text += f'\n<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'
    return text


GT = {
    "state_annotations": [("CHECK", "CHECK")],
    "tool_calls": [{"name": "book", "arguments": {"order_id": "ORD-9912", "qty": 2, "date": "2026-05-15"}}],
    "terminal_state": "TERMINAL",
    "terminal_reached": False,
}
PERFECT = _completion(("CHECK", "CHECK"), [("book", {"order_id": "ORD-9912", "qty": 2, "date": "2026-05-15"})])
ONE_WRONG = _completion(("CHECK", "CHECK"), [("book", {"order_id": "ORD-9917", "qty": 2, "date": "2026-05-15"})])
WRONG_NAME = _completion(("CHECK", "CHECK"), [("reserve", {"order_id": "ORD-9912", "qty": 2, "date": "2026-05-15"})])
NO_CALL = _completion(("CHECK", "CHECK"), [])


def _r(fn, completion, gt=GT):
    return fn([""], [completion], [gt])[0]


def test_weights_sum_to_one():
    assert W_STATE + W_TOOL == pytest.approx(1.0)


def test_a_perfect_turn_scores_one():
    assert _r(reward_business_logic_strict, PERFECT) == pytest.approx(1.0)


def test_one_wrong_argument_of_three_costs_a_third_of_the_tool_weight():
    assert _r(reward_business_logic_strict, ONE_WRONG) == pytest.approx(W_STATE + W_TOOL * 2 / 3)


def test_a_wrong_tool_name_earns_no_tool_credit():
    assert _r(reward_business_logic_strict, WRONG_NAME) == pytest.approx(W_STATE)


def test_no_call_where_one_is_needed_earns_no_tool_credit():
    assert _r(reward_business_logic_strict, NO_CALL) == pytest.approx(W_STATE)


def test_it_separates_a_wrong_argument_more_than_the_graded_reward_does():
    """The point of the strict reward: a one-argument slip must move the score."""
    graded_gap = _r(reward_business_logic, PERFECT) - _r(reward_business_logic, ONE_WRONG)
    strict_gap = _r(reward_business_logic_strict, PERFECT) - _r(reward_business_logic_strict, ONE_WRONG)
    assert strict_gap > graded_gap


def test_a_number_written_as_a_string_still_matches():
    assert strict_tool_call_score(
        [{"name": "t", "arguments": {"qty": "2"}}], [{"name": "t", "arguments": {"qty": 2}}]
    ) == 1.0


def test_an_identifier_is_compared_exactly():
    assert strict_tool_call_score(
        [{"name": "t", "arguments": {"id": "ORD-042"}}], [{"name": "t", "arguments": {"id": "ORD-42"}}]
    ) == 0.0


def test_an_invented_argument_costs_as_much_as_a_missing_one():
    gt = [{"name": "t", "arguments": {"a": "alpha", "b": "bravo"}}]
    invented = [{"name": "t", "arguments": {"a": "alpha", "b": "bravo", "c": "charlie"}}]
    missing = [{"name": "t", "arguments": {"a": "alpha"}}]
    assert strict_tool_call_score(invented, gt) == pytest.approx(2 / 3)
    assert strict_tool_call_score(missing, gt) == pytest.approx(1 / 2)


def test_a_spurious_extra_call_lowers_the_score():
    gt = [{"name": "t", "arguments": {"a": "alpha"}}]
    two = [{"name": "t", "arguments": {"a": "alpha"}}, {"name": "u", "arguments": {}}]
    assert strict_tool_call_score(two, gt) == pytest.approx(0.5)


def test_placeholder_arguments_fall_back_to_name_only():
    gt = [{"name": "t", "arguments": {"placeholder": "value"}}]
    assert strict_tool_call_score([{"name": "t", "arguments": {"x": "whatever"}}], gt) == 1.0


def test_no_tool_turns_reward_silence_and_penalise_a_call():
    assert strict_tool_call_score([], []) == 1.0
    assert strict_tool_call_score([{"name": "t", "arguments": {}}], []) == 0.0


def test_state_has_no_partial_credit():
    assert exact_state_score([("A", "B")], [("A", "B")]) == 1.0
    assert exact_state_score([("A", "C")], [("A", "B")]) == 0.0  # graded reward pays 0.5
    assert exact_state_score([("B", "A")], [("A", "B")]) == 0.0  # graded reward pays 0.3


def test_extra_state_annotations_count_against_the_turn():
    assert exact_state_score([("A", "B"), ("B", "C")], [("A", "B")]) == pytest.approx(0.5)


def test_rewards_stay_in_range_for_garbage():
    for text in ["", "no annotation", "<tool_call>not json</tool_call>", WRONG_NAME * 5]:
        assert 0.0 <= _r(reward_business_logic_strict, text) <= 1.0


def test_it_is_registered_for_grpo():
    from llm_workflow_agents.training.grpo import _resolve_reward_fn

    assert _resolve_reward_fn("reward_business_logic_strict") is reward_business_logic_strict
