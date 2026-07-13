"""Unit tests for GRPO GT tool-call sanitization (grpo._sanitize_gt_tool_calls)."""

from __future__ import annotations

from llm_workflow_agents.training.grpo import (
    _gt_tool_call_is_invalid,
    _required_args_by_tool,
    _sanitize_gt_tool_calls,
)


def test_required_args_by_tool_parses_function_wrapper():
    schemas = [
        {"type": "function", "function": {"name": "foo",
         "parameters": {"required": ["a", "b"], "properties": {}}}},
        {"name": "bar", "parameters": {"required": ["x"]}},
    ]
    req = _required_args_by_tool(schemas)
    assert req == {"foo": {"a", "b"}, "bar": {"x"}}


def test_null_sentinel_on_required_arg_is_invalid():
    req = {"customer_id"}
    assert _gt_tool_call_is_invalid(
        {"name": "apply_for_loan", "arguments": {"customer_id": "UNKNOWN"}}, req
    ) is True
    assert _gt_tool_call_is_invalid(
        {"name": "dispute_bill", "arguments": {"account_id": "000000"}}, {"account_id"}
    ) is True
    assert _gt_tool_call_is_invalid(
        {"name": "x", "arguments": {"claim_id": "N/A"}}, {"claim_id"}
    ) is True


def test_null_sentinel_on_optional_arg_is_kept():
    # region is NOT required for log_complaint_trend -> harmless default, keep it.
    assert _gt_tool_call_is_invalid(
        {"name": "log_complaint_trend",
         "arguments": {"category": "x", "description": "y", "region": "unknown"}},
        {"category", "description"},
    ) is False


def test_out_of_range_score_is_invalid_even_if_not_marked_required():
    # A score outside [0,10] is invalid regardless of the required set.
    assert _gt_tool_call_is_invalid(
        {"name": "collect_nps", "arguments": {"score": 11}}, set()
    ) is True
    assert _gt_tool_call_is_invalid(
        {"name": "collect_nps", "arguments": {"score": -1}}, {"score"}
    ) is True


def test_valid_score_and_real_ids_are_kept():
    assert _gt_tool_call_is_invalid(
        {"name": "collect_nps", "arguments": {"customer_id": "CUST-991", "score": 9}},
        {"customer_id", "score"},
    ) is False
    # synthetic-style IDs the user actually provided are NOT sentinels
    assert _gt_tool_call_is_invalid(
        {"name": "create_ticket", "arguments": {"customer_id": "TH12345"}},
        {"customer_id"},
    ) is False


def test_bool_is_not_treated_as_out_of_range_score():
    # score=True would coerce to 1 numerically; guard against bool false-positives.
    assert _gt_tool_call_is_invalid(
        {"name": "x", "arguments": {"score": True}}, {"score"}
    ) is False


def test_sanitize_drops_only_invalid_and_counts():
    tcs = [
        {"name": "apply_for_loan", "arguments": {"customer_id": "UNKNOWN"}},  # drop
        {"name": "check_balance", "arguments": {"account_id": "ACC-1029"}},   # keep
        {"name": "collect_nps", "arguments": {"score": 11}},                  # drop
    ]
    req = {"apply_for_loan": {"customer_id"}, "check_balance": {"account_id"},
           "collect_nps": {"score"}}
    kept, n_removed = _sanitize_gt_tool_calls(tcs, req)
    assert n_removed == 2
    assert [tc["name"] for tc in kept] == ["check_balance"]


def test_sanitize_empty_is_noop():
    assert _sanitize_gt_tool_calls([], {}) == ([], 0)
