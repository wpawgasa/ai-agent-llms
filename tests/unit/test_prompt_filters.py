"""Prompt filters for chain-dependent tool turns (CLAUDE.md R23)."""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.prompt_filters import (
    PROMPT_FILTERS,
    is_chain_dependent,
    is_tool_bearing,
    keep_row,
    propagated_arguments,
)

PROMPT = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Cancel my order please, my name is Alice."},
    {"role": "assistant", "content": "[STATE: A → A]\n<tool_call>...</tool_call>"},
    {"role": "tool", "content": '{"orders": [{"order_id": "ORD-9912", "qty": 2}]}'},
    {"role": "user", "content": "Yes, that one."},
]
CARRIED = {"tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-9912"}}]}


def test_an_argument_copied_from_a_tool_result_is_carried():
    assert propagated_arguments(PROMPT, CARRIED["tool_calls"]) == [
        {"tool": "cancel_order", "argument": "order_id", "value": "ORD-9912"}
    ]
    assert is_chain_dependent(PROMPT, CARRIED)


def test_a_value_only_the_user_typed_is_not_carried():
    gt = {"tool_calls": [{"name": "lookup", "arguments": {"name": "Alice"}}]}
    assert propagated_arguments(PROMPT, gt["tool_calls"]) == []
    assert not is_chain_dependent(PROMPT, gt)


def test_short_values_and_booleans_are_ignored_as_coincidences():
    prompt = [{"role": "tool", "content": '{"ok": true, "lang": "en", "n": 12}'}]
    gt = [{"name": "t", "arguments": {"ok": True, "lang": "en", "n": 12}}]
    assert propagated_arguments(prompt, gt) == []


def test_long_numbers_count():
    prompt = [{"role": "tool", "content": '{"account": 55012}'}]
    gt = [{"name": "t", "arguments": {"account": 55012}}]
    assert propagated_arguments(prompt, gt)[0]["value"] == "55012"


def test_nested_and_json_string_arguments_are_searched():
    prompt = [{"role": "tool", "content": '{"id": "PAY-77120"}'}]
    gt = [{"name": "t", "arguments": '{"payment": {"id": "PAY-77120"}}'}]
    assert propagated_arguments(prompt, gt)[0]["argument"] == "payment.id"


def test_no_tool_result_in_the_prompt_means_nothing_is_carried():
    prompt = [m for m in PROMPT if m["role"] != "tool"]
    assert not is_chain_dependent(prompt, CARRIED)


def test_a_turn_without_a_tool_call_is_neither_tool_bearing_nor_chain_dependent():
    assert not is_tool_bearing({"tool_calls": []})
    assert not is_chain_dependent(PROMPT, {"tool_calls": []})


@pytest.mark.parametrize("name", PROMPT_FILTERS)
def test_every_named_filter_is_accepted(name):
    keep_row(name, PROMPT, CARRIED)


def test_filters_nest():
    no_tool = {"tool_calls": []}
    uncarried = {"tool_calls": [{"name": "lookup", "arguments": {"name": "Alice"}}]}
    assert keep_row("none", PROMPT, no_tool)
    assert not keep_row("tool_bearing", PROMPT, no_tool)
    assert keep_row("tool_bearing", PROMPT, uncarried)
    assert not keep_row("chain_dependent", PROMPT, uncarried)
    assert keep_row("chain_dependent", PROMPT, CARRIED)


def test_an_unknown_filter_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt filter"):
        keep_row("everything", PROMPT, CARRIED)
