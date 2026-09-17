"""Tool calls as text and tool results as prefixed user turns (data/tool_turns.py).

The Gemma-4 chat template drops a `tool` message unless it answers structured
`tool_calls`, so no Gemma-4 training sequence on the Task A corpus contained a
tool result. This transform is what every render site now applies.
"""

from __future__ import annotations

import json

from llm_workflow_agents.data.tool_turns import (
    TOOL_RESULT_PREFIX,
    is_tool_result_turn,
    to_text_tool_turns,
    tool_result_text,
)

CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "cancel ORD-1"},
    {
        "role": "assistant",
        "content": '[STATE: A → A]\n<tool_call>{"name": "cancel", "arguments": {"id": "ORD-1"}}</tool_call>',
        "annotations": {"state_transition": {"from": "A", "to": "A"}},
    },
    {"role": "tool", "content": '{"status": "cancelled"}', "tool_call_id": "c1", "name": "cancel"},
    {"role": "assistant", "content": "[STATE: A → B]\nDone.", "loss": False},
]


def test_a_tool_message_becomes_a_prefixed_user_turn():
    out = to_text_tool_turns(CONVERSATION)
    assert out[3] == {"role": "user", "content": TOOL_RESULT_PREFIX + '{"status": "cancelled"}'}


def test_text_tool_calls_and_other_keys_are_untouched():
    out = to_text_tool_turns(CONVERSATION)
    assert out[2] == CONVERSATION[2]
    assert out[4]["loss"] is False


def test_structured_tool_calls_become_corpus_text():
    message = {
        "role": "assistant",
        "content": "[STATE: A → A]",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "book", "arguments": json.dumps({"name": "สมชาย", "n": 2})}}
        ],
    }
    (out,) = to_text_tool_turns([message])
    assert "tool_calls" not in out
    assert out["content"] == '[STATE: A → A]\n<tool_call>{"name": "book", "arguments": {"name": "สมชาย", "n": 2}}</tool_call>'


def test_a_structured_call_with_no_text_content():
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]}
    (out,) = to_text_tool_turns([message])
    assert out["content"] == '<tool_call>{"name": "t", "arguments": {}}</tool_call>'


def test_idempotent():
    once = to_text_tool_turns(CONVERSATION)
    assert to_text_tool_turns(once) == once


def test_input_is_not_mutated():
    snapshot = json.dumps(CONVERSATION, ensure_ascii=False, sort_keys=True)
    to_text_tool_turns(CONVERSATION)
    assert json.dumps(CONVERSATION, ensure_ascii=False, sort_keys=True) == snapshot


def test_no_tool_role_survives():
    assert all(m["role"] != "tool" for m in to_text_tool_turns(CONVERSATION))


def test_tool_result_turns_are_recognised_before_and_after():
    converted = to_text_tool_turns(CONVERSATION)[3]
    assert is_tool_result_turn(CONVERSATION[3]) and is_tool_result_turn(converted)
    assert tool_result_text(CONVERSATION[3]) == tool_result_text(converted) == '{"status": "cancelled"}'
    assert not is_tool_result_turn(CONVERSATION[1])
