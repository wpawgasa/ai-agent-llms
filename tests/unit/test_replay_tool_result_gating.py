"""The benchmark replay may only show the model a tool result for a call it made.

Before this fix ``_replay_conversation`` appended every ground-truth tool
result, whether or not the model's reply called the tool. Two things followed:

- Native format: the Gemma-4 chat template (correctly) renders a tool result
  only under the assistant message whose ``tool_calls`` it answers, so a result
  with no call was dropped from the prompt, while the harness went on as if the
  model had seen it.
- Text format: the result became a ``[Tool result]: `` user turn, so the model
  saw the output of a call it never made — free information that inflates the
  scores of models that miss calls (docs/cat_a_benchmark_tool_result_gating.md).

Three rules are tested here:

1. A ground-truth tool result reaches the context only when the preceding
   assistant turn (model reply or copied ground-truth turn) made a tool call,
   one result per call.
2. A copied ground-truth turn writes its tool calls as ``<tool_call>`` text;
   native mode lifts them into structured ``tool_calls`` so their results render.
3. With ``split_tool_call_content`` (Gemma-4), a reply carrying both text and a
   tool call goes to the server as two assistant messages, text first. The
   Gemma-4 template renders a merged message's text AFTER the tool response and
   closes the turn, leaving no generation cue — the empty replies seen in the
   untrained-E4B runs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_workflow_agents.eval import agent_benchmark as ab

RESULT = '{"status": "verified"}'
CALL_TEXT = '<tool_call>{"name": "verify", "arguments": {"pin": "5542"}}</tool_call>'

SAMPLE = {
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "my PIN is 5542"},
        {"role": "assistant", "content": f"[STATE: V → V]\n{CALL_TEXT}"},
        {"role": "tool", "content": RESULT},
        {"role": "assistant", "content": "[STATE: V → DONE]\nYou are verified."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "[STATE: DONE → DONE]\nBye."},
    ],
    "tool_schemas": [{"type": "function", "function": {"name": "verify"}}],
    "workflow_graph": {},
}


def _run(monkeypatch, replies: list[str], sample=SAMPLE, **kw):
    seen: list[list[dict[str, Any]]] = []
    queue = list(replies)

    def fake_call(endpoint, model, messages, temperature, tools=None, **_):
        seen.append(json.loads(json.dumps(messages)))
        return queue.pop(0), [], 1.0, 1.0

    monkeypatch.setattr(ab, "_call_vllm", fake_call)
    predicted, latencies, _ = ab._replay_conversation("http://x", "m", sample, **kw)
    return seen, predicted, latencies


def _mentions_result(messages) -> bool:
    return any(RESULT in (m.get("content") or "") for m in messages)


@pytest.mark.parametrize("fmt", ["native", "text"])
def test_no_call_means_no_tool_result(monkeypatch, fmt):
    seen, predicted, latencies = _run(
        monkeypatch,
        ["[STATE: V → V]\nCould you repeat the PIN?", "[STATE: DONE → DONE]\nBye."],
        tool_turn_format=fmt,
    )
    assert all(not _mentions_result(msgs) for msgs in seen)
    # The turn that answers the result cannot be asked for, so it is scored as
    # a miss (empty), never copied from ground truth.
    assert predicted[4]["content"] == ""
    assert len(seen) == 2 and len(latencies) == 2


@pytest.mark.parametrize("fmt", ["native", "text"])
def test_call_means_tool_result_is_shown(monkeypatch, fmt):
    seen, predicted, _ = _run(
        monkeypatch,
        [f"[STATE: V → V]\n{CALL_TEXT}", "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
        tool_turn_format=fmt,
    )
    assert len(seen) == 3
    assert _mentions_result(seen[1])
    if fmt == "native":
        tool_msg = seen[1][-1]
        call_msg = seen[1][-2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == call_msg["tool_calls"][0]["id"]


def test_one_call_admits_only_one_of_two_results(monkeypatch):
    sample = json.loads(json.dumps(SAMPLE))
    sample["messages"].insert(4, {"role": "tool", "content": '{"second": "result"}'})
    seen, _, _ = _run(
        monkeypatch,
        [f"[STATE: V → V]\n{CALL_TEXT}", "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
        sample=sample,
    )
    assert _mentions_result(seen[1])
    assert not any("second" in (m.get("content") or "") for m in seen[1])


def test_stale_call_does_not_admit_a_later_result(monkeypatch):
    """A call from an earlier turn must not license a result after a turn with no call."""
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": f"[STATE: V → V]\n{CALL_TEXT}"},
            {"role": "user", "content": "and again"},
            {"role": "assistant", "content": f"[STATE: V → V]\n{CALL_TEXT}"},
            {"role": "tool", "content": RESULT},
            {"role": "assistant", "content": "[STATE: V → DONE]\nok"},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    seen, predicted, _ = _run(
        monkeypatch, [f"[STATE: V → V]\n{CALL_TEXT}", "[STATE: V → V]\nno call"], sample=sample,
    )
    assert all(not _mentions_result(msgs) for msgs in seen)
    assert predicted[6]["content"] == ""


def test_copied_ground_truth_turn_keeps_its_tool_call_native(monkeypatch):
    """Consecutive assistant turns: the second is copied from ground truth. Its
    text tool call must become structured so the following result renders."""
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "PIN 5542"},
            {"role": "assistant", "content": "[STATE: V → V]\nOne moment."},
            {"role": "assistant", "content": f"[STATE: V → V]\n{CALL_TEXT}"},
            {"role": "tool", "content": RESULT},
            {"role": "assistant", "content": "[STATE: V → DONE]\nVerified."},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    seen, predicted, _ = _run(monkeypatch, ["[STATE: V → V]\nOne moment.", "[STATE: V → DONE]\nok"], sample=sample)
    assert len(seen) == 2
    last = seen[1]
    assert last[-1]["role"] == "tool" and RESULT in last[-1]["content"]
    copied = last[-2]
    assert copied["role"] == "assistant"
    assert copied["tool_calls"][0]["function"]["name"] == "verify"
    assert "<tool_call>" not in copied["content"]
    assert last[-1]["tool_call_id"] == copied["tool_calls"][0]["id"]
    # Predictions still carry the ground-truth turn unchanged.
    assert predicted[3]["content"] == sample["messages"][3]["content"]


def test_copied_ground_truth_turn_keeps_its_tool_call_text(monkeypatch):
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "PIN 5542"},
            {"role": "assistant", "content": "[STATE: V → V]\nOne moment."},
            {"role": "assistant", "content": f"[STATE: V → V]\n{CALL_TEXT}"},
            {"role": "tool", "content": RESULT},
            {"role": "assistant", "content": "[STATE: V → DONE]\nVerified."},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    seen, _, _ = _run(
        monkeypatch, ["[STATE: V → V]\nOne moment.", "[STATE: V → DONE]\nok"], sample=sample,
        tool_turn_format="text",
    )
    assert _mentions_result(seen[1])


def test_split_sends_text_then_call_as_two_messages(monkeypatch):
    seen, _, _ = _run(
        monkeypatch,
        [f"[STATE: V → V]\nChecking.\n{CALL_TEXT}", "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
        split_tool_call_content=True,
    )
    tail = seen[1][-3:]
    assert [m["role"] for m in tail] == ["assistant", "assistant", "tool"]
    assert tail[0]["content"] == "[STATE: V → V]\nChecking." and not tail[0].get("tool_calls")
    assert tail[1]["content"] == "" and tail[1]["tool_calls"]


def test_split_leaves_a_call_only_reply_as_one_message(monkeypatch):
    seen, _, _ = _run(
        monkeypatch,
        [CALL_TEXT, "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
        split_tool_call_content=True,
    )
    assert [m["role"] for m in seen[1][-2:]] == ["assistant", "tool"]
    assert seen[1][-3]["role"] == "user"


def test_without_split_text_and_call_stay_merged(monkeypatch):
    seen, _, _ = _run(
        monkeypatch,
        [f"[STATE: V → V]\nChecking.\n{CALL_TEXT}", "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
    )
    call_msg = seen[1][-2]
    assert call_msg["tool_calls"] and "Checking." in call_msg["content"]
    assert seen[1][-3]["role"] == "user"


def test_split_is_ignored_in_text_format(monkeypatch):
    seen, _, _ = _run(
        monkeypatch,
        [f"[STATE: V → V]\nChecking.\n{CALL_TEXT}", "[STATE: V → DONE]\nok", "[STATE: DONE → DONE]\nBye."],
        tool_turn_format="text", split_tool_call_content=True,
    )
    assert [m["role"] for m in seen[1][-3:]] == ["user", "assistant", "user"]
