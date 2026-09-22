"""Turns the replay cannot ordinarily ask for: openers and back-to-back turns.

History. Gemini (via BiFrost) rejects any request whose trailing message is
the model's own ("Requests ending with a model turn are not supported"), and
the corpus often has two ground-truth assistant turns in a row — a speech turn
then the tool-call turn, 295/296 of such pairs (R25). The first fix stopped
asking for the second turn and COPIED it from ground truth into the
predictions, where it scored as perfect: ~327 turns per run the model never
produced.

Now (2026-09-22) the replay works per segment — a run of consecutive
ground-truth assistant turns (``eval/segment_scoring.py``):

- The model is asked once at the start of each segment, and once more when
  the segment owes a tool call the reply did not make. vLLM and the Gemma-4
  template accept a request ending on the model's own reply as the cue for a
  fresh turn (verified on 40 real pairs, both tool-turn formats).
- BiFrost refuses that request, so there the second ask is skipped and counted.
- A second reply is kept only if it makes a tool call; otherwise it is
  discarded (live smoke 2026-09-22: asked again with no call to make, the E4B
  sometimes wrote the customer's next line).
- An opening turn (outbound, no user message yet) is not asked for on any
  engine: the served prompt does not say why the agent is calling. The
  ground-truth opener goes into the CONTEXT only and the segment is marked
  ``unscored``.
- Ground truth is never copied into a scored prediction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from llm_workflow_agents.eval.agent_benchmark import _replay_conversation

ENDPOINT = "http://localhost:8000"
MODEL = "test-model"
CALL = '<tool_call>{"name": "verify_identity", "arguments": {}}</tool_call>'


def _fake(recorded: list[list[dict[str, Any]]], replies: list[str]):
    queue = list(replies)

    def _fake_call_vllm(endpoint, model, messages, temperature=0.0, tools=None, **kwargs):
        recorded.append([dict(m) for m in messages])
        return (queue.pop(0) if len(queue) > 1 else queue[0]), [], 10.0, 5.0

    return _fake_call_vllm


def _replay(sample, replies, **kw):
    recorded: list[list[dict[str, Any]]] = []
    stats: dict[str, int] = {}
    with patch(
        "llm_workflow_agents.eval.agent_benchmark._call_vllm",
        side_effect=_fake(recorded, replies),
    ):
        predicted, latencies, _ = _replay_conversation(ENDPOINT, MODEL, sample, stats=stats, **kw)
    return recorded, predicted, latencies, stats


OUTBOUND = {
    "messages": [
        {"role": "system", "content": "You are an agent."},
        {"role": "assistant", "content": "[STATE: GREETING → GREETING]\nHi, calling about your account."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "[STATE: GREETING → TERMINAL]\ndone"},
    ],
    "tool_schemas": [],
    "workflow_graph": {},
}

PAIR = {
    "messages": [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "check my identity, id ACCT-1"},
        {"role": "assistant", "content": "[STATE: VERIFY → VERIFY]\nOne moment."},
        {
            "role": "assistant",
            "content": f"[STATE: VERIFY → VERIFY]\n{CALL}",
            "annotations": {"tool_calls": [{"name": "verify_identity", "arguments": {}}]},
        },
        {"role": "tool", "content": '{"status": "success"}'},
        {"role": "assistant", "content": "[STATE: VERIFY → TERMINAL]\nAll set."},
    ],
    "tool_schemas": [{"function": {"name": "verify_identity", "parameters": {}}}],
    "workflow_graph": {},
}


def test_ordinary_alternating_conversation_asks_once_per_assistant_turn():
    sample = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[STATE: A → A]\nhello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "[STATE: A → TERMINAL]\nbye"},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    recorded, _, latencies, stats = _replay(sample, ["[STATE: A → A]\nreply"])
    assert len(recorded) == 2 and len(latencies) == 2
    assert stats["segments"] == 2 and "second_asks" not in stats


@pytest.mark.parametrize("engine", ["vllm", "bifrost"])
def test_opener_goes_into_context_unscored(engine):
    recorded, predicted, _, stats = _replay(OUTBOUND, ["[STATE: GREETING → TERMINAL]\ndone"], engine=engine)
    assert len(recorded) == 1  # only the turn after the customer spoke
    assert recorded[0][1]["content"] == OUTBOUND["messages"][1]["content"]  # opener in context
    assert predicted[1]["unscored"] is True
    assert stats["unscored_opening"] == 1


def test_bifrost_never_asks_without_a_user_message_or_after_its_own_turn():
    for sample in (OUTBOUND, PAIR):
        recorded, _, _, _ = _replay(sample, ["[STATE: VERIFY → VERIFY]\nOne moment."], engine="bifrost")
        for ctx in recorded:
            assert ctx[-1]["role"] in ("user", "tool")


def test_owed_call_after_an_announcement_gets_a_second_ask():
    recorded, predicted, latencies, stats = _replay(
        PAIR,
        ["[STATE: VERIFY → VERIFY]\nOne moment.", f"[STATE: VERIFY → VERIFY]\n{CALL}", "[STATE: VERIFY → TERMINAL]\nok"],
    )
    assert len(recorded) == 3 and len(latencies) == 3
    assert recorded[1][-1]["role"] == "assistant"  # the second ask follows the model's own reply
    assert recorded[2][-1]["role"] == "tool"  # the call it then made admits the result
    assert stats["second_asks"] == 1 and stats["second_ask_made_call"] == 1
    # Both replies land in the segment's first slot; ground truth is never copied.
    assert "One moment." in predicted[2]["content"] and CALL in predicted[2]["content"]
    assert predicted[2]["replies"][1] == f"[STATE: VERIFY → VERIFY]\n{CALL}"
    assert predicted[3]["content"] == ""


def test_second_ask_without_a_call_is_discarded():
    recorded, predicted, _, stats = _replay(
        PAIR,
        [
            "[STATE: VERIFY → VERIFY]\nShall I verify you now?",
            "[STATE: VERIFY → VERIFY]\nYes please, go ahead.",  # the customer's line, written by the model
            "[STATE: VERIFY → TERMINAL]\nok",
        ],
    )
    assert stats["second_asks"] == 1 and stats["second_ask_no_call_discarded"] == 1
    assert "go ahead" not in predicted[2]["content"]
    assert predicted[2]["replies"] == ["[STATE: VERIFY → VERIFY]\nShall I verify you now?"]
    # Discarded from the context too: no later request carries it.
    assert all("go ahead" not in (m.get("content") or "") for ctx in recorded for m in ctx)


def test_a_reply_that_speaks_and_calls_needs_no_second_ask():
    recorded, predicted, _, stats = _replay(
        PAIR, [f"[STATE: VERIFY → VERIFY]\nOne moment.\n{CALL}", "[STATE: VERIFY → TERMINAL]\nok"],
    )
    assert len(recorded) == 2 and "second_asks" not in stats
    assert predicted[3]["content"] == ""


def test_bifrost_skips_the_second_ask_and_counts_it():
    recorded, predicted, _, stats = _replay(
        PAIR, ["[STATE: VERIFY → VERIFY]\nOne moment.", "[STATE: VERIFY → TERMINAL]\nok"], engine="bifrost",
    )
    assert stats["second_ask_unavailable"] == 1
    # No call was made, so the result is withheld and the segment answering
    # it is a miss.
    assert predicted[5]["content"] == ""
    assert stats["missed_after_withheld_result"] == 1
    assert len(recorded) == 1


def test_three_prose_turns_in_a_row_are_one_segment_and_one_ask():
    sample = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[STATE: A → A]\nfirst"},
            {"role": "assistant", "content": "[STATE: A → A]\nsecond"},
            {"role": "assistant", "content": "[STATE: A → TERMINAL]\nthird"},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    recorded, predicted, _, _ = _replay(sample, ["[STATE: A → A]\nreply"])
    assert len(recorded) == 1
    assert predicted[2]["content"] == "[STATE: A → A]\nreply"
    assert predicted[3]["content"] == "" and predicted[4]["content"] == ""
