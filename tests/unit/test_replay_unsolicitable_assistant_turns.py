"""``_replay_conversation`` must never solicit a completion with a trailing
context entry that is itself model-authored.

Root cause (2026-09-07): Gemini (via BiFrost) rejects any chat-completion
request whose trailing message has role ``assistant`` — "Requests ending
with a model turn are not supported" — because there is no such thing as
"continue the model's own last utterance" in a turn-based chat API when
nothing new (a user utterance or a tool result) has arrived since. Before
this fix, ``_replay_conversation`` only guarded the OPENING case (an
assistant turn with no user turn before it at all — the pre-existing
``skip_preamble_assistant_turn`` path). It did not guard the much more
common case: two GT assistant turns adjacent with nothing between them,
which the benchmark corpus contains routinely (295/296 measured instances
are a text-only state-transition turn immediately followed by a tool-call
turn). Every one of those calls the harness previously made was
structurally unanswerable and always failed with the same 400, scoring as
both a missing state annotation and a missing tool call — not a model
capability gap.

The invariant under test: for ANY sample, ``_call_vllm`` is only ever
invoked with a context whose last message has role ``user`` or ``tool``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from llm_workflow_agents.eval.agent_benchmark import _replay_conversation

ENDPOINT = "http://localhost:8000"
MODEL = "test-model"


def _fake_call_vllm_factory(recorded_contexts: list[list[dict[str, Any]]], reply: str):
    def _fake_call_vllm(endpoint, model, messages, temperature=0.0, tools=None,
                         enable_thinking=False, engine="vllm"):
        recorded_contexts.append([dict(m) for m in messages])
        return reply, [], 10.0, 5.0

    return _fake_call_vllm


def _assert_no_call_ends_in_assistant(recorded_contexts):
    for ctx in recorded_contexts:
        assert ctx, "a call was made with empty context"
        assert ctx[-1]["role"] in ("user", "tool"), (
            f"_call_vllm was invoked with a context ending in role "
            f"{ctx[-1]['role']!r} — this is exactly the shape Gemini's API "
            f"rejects with 'Requests ending with a model turn are not "
            f"supported'"
        )


def test_ordinary_alternating_conversation_calls_model_for_every_assistant_turn():
    """Baseline: a normal system/user/assistant/user/assistant conversation
    must still call the model once per assistant turn — the fix must not
    over-skip."""
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
    recorded: list[list[dict[str, Any]]] = []
    with patch(
        "llm_workflow_agents.eval.agent_benchmark._call_vllm",
        side_effect=_fake_call_vllm_factory(recorded, "[STATE: A → A]\nreply"),
    ):
        predicted, latencies, ttfts = _replay_conversation(ENDPOINT, MODEL, sample)
    assert len(recorded) == 2
    assert len(latencies) == 2
    _assert_no_call_ends_in_assistant(recorded)


def test_opening_assistant_preamble_is_not_solicited():
    """Pre-existing case: an outbound/system-initiated conversation
    (system → assistant → user → ...) must not call the model for the
    opener — there is no user turn yet for it to react to."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "assistant", "content": "[STATE: GREETING → GREETING]\nHi, calling about your account."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "[STATE: GREETING → TERMINAL]\ndone"},
        ],
        "tool_schemas": [],
        "workflow_graph": {},
    }
    recorded: list[list[dict[str, Any]]] = []
    with patch(
        "llm_workflow_agents.eval.agent_benchmark._call_vllm",
        side_effect=_fake_call_vllm_factory(recorded, "[STATE: GREETING → TERMINAL]\ndone"),
    ):
        predicted, latencies, ttfts = _replay_conversation(ENDPOINT, MODEL, sample)
    # Only the second assistant turn (after the user spoke) is solicited.
    assert len(recorded) == 1
    assert len(latencies) == 1
    _assert_no_call_ends_in_assistant(recorded)
    # The opener is carried through verbatim in the predictions.
    assert predicted[1]["content"] == sample["messages"][1]["content"]


def test_consecutive_assistant_turn_is_not_solicited():
    """The newly-guarded case: two GT assistant turns back to back with no
    user/tool message between them — the shape measured at 295/296
    instances across the real benchmark corpus (a text-only state-
    transition turn immediately followed by a tool-call turn), e.g.
    conversation L1_005 (announce a tool error, then retry the same call
    with no new user input in between)."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "check my identity, id ACCT-1"},
            {
                "role": "assistant",
                "content": '[STATE: VERIFY → VERIFY]\n<tool_call>{"name": "verify_identity", "arguments": {}}</tool_call>',
                "annotations": {"tool_calls": [{"name": "verify_identity", "arguments": {}}]},
            },
            {"role": "tool", "content": '{"error": "unavailable"}'},
            {
                # Text-only turn, no tool call — the "prev" half of the pair.
                "role": "assistant",
                "content": "[STATE: VERIFY → VERIFY]\nApologies, hit a temporary issue.",
            },
            {
                # Immediately-following tool-call turn, no user message
                # between this and the previous assistant turn — the "cur"
                # half of the pair, and the one that must be skipped.
                "role": "assistant",
                "content": '[STATE: VERIFY → VERIFY]\n<tool_call>{"name": "verify_identity", "arguments": {}}</tool_call>',
                "annotations": {"tool_calls": [{"name": "verify_identity", "arguments": {}}]},
            },
            {"role": "tool", "content": '{"status": "success"}'},
            {"role": "assistant", "content": "[STATE: VERIFY → TERMINAL]\nAll set."},
        ],
        "tool_schemas": [{"function": {"name": "verify_identity", "parameters": {}}}],
        "workflow_graph": {},
    }
    recorded: list[list[dict[str, Any]]] = []
    with patch(
        "llm_workflow_agents.eval.agent_benchmark._call_vllm",
        # The reply calls the tool: a tool result is only shown after a call
        # the model made (test_replay_tool_result_gating.py), and this test is
        # about the consecutive-turn skip, not about a missed call.
        side_effect=_fake_call_vllm_factory(
            recorded,
            '[STATE: VERIFY → VERIFY]\n<tool_call>{"name": "verify_identity", "arguments": {}}</tool_call>',
        ),
    ):
        predicted, latencies, ttfts = _replay_conversation(ENDPOINT, MODEL, sample)

    # 4 GT assistant turns total; the 3rd one (index 5, the retry) must be
    # skipped, so only 3 model calls are made.
    assert len(recorded) == 3
    assert len(latencies) == 3
    _assert_no_call_ends_in_assistant(recorded)
    # The skipped turn is carried through verbatim, tool_calls included.
    skipped_gt = sample["messages"][5]
    skipped_pred = predicted[5]
    assert skipped_pred["content"] == skipped_gt["content"]
    assert skipped_pred.get("tool_calls") == skipped_gt.get("tool_calls")


def test_multiple_consecutive_assistant_turns_in_a_row():
    """Three GT assistant turns in a row (no user/tool between any of
    them): only the first is solicited; the following two are structurally
    unanswerable in the same way and must both be skipped."""
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
    recorded: list[list[dict[str, Any]]] = []
    with patch(
        "llm_workflow_agents.eval.agent_benchmark._call_vllm",
        side_effect=_fake_call_vllm_factory(recorded, "[STATE: A → A]\nreply"),
    ):
        predicted, latencies, ttfts = _replay_conversation(ENDPOINT, MODEL, sample)
    assert len(recorded) == 1
    _assert_no_call_ends_in_assistant(recorded)
    assert predicted[3]["content"] == sample["messages"][3]["content"]
    assert predicted[4]["content"] == sample["messages"][4]["content"]
