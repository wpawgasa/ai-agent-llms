"""Contrastive preference pairs for Cat A.

Why pairs at all: R18 / docs/grpo_reward_resolution_investigation.md established
that GRPO cannot learn here — the reward takes 11 distinct values on 206 real
completions and is exactly 1.0 on 81.1% of them, so groups tie and the advantage
is zero. Three fixes (tool-bearing mix, trajectory aggregation, higher
temperature) each moved the needle and none sufficed. A preference objective
needs no reward variance: every pair carries a guaranteed margin.

The three corruptions mirror the failures C2 actually makes on held-out data,
not invented ones:

  drop_tool_calls      the announce-but-don't-call gap — narrates the action,
                       emits no <tool_call>. 9 of 71 tool-bearing rows.
  flip_state_advance   advances where the convention wants a self-loop, or
                       self-loops where it should advance. Spurious self-loops
                       ran 26.9% in v4 and 2.8% in C2.
  corrupt_tool_args    right tool name, wrong arguments — 18 of 71 rows, the
                       current bottleneck.

Every corruption returns None when it does not apply, so callers can never
emit a pair whose "rejected" is identical to its "chosen" — a zero-margin pair
is worse than no pair, because it teaches the model that the gold answer is
also the bad one.
"""

import json

import pytest

from llm_workflow_agents.data.preference_pairs import (
    corrupt_tool_args,
    drop_tool_calls,
    flip_state_transition,
)

TOOL_TURN = (
    "[STATE: LOOKUP_ORDER → LOOKUP_ORDER]\n"
    "Let me look that order up for you.\n"
    '<tool_call>{"name": "lookup_order", "arguments": {"order_id": "TH-99823"}}</tool_call>'
)
ADVANCE_TURN = "[STATE: GREETING → VERIFY_IDENTITY]\nCould I take your name?"
VALID = [["GREETING", "VERIFY_IDENTITY"], ["LOOKUP_ORDER", "CHECK_STATUS"]]


# --------------------------------------------------------------------------- #
# drop_tool_calls — announce-but-don't-call
# --------------------------------------------------------------------------- #


def test_drop_tool_calls_removes_the_call_but_keeps_the_narration():
    out = drop_tool_calls(TOOL_TURN)
    assert "<tool_call>" not in out
    assert "Let me look that order up for you." in out
    assert "[STATE: LOOKUP_ORDER → LOOKUP_ORDER]" in out


def test_drop_tool_calls_is_none_when_there_is_no_call():
    assert drop_tool_calls(ADVANCE_TURN) is None


def test_drop_tool_calls_removes_every_call():
    two = TOOL_TURN + '\n<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    assert "<tool_call>" not in drop_tool_calls(two)


# --------------------------------------------------------------------------- #
# flip_state_transition
# --------------------------------------------------------------------------- #


def test_self_loop_becomes_a_legal_advance():
    """The v4 failure: advancing on a turn the convention says should stay."""
    out = flip_state_transition(TOOL_TURN, VALID, seed=0)
    assert "[STATE: LOOKUP_ORDER → CHECK_STATUS]" in out
    assert "[STATE: LOOKUP_ORDER → LOOKUP_ORDER]" not in out


def test_advance_becomes_a_spurious_self_loop():
    out = flip_state_transition(ADVANCE_TURN, VALID, seed=0)
    assert "[STATE: GREETING → GREETING]" in out


def test_flip_preserves_everything_but_the_annotation():
    out = flip_state_transition(TOOL_TURN, VALID, seed=0)
    assert "<tool_call>" in out
    assert "Let me look that order up for you." in out


def test_flip_is_none_when_no_legal_successor_exists():
    """A self-loop with no outgoing edge cannot be corrupted into a legal one."""
    assert flip_state_transition(TOOL_TURN, [["OTHER", "ELSEWHERE"]], seed=0) is None


def test_flip_is_none_without_a_state_annotation():
    assert flip_state_transition("just prose", VALID, seed=0) is None


def test_flip_never_returns_the_input_unchanged():
    for text in (TOOL_TURN, ADVANCE_TURN):
        out = flip_state_transition(text, VALID, seed=0)
        assert out != text


# --------------------------------------------------------------------------- #
# corrupt_tool_args
# --------------------------------------------------------------------------- #


def test_corrupt_tool_args_keeps_the_name_and_changes_an_argument():
    out = corrupt_tool_args(TOOL_TURN, seed=0)
    payload = json.loads(out.split("<tool_call>")[1].split("</tool_call>")[0])
    assert payload["name"] == "lookup_order"
    assert payload["arguments"]["order_id"] != "TH-99823"


def test_corrupt_tool_args_emits_valid_json():
    out = corrupt_tool_args(TOOL_TURN, seed=0)
    body = out.split("<tool_call>")[1].split("</tool_call>")[0]
    json.loads(body)  # must not raise


def test_corrupt_tool_args_is_none_without_a_call():
    assert corrupt_tool_args(ADVANCE_TURN, seed=0) is None


def test_corrupt_tool_args_is_none_when_the_call_has_no_arguments():
    """Nothing to corrupt — must not emit a zero-margin pair."""
    turn = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
    assert corrupt_tool_args(turn, seed=0) is None


def test_corrupt_tool_args_is_deterministic_under_seed():
    assert corrupt_tool_args(TOOL_TURN, seed=7) == corrupt_tool_args(TOOL_TURN, seed=7)


def test_corrupt_tool_args_preserves_the_state_annotation():
    """Only the arguments may differ, or the pair tests two things at once."""
    out = corrupt_tool_args(TOOL_TURN, seed=0)
    assert "[STATE: LOOKUP_ORDER → LOOKUP_ORDER]" in out


@pytest.mark.parametrize("fn", [drop_tool_calls])
def test_corruptions_never_return_the_input_unchanged(fn):
    assert fn(TOOL_TURN) != TOOL_TURN


# --------------------------------------------------------------------------- #
# corrupt_tool_args must stay in-distribution
# --------------------------------------------------------------------------- #

MULTI_ARG_TURN = (
    "[STATE: CHECK → CHECK]\n"
    '<tool_call>{"name": "check_visa", "arguments": '
    '{"nationality": "Thai", "destination": "United Kingdom"}}</tool_call>'
)
SINGLE_TEXT_ARG_TURN = (
    '<tool_call>{"name": "note", "arguments": {"reason": "customer request"}}</tool_call>'
)


def _args_of(text):
    return json.loads(text.split("<tool_call>")[1].split("</tool_call>")[0])["arguments"]


def test_no_synthetic_marker_is_ever_introduced():
    """An `_x`-style marker would be trivially separable.

    The model would learn to reject the marker instead of learning argument
    fidelity — scoring well on the pair set while being no better at the task.
    """
    for seed in range(25):
        for turn in (TOOL_TURN, MULTI_ARG_TURN, SINGLE_TEXT_ARG_TURN):
            out = corrupt_tool_args(turn, seed=seed)
            if out is None:
                continue
            for value in _args_of(out).values():
                if isinstance(value, str):
                    assert not value.endswith("_x"), value
                    assert "CORRUPT" not in value.upper()


def test_two_string_args_are_swapped_not_mangled():
    """A value-for-value mix-up is the most in-distribution wrong-arg error."""
    args = _args_of(corrupt_tool_args(MULTI_ARG_TURN, seed=1))
    assert args == {"nationality": "United Kingdom", "destination": "Thai"}


def test_single_non_numeric_arg_is_dropped_not_marked():
    """Missing required argument — a real failure, and no invented token."""
    args = _args_of(corrupt_tool_args(SINGLE_TEXT_ARG_TURN, seed=1))
    assert "reason" not in args


def test_numeric_identifier_keeps_its_shape():
    """Digit perturbation models 'right tool, wrong record'."""
    args = _args_of(corrupt_tool_args(TOOL_TURN, seed=3))
    assert args["order_id"] != "TH-99823"
    assert args["order_id"].startswith("TH-")
    assert len(args["order_id"]) == len("TH-99823")


# --------------------------------------------------------------------------- #
# Contamination guard must actually be able to fire
# --------------------------------------------------------------------------- #


def test_prefix_fingerprints_match_a_turn_row_prompt():
    """The guard's whole point: a per-turn prompt must match its conversation.

    `user_turn_fingerprint` hashes ALL user turns, but a training row's prompt
    holds only the first k. Comparing a prompt fingerprint against whole
    conversation fingerprints therefore never matches — the guard passes
    everything and looks clean while doing nothing. Expanding to prefixes is
    what makes exclusion real.
    """
    from llm_workflow_agents.data.heldout_clean_set import (
        user_turn_fingerprint,
        user_turn_prefix_fingerprints,
    )

    conv = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
    prompt_at_turn_1 = {"messages": conv["messages"][:2]}  # system + u1

    whole = user_turn_fingerprint(conv)
    prefix_fp = user_turn_fingerprint(prompt_at_turn_1)

    assert prefix_fp != whole, "prefix must not equal the whole — the inert case"
    assert prefix_fp in user_turn_prefix_fingerprints(conv)
    assert user_turn_fingerprint({"messages": conv["messages"][:4]}) in (
        user_turn_prefix_fingerprints(conv)
    )


def test_prefix_fingerprints_do_not_match_a_different_conversation():
    from llm_workflow_agents.data.heldout_clean_set import (
        user_turn_fingerprint,
        user_turn_prefix_fingerprints,
    )

    a = {"messages": [{"role": "user", "content": "alpha"}]}
    b = {"messages": [{"role": "user", "content": "beta"}]}
    assert user_turn_fingerprint(b) not in user_turn_prefix_fingerprints(a)
