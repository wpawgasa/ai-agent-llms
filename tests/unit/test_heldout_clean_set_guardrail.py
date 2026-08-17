"""Tests for reserve_guardrail_slice — the fingerprint partition that keeps
scripts/mine_model_negatives.py's validation-split mining and dpo.py's R5
held-out guardrail from ever drawing on the same conversations.

Returns PREFIX fingerprints, not whole-conversation fingerprints, because a
per-turn row's prompt (what both consumers actually fingerprint at call time)
is only a prefix of its conversation's user turns — matching it against a
whole-conversation fingerprint would silently match nothing, exactly the bug
fixed in commit 165f2cc for the held-out contamination guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.heldout_clean_set import (
    reserve_guardrail_slice,
    user_turn_fingerprint,
    user_turn_prefix_fingerprints,
)


def _conv(user_turns: list[str], cid: str) -> dict:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for u in user_turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": "ok"})
    return {"conversation_id": cid, "messages": msgs}


def _write_split(tmp_path: Path, conversations: list[dict], split: str = "validation") -> Path:
    path = tmp_path / f"{split}.jsonl"
    with open(path, "w") as fh:
        for c in conversations:
            fh.write(json.dumps(c) + "\n")
    return tmp_path


def test_returns_prefix_fingerprints_not_whole_conversation_ones(tmp_path):
    conv = _conv(["turn one", "turn two"], "c1")
    _write_split(tmp_path, [conv] * 10)  # all identical -> all reserved regardless of fraction
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.5, seed=1)
    # A per-turn row's prompt after the FIRST user turn only ("turn one") must
    # match — that is a prefix, and prefix fingerprints are what both real
    # consumers compute against a prompt's own message list.
    prefix_fp = user_turn_fingerprint({"messages": [{"role": "user", "content": "turn one"}]})
    assert prefix_fp in reserved
    # And it must be a SUBSET of the full prefix-fingerprint expansion for
    # every reserved conversation, matching user_turn_prefix_fingerprints exactly.
    assert reserved == user_turn_prefix_fingerprints(conv)


def test_reserved_and_minable_conversations_are_disjoint(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(50)]
    _write_split(tmp_path, conversations)
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.2, seed=7)

    # Every conversation is either fully inside `reserved` (all its prefixes
    # present) or fully outside (none present).
    for c in conversations:
        prefixes = user_turn_prefix_fingerprints(c)
        overlap = prefixes & reserved
        assert overlap == prefixes or overlap == set()


def test_reserved_fraction_is_honored_within_rounding(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(100)]
    _write_split(tmp_path, conversations)
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.2, seed=3)
    reserved_conv_count = sum(
        1 for c in conversations if user_turn_fingerprint(c) in reserved
    )
    assert reserved_conv_count == 20


def test_deterministic_across_repeated_calls_with_same_seed(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(30)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    assert a == b


def test_different_seeds_give_different_partitions(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(30)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=99)
    assert a != b


def test_result_is_stable_if_the_file_is_rewritten_in_a_different_row_order(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(20)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.25, seed=5)

    _write_split(tmp_path, list(reversed(conversations)))
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.25, seed=5)
    assert a == b


def test_rejects_fraction_out_of_range(tmp_path):
    _write_split(tmp_path, [_conv(["q"], "c1")])
    with pytest.raises(ValueError, match="reserved_fraction"):
        reserve_guardrail_slice(tmp_path, reserved_fraction=0.0, seed=1)
    with pytest.raises(ValueError, match="reserved_fraction"):
        reserve_guardrail_slice(tmp_path, reserved_fraction=1.0, seed=1)
