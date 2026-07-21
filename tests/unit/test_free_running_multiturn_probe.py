"""Unit tests for scripts/free_running_multiturn_probe.py pure functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from free_running_multiturn_probe import (  # noqa: E402
    MATERIAL_DELTA,
    RELIABLE_WINDOW,
    classify_free_running_outcome,
    classify_gate,
    classify_target_shape,
    find_anchors,
    strip_tool_calls,
    summarize_probe,
)

CALL = '<tool_call>{"name":"check_balance","arguments":{"id":"1"}}</tool_call>'


def _asst(content: str, tool_calls: list | None = None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "annotations": {"tool_calls": tool_calls or []},
    }


def _conv(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"conversation_id": "c1", "complexity_level": "L3", "messages": messages}


class TestStripToolCalls:
    def test_strips_call_blocks_and_state_markers(self):
        text = f"[STATE: A → B]\n{CALL}"
        assert strip_tool_calls(text) == ""

    def test_keeps_narration(self):
        assert strip_tool_calls(f"[STATE: A → B]\nLet me check that.\n{CALL}") == (
            "Let me check that."
        )

    def test_handles_empty(self):
        assert strip_tool_calls("") == ""


class TestClassifyTargetShape:
    def test_no_tool_calls_is_no_tool(self):
        assert classify_target_shape("I will look that up now.", []) == "no_tool"

    def test_call_only_is_bare_call(self):
        assert classify_target_shape(f"[STATE: A → B]\n{CALL}", [{"name": "x"}]) == (
            "bare_call"
        )

    def test_narration_plus_call_is_fused(self):
        content = f"[STATE: A → B]\nChecking your balance right away.\n{CALL}"
        assert classify_target_shape(content, [{"name": "x"}]) == "fused"

    def test_trivial_residue_stays_bare_call(self):
        # Below NARRATION_MIN_CHARS — punctuation noise is not narration.
        assert classify_target_shape(f"{CALL}\nOK.", [{"name": "x"}]) == "bare_call"


class TestFindAnchors:
    def test_finds_announce_then_bare_call(self):
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst("I'll check your balance, may I confirm your ID?"),
            {"role": "user", "content": "yes, 1"},
            _asst(CALL, [{"name": "check_balance"}]),
        ])
        anchors = find_anchors(conv)
        assert len(anchors) == 1
        a = anchors[0]
        assert (a["target_index"], a["announce_index"], a["user_index"]) == (4, 2, 3)
        assert a["announce_prev_role"] == "user"
        assert a["announce_is_sliceable"] is True

    def test_marks_tool_tailed_announce_not_sliceable(self):
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst(CALL, [{"name": "lookup"}]),
            {"role": "tool", "content": "{}"},
            _asst("Now I'll check your balance, confirm ID?"),
            {"role": "user", "content": "yes"},
            _asst(CALL, [{"name": "check_balance"}]),
        ])
        anchors = find_anchors(conv)
        assert len(anchors) == 1
        assert anchors[0]["announce_prev_role"] == "tool"
        assert anchors[0]["announce_is_sliceable"] is False

    def test_skips_fused_target(self):
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst("I'll check your balance now, one moment."),
            {"role": "user", "content": "ok"},
            _asst(f"Checking your balance right now.\n{CALL}", [{"name": "cb"}]),
        ])
        assert find_anchors(conv) == []

    def test_skips_when_predecessor_assistant_already_called(self):
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst(CALL, [{"name": "lookup"}]),
            {"role": "user", "content": "ok"},
            _asst(CALL, [{"name": "check_balance"}]),
        ])
        assert find_anchors(conv) == []

    def test_skips_target_not_preceded_by_user(self):
        # Adjacent assistant turns — _load_grpo_jsonl never emits this as a row.
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst("I'll check your balance."),
            _asst(CALL, [{"name": "check_balance"}]),
        ])
        assert find_anchors(conv) == []

    def test_skips_target_with_no_gt_tool_calls(self):
        conv = _conv([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _asst("I'll check your balance, confirm ID?"),
            {"role": "user", "content": "yes"},
            _asst("Thanks, all set."),
        ])
        assert find_anchors(conv) == []

    def test_empty_conversation(self):
        assert find_anchors({"messages": []}) == []


class TestClassifyFreeRunningOutcome:
    def test_fired_at_t1_when_first_turn_calls(self):
        assert classify_free_running_outcome([{"name": "x"}], []) == "fired_at_t1"

    def test_t1_wins_when_both_turns_call(self):
        assert classify_free_running_outcome([{"name": "x"}], [{"name": "y"}]) == (
            "fired_at_t1"
        )

    def test_fired_at_t2_after_user_reply(self):
        assert classify_free_running_outcome([], [{"name": "y"}]) == "fired_at_t2"

    def test_never_fired(self):
        assert classify_free_running_outcome([], []) == "never_fired"


def _record(
    a_emitted: bool,
    b_outcome: str,
    *,
    a_f1: float = 0.0,
    b_f1: float = 0.0,
    t1_shape: str = "announce_no_call",
) -> dict[str, Any]:
    return {
        "a_emitted": a_emitted,
        "a_name_match": a_emitted,
        "a_f1": a_f1,
        "a_graded_f1": a_f1,
        "b_outcome": b_outcome,
        "b_emitted": b_outcome != "never_fired",
        "b_name_match": b_outcome != "never_fired",
        "b_f1": b_f1,
        "b_graded_f1": b_f1,
        "t1_shape": t1_shape,
    }


class TestSummarizeProbe:
    def test_empty_records_are_all_zero(self):
        s = summarize_probe([])
        assert s["n_anchors"] == 0
        assert s["single_turn_emission_rate"] == 0.0
        assert s["free_running_emission_rate"] == 0.0
        assert s["frac_recovered_of_single_turn_failures"] == 0.0

    def test_rates_and_delta(self):
        records = [
            _record(False, "fired_at_t2"),
            _record(False, "fired_at_t2"),
            _record(False, "never_fired"),
            _record(True, "fired_at_t1"),
        ]
        s = summarize_probe(records)
        assert s["n_anchors"] == 4
        assert s["single_turn_emission_rate"] == 0.25
        assert s["free_running_emission_rate"] == 0.75
        assert s["emission_rate_delta"] == 0.5

    def test_outcome_counts_and_fracs(self):
        records = [
            _record(False, "fired_at_t1"),
            _record(False, "fired_at_t2"),
            _record(False, "never_fired"),
            _record(False, "never_fired"),
        ]
        s = summarize_probe(records)
        assert s["free_running_outcomes"] == {
            "fired_at_t1": 1,
            "fired_at_t2": 1,
            "never_fired": 2,
        }
        assert s["free_running_outcome_fracs"]["never_fired"] == 0.5

    def test_recovered_and_lost_are_both_reported(self):
        records = [
            _record(False, "fired_at_t2"),   # recovered
            _record(False, "fired_at_t2"),   # recovered
            _record(True, "never_fired"),    # lost
            _record(False, "never_fired"),
        ]
        s = summarize_probe(records)
        assert s["n_recovered_by_free_running"] == 2
        assert s["n_lost_by_free_running"] == 1
        # 3 single-turn failures, 2 recovered
        assert s["frac_recovered_of_single_turn_failures"] == pytest.approx(2 / 3)

    def test_mean_f1_uses_all_rows(self):
        records = [
            _record(True, "fired_at_t1", a_f1=1.0, b_f1=1.0),
            _record(False, "never_fired", a_f1=0.0, b_f1=0.0),
        ]
        s = summarize_probe(records)
        assert s["single_turn_mean_f1"] == 0.5
        assert s["free_running_mean_f1"] == 0.5

    def test_t1_shape_census(self):
        records = [
            _record(False, "fired_at_t1", t1_shape="fused_with_call"),
            _record(False, "fired_at_t2", t1_shape="announce_no_call"),
            _record(False, "never_fired", t1_shape="empty"),
        ]
        assert summarize_probe(records)["t1_shapes"] == {
            "announce_no_call": 1,
            "fused_with_call": 1,
            "empty": 1,
        }


class TestClassifyGate:
    def test_artifact_dominant(self):
        summary = {
            "emission_rate_delta": MATERIAL_DELTA + 0.1,
            "free_running_emission_rate": RELIABLE_WINDOW + 0.1,
        }
        verdict, detail = classify_gate(summary)
        assert verdict == "ARTIFACT_DOMINANT"
        assert all(detail["checks"].values())

    def test_partial_artifact_when_material_but_unreliable(self):
        summary = {
            "emission_rate_delta": MATERIAL_DELTA + 0.05,
            "free_running_emission_rate": RELIABLE_WINDOW - 0.1,
        }
        verdict, _ = classify_gate(summary)
        assert verdict == "PARTIAL_ARTIFACT"

    def test_policy_defect_when_no_material_gain(self):
        summary = {
            "emission_rate_delta": 0.01,
            "free_running_emission_rate": 0.9,
        }
        verdict, _ = classify_gate(summary)
        assert verdict == "POLICY_DEFECT"

    def test_boundaries_are_inclusive(self):
        summary = {
            "emission_rate_delta": MATERIAL_DELTA,
            "free_running_emission_rate": RELIABLE_WINDOW,
        }
        assert classify_gate(summary)[0] == "ARTIFACT_DOMINANT"
