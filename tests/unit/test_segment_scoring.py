"""Segment scoring: one scored turn per run of back-to-back gold assistant turns.

The corpus splits speech and a tool call across two turns in about a third of
its tool turns; a model may do both in one reply. Scored turn by turn, the
combined reply lost twice — a spurious call on the speech turn, a missing call
on the call turn. Scored by segment, the split and the combined answer score
the same, and the stay rule (a tool call sits under a self-loop annotation) is
still enforced.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pytest

from llm_workflow_agents.eval.agent_benchmark import build_state_machine_inputs, dataset_fingerprints
from llm_workflow_agents.eval.segment_scoring import (
    ILLEGAL_STATE,
    SegmentStats,
    normalize_segment_states,
    reply_texts,
    segment_scoring_view,
)
from llm_workflow_agents.eval.state_accuracy import (
    evaluate_state_machine,
    extract_ground_truth_transitions,
    parse_state_transitions,
)
from llm_workflow_agents.eval.tool_call_f1 import TurnGroundTruth, TurnPrediction, evaluate_tool_calls

CALL = {"name": "check_balance", "arguments": {"account_id": "ACC-1"}}
CALL_TEXT = f"<tool_call>{json.dumps(CALL)}</tool_call>"

GOLD = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "balance for ACC-1"},
    {
        "role": "assistant",
        "content": "[STATE: GREET → LOOKUP] Let me check that.",
        "annotations": {"state_transition": {"from": "GREET", "to": "LOOKUP"}, "tool_calls": []},
    },
    {
        "role": "assistant",
        "content": f"[STATE: LOOKUP → LOOKUP]\n{CALL_TEXT}",
        "annotations": {"state_transition": {"from": "LOOKUP", "to": "LOOKUP"}, "tool_calls": [CALL]},
    },
    {"role": "tool", "content": '{"balance": 10}'},
    {
        "role": "assistant",
        "content": "[STATE: LOOKUP → DONE] Your balance is 10.",
        "annotations": {"state_transition": {"from": "LOOKUP", "to": "DONE"}, "tool_calls": []},
    },
]


def _pred(first_slot: str, last: str = "[STATE: LOOKUP → DONE] It is 10.") -> list[dict]:
    return [
        GOLD[0], GOLD[1],
        {"role": "assistant", "content": first_slot},
        {"role": "assistant", "content": ""},
        GOLD[4],
        {"role": "assistant", "content": last},
    ]


def _scores(pred: list[dict]) -> tuple[float, float]:
    gt_view, pred_view = segment_scoring_view(GOLD, pred)
    turns = [(p, g) for p, g in zip(pred_view, gt_view) if g["role"] == "assistant"]
    tool = evaluate_tool_calls(
        [TurnPrediction(turn_id=k, content=p["content"]) for k, (p, _) in enumerate(turns)],
        [TurnGroundTruth(turn_id=k, tool_calls=g["annotations"]["tool_calls"]) for k, (_, g) in enumerate(turns)],
    )
    preds, gts = build_state_machine_inputs([{"conversation_id": "c"}], [pred_view], [gt_view])
    state = evaluate_state_machine(preds, gts, num_stochastic_trials=0)
    return tool.tool_call_f1, state.state_transition_accuracy


def test_split_and_combined_answers_score_the_same():
    split = _pred(f"[STATE: GREET → LOOKUP] Let me check that.\n[STATE: LOOKUP → LOOKUP]\n{CALL_TEXT}")
    combined = _pred(f"[STATE: GREET → LOOKUP] Checking.\n[STATE: LOOKUP → LOOKUP] {CALL_TEXT}")
    assert _scores(split) == _scores(combined) == (1.0, 1.0)


def test_a_call_under_an_advancing_annotation_breaks_the_stay_rule():
    tool_f1, state_acc = _scores(_pred(f"[STATE: GREET → LOOKUP] Checking. {CALL_TEXT}"))
    assert tool_f1 == 1.0  # the call itself is right
    assert state_acc == 0.5  # but the segment's transition is illegal


def test_a_missing_call_is_a_miss_over_the_whole_segment():
    tool_f1, state_acc = _scores(_pred("[STATE: GREET → LOOKUP] Let me check that."))
    assert tool_f1 < 1.0
    assert state_acc == 1.0


def test_normalize_legal_chain():
    content, problem = normalize_segment_states(f"[STATE: A → B] hi\n[STATE: B → B]\n{CALL_TEXT}")
    assert problem is None
    assert parse_state_transitions([{"role": "assistant", "content": content}]) == [("A", "B")]
    assert CALL_TEXT in content and "hi" in content


def test_normalize_discontinuous_chain():
    content, problem = normalize_segment_states("[STATE: A → B] hi\n[STATE: C → C] there")
    assert problem == "discontinuous"
    assert parse_state_transitions([{"role": "assistant", "content": content}]) == [("A", ILLEGAL_STATE)]


def test_normalize_without_annotation_is_unchanged():
    assert normalize_segment_states("just words") == ("just words", None)


def test_ground_truth_calls_and_transition_merge_per_segment():
    gt_view, pred_view = segment_scoring_view(GOLD, GOLD)
    assert len(gt_view) == len(pred_view) == 5
    merged = gt_view[2]
    assert merged["annotations"]["tool_calls"] == [CALL]
    assert merged["annotations"]["state_transition"] == {"from": "GREET", "to": "LOOKUP"}


def test_unscored_segment_is_dropped_from_both_views():
    pred = _pred("x")
    pred[2] = {**GOLD[2], "unscored": True}
    pred[3] = {**GOLD[3], "unscored": True}
    stats = SegmentStats()
    gt_view, pred_view = segment_scoring_view(GOLD, pred, stats)
    assert len(gt_view) == len(pred_view) == 4
    assert stats.unscored_segments == 1 and stats.segments == 2


def test_reply_texts_lists_each_request_once():
    pred = _pred("a\nb")
    pred[2]["replies"] = ["a", "b"]
    assert reply_texts(pred) == ["a", "b", "[STATE: LOOKUP → DONE] It is 10."]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        segment_scoring_view(GOLD, GOLD[:-1])


def test_dataset_fingerprint_changes_with_content(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n')
    first = dataset_fingerprints([tmp_path])[str(tmp_path)]
    (tmp_path / "a.jsonl").write_text('{"x": 2}\n')
    assert dataset_fingerprints([tmp_path])[str(tmp_path)] != first
    assert len(first) == len(hashlib.sha256().hexdigest())


V4 = sorted(glob.glob("data/output/benchmark/task_a_v4/*.jsonl")) + sorted(
    glob.glob("data/output/benchmark/task_a_voice_v3/*.jsonl")
)


@pytest.mark.skipif(not V4, reason="v4 benchmark not materialized (dvc pull)")
def test_ground_truth_is_legal_and_scores_itself_exactly():
    stats = SegmentStats()
    for path in V4:
        for line in open(path):
            messages = json.loads(line)["messages"]
            gt_view, pred_view = segment_scoring_view(messages, messages, stats)
            assert parse_state_transitions(pred_view) == extract_ground_truth_transitions(gt_view)
    assert stats.illegal_discontinuous == 0 and stats.illegal_stay_rule == 0
    assert stats.multi_turn_segments > 0
