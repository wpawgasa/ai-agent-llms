"""The benchmark's own tool inputs honour the declared argument owners.

`eval/argument_scoring.py` decides what a paraphrase is worth;
`build_tool_inputs` is where the Phase 1 harness applies it. Both need cover:
a metric that works in isolation and is never reached is the R18(c) shape.
"""

from __future__ import annotations

from llm_workflow_agents.eval.agent_benchmark import build_tool_inputs
from llm_workflow_agents.eval.argument_scoring import EXACT_MATCH, ArgumentScoring
from llm_workflow_agents.eval.tool_call_f1 import compute_ast_f1

SOURCES = ArgumentScoring(sources={"close_case.resolution_summary": "free_text"})

GOLD_CALL = {
    "name": "close_case",
    "arguments": {"case_id": "CASE-1", "resolution_summary": "Refunded the duplicate charge"},
}


def _views(written: str) -> tuple[list[dict], list[dict]]:
    gt_view = [
        {"role": "user", "content": "please close it"},
        {"role": "assistant", "content": "[STATE: A -> A]", "annotations": {"tool_calls": [GOLD_CALL]}},
    ]
    pred_view = [
        {"role": "user", "content": "please close it"},
        {
            "role": "assistant",
            "content": (
                '[STATE: A -> A]<tool_call>{"name": "close_case", "arguments": '
                '{"case_id": "CASE-1", "resolution_summary": "%s"}}</tool_call>' % written
            ),
        },
    ]
    return gt_view, pred_view


def _f1(scoring: ArgumentScoring, written: str) -> float:
    gt_view, pred_view = _views(written)
    predictions, ground_truths, _ = build_tool_inputs(pred_view, gt_view, [], scoring)
    return compute_ast_f1(predictions[0].tool_calls, ground_truths[0].tool_calls)


PARAPHRASE = "Refunded the customer for the duplicated charge"
DIFFERENT = "Escalated the case to the supervisor"


def test_exact_match_scores_a_paraphrase_zero_as_it_always_has() -> None:
    assert _f1(EXACT_MATCH, PARAPHRASE) == 0.0


def test_the_declared_rule_scores_the_same_paraphrase_correct() -> None:
    assert _f1(SOURCES, PARAPHRASE) == 1.0


def test_the_declared_rule_still_scores_a_different_answer_zero() -> None:
    assert _f1(SOURCES, DIFFERENT) == 0.0


def test_an_undeclared_argument_is_untouched_by_the_declared_rule() -> None:
    gt_view, pred_view = _views(GOLD_CALL["arguments"]["resolution_summary"])
    pred_view[1]["content"] = pred_view[1]["content"].replace("CASE-1", "CASE-9")
    predictions, ground_truths, _ = build_tool_inputs(pred_view, gt_view, [], SOURCES)
    assert compute_ast_f1(predictions[0].tool_calls, ground_truths[0].tool_calls) == 0.0


def test_only_ground_truth_assistant_turns_become_scored_turns() -> None:
    gt_view, pred_view = _views(PARAPHRASE)
    predictions, ground_truths, _ = build_tool_inputs(pred_view, gt_view, [], SOURCES)
    assert len(predictions) == len(ground_truths) == 1
    assert predictions[0].turn_id == 1


def test_the_agent_decidable_pair_carries_the_same_canonicalized_calls() -> None:
    gt_view, pred_view = _views(PARAPHRASE)
    predictions, _, (decidable_preds, decidable_gts) = build_tool_inputs(
        pred_view, gt_view, [], SOURCES
    )
    assert decidable_preds[0].tool_calls == predictions[0].tool_calls
    assert len(decidable_gts) == 1


def test_counters_report_what_the_rule_changed() -> None:
    counters: dict[str, int] = {}
    gt_view, pred_view = _views(PARAPHRASE)
    build_tool_inputs(pred_view, gt_view, [], SOURCES, counters)
    assert counters == {"free_text_accepted": 1}
