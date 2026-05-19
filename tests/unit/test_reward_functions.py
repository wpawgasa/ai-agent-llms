"""Tests for reward functions, reward_utils, SFT/GRPO result types, and pilot_check.

These tests run without GPU by testing the logic/wiring only.
Heavy deps are mocked where needed.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.tool_call_f1 import (
    compute_argument_graded_f1,
    compute_ast_f1,
)
from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
    format_compliance_check,
    graded_tool_call_f1,
    reached_terminal,
    state_sequence_match,
    tool_call_f1,
)
from llm_workflow_agents.training.rewards.reward_business_logic import (
    W_LENGTH_BAND,
    W_STATE_TRANSITION,
    W_TASK_COMPLETION,
    W_TOOL_CALL_F1,
    _graded_state_match,
    _partial_state_match,
    reward_business_logic,
)
from llm_workflow_agents.training.rewards.reward_subagent import (
    _compute_slot_accuracy,
    reward_subagent,
)
from llm_workflow_agents.training.rewards.reward_graph_extraction import (
    reward_graph_extraction,
)
from llm_workflow_agents.training.sft import SFTResult
from llm_workflow_agents.training.grpo import GRPOResult, _resolve_reward_fn
from llm_workflow_agents.training.pilot_check import PilotResult


# --- Reward Utils Tests ---


class TestRewardUtils:
    """Tests for shared reward computation helpers."""

    def test_extract_state_annotations(self) -> None:
        text = "[STATE: INIT → GREETING] Hello! [STATE: GREETING → COLLECT_INFO]"
        annotations = extract_state_annotations(text)
        assert len(annotations) == 2
        assert annotations[0] == ("INIT", "GREETING")
        assert annotations[1] == ("GREETING", "COLLECT_INFO")

    def test_extract_tool_calls(self) -> None:
        text = '<tool_call>{"name": "lookup", "arguments": {"id": "123"}}</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "lookup"

    def test_state_sequence_match_perfect(self) -> None:
        transitions = [("A", "B"), ("B", "C")]
        score = state_sequence_match(transitions, transitions)
        assert score == 1.0

    def test_state_sequence_match_empty(self) -> None:
        score = state_sequence_match([], [("A", "B")])
        assert score == 0.0

    def test_tool_call_f1_perfect(self) -> None:
        tools = [{"name": "fn", "arguments": {"x": 1}}]
        score = tool_call_f1(tools, tools)
        assert score == 1.0

    def test_tool_call_f1_empty(self) -> None:
        score = tool_call_f1([], [{"name": "fn", "arguments": {}}])
        assert score == 0.0

    def test_format_compliance_valid(self) -> None:
        text = "<tool_call>{}</tool_call> done"
        assert format_compliance_check(text) == 1.0

    def test_format_compliance_mismatched_tags(self) -> None:
        text = "<tool_call>{} done"
        score = format_compliance_check(text)
        assert score == 0.5

    def test_format_compliance_traceback(self) -> None:
        text = "Traceback (most recent call last):"
        score = format_compliance_check(text)
        assert score < 1.0

    def test_reached_terminal_true(self) -> None:
        text = "[STATE: COLLECT → DONE]"
        assert reached_terminal(text, "DONE") is True

    def test_reached_terminal_false(self) -> None:
        text = "[STATE: INIT → COLLECT]"
        assert reached_terminal(text, "DONE") is False

    def test_reached_terminal_no_annotations(self) -> None:
        assert reached_terminal("plain text", "DONE") is False


# --- Reward Business Logic Tests ---


class TestRewardBusinessLogic:
    """Known-answer tests for Cat A reward function."""

    def test_weights_sum_to_one(self) -> None:
        total = (
            W_STATE_TRANSITION + W_TOOL_CALL_F1
            + W_TASK_COMPLETION + W_LENGTH_BAND
        )
        assert abs(total - 1.0) < 1e-9

    def test_weights_match_2026_05_19_redesign(self) -> None:
        # Pins the post-pre-flight weights so future drift is caught.
        # See docs/grpo_diagnosis_gemma4_26b.md addendum.
        assert W_STATE_TRANSITION == 0.40
        assert W_TOOL_CALL_F1 == 0.40
        assert W_TASK_COMPLETION == 0.10
        assert W_LENGTH_BAND == 0.10

    def test_format_only_score(self) -> None:
        """Well-formatted but wrong content scores at least format component."""
        result = reward_business_logic(
            prompts=["prompt"],
            completions=["<tool_call>{}</tool_call>"],
            ground_truths=[{
                "state_annotations": [("A", "B")],
                "tool_calls": [{"name": "fn", "arguments": {"x": 1}}],
                "messages": [],
                "terminal_state": "END",
            }],
        )
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_empty_completion_low_score(self) -> None:
        result = reward_business_logic(
            prompts=["prompt"],
            completions=[""],
            ground_truths=[{
                "state_annotations": [("A", "B")],
                "tool_calls": [{"name": "fn", "arguments": {}}],
                "messages": [],
                "terminal_state": "END",
            }],
        )
        assert result[0] < 0.5

    def test_batch_processing(self) -> None:
        result = reward_business_logic(
            prompts=["p1", "p2"],
            completions=["c1", "c2"],
            ground_truths=[
                {"state_annotations": [], "tool_calls": [], "messages": [], "terminal_state": ""},
                {"state_annotations": [], "tool_calls": [], "messages": [], "terminal_state": ""},
            ],
        )
        assert len(result) == 2


# --- Graded State Match Tests (fix c2 — see grpo_diagnosis addendum) ---


class TestGradedStateMatch:
    """Continuous variant of _partial_state_match used by the GRPO reward."""

    def test_exact_match_scores_one(self) -> None:
        assert _graded_state_match([("A", "B")], [("A", "B")]) == 1.0

    def test_from_only_scores_half(self) -> None:
        assert _graded_state_match([("A", "X")], [("A", "B")]) == 0.5

    def test_to_only_scores_half(self) -> None:
        # New credit not present in _partial_state_match.
        assert _graded_state_match([("X", "B")], [("A", "B")]) == 0.5

    def test_reverse_direction_scores_three_tenths(self) -> None:
        assert _graded_state_match([("B", "A")], [("A", "B")]) == pytest.approx(0.3)

    def test_no_overlap_scores_zero(self) -> None:
        assert _graded_state_match([("X", "Y")], [("A", "B")]) == 0.0

    def test_empty_gt_empty_pred_scores_one(self) -> None:
        assert _graded_state_match([], []) == 1.0

    def test_empty_gt_with_pred_scores_zero(self) -> None:
        assert _graded_state_match([("A", "B")], []) == 0.0

    def test_dominates_partial_state_match_on_to_only(self) -> None:
        # The whole point of the redesign: to-only matches must give credit.
        pred, gt = [("X", "B")], [("A", "B")]
        assert _graded_state_match(pred, gt) > _partial_state_match(pred, gt)

    def test_matches_partial_state_match_on_exact_and_from_only(self) -> None:
        # Where the old metric did give credit, the new one must agree.
        for pred, gt in [
            ([("A", "B")], [("A", "B")]),     # exact
            ([("A", "X")], [("A", "B")]),     # from-only
            ([("X", "Y")], [("A", "B")]),     # no overlap
        ]:
            assert _graded_state_match(pred, gt) == _partial_state_match(pred, gt)


# --- Graded Tool F1 Tests (fix c1 — see grpo_diagnosis addendum) ---


class TestArgumentGradedF1:
    """Continuous variant of compute_ast_f1 used by the GRPO reward."""

    def test_perfect_match_scores_one(self) -> None:
        tools = [{"name": "fn", "arguments": {"x": 1, "y": 2}}]
        assert compute_argument_graded_f1(tools, tools) == 1.0

    def test_both_empty_scores_one(self) -> None:
        assert compute_argument_graded_f1([], []) == 1.0

    def test_one_empty_scores_zero(self) -> None:
        assert compute_argument_graded_f1([], [{"name": "fn", "arguments": {}}]) == 0.0
        assert compute_argument_graded_f1([{"name": "fn", "arguments": {}}], []) == 0.0

    def test_wrong_name_scores_zero(self) -> None:
        pred = [{"name": "wrong", "arguments": {"x": 1}}]
        gt = [{"name": "fn", "arguments": {"x": 1}}]
        assert compute_argument_graded_f1(pred, gt) == 0.0

    def test_right_name_no_args_scores_full(self) -> None:
        pred = [{"name": "fn", "arguments": {}}]
        gt = [{"name": "fn", "arguments": {}}]
        assert compute_argument_graded_f1(pred, gt) == 1.0

    def test_right_name_half_args_correct(self) -> None:
        # Pair score = 0.4 (name) + 0.6 * (1/2 args) = 0.7
        # F1 with single pred & gt: precision=recall=0.7 → F1 = 0.7
        pred = [{"name": "fn", "arguments": {"x": 1, "y": 999}}]
        gt = [{"name": "fn", "arguments": {"x": 1, "y": 2}}]
        assert compute_argument_graded_f1(pred, gt) == pytest.approx(0.7)

    def test_right_name_zero_args_correct(self) -> None:
        # Pair score = 0.4 (name) + 0.6 * 0/2 = 0.4. F1 = 0.4.
        pred = [{"name": "fn", "arguments": {"x": 999, "y": 999}}]
        gt = [{"name": "fn", "arguments": {"x": 1, "y": 2}}]
        assert compute_argument_graded_f1(pred, gt) == pytest.approx(0.4)

    def test_dominates_strict_on_partial_match(self) -> None:
        # The whole point of the redesign: a partial-arg match must not
        # collapse to 0 (which is what compute_ast_f1 does on this input).
        pred = [{"name": "fn", "arguments": {"x": 1, "y": 999}}]
        gt = [{"name": "fn", "arguments": {"x": 1, "y": 2}}]
        graded = compute_argument_graded_f1(pred, gt)
        strict = compute_ast_f1(pred, gt)
        assert graded > strict
        assert strict == 0.0

    def test_matches_strict_on_exact_and_no_match(self) -> None:
        # Where the strict metric gives a clean 1.0 or 0.0, graded agrees.
        exact_pred = [{"name": "fn", "arguments": {"x": 1}}]
        wrong_name = [{"name": "wrong", "arguments": {"x": 1}}]
        assert compute_argument_graded_f1(exact_pred, exact_pred) == 1.0
        assert compute_argument_graded_f1(wrong_name, exact_pred) == 0.0

    def test_hallucinated_extra_call_lowers_precision(self) -> None:
        # 1 correct + 1 hallucinated: partial_tp=1.0, p=1/2, r=1/1 → F1 = 2/3.
        gt = [{"name": "fn", "arguments": {"x": 1}}]
        pred = [
            {"name": "fn", "arguments": {"x": 1}},
            {"name": "ghost", "arguments": {}},
        ]
        assert compute_argument_graded_f1(pred, gt) == pytest.approx(2 / 3)

    def test_graded_tool_call_f1_wrapper_matches(self) -> None:
        # reward_utils wrapper delegates to compute_argument_graded_f1.
        pred = [{"name": "fn", "arguments": {"x": 1, "y": 999}}]
        gt = [{"name": "fn", "arguments": {"x": 1, "y": 2}}]
        assert graded_tool_call_f1(pred, gt) == compute_argument_graded_f1(pred, gt)


# --- Reward Subagent Tests ---


class TestRewardSubagent:
    """Known-answer tests for Cat B reward function."""

    def test_slot_accuracy_perfect(self) -> None:
        tools = [{"name": "fn", "arguments": {"slot1": "val1", "slot2": "val2"}}]
        assert _compute_slot_accuracy(tools, tools) == 1.0

    def test_slot_accuracy_partial(self) -> None:
        pred = [{"name": "fn", "arguments": {"slot1": "val1"}}]
        gt = [{"name": "fn", "arguments": {"slot1": "val1", "slot2": "val2"}}]
        assert _compute_slot_accuracy(pred, gt) == 0.5

    def test_slot_accuracy_no_gt(self) -> None:
        assert _compute_slot_accuracy([], []) == 1.0

    def test_slot_accuracy_no_match(self) -> None:
        pred = [{"name": "wrong", "arguments": {}}]
        gt = [{"name": "fn", "arguments": {"slot1": "val1"}}]
        assert _compute_slot_accuracy(pred, gt) == 0.0

    def test_empty_completion(self) -> None:
        result = reward_subagent(
            prompts=["p"],
            completions=[""],
            ground_truths=[{"tool_calls": [], "state_annotations": []}],
        )
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0


# --- Reward Graph Extraction Tests ---


class TestRewardGraphExtraction:
    """Known-answer tests for Cat C reward function."""

    def test_invalid_json_early_exit(self) -> None:
        result = reward_graph_extraction(
            prompts=["p"],
            completions=["this is not json at all"],
            ground_truths=[{"nodes": [], "edges": [], "initial_state": "", "terminal_states": []}],
        )
        assert result[0] == 0.0

    def test_valid_json_scores_above_zero(self) -> None:
        import json

        graph = {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"from": "A", "to": "B"}],
            "initial_state": "A",
            "terminal_states": ["B"],
        }
        completion = json.dumps(graph)
        result = reward_graph_extraction(
            prompts=["p"],
            completions=[completion],
            ground_truths=[graph],
        )
        assert result[0] > 0.0

    def test_batch_with_mixed_validity(self) -> None:
        import json

        valid_graph = {
            "nodes": [{"id": "X"}],
            "edges": [],
            "initial_state": "X",
            "terminal_states": ["X"],
        }
        result = reward_graph_extraction(
            prompts=["p1", "p2"],
            completions=[json.dumps(valid_graph), "not json"],
            ground_truths=[valid_graph, valid_graph],
        )
        assert len(result) == 2
        assert result[0] > 0.0
        assert result[1] == 0.0


# --- Result Dataclass Tests ---


class TestSFTResult:
    """Tests for SFTResult dataclass."""

    def test_defaults(self) -> None:
        r = SFTResult()
        assert r.checkpoint_path is None
        assert r.best_eval_loss is None
        assert r.total_steps == 0
        assert r.error is None
        assert r.metrics == {}

    def test_with_error(self) -> None:
        r = SFTResult(error="something went wrong")
        assert r.error == "something went wrong"


class TestGRPOResult:
    """Tests for GRPOResult dataclass."""

    def test_defaults(self) -> None:
        r = GRPOResult()
        assert r.checkpoint_path is None
        assert r.total_steps == 0
        assert r.early_stopped is False
        assert r.reward_curves == []
        assert r.kl_divergence == []

    def test_early_stopped(self) -> None:
        r = GRPOResult(early_stopped=True, total_steps=250)
        assert r.early_stopped is True
        assert r.total_steps == 250


class TestResolveRewardFn:
    """Tests for dynamic reward function resolution."""

    def test_resolve_business_logic(self) -> None:
        fn = _resolve_reward_fn("reward_business_logic")
        assert callable(fn)

    def test_resolve_subagent(self) -> None:
        fn = _resolve_reward_fn("reward_subagent")
        assert callable(fn)

    def test_resolve_graph_extraction(self) -> None:
        fn = _resolve_reward_fn("reward_graph_extraction")
        assert callable(fn)

    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward function"):
            _resolve_reward_fn("nonexistent")


# --- Pilot Check Tests ---


class TestPilotResult:
    """Tests for PilotResult dataclass."""

    def test_defaults(self) -> None:
        r = PilotResult()
        assert r.model_name == ""
        assert r.initial_loss is None
        assert r.degraded is False

    def test_degraded(self) -> None:
        r = PilotResult(
            model_name="test/model",
            initial_loss=2.0,
            final_loss=3.0,
            degraded=True,
        )
        assert r.degraded is True
        assert r.final_loss > r.initial_loss
