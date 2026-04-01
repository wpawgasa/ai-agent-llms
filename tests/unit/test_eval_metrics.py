"""Tests for evaluation metrics — known-answer tests for F1, state accuracy, chain propagation."""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.composite_score import (
    CompositeResult,
    compute_weighted_workflow_score,
    full_workflow_success_rate,
)
from llm_workflow_agents.eval.quant_benchmark import (
    CellResult,
    QuantBenchmarkMatrix,
    QualityMetrics,
    RunResult,
    _aggregate_quality,
    _compute_mean_std,
)
from llm_workflow_agents.eval.state_accuracy import (
    ConversationGroundTruth,
    ConversationPrediction,
    StateMachineMetrics,
    check_recovery,
    check_task_completion,
    compute_pass5_consistency,
    compute_transition_accuracy,
    evaluate_state_machine,
    extract_ground_truth_transitions,
    parse_state_transitions,
)
from llm_workflow_agents.eval.tool_call_f1 import (
    ToolCallMetrics,
    TurnGroundTruth,
    TurnPrediction,
    compute_argument_match,
    compute_ast_f1,
    compute_name_accuracy,
    detect_hallucinated_tools,
    evaluate_tool_calls,
    parse_tool_calls,
)
from llm_workflow_agents.eval.tool_chain_propagation import (
    ChainPropagationMetrics,
    ToolChainLink,
    check_value_propagation,
    evaluate_chain_propagation,
    extract_tool_chains,
)
from llm_workflow_agents.eval.agent_benchmark import (
    WorkflowQualityMetrics,
    compute_full_workflow_success,
    compute_latency_median,
    compute_weighted_score,
    evaluate_workflow_quality,
)


# ============================================================
# State Accuracy Tests
# ============================================================


class TestParseStateTransitions:

    def test_arrow_unicode(self) -> None:
        messages = [
            {"role": "assistant", "content": "[STATE: GREETING → COLLECT_INFO] Hello!"},
        ]
        result = parse_state_transitions(messages)
        assert result == [("GREETING", "COLLECT_INFO")]

    def test_arrow_ascii(self) -> None:
        messages = [
            {"role": "assistant", "content": "[STATE: A -> B] text"},
        ]
        assert parse_state_transitions(messages) == [("A", "B")]

    def test_multiple_transitions(self) -> None:
        messages = [
            {"role": "assistant", "content": "[STATE: A → B] first"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "[STATE: B → C] second"},
        ]
        result = parse_state_transitions(messages)
        assert result == [("A", "B"), ("B", "C")]

    def test_no_transitions(self) -> None:
        messages = [
            {"role": "assistant", "content": "No state info here"},
            {"role": "user", "content": "Hello"},
        ]
        assert parse_state_transitions(messages) == []

    def test_ignores_non_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "[STATE: A → B]"},
        ]
        assert parse_state_transitions(messages) == []


class TestExtractGroundTruthTransitions:

    def test_from_annotations(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "response",
                "annotations": {
                    "state_transition": {"from": "GREETING", "to": "COLLECT"},
                },
            },
        ]
        result = extract_ground_truth_transitions(messages)
        assert result == [("GREETING", "COLLECT")]

    def test_missing_annotations(self) -> None:
        messages = [{"role": "assistant", "content": "no annotations"}]
        assert extract_ground_truth_transitions(messages) == []


class TestComputeTransitionAccuracy:

    def test_perfect_match(self) -> None:
        acc, invalid = compute_transition_accuracy(
            [("A", "B"), ("B", "C")],
            [("A", "B"), ("B", "C")],
        )
        assert acc == 1.0
        assert invalid == 0

    def test_partial_match(self) -> None:
        acc, invalid = compute_transition_accuracy(
            [("A", "B"), ("B", "D")],
            [("A", "B"), ("B", "C")],
        )
        assert acc == 0.5

    def test_empty_ground_truth(self) -> None:
        acc, _ = compute_transition_accuracy([("A", "B")], [])
        assert acc == 0.0

    def test_both_empty(self) -> None:
        acc, _ = compute_transition_accuracy([], [])
        assert acc == 1.0

    def test_invalid_transitions(self) -> None:
        _, invalid = compute_transition_accuracy(
            [("X", "Y")],
            [("A", "B")],
        )
        assert invalid == 1


class TestCheckTaskCompletion:

    def test_reaches_terminal(self) -> None:
        assert check_task_completion([("A", "B"), ("B", "END")], ["END"])

    def test_does_not_reach_terminal(self) -> None:
        assert not check_task_completion([("A", "B")], ["END"])

    def test_empty_predictions(self) -> None:
        assert not check_task_completion([], ["END"])


class TestCheckRecovery:

    def test_successful_recovery(self) -> None:
        messages = [
            {"role": "tool", "content": '{"error": "timeout"}'},
            {"role": "assistant", "content": "[STATE: A → B] Retrying..."},
        ]
        recoveries, errors = check_recovery(messages)
        assert errors == 1
        assert recoveries == 1

    def test_failed_recovery(self) -> None:
        messages = [
            {"role": "tool", "content": '{"error": "timeout"}'},
            {"role": "assistant", "content": "I'm sorry, something went wrong."},
        ]
        recoveries, errors = check_recovery(messages)
        assert errors == 1
        assert recoveries == 0

    def test_no_errors(self) -> None:
        messages = [
            {"role": "tool", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": "[STATE: A → B] Done"},
        ]
        _, errors = check_recovery(messages)
        assert errors == 0

    def test_recovery_through_user_message(self) -> None:
        # A user message between the tool error and assistant recovery should not
        # break the recovery detection (prev_was_error must survive user turns).
        messages = [
            {"role": "tool", "content": '{"error": "timeout"}'},
            {"role": "user", "content": "Please try again"},
            {"role": "assistant", "content": "[STATE: A → B] Retrying..."},
        ]
        recoveries, errors = check_recovery(messages)
        assert errors == 1
        assert recoveries == 1


class TestPass5Consistency:

    def test_all_trials_pass(self) -> None:
        trial = [{"role": "assistant", "content": "[STATE: A → END]"}]
        trials = [trial] * 5  # 5 stochastic trials as per pass^5 protocol
        assert compute_pass5_consistency(trials, ["END"])

    def test_one_trial_fails(self) -> None:
        trials = [
            [{"role": "assistant", "content": "[STATE: A → END]"}],
            [{"role": "assistant", "content": "[STATE: A → B]"}],
        ]
        assert not compute_pass5_consistency(trials, ["END"])

    def test_empty_trials(self) -> None:
        assert not compute_pass5_consistency([], ["END"])


class TestEvaluateStateMachine:

    def test_full_evaluation(self) -> None:
        preds = [
            ConversationPrediction(
                conversation_id="c1",
                messages=[
                    {"role": "assistant", "content": "[STATE: GREETING → COLLECT]"},
                    {"role": "assistant", "content": "[STATE: COLLECT → DONE]"},
                ],
            ),
        ]
        gts = [
            ConversationGroundTruth(
                conversation_id="c1",
                messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "annotations": {"state_transition": {"from": "GREETING", "to": "COLLECT"}},
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "annotations": {"state_transition": {"from": "COLLECT", "to": "DONE"}},
                    },
                ],
                terminal_states=["DONE"],
            ),
        ]
        metrics = evaluate_state_machine(preds, gts)
        assert metrics.state_transition_accuracy == 1.0
        assert metrics.task_completion_rate == 1.0

    def test_empty_inputs(self) -> None:
        metrics = evaluate_state_machine([], [])
        assert metrics.state_transition_accuracy == 0.0


# ============================================================
# Tool Call F1 Tests
# ============================================================


class TestParseToolCalls:

    def test_standard_format(self) -> None:
        content = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == {"q": "test"}

    def test_hermes_format(self) -> None:
        content = '<|tool_call|>{"name": "fn", "arguments": {}}<|/tool_call|>'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "fn"

    def test_nested_function_format(self) -> None:
        content = '<tool_call>{"function": {"name": "lookup", "arguments": {"id": "42"}}}</tool_call>'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "lookup"
        assert calls[0]["arguments"] == {"id": "42"}

    def test_no_tool_calls(self) -> None:
        assert parse_tool_calls("Just a regular message") == []

    def test_multiple_calls(self) -> None:
        content = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call> text '
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = parse_tool_calls(content)
        assert len(calls) == 2

    def test_string_arguments_parsed(self) -> None:
        content = '<tool_call>{"name": "fn", "arguments": "{\\"x\\": 1}"}</tool_call>'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"x": 1}


class TestComputeNameAccuracy:

    def test_perfect(self) -> None:
        assert compute_name_accuracy(
            [{"name": "a"}, {"name": "b"}],
            [{"name": "a"}, {"name": "b"}],
        ) == 1.0

    def test_half(self) -> None:
        assert compute_name_accuracy(
            [{"name": "a"}, {"name": "x"}],
            [{"name": "a"}, {"name": "b"}],
        ) == 0.5

    def test_empty_gt(self) -> None:
        assert compute_name_accuracy([{"name": "a"}], []) == 0.0

    def test_both_empty(self) -> None:
        assert compute_name_accuracy([], []) == 1.0


class TestComputeArgumentMatch:

    def test_exact_match(self) -> None:
        assert compute_argument_match(
            {"arguments": {"x": 1, "y": "hello"}},
            {"arguments": {"x": 1, "y": "hello"}},
        )

    def test_mismatch(self) -> None:
        assert not compute_argument_match(
            {"arguments": {"x": 1}},
            {"arguments": {"x": 2}},
        )

    def test_nested(self) -> None:
        assert compute_argument_match(
            {"arguments": {"obj": {"a": [1, 2]}}},
            {"arguments": {"obj": {"a": [1, 2]}}},
        )


class TestComputeAstF1:

    def test_perfect_f1(self) -> None:
        pred = [{"name": "fn", "arguments": {"x": 1}}]
        gt = [{"name": "fn", "arguments": {"x": 1}}]
        assert compute_ast_f1(pred, gt) == 1.0

    def test_zero_f1_wrong_name(self) -> None:
        pred = [{"name": "wrong", "arguments": {"x": 1}}]
        gt = [{"name": "fn", "arguments": {"x": 1}}]
        assert compute_ast_f1(pred, gt) == 0.0

    def test_subtree_match(self) -> None:
        # Prediction has extra args — should still match
        pred = [{"name": "fn", "arguments": {"x": 1, "extra": 2}}]
        gt = [{"name": "fn", "arguments": {"x": 1}}]
        assert compute_ast_f1(pred, gt) == 1.0

    def test_both_empty(self) -> None:
        assert compute_ast_f1([], []) == 1.0

    def test_pred_empty_gt_not(self) -> None:
        assert compute_ast_f1([], [{"name": "fn", "arguments": {}}]) == 0.0


class TestDetectHallucinatedTools:

    def test_no_hallucination(self) -> None:
        result = detect_hallucinated_tools(
            [{"name": "search"}], ["search", "lookup"]
        )
        assert result == []

    def test_hallucinated(self) -> None:
        result = detect_hallucinated_tools(
            [{"name": "search"}, {"name": "fake_tool"}],
            ["search"],
        )
        assert result == ["fake_tool"]


class TestEvaluateToolCalls:

    def test_full_evaluation(self) -> None:
        preds = [
            TurnPrediction(
                turn_id=0,
                content='<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>',
            ),
        ]
        gts = [
            TurnGroundTruth(
                turn_id=0,
                tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
            ),
        ]
        metrics = evaluate_tool_calls(preds, gts)
        assert metrics.tool_name_accuracy == 1.0
        assert metrics.tool_call_f1 == 1.0
        assert metrics.argument_exact_match == 1.0

    def test_empty_inputs(self) -> None:
        metrics = evaluate_tool_calls([], [])
        assert metrics.tool_call_f1 == 0.0


# ============================================================
# Tool Chain Propagation Tests
# ============================================================


class TestCheckValuePropagation:

    def test_value_propagated(self) -> None:
        assert check_value_propagation(
            {"order_id": "123", "status": "active"},
            {"order_id": "123", "action": "cancel"},
        )

    def test_value_not_propagated(self) -> None:
        assert not check_value_propagation(
            {"order_id": "123"},
            {"customer": "john"},
        )

    def test_empty_response(self) -> None:
        assert not check_value_propagation({}, {"x": "y"})

    def test_nested_value(self) -> None:
        assert check_value_propagation(
            {"data": {"id": "abc"}},
            {"ref_id": "abc"},
        )


class TestExtractToolChains:

    def test_simple_chain(self) -> None:
        messages = [
            {"role": "assistant", "content": '<tool_call>{"name": "a", "arguments": {}}</tool_call>'},
            {"role": "tool", "content": '{"id": "123"}'},
            {"role": "assistant", "content": '<tool_call>{"name": "b", "arguments": {"id": "123"}}</tool_call>'},
            {"role": "tool", "content": '{"status": "ok"}'},
        ]
        links = extract_tool_chains(messages)
        assert len(links) == 2
        assert links[0].tool_name == "a"
        assert links[1].tool_name == "b"

    def test_no_tools(self) -> None:
        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ]
        assert extract_tool_chains(messages) == []


class TestEvaluateChainPropagation:

    def test_perfect_propagation(self) -> None:
        conv = {
            "messages": [
                {"role": "assistant", "content": '<tool_call>{"name": "lookup", "arguments": {"id": "1"}}</tool_call>'},
                {"role": "tool", "content": '{"order_id": "ORD-42"}'},
                {"role": "assistant", "content": '<tool_call>{"name": "cancel", "arguments": {"order_id": "ORD-42"}}</tool_call>'},
                {"role": "tool", "content": '{"status": "cancelled"}'},
            ]
        }
        metrics = evaluate_chain_propagation([conv], [conv])
        assert metrics.chain_propagation_accuracy == 1.0

    def test_empty(self) -> None:
        metrics = evaluate_chain_propagation([], [])
        assert metrics.chain_propagation_accuracy == 0.0


# ============================================================
# Agent Benchmark Tests
# ============================================================


class TestComputeWeightedScore:

    def test_perfect_score(self) -> None:
        state = StateMachineMetrics(state_transition_accuracy=1.0)
        tool = ToolCallMetrics(tool_call_f1=1.0)
        score = compute_weighted_score(state, tool, completion=1.0)
        assert score == pytest.approx(1.0)

    def test_zero_score(self) -> None:
        state = StateMachineMetrics()
        tool = ToolCallMetrics()
        assert compute_weighted_score(state, tool, completion=0.0) == 0.0

    def test_weighted_components(self) -> None:
        state = StateMachineMetrics(state_transition_accuracy=0.5)
        tool = ToolCallMetrics(tool_call_f1=0.5)
        score = compute_weighted_score(state, tool, completion=0.5)
        # 0.4*0.5 + 0.4*0.5 + 0.2*0.5 = 0.5
        assert score == pytest.approx(0.5)


class TestComputeLatencyMedian:

    def test_odd_count(self) -> None:
        assert compute_latency_median([100, 200, 300]) == 200

    def test_even_count(self) -> None:
        assert compute_latency_median([100, 200, 300, 400]) == 250

    def test_empty(self) -> None:
        assert compute_latency_median([]) == 0.0

    def test_single(self) -> None:
        assert compute_latency_median([42.0]) == 42.0


class TestComputeFullWorkflowSuccess:

    def test_no_chains_not_penalized(self) -> None:
        # total_chains=0 means workflow has no tool chains — should not penalize
        state = StateMachineMetrics(task_completion_rate=1.0)
        tool = ToolCallMetrics(tool_call_f1=1.0)
        chain = ChainPropagationMetrics(chain_propagation_accuracy=0.0, total_chains=0)
        result = compute_full_workflow_success(state, tool, chain)
        assert result == pytest.approx(1.0)

    def test_zero_accuracy_with_chains_penalized(self) -> None:
        # chain_propagation_accuracy=0.0 with actual chains should yield 0.0 chain_factor
        state = StateMachineMetrics(task_completion_rate=1.0)
        tool = ToolCallMetrics(tool_call_f1=1.0)
        chain = ChainPropagationMetrics(chain_propagation_accuracy=0.0, total_chains=5)
        result = compute_full_workflow_success(state, tool, chain)
        assert result == pytest.approx(0.0)

    def test_partial_chain_accuracy(self) -> None:
        state = StateMachineMetrics(task_completion_rate=1.0)
        tool = ToolCallMetrics(tool_call_f1=1.0)
        chain = ChainPropagationMetrics(chain_propagation_accuracy=0.7, total_chains=3)
        result = compute_full_workflow_success(state, tool, chain)
        assert result == pytest.approx(1.0)  # 0.7/0.7 = 1.0 factor


class TestEvaluateWorkflowQuality:

    def test_composite_evaluation(self) -> None:
        state = StateMachineMetrics(
            state_transition_accuracy=0.9,
            task_completion_rate=0.8,
        )
        tool = ToolCallMetrics(tool_call_f1=0.85)
        chain = ChainPropagationMetrics(chain_propagation_accuracy=0.75, total_chains=4)

        metrics = evaluate_workflow_quality(
            state, tool, chain, latencies_ms=[100, 200, 150]
        )
        assert isinstance(metrics, WorkflowQualityMetrics)
        assert metrics.weighted_workflow_score > 0
        assert metrics.latency_per_turn_median_ms == 150
        assert metrics.state_metrics is state
        assert metrics.tool_metrics is tool


# --- Composite Score Tests ---


class TestCompositeScore:

    def test_weighted_workflow_score_formula(self) -> None:
        state = StateMachineMetrics(
            state_transition_accuracy=1.0,
            task_completion_rate=1.0,
        )
        tool = ToolCallMetrics(tool_call_f1=1.0)
        score = compute_weighted_workflow_score(state, tool)
        assert abs(score - 1.0) < 1e-9

    def test_weighted_workflow_score_zero(self) -> None:
        state = StateMachineMetrics()
        tool = ToolCallMetrics()
        score = compute_weighted_workflow_score(state, tool)
        assert score == 0.0

    def test_weighted_workflow_score_partial(self) -> None:
        state = StateMachineMetrics(
            state_transition_accuracy=0.8,
            task_completion_rate=0.6,
        )
        tool = ToolCallMetrics(tool_call_f1=0.9)
        score = compute_weighted_workflow_score(state, tool)
        expected = 0.4 * 0.8 + 0.4 * 0.9 + 0.2 * 0.6
        assert abs(score - expected) < 1e-9

    def test_full_workflow_success_rate_empty(self) -> None:
        assert full_workflow_success_rate([], []) == 0.0

    def test_full_workflow_success_rate_perfect(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "[STATE: INIT → DONE]"},
        ]
        pred = [ConversationPrediction(
            conversation_id="c1",
            messages=msgs,
        )]
        gt = [ConversationGroundTruth(
            conversation_id="c1",
            messages=msgs,
            terminal_states=["DONE"],
        )]
        rate = full_workflow_success_rate(pred, gt)
        assert rate == 1.0

    def test_full_workflow_success_rate_wrong_transition(self) -> None:
        pred_msgs = [
            {"role": "assistant", "content": "[STATE: INIT → WRONG]"},
        ]
        gt_msgs = [
            {"role": "assistant", "content": "[STATE: INIT → DONE]"},
        ]
        pred = [ConversationPrediction(conversation_id="c1", messages=pred_msgs)]
        gt = [ConversationGroundTruth(
            conversation_id="c1", messages=gt_msgs, terminal_states=["DONE"]
        )]
        rate = full_workflow_success_rate(pred, gt)
        assert rate == 0.0

    def test_composite_result_to_dict(self) -> None:
        r = CompositeResult(
            model_name="test", category="A",
            weighted_workflow_score=0.8,
            full_workflow_success_rate=0.6,
            num_conversations=100,
        )
        d = r.to_dict()
        assert d["model_name"] == "test"
        assert d["weighted_workflow_score"] == 0.8


# --- Quant Benchmark Tests ---


class TestQuantBenchmark:

    def test_compute_mean_std_basic(self) -> None:
        mean, std = _compute_mean_std([10.0, 10.0, 10.0])
        assert mean == 10.0
        assert std == 0.0

    def test_compute_mean_std_empty(self) -> None:
        mean, std = _compute_mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_compute_mean_std_single(self) -> None:
        mean, std = _compute_mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_compute_mean_std_nonzero(self) -> None:
        mean, std = _compute_mean_std([1.0, 3.0])
        assert abs(mean - 2.0) < 1e-9
        assert std > 0

    def test_aggregate_quality(self) -> None:
        runs = [
            RunResult(run_id=0, quality=QualityMetrics(wikitext2_ppl=10.0, c4_ppl=20.0)),
            RunResult(run_id=1, quality=QualityMetrics(wikitext2_ppl=12.0, c4_ppl=22.0)),
        ]
        mean, std = _aggregate_quality(runs)
        assert abs(mean.wikitext2_ppl - 11.0) < 1e-9
        assert abs(mean.c4_ppl - 21.0) < 1e-9
        assert std.wikitext2_ppl > 0

    def test_aggregate_quality_empty(self) -> None:
        mean, std = _aggregate_quality([])
        assert mean.wikitext2_ppl == 0.0

    def test_matrix_get_cell(self) -> None:
        matrix = QuantBenchmarkMatrix(
            models=["m1"], methods=["fp8"],
            results={"m1::fp8": CellResult(model="m1", method="fp8")},
        )
        cell = matrix.get_cell("m1", "fp8")
        assert cell is not None
        assert cell.model == "m1"
        assert matrix.get_cell("m1", "kivi") is None

    def test_matrix_to_dict(self) -> None:
        matrix = QuantBenchmarkMatrix(
            models=["m1"], methods=["fp8"],
            results={"m1::fp8": CellResult(model="m1", method="fp8")},
            total_runs=5,
        )
        d = matrix.to_dict()
        assert d["models"] == ["m1"]
        assert d["total_runs"] == 5
        assert "m1::fp8" in d["results"]

    def test_quality_metrics_to_dict(self) -> None:
        q = QualityMetrics(wikitext2_ppl=5.5, c4_ppl=6.6)
        d = q.to_dict()
        assert d["wikitext2_ppl"] == 5.5
        assert d["c4_ppl"] == 6.6
