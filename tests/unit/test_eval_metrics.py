"""Tests for evaluation metrics — known-answer tests for F1, state accuracy, chain propagation."""

from __future__ import annotations

import pytest

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


class TestPass5Consistency:

    def test_all_trials_pass(self) -> None:
        trials = [
            [{"role": "assistant", "content": "[STATE: A → END]"}],
            [{"role": "assistant", "content": "[STATE: A → END]"}],
        ]
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


class TestEvaluateWorkflowQuality:

    def test_composite_evaluation(self) -> None:
        state = StateMachineMetrics(
            state_transition_accuracy=0.9,
            task_completion_rate=0.8,
        )
        tool = ToolCallMetrics(tool_call_f1=0.85)
        chain = ChainPropagationMetrics(chain_propagation_accuracy=0.75)

        metrics = evaluate_workflow_quality(
            state, tool, chain, latencies_ms=[100, 200, 150]
        )
        assert isinstance(metrics, WorkflowQualityMetrics)
        assert metrics.weighted_workflow_score > 0
        assert metrics.latency_per_turn_median_ms == 150
        assert metrics.state_metrics is state
        assert metrics.tool_metrics is tool
