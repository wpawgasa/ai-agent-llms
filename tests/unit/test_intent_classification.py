"""Tests for intent classification benchmark."""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.intent_classification import (
    IntentClassificationMetrics,
    IntentSample,
    _classify_intent_to_routing,
    _compute_prf1,
    evaluate_intent_classification,
    generate_intent_samples,
)


class TestIntentSampleGeneration:

    def test_generates_correct_count(self) -> None:
        samples = generate_intent_samples(num_samples=50, seed=42)
        assert len(samples) == 50

    def test_all_samples_have_required_fields(self) -> None:
        samples = generate_intent_samples(num_samples=20, seed=42)
        for s in samples:
            assert s.text, "text should not be empty"
            assert s.domain, "domain should not be empty"
            assert s.intent, "intent should not be empty"
            assert s.routing_target in ("tool_execution", "graph_extraction", "self_handle")

    def test_routing_distribution_approximate(self) -> None:
        samples = generate_intent_samples(num_samples=500, seed=42)
        counts = {"tool_execution": 0, "graph_extraction": 0, "self_handle": 0}
        for s in samples:
            counts[s.routing_target] += 1
        # Default: 60% tool, 20% graph, 20% self
        assert counts["tool_execution"] > counts["graph_extraction"]
        assert counts["tool_execution"] > counts["self_handle"]
        assert counts["tool_execution"] / 500 > 0.45  # At least 45%

    def test_seed_determinism(self) -> None:
        s1 = generate_intent_samples(num_samples=10, seed=42)
        s2 = generate_intent_samples(num_samples=10, seed=42)
        for a, b in zip(s1, s2):
            assert a.text == b.text
            assert a.routing_target == b.routing_target

    def test_custom_routing_distribution(self) -> None:
        samples = generate_intent_samples(
            num_samples=100,
            seed=42,
            routing_distribution={"tool_execution": 0.0, "graph_extraction": 1.0, "self_handle": 0.0},
        )
        assert all(s.routing_target == "graph_extraction" for s in samples)

    def test_multiple_domains_covered(self) -> None:
        samples = generate_intent_samples(num_samples=200, seed=42)
        domains = {s.domain for s in samples if s.domain != "general"}
        assert len(domains) > 5, f"Expected diverse domains, got {domains}"


class TestIntentToRoutingMapping:

    def test_tool_execution_default(self) -> None:
        assert _classify_intent_to_routing("balance_inquiry") == "tool_execution"
        assert _classify_intent_to_routing("order_tracking") == "tool_execution"

    def test_graph_extraction(self) -> None:
        assert _classify_intent_to_routing("workflow_visualization") == "graph_extraction"
        assert _classify_intent_to_routing("process_mapping") == "graph_extraction"

    def test_self_handle(self) -> None:
        assert _classify_intent_to_routing("greeting") == "self_handle"
        assert _classify_intent_to_routing("closing") == "self_handle"
        assert _classify_intent_to_routing("out_of_scope") == "self_handle"


class TestPRF1Computation:

    def test_perfect_classification(self) -> None:
        preds = ["a", "b", "c", "a", "b"]
        gts = ["a", "b", "c", "a", "b"]
        precision, recall, f1, confusion = _compute_prf1(preds, gts, ["a", "b", "c"])
        assert precision["a"] == 1.0
        assert recall["a"] == 1.0
        assert f1["a"] == 1.0

    def test_all_wrong(self) -> None:
        preds = ["b", "c", "a"]
        gts = ["a", "b", "c"]
        precision, recall, f1, confusion = _compute_prf1(preds, gts, ["a", "b", "c"])
        for cls in ["a", "b", "c"]:
            assert f1[cls] == 0.0

    def test_confusion_matrix_counts(self) -> None:
        preds = ["a", "b", "a", "a"]
        gts = ["a", "a", "b", "a"]
        _, _, _, confusion = _compute_prf1(preds, gts, ["a", "b"])
        assert confusion["a"]["a"] == 2  # True positives for 'a'
        assert confusion["a"]["b"] == 1  # 'a' misclassified as 'b'
        assert confusion["b"]["a"] == 1  # 'b' misclassified as 'a'


class TestEvaluateIntentClassification:

    def test_empty_inputs(self) -> None:
        metrics = evaluate_intent_classification([], [])
        assert metrics.num_samples == 0
        assert metrics.routing_accuracy == 0.0

    def test_perfect_predictions(self) -> None:
        gt = [
            IntentSample(text="t1", domain="banking", intent="balance", routing_target="tool_execution"),
            IntentSample(text="t2", domain="general", intent="greeting", routing_target="self_handle"),
            IntentSample(text="t3", domain="healthcare", intent="graph_extraction", routing_target="graph_extraction"),
        ]
        preds = [
            {"routing_target": "tool_execution", "domain": "banking"},
            {"routing_target": "self_handle", "domain": "general"},
            {"routing_target": "graph_extraction", "domain": "healthcare"},
        ]
        metrics = evaluate_intent_classification(preds, gt)
        assert metrics.routing_accuracy == 1.0
        assert metrics.domain_accuracy == 1.0
        assert metrics.num_samples == 3

    def test_partial_predictions(self) -> None:
        gt = [
            IntentSample(text="t1", domain="banking", intent="balance", routing_target="tool_execution"),
            IntentSample(text="t2", domain="telecom", intent="plan_change", routing_target="tool_execution"),
        ]
        preds = [
            {"routing_target": "tool_execution", "domain": "banking"},
            {"routing_target": "self_handle", "domain": "telecom"},  # Wrong routing
        ]
        metrics = evaluate_intent_classification(preds, gt)
        assert metrics.routing_accuracy == 0.5
        assert metrics.domain_accuracy == 1.0  # Domain was correct

    def test_to_dict(self) -> None:
        metrics = IntentClassificationMetrics(
            routing_accuracy=0.85,
            domain_accuracy=0.72,
            num_samples=100,
        )
        d = metrics.to_dict()
        assert d["routing_accuracy"] == 0.85
        assert d["domain_accuracy"] == 0.72
        assert d["num_samples"] == 100

    def test_top3_domain_accuracy(self) -> None:
        gt = [
            IntentSample(text="t1", domain="banking", intent="balance", routing_target="tool_execution"),
        ]
        preds = [
            {"routing_target": "tool_execution", "domain": "telecom", "domain_top3": ["telecom", "banking", "healthcare"]},
        ]
        metrics = evaluate_intent_classification(preds, gt)
        assert metrics.domain_accuracy == 0.0  # Exact domain wrong
        assert metrics.domain_top3_accuracy == 1.0  # But in top-3


class TestModuleImport:

    def test_import_from_eval(self) -> None:
        from llm_workflow_agents.eval import (
            IntentClassificationMetrics,
            evaluate_intent_classification,
            generate_intent_samples,
            run_intent_benchmark,
        )
        assert callable(generate_intent_samples)
        assert callable(evaluate_intent_classification)
        assert callable(run_intent_benchmark)
