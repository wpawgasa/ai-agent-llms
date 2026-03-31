"""Tests for Phase 5 benchmark evaluation modules — perplexity, LongBench, needle-in-haystack."""

from __future__ import annotations

import math

import pytest

from llm_workflow_agents.eval.perplexity import (
    PerplexityResult,
    compute_perplexity_from_losses,
)
from llm_workflow_agents.eval.longbench import (
    ALL_TASK_NAMES,
    LONGBENCH_TASKS,
    LongBenchResult,
    LongBenchTaskResult,
    _compute_f1,
    _compute_rouge_l,
    _get_task_category,
    score_task,
)
from llm_workflow_agents.eval.needle_haystack import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_DEPTH_POSITIONS,
    NeedleHaystackResult,
    NeedleResult,
    _build_haystack,
    _compute_aggregated_accuracy,
    check_needle_found,
)


# ============================================================
# Perplexity Tests
# ============================================================


class TestComputePerplexityFromLosses:

    def test_known_value(self) -> None:
        # exp(1.0) ≈ 2.718
        result = compute_perplexity_from_losses([1.0, 1.0, 1.0])
        assert result == pytest.approx(math.e, abs=0.01)

    def test_zero_loss(self) -> None:
        # exp(0) = 1.0 — perfect model
        assert compute_perplexity_from_losses([0.0]) == pytest.approx(1.0)

    def test_empty_losses(self) -> None:
        assert compute_perplexity_from_losses([]) == float("inf")

    def test_mixed_losses(self) -> None:
        # exp((2.0 + 0.0) / 2) = exp(1.0) ≈ 2.718
        result = compute_perplexity_from_losses([2.0, 0.0])
        assert result == pytest.approx(math.e, abs=0.01)

    def test_high_loss(self) -> None:
        # exp(10.0) ≈ 22026
        result = compute_perplexity_from_losses([10.0])
        assert result > 20000


class TestPerplexityResult:

    def test_to_dict(self) -> None:
        r = PerplexityResult(
            dataset="wikitext2",
            perplexity=15.3,
            avg_neg_log_likelihood=2.73,
            num_tokens=10000,
            num_sequences=50,
        )
        d = r.to_dict()
        assert d["dataset"] == "wikitext2"
        assert d["perplexity"] == 15.3
        assert d["num_tokens"] == 10000


# ============================================================
# LongBench Tests
# ============================================================


class TestLongBenchTasks:

    def test_all_15_tasks(self) -> None:
        assert len(ALL_TASK_NAMES) == 15

    def test_task_categories(self) -> None:
        assert len(LONGBENCH_TASKS) == 6
        assert "single_doc_qa" in LONGBENCH_TASKS
        assert "code" in LONGBENCH_TASKS

    def test_get_task_category(self) -> None:
        assert _get_task_category("narrativeqa") == "single_doc_qa"
        assert _get_task_category("hotpotqa") == "multi_doc_qa"
        assert _get_task_category("gov_report") == "summarization"
        assert _get_task_category("lcc") == "code"
        assert _get_task_category("unknown_task") == "unknown"


class TestComputeF1:

    def test_perfect_match(self) -> None:
        assert _compute_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        f1 = _compute_f1("the cat", "the cat sat")
        # precision = 2/2, recall = 2/3, F1 = 2*(1*2/3)/(1+2/3) = 0.8
        assert f1 == pytest.approx(0.8, abs=0.01)

    def test_no_match(self) -> None:
        assert _compute_f1("hello world", "foo bar") == 0.0

    def test_empty_prediction(self) -> None:
        assert _compute_f1("", "some reference") == 0.0

    def test_empty_reference(self) -> None:
        assert _compute_f1("some prediction", "") == 0.0


class TestComputeRougeL:

    def test_identical(self) -> None:
        assert _compute_rouge_l("a b c d", "a b c d") == pytest.approx(1.0)

    def test_partial_subsequence(self) -> None:
        # LCS of "a b c" and "a c" is "a c" (length 2)
        # precision = 2/3, recall = 2/2, F = 2*(2/3*1)/(2/3+1) = 0.8
        result = _compute_rouge_l("a b c", "a c")
        assert result == pytest.approx(0.8, abs=0.01)

    def test_no_common(self) -> None:
        assert _compute_rouge_l("a b c", "x y z") == 0.0

    def test_empty(self) -> None:
        assert _compute_rouge_l("", "abc") == 0.0


class TestScoreTask:

    def test_qa_task_scoring(self) -> None:
        score = score_task(
            "narrativeqa",
            predictions=["the answer is 42"],
            references=["the answer is 42"],
        )
        assert score == pytest.approx(100.0)

    def test_summarization_scoring(self) -> None:
        score = score_task(
            "gov_report",
            predictions=["the government report discusses policy"],
            references=["the government report discusses policy"],
        )
        assert score == pytest.approx(100.0)

    def test_empty_predictions(self) -> None:
        assert score_task("narrativeqa", [], []) == 0.0

    def test_zero_overlap(self) -> None:
        score = score_task(
            "narrativeqa",
            predictions=["xyz abc"],
            references=["hello world foo"],
        )
        assert score == 0.0


class TestLongBenchResult:

    def test_to_dict(self) -> None:
        result = LongBenchResult(
            task_results=[
                LongBenchTaskResult("narrativeqa", "single_doc_qa", 75.0, 100),
                LongBenchTaskResult("hotpotqa", "multi_doc_qa", 65.0, 100),
            ],
            overall_score=70.0,
            category_scores={"single_doc_qa": 75.0, "multi_doc_qa": 65.0},
        )
        d = result.to_dict()
        assert d["overall_score"] == 70.0
        assert "narrativeqa" in d["tasks"]


# ============================================================
# Needle-in-a-Haystack Tests
# ============================================================


class TestCheckNeedleFound:

    def test_found(self) -> None:
        assert check_needle_found("The answer is 12345.", "12345")

    def test_not_found(self) -> None:
        assert not check_needle_found("I don't know the answer.", "12345")

    def test_empty_response(self) -> None:
        assert not check_needle_found("", "12345")


class TestBuildHaystack:

    def test_contains_needle(self) -> None:
        needle = "The magic number is 42."
        haystack = _build_haystack(1024, needle, 0.5)
        assert "magic number is 42" in haystack

    def test_depth_zero_near_start(self) -> None:
        needle = "UNIQUE_NEEDLE"
        haystack = _build_haystack(512, needle, 0.0)
        # Needle should be near the beginning
        pos = haystack.find("UNIQUE_NEEDLE")
        assert pos >= 0
        assert pos < len(haystack) // 4  # First quarter

    def test_depth_one_near_end(self) -> None:
        needle = "UNIQUE_NEEDLE"
        haystack = _build_haystack(512, needle, 1.0)
        pos = haystack.find("UNIQUE_NEEDLE")
        assert pos >= 0
        assert pos > len(haystack) // 2  # Second half

    def test_approximate_length(self) -> None:
        haystack = _build_haystack(1024, "needle text", 0.5)
        # Should be approximately 1024 * 4 = 4096 chars
        assert 3000 < len(haystack) < 5000


class TestComputeAggregatedAccuracy:

    def test_perfect_accuracy(self) -> None:
        result = NeedleHaystackResult(
            probes=[
                NeedleResult(2048, 0.0, True),
                NeedleResult(2048, 0.5, True),
                NeedleResult(4096, 0.0, True),
                NeedleResult(4096, 0.5, True),
            ]
        )
        _compute_aggregated_accuracy(result)
        assert result.overall_accuracy == 1.0
        assert result.accuracy_by_length[2048] == 1.0
        assert result.accuracy_by_length[4096] == 1.0

    def test_partial_accuracy(self) -> None:
        result = NeedleHaystackResult(
            probes=[
                NeedleResult(2048, 0.0, True),
                NeedleResult(2048, 0.5, False),
                NeedleResult(4096, 0.0, False),
                NeedleResult(4096, 0.5, False),
            ]
        )
        _compute_aggregated_accuracy(result)
        assert result.overall_accuracy == 0.25
        assert result.accuracy_by_length[2048] == 0.5
        assert result.accuracy_by_length[4096] == 0.0

    def test_accuracy_by_depth(self) -> None:
        result = NeedleHaystackResult(
            probes=[
                NeedleResult(2048, 0.0, True),
                NeedleResult(4096, 0.0, True),
                NeedleResult(2048, 1.0, False),
                NeedleResult(4096, 1.0, False),
            ]
        )
        _compute_aggregated_accuracy(result)
        assert result.accuracy_by_depth[0.0] == 1.0
        assert result.accuracy_by_depth[1.0] == 0.0

    def test_empty_probes(self) -> None:
        result = NeedleHaystackResult()
        _compute_aggregated_accuracy(result)
        assert result.overall_accuracy == 0.0


class TestNeedleHaystackResult:

    def test_to_dict(self) -> None:
        result = NeedleHaystackResult(
            probes=[NeedleResult(2048, 0.5, True)],
            accuracy_by_length={2048: 1.0},
            accuracy_by_depth={0.5: 1.0},
            overall_accuracy=1.0,
        )
        d = result.to_dict()
        assert d["overall_accuracy"] == 1.0
        assert d["total_probes"] == 1


class TestDefaultConstants:

    def test_default_context_lengths(self) -> None:
        assert DEFAULT_CONTEXT_LENGTHS == [2048, 4096, 8192, 16384, 32768]

    def test_default_depth_positions(self) -> None:
        assert DEFAULT_DEPTH_POSITIONS == [0.0, 0.25, 0.5, 0.75, 1.0]


# ============================================================
# Module Import Tests
# ============================================================


class TestModuleImports:

    def test_import_perplexity(self) -> None:
        from llm_workflow_agents.eval import PerplexityResult, compute_perplexity_from_losses
        assert PerplexityResult is not None
        assert compute_perplexity_from_losses is not None

    def test_import_longbench(self) -> None:
        from llm_workflow_agents.eval import LongBenchResult, score_task
        assert LongBenchResult is not None
        assert score_task is not None

    def test_import_needle(self) -> None:
        from llm_workflow_agents.eval import NeedleHaystackResult, check_needle_found
        assert NeedleHaystackResult is not None
        assert check_needle_found is not None
