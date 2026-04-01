"""Tests for Phase 1 benchmark module — model selection, composite scoring, latency profiling."""

from __future__ import annotations

import pytest

from llm_workflow_agents.benchmark.latency_profiler import (
    LatencyProfile,
    PercentileStats,
    _compute_percentiles,
)
from llm_workflow_agents.benchmark.model_selector import (
    CompositeScore,
    _normalize,
    compute_composite_scores,
    select_winners,
)
from llm_workflow_agents.benchmark.results_aggregator import (
    Phase1Results,
    RankingTable,
    aggregate_results,
    save_results,
)
from llm_workflow_agents.benchmark.task_runner import TaskResult


# --- Latency Profiler Tests ---


class TestPercentileStats:

    def test_defaults(self) -> None:
        ps = PercentileStats()
        assert ps.p50 == 0.0
        assert ps.p95 == 0.0
        assert ps.p99 == 0.0

    def test_to_dict(self) -> None:
        ps = PercentileStats(p50=10.0, p95=20.0, p99=30.0)
        d = ps.to_dict()
        assert d == {"p50": 10.0, "p95": 20.0, "p99": 30.0}


class TestComputePercentiles:

    def test_empty(self) -> None:
        ps = _compute_percentiles([])
        assert ps.p50 == 0.0

    def test_single(self) -> None:
        ps = _compute_percentiles([42.0])
        assert ps.p50 == 42.0
        assert ps.p95 == 42.0

    def test_basic(self) -> None:
        values = list(range(1, 101))  # 1..100
        ps = _compute_percentiles(values)
        assert ps.p50 == 50.0
        assert ps.p95 >= 95.0


class TestLatencyProfile:

    def test_defaults(self) -> None:
        lp = LatencyProfile()
        assert lp.throughput_prefill_tok_s == 0.0
        assert lp.peak_vram_gb == 0.0

    def test_to_dict(self) -> None:
        lp = LatencyProfile(throughput_decode_tok_s=1000.0)
        d = lp.to_dict()
        assert d["throughput_decode_tok_s"] == 1000.0
        assert "ttft_ms" in d


# --- Model Selector Tests ---


class TestNormalize:

    def test_empty(self) -> None:
        assert _normalize([]) == []

    def test_identical(self) -> None:
        assert _normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]

    def test_ascending(self) -> None:
        result = _normalize([0.0, 50.0, 100.0])
        assert result == [0.0, 0.5, 1.0]

    def test_inverted(self) -> None:
        result = _normalize([0.0, 50.0, 100.0], higher_is_better=False)
        assert result == [1.0, 0.5, 0.0]


class TestCompositeScore:

    def test_to_dict(self) -> None:
        cs = CompositeScore(model_name="m1", category="A", weighted_composite=0.8)
        d = cs.to_dict()
        assert d["model_name"] == "m1"
        assert d["weighted_composite"] == 0.8

    def test_compute_composite_scores_empty(self) -> None:
        assert compute_composite_scores([]) == []

    def test_compute_composite_scores_sorted(self) -> None:
        results = [
            TaskResult(model_name="low", quality_score=0.3,
                       latency=LatencyProfile(ttft_ms=PercentileStats(p95=100.0), throughput_decode_tok_s=50.0, peak_vram_gb=20.0)),
            TaskResult(model_name="high", quality_score=0.9,
                       latency=LatencyProfile(ttft_ms=PercentileStats(p95=50.0), throughput_decode_tok_s=100.0, peak_vram_gb=10.0)),
        ]
        scores = compute_composite_scores(results, category="A")
        assert len(scores) == 2
        assert scores[0].model_name == "high"
        assert scores[0].weighted_composite > scores[1].weighted_composite


class TestSelectWinners:

    def test_basic_selection(self) -> None:
        scores = {
            "A": [CompositeScore(model_name="winner_a", category="A", weighted_composite=0.9)],
            "B": [CompositeScore(model_name="winner_b", category="B", weighted_composite=0.8)],
        }
        winners = select_winners(scores)
        assert winners["A"] == "winner_a"
        assert winners["B"] == "winner_b"

    def test_empty_category(self) -> None:
        winners = select_winners({"A": []})
        assert "A" not in winners

    def test_same_bc_winner_logged(self) -> None:
        scores = {
            "B": [CompositeScore(model_name="shared", category="B", weighted_composite=0.9)],
            "C": [CompositeScore(model_name="shared", category="C", weighted_composite=0.8)],
        }
        winners = select_winners(scores)
        assert winners["B"] == "shared"
        assert winners["C"] == "shared"


# --- Task Runner Tests ---


class TestTaskResult:

    def test_defaults(self) -> None:
        r = TaskResult()
        assert r.model_name == ""
        assert r.quality_score == 0.0
        assert r.error is None

    def test_to_dict(self) -> None:
        r = TaskResult(model_name="m1", task="task_a", quality_score=0.85)
        d = r.to_dict()
        assert d["model_name"] == "m1"
        assert d["quality_score"] == 0.85


# --- Results Aggregator Tests ---


class TestResultsAggregator:

    def test_aggregate_results(self) -> None:
        task_results = [TaskResult(model_name="m1", task="task_a")]
        scores = {"A": [CompositeScore(model_name="m1", category="A", weighted_composite=0.9)]}
        winners = {"A": "m1"}

        results = aggregate_results(task_results, scores, winners)
        assert isinstance(results, Phase1Results)
        assert results.winners == {"A": "m1"}
        assert "A" in results.ranking_tables

    def test_phase1_results_to_dict(self) -> None:
        results = Phase1Results(winners={"A": "m1"})
        d = results.to_dict()
        assert d["winners"] == {"A": "m1"}
        assert d["num_evaluations"] == 0

    def test_save_results(self, tmp_path) -> None:
        results = Phase1Results(winners={"A": "model1"})
        path = save_results(results, tmp_path / "results.json")
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["winners"]["A"] == "model1"


# --- Module Import Tests ---


class TestBenchmarkImports:

    def test_import_benchmark(self) -> None:
        from llm_workflow_agents import benchmark
        assert hasattr(benchmark, "Phase1Orchestrator")
        assert hasattr(benchmark, "CompositeScore")
        assert hasattr(benchmark, "LatencyProfile")
        assert hasattr(benchmark, "run_task")

    def test_import_integration(self) -> None:
        from llm_workflow_agents import integration
        assert hasattr(integration, "MultiAgentOrchestrator")
        assert hasattr(integration, "benchmark_concurrency")
        assert hasattr(integration, "find_pareto_frontier")

    def test_import_analysis_plots(self) -> None:
        from llm_workflow_agents.analysis import plot_phase1_rankings
        from llm_workflow_agents.analysis import plot_sft_vs_rl
        from llm_workflow_agents.analysis import plot_quant_matrix
        from llm_workflow_agents.analysis import plot_pareto
        assert hasattr(plot_phase1_rankings, "plot_phase1_rankings")
        assert hasattr(plot_sft_vs_rl, "plot_sft_vs_rl")
        assert hasattr(plot_quant_matrix, "plot_quant_matrix")
        assert hasattr(plot_pareto, "plot_pareto_projections")
