"""Integration tests for Phase 7: serving orchestrator, benchmark, and analysis modules."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_workflow_agents.analysis.pareto import (
    ParetoPoint,
    _dominates,
    find_pareto_frontier,
)
from llm_workflow_agents.serving.benchmark_e2e import (
    BenchmarkResult,
    compute_pareto_frontier,
)
from llm_workflow_agents.serving.orchestrator import (
    MultiAgentOrchestrator,
    WorkflowResult,
)


# ============================================================
# ParetoPoint Tests
# ============================================================


class TestParetoPoint:

    def test_to_dict(self) -> None:
        p = ParetoPoint(config_name="cfg_a", metrics={"quality": 0.9, "latency": 100.0})
        d = p.to_dict()
        assert d["config_name"] == "cfg_a"
        assert d["metrics"]["quality"] == 0.9
        assert d["metrics"]["latency"] == 100.0

    def test_default_metrics(self) -> None:
        p = ParetoPoint(config_name="cfg_b")
        assert p.metrics == {}
        assert p.to_dict()["metrics"] == {}

    def test_metrics_stored(self) -> None:
        metrics = {"a": 1.0, "b": 2.0, "c": 3.0}
        p = ParetoPoint(config_name="x", metrics=metrics)
        assert p.metrics == metrics


# ============================================================
# Dominance Helper Tests
# ============================================================


class TestDominates:

    def test_strictly_better_on_maximize(self) -> None:
        a = ParetoPoint("a", {"quality": 0.9})
        b = ParetoPoint("b", {"quality": 0.8})
        assert _dominates(a, b, maximize=["quality"], minimize=[])

    def test_strictly_better_on_minimize(self) -> None:
        a = ParetoPoint("a", {"latency": 100.0})
        b = ParetoPoint("b", {"latency": 200.0})
        assert _dominates(a, b, maximize=[], minimize=["latency"])

    def test_not_dominated_equal(self) -> None:
        a = ParetoPoint("a", {"quality": 0.8})
        b = ParetoPoint("b", {"quality": 0.8})
        # Equal: no strict improvement, so neither dominates the other
        assert not _dominates(a, b, maximize=["quality"], minimize=[])

    def test_worse_on_one_axis(self) -> None:
        a = ParetoPoint("a", {"quality": 0.9, "latency": 300.0})
        b = ParetoPoint("b", {"quality": 0.8, "latency": 100.0})
        # a is better on quality, b is better on latency → neither dominates
        assert not _dominates(a, b, maximize=["quality"], minimize=["latency"])
        assert not _dominates(b, a, maximize=["quality"], minimize=["latency"])


# ============================================================
# Pareto Frontier Tests
# ============================================================


class TestFindParetoFrontier:

    def test_empty_input(self) -> None:
        assert find_pareto_frontier([], maximize=["q"], minimize=[]) == []

    def test_single_point(self) -> None:
        p = ParetoPoint("a", {"q": 0.9})
        result = find_pareto_frontier([p], maximize=["q"], minimize=[])
        assert len(result) == 1
        assert result[0].config_name == "a"

    def test_two_non_dominated(self) -> None:
        a = ParetoPoint("a", {"quality": 0.9, "latency": 200.0})
        b = ParetoPoint("b", {"quality": 0.7, "latency": 50.0})
        # a better on quality, b better on latency — neither dominates
        result = find_pareto_frontier([a, b], maximize=["quality"], minimize=["latency"])
        names = {p.config_name for p in result}
        assert names == {"a", "b"}

    def test_dominated_point_removed(self) -> None:
        a = ParetoPoint("a", {"quality": 0.9, "latency": 100.0})
        b = ParetoPoint("b", {"quality": 0.8, "latency": 200.0})
        # a is better on both axes → a dominates b
        result = find_pareto_frontier([a, b], maximize=["quality"], minimize=["latency"])
        assert len(result) == 1
        assert result[0].config_name == "a"

    def test_all_equal_all_returned(self) -> None:
        points = [ParetoPoint(f"p{i}", {"q": 0.5}) for i in range(3)]
        result = find_pareto_frontier(points, maximize=["q"], minimize=[])
        # No strict improvement → none dominates any other → all returned
        assert len(result) == 3

    def test_three_points_mixed(self) -> None:
        a = ParetoPoint("a", {"quality": 0.9, "latency": 300.0})
        b = ParetoPoint("b", {"quality": 0.7, "latency": 100.0})
        c = ParetoPoint("c", {"quality": 0.6, "latency": 200.0})
        # c is dominated by both a (q) and b (lat), a and b are incomparable
        result = find_pareto_frontier([a, b, c], maximize=["quality"], minimize=["latency"])
        names = {p.config_name for p in result}
        assert names == {"a", "b"}
        assert "c" not in names

    def test_high_dimensional(self) -> None:
        a = ParetoPoint("a", {"q": 0.9, "v": 20.0, "l": 100.0, "t": 50.0})
        b = ParetoPoint("b", {"q": 0.8, "v": 30.0, "l": 200.0, "t": 60.0})
        # a dominates b on all 4 axes
        result = find_pareto_frontier(
            [a, b], maximize=["q"], minimize=["v", "l", "t"]
        )
        assert len(result) == 1
        assert result[0].config_name == "a"

    def test_maximize_and_minimize_mixed(self) -> None:
        # a: high quality, high vram — b: low quality, low vram
        a = ParetoPoint("a", {"quality": 0.9, "vram": 80.0})
        b = ParetoPoint("b", {"quality": 0.5, "vram": 20.0})
        result = find_pareto_frontier([a, b], maximize=["quality"], minimize=["vram"])
        # Both are on the frontier
        assert len(result) == 2


# ============================================================
# WorkflowResult Tests
# ============================================================


class TestWorkflowResult:

    def test_defaults(self) -> None:
        r = WorkflowResult()
        assert r.turns == []
        assert r.total_latency_ms == 0.0
        assert r.tool_calls == []
        assert r.state_transitions == []
        assert r.success is False

    def test_to_dict(self) -> None:
        r = WorkflowResult(
            turns=[{"turn": 0}],
            total_latency_ms=123.4,
            tool_calls=[{"name": "search"}],
            state_transitions=[{"from_state": "A", "to_state": "B"}],
            success=True,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["total_latency_ms"] == pytest.approx(123.4)
        assert d["turns"][0]["turn"] == 0
        assert d["state_transitions"][0]["from_state"] == "A"

    def test_success_flag(self) -> None:
        assert WorkflowResult(success=True).success is True
        assert WorkflowResult(success=False).success is False


# ============================================================
# MultiAgentOrchestrator Initialization Tests
# ============================================================


class TestOrchestratorInit:

    def test_stores_config(self) -> None:
        orch_cfg = {"model_name": "Qwen/Qwen3-32B", "base_url": "http://localhost:8000/v1"}
        spec_cfgs = [{"model_name": "Qwen/Qwen2.5-3B", "tag": "billing"}]
        orch = MultiAgentOrchestrator(orch_cfg, spec_cfgs, kv_cache_dtype="fp8")
        assert orch.orchestrator_config is orch_cfg
        assert orch.specialist_configs is spec_cfgs
        assert orch.kv_cache_dtype == "fp8"

    def test_default_kv_cache_dtype(self) -> None:
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])
        assert orch.kv_cache_dtype == "turboquant"

    def test_base_url_from_config(self) -> None:
        orch = MultiAgentOrchestrator(
            {"model_name": "m", "base_url": "http://myserver:9000/v1"}, []
        )
        assert orch.base_url == "http://myserver:9000/v1"

    def test_base_url_default(self) -> None:
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])
        assert orch.base_url == "http://localhost:8000/v1"

    def test_no_openai_import_at_init(self) -> None:
        """Verify openai is not imported at __init__ time."""
        import sys
        # Remove openai from sys.modules to ensure it's not pre-imported
        openai_mod = sys.modules.pop("openai", None)
        try:
            orch = MultiAgentOrchestrator({"model_name": "m"}, [])
            # If openai were imported at __init__, this would fail if openai isn't installed
            assert orch is not None
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_extract_state_transitions(self) -> None:
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])
        content = "Processing... [STATE: INTAKE -> CLASSIFY] Done."
        transitions = orch._extract_state_transitions(content)
        assert len(transitions) == 1
        assert transitions[0]["from_state"] == "INTAKE"
        assert transitions[0]["to_state"] == "CLASSIFY"

    def test_extract_state_transitions_multiple(self) -> None:
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])
        content = "[STATE: A -> B] then [STATE: B -> C]"
        transitions = orch._extract_state_transitions(content)
        assert len(transitions) == 2

    def test_extract_state_transitions_no_match(self) -> None:
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])
        assert orch._extract_state_transitions("no transitions here") == []


# ============================================================
# Async Orchestrator Test
# ============================================================


class TestOrchestratorAsync:

    @staticmethod
    def _make_mock_openai(content: str = "") -> tuple[MagicMock, MagicMock]:
        """Build a mock openai module with AsyncOpenAI that returns a single response."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        return mock_openai, mock_client

    @pytest.mark.asyncio
    async def test_run_workflow_mocked(self) -> None:
        """Test run_workflow with mocked openai AsyncOpenAI client."""
        import sys

        mock_openai, _ = self._make_mock_openai(
            "Processing request. [STATE: INTAKE -> RESOLVED]"
        )

        orch = MultiAgentOrchestrator(
            {"model_name": "Qwen/Qwen3-32B", "base_url": "http://localhost:8000/v1"},
            [],
            kv_cache_dtype="fp8",
        )

        workflow_graph = {
            "initial_state": "INTAKE",
            "terminal_states": ["RESOLVED"],
            "nodes": [{"id": "INTAKE"}, {"id": "RESOLVED"}],
            "edges": [{"from_state": "INTAKE", "to_state": "RESOLVED"}],
        }
        conversation = [
            {"role": "system", "content": "You are a workflow assistant."},
            {"role": "user", "content": "Handle this request."},
        ]

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = await orch.run_workflow(conversation, workflow_graph)

        assert isinstance(result, WorkflowResult)
        assert result.total_latency_ms >= 0.0
        assert isinstance(result.turns, list)
        assert isinstance(result.state_transitions, list)

    @pytest.mark.asyncio
    async def test_run_workflow_empty_conversation(self) -> None:
        """Workflow with no user turns returns empty WorkflowResult."""
        import sys

        mock_openai, _ = self._make_mock_openai()
        orch = MultiAgentOrchestrator({"model_name": "m"}, [])

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = await orch.run_workflow(
                [],
                {"initial_state": "S0", "terminal_states": ["S0"], "nodes": [], "edges": []},
            )

        assert result.turns == []
        assert result.total_latency_ms == 0.0


# ============================================================
# BenchmarkResult Tests
# ============================================================


class TestBenchmarkResult:

    def test_to_dict(self) -> None:
        br = BenchmarkResult(
            model="Qwen/Qwen3-32B",
            kv_cache_dtype="fp8",
            context_length=4096,
            max_concurrent=350,
            p95_latency_ms=2500.0,
            peak_vram_gb=72.0,
            task_completion_rate=0.82,
        )
        d = br.to_dict()
        assert d["model"] == "Qwen/Qwen3-32B"
        assert d["max_concurrent"] == 350
        assert d["p95_latency_ms"] == pytest.approx(2500.0)

    def test_field_access(self) -> None:
        br = BenchmarkResult(
            model="m", kv_cache_dtype="auto", context_length=2048,
            max_concurrent=100, p95_latency_ms=1000.0,
            peak_vram_gb=40.0, task_completion_rate=0.75,
        )
        assert br.kv_cache_dtype == "auto"
        assert br.task_completion_rate == pytest.approx(0.75)


# ============================================================
# compute_pareto_frontier Tests
# ============================================================


class TestComputeParetoFrontier:

    def test_empty_returns_empty(self) -> None:
        assert compute_pareto_frontier([]) == []

    def test_single_config_returned(self) -> None:
        br = BenchmarkResult("m", "fp8", 4096, 100, 500.0, 40.0, 0.8)
        result = compute_pareto_frontier([br])
        assert len(result) == 1

    def test_dominated_config_removed(self) -> None:
        # a: high completion, low vram, low latency → dominates b
        a = BenchmarkResult("m", "fp8", 4096, 350, 200.0, 30.0, 0.9)
        b = BenchmarkResult("m", "bf16", 4096, 100, 500.0, 70.0, 0.5)
        result = compute_pareto_frontier([a, b])
        assert len(result) == 1
        assert result[0].kv_cache_dtype == "fp8"

    def test_incomparable_configs_both_returned(self) -> None:
        # a: high completion, high vram; b: low completion, low vram
        a = BenchmarkResult("m", "fp8", 4096, 350, 500.0, 70.0, 0.9)
        b = BenchmarkResult("m", "bf16", 4096, 100, 200.0, 20.0, 0.5)
        result = compute_pareto_frontier([a, b])
        assert len(result) == 2

    def test_axis_directions(self) -> None:
        """task_completion_rate is maximized; vram and latency are minimized."""
        # c is strictly better on all 3 axes than d
        c = BenchmarkResult("m", "turboquant", 4096, 925, 150.0, 25.0, 0.88)
        d = BenchmarkResult("m", "rotorquant", 4096, 200, 600.0, 60.0, 0.60)
        result = compute_pareto_frontier([c, d])
        assert len(result) == 1
        assert result[0].kv_cache_dtype == "turboquant"


# ============================================================
# Plot Function Signature Tests (no actual rendering)
# ============================================================


class TestPlotFunctionSignatures:

    def _check_callable(self, func_name: str) -> None:
        from llm_workflow_agents.analysis import plot_results
        func = getattr(plot_results, func_name)
        assert callable(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert "results" in params
        assert "output_path" in params

    def test_plot_accuracy_vs_complexity_callable(self) -> None:
        self._check_callable("plot_accuracy_vs_complexity")

    def test_plot_specialist_vs_baseline_callable(self) -> None:
        self._check_callable("plot_specialist_vs_baseline")

    def test_plot_graph_metrics_callable(self) -> None:
        self._check_callable("plot_graph_metrics")

    def test_plot_quality_degradation_callable(self) -> None:
        self._check_callable("plot_quality_degradation")

    def test_plot_pareto_frontier_callable(self) -> None:
        self._check_callable("plot_pareto_frontier")


# ============================================================
# Module Import Tests
# ============================================================


class TestModuleImports:

    def test_import_serving(self) -> None:
        from llm_workflow_agents.serving import MultiAgentOrchestrator, WorkflowResult
        assert MultiAgentOrchestrator is not None
        assert WorkflowResult is not None

    def test_import_analysis(self) -> None:
        from llm_workflow_agents.analysis import ParetoPoint, find_pareto_frontier
        assert ParetoPoint is not None
        assert find_pareto_frontier is not None
