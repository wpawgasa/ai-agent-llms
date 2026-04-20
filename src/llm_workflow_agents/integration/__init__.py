"""Phase 4 (v3): Multi-agent integration, E2E benchmark, and Pareto analysis.

Re-exports from serving/ (v2 location) and analysis/ for unified v3 API.
"""

from llm_workflow_agents.serving.orchestrator import (
    MultiAgentOrchestrator,
    WorkflowResult,
)
from llm_workflow_agents.serving.benchmark_e2e import (
    BenchmarkResult,
    benchmark_concurrency,
    compute_pareto_frontier,
)
from llm_workflow_agents.analysis.pareto import (
    ParetoPoint,
    find_pareto_frontier,
)
from llm_workflow_agents.eval.concurrency_benchmark import (
    ConcurrencySweepResult,
    ContextSweepResult,
    LevelResult,
    run_concurrency_sweep,
)

__all__ = [
    "BenchmarkResult",
    "ConcurrencySweepResult",
    "ContextSweepResult",
    "LevelResult",
    "MultiAgentOrchestrator",
    "ParetoPoint",
    "WorkflowResult",
    "benchmark_concurrency",
    "compute_pareto_frontier",
    "find_pareto_frontier",
    "run_concurrency_sweep",
]
