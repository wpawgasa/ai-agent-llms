# Phase 7: Integration & E2E

**Date**: 2026-03-31
**Branch**: `feature/phase7-integration`
**Author**: Claude

## Overview

Phase 7 implements the final integration layer for the LLM Workflow-Orchestrating Agents project, wiring together the multi-agent orchestrator, E2E benchmarking, Pareto frontier analysis, result visualization, and experiment runner scripts.

## Changes

### New Files
- `src/llm_workflow_agents/analysis/pareto.py` — Pareto frontier computation (pure logic, no heavy deps)
- `src/llm_workflow_agents/serving/orchestrator.py` — MultiAgentOrchestrator with async run_workflow()
- `src/llm_workflow_agents/serving/benchmark_e2e.py` — Concurrency benchmark + Pareto wrapper
- `src/llm_workflow_agents/analysis/plot_results.py` — 5 chart generation functions (deferred matplotlib/seaborn)
- `scripts/run_exp_a.sh` — Experiment A runner (prompt-encoded 15-30B models)
- `scripts/run_exp_b.sh` — Experiment B runner (fine-tuned specialist subagents)
- `scripts/run_exp_c.sh` — Experiment C runner (graph extraction)
- `scripts/run_exp_d.sh` — Experiment D runner (KV cache quantization benchmark matrix)
- `scripts/run_exp_e2e.sh` — Full E2E integration experiment runner
- `tests/unit/test_integration.py` — 42 unit tests covering all Phase 7 modules

### Modified Files
- `src/llm_workflow_agents/serving/__init__.py` — Exports MultiAgentOrchestrator, WorkflowResult
- `src/llm_workflow_agents/analysis/__init__.py` — Exports ParetoPoint, find_pareto_frontier
- `.github/workflows/ci.yml` — Added networkx and pytest-asyncio to CI deps
- `pyproject.toml` — Added asyncio_mode = "auto" for pytest-asyncio
- `CLAUDE.md` — Phase 7 marked complete

## Technical Details

### Pareto Frontier (`analysis/pareto.py`)
- `ParetoPoint` dataclass with arbitrary `metrics: dict[str, float]`
- `find_pareto_frontier()` performs O(n²) pairwise dominance check, supporting mixed maximize/minimize axes
- `_dominates(a, b)` helper: a dominates b iff a ≥ b on all maximize axes, a ≤ b on all minimize axes, and a is strictly better on at least one

### Orchestrator (`serving/orchestrator.py`)
- `MultiAgentOrchestrator` routes workflow turns between a 15-30B orchestrator and 2-5B specialists
- `run_workflow()` is async; uses `openai.AsyncOpenAI` (deferred import) to call vLLM
- Extracts `[STATE: X -> Y]` annotations via regex to track state transitions
- Records per-turn latency via `time.perf_counter()`
- Nemotron fallback: catches exceptions for nemotron models, logs warning, continues

### Benchmark E2E (`serving/benchmark_e2e.py`)
- `benchmark_concurrency()`: binary-search for max concurrent requests under a P95 latency threshold
- `compute_pareto_frontier()`: wraps `analysis.pareto.find_pareto_frontier` with axis direction mapping (task_completion=maximize; vram, latency=minimize)
- Expected concurrency: BF16 ~175, FP8 ~350, TurboQuant-3bit ~925 at 4096-token context

### Visualization (`analysis/plot_results.py`)
- All matplotlib/seaborn/pandas imports are deferred inside each function
- `plot_accuracy_vs_complexity()` — Exp A line plot per model
- `plot_specialist_vs_baseline()` — Exp B grouped bar chart
- `plot_graph_metrics()` — Exp C node F1, edge F1, GED bars
- `plot_quality_degradation()` — Exp D scatter (compression vs. perplexity delta)
- `plot_pareto_frontier()` — E2E Pareto scatter with annotated optimal configs

## Testing

- Unit tests: `tests/unit/test_integration.py` — 42 tests
- All tests pass without heavy deps (openai mocked via `sys.modules`, matplotlib not exercised)
- Full suite: 334 tests, 79% coverage (threshold: 70%)

## Usage

```bash
# Run each experiment
./scripts/run_exp_a.sh --kv-cache-dtype fp8
./scripts/run_exp_b.sh --skip-training
./scripts/run_exp_c.sh
./scripts/run_exp_d.sh --models-only exp_a
./scripts/run_exp_e2e.sh --kv-cache-dtype turboquant

# From Python
from llm_workflow_agents.serving import MultiAgentOrchestrator
from llm_workflow_agents.analysis import ParetoPoint, find_pareto_frontier
from llm_workflow_agents.serving.benchmark_e2e import compute_pareto_frontier
```
