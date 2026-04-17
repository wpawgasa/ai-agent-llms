# Phase 4: Integration & Pareto (`integration/`)

## Overview
`integration/` implements Phase 4: multi-agent deployment of the 3 fine-tuned + quantized models, end-to-end benchmarking, and Pareto frontier analysis.

## Files
- `orchestrator.py` — Multi-agent orchestrator (Cat A + Cat B + Cat C models)
- `benchmark_e2e.py` — Concurrency + latency measurement
- `pareto.py` — Pareto frontier computation + 2D projection plots

## orchestrator.py — MultiAgentOrchestrator

```python
class MultiAgentOrchestrator:
    """
    Production deployment:
      Orchestrator: Cat A winner (15–35B) — routes intent, manages workflow state
      Specialist:   Cat B winner (2–5B)   — executes tool calls + state transitions
      Visualizer:   Cat C winner (2–5B)   — converts prompts to workflow graphs on demand

    All served via vLLM with best quantization method from Phase 3.
    vLLM LoRA multi-adapter serving if Cat B and Cat C share same base model.
    """

    def __init__(
        self,
        orchestrator_config: Path,
        specialist_config: Path,
        visualizer_config: Path,
        kv_cache_dtype: str = "turboquant",
    )

    async def run_workflow(self, user_input: str, workflow_graph: dict) -> WorkflowResult
    async def run_scenario_battery(self, num_scenarios=50, trials_per_scenario=5) -> IntegrationResults
```

### Workflow Routing
1. Orchestrator receives user input
2. Classifies intent → selects specialist
3. Specialist executes tool calls + state transitions
4. Returns to orchestrator for confirmation / next step
5. Optional: Cat C model generates workflow visualization

Records per-turn latency, tool calls, state transitions.

## benchmark_e2e.py

```python
def benchmark_concurrency(
    deployment_config: Path,
    context_length: int = 4096,
) -> ConcurrencyResult
```

### Expected Results (4096 ctx, H100 80GB)
| Method | Concurrent Sessions |
|--------|-------------------|
| BF16 | ~175 |
| FP8 | ~350 |
| TQ 3-bit | ~925 |

## pareto.py — Pareto Frontier

```python
def compute_pareto_frontier(
    results: list[ConfigResult],
    axes: tuple[str, str, str] = ("task_completion", "peak_vram_gb", "p95_latency_ms"),
) -> list[ConfigResult]
    """Identify Pareto-optimal (model, quantization) configs across quality × memory × latency."""

def plot_pareto_projections(
    pareto_configs: list[ConfigResult],
    output_dir: Path = Path("analysis/figures"),
) -> list[Path]
    """Generate 3 × 2D scatter plots (calls analysis/plot_pareto.py)."""
```

### Expected Pareto Optima
- Lowest latency: Gemma-2B + FP8
- Best quality: Qwen2.5-3B + TurboQuant-3.5bit
- Max concurrency: Gemma-2B + TurboQuant-3bit

## Checklist
- [x] Implement orchestrator.py with 3-role routing (orchestrator, specialist, visualizer)
- [x] Handle vLLM LoRA multi-adapter for shared Cat B/C base model
- [x] Implement benchmark_e2e.py concurrency measurement
- [x] Implement compute_pareto_frontier (3-objective)
- [x] Implement plot_pareto_projections (delegates to analysis/plot_pareto.py)
- [x] Run 50 scenarios × 5 trials = 250+ multi-agent integration runs
