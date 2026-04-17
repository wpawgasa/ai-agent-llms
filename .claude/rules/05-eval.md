# Evaluation Module

## Overview
`eval/` contains all evaluation metrics for Phase 1 benchmarking (Tasks A–C) and Phase 3 quantization, plus composite scoring for model selection.

## State Machine Adherence (`state_accuracy.py`)
```python
@dataclass
class StateMachineMetrics:
    state_transition_accuracy: float    # Target: >=85%
    task_completion_rate: float         # Target: >=70%
    invalid_transition_rate: float      # Target: <=5%
    recovery_rate: float                # Target: >=60%
    consistency_pass5: float            # Target: >=0.40

def evaluate_state_machine(
    predictions, ground_truth, num_stochastic_trials=5
) -> StateMachineMetrics
```
- Parse `[STATE: X → Y]` annotations from model output
- pass^5: all 5 temperature=0.7 trials must reach correct terminal state

## Tool-Calling Accuracy (`tool_call_f1.py`)
```python
@dataclass
class ToolCallMetrics:
    tool_name_accuracy: float           # Target: >=90%
    argument_exact_match: float         # Target: >=75%
    tool_call_f1: float                 # Target: >=85% (BFCL-style AST match)
    chain_propagation_accuracy: float   # Target: >=70%
    hallucinated_tool_rate: float       # Target: <=3%
    error_recovery_rate: float          # Target: >=60%

def evaluate_tool_calls(predictions, ground_truth, tool_schemas) -> ToolCallMetrics
```
- Parse `<tool_call>{JSON}</tool_call>` from model output
- AST sub-tree matching for argument comparison (BFCL style)
- Normalize all tool-call formats to canonical JSON before scoring

## Graph Extraction (`graph_extraction_eval.py`)
```python
@dataclass
class GraphExtractionMetrics:
    node_f1: float                       # Target: >=85%
    edge_f1: float                       # Target: >=75%
    graph_edit_distance: float           # Target: <=0.20 (normalized)
    json_validity: float                 # Target: >=95%
    structural_validity: float           # Target: >=90%
    mermaid_renderability: float         # Target: >=90%

def evaluate_graph_extraction(predicted_graphs, gold_graphs) -> GraphExtractionMetrics
```
Structural checks: valid initial state, reachable terminals, no orphan nodes

## Combined Workflow Quality (`composite_score.py`)
```python
def compute_weighted_workflow_score(state: StateMachineMetrics, tool: ToolCallMetrics) -> float:
    """0.4 × StateTransAcc + 0.4 × ToolCallF1 + 0.2 × TaskCompletion. Target: >=0.75."""

def full_workflow_success_rate(predictions, ground_truth) -> float:
    """% of conversations with ALL correct transitions AND tool calls. Target: >=55%."""
```

## Quantization Benchmark Harness (`quant_benchmark.py`)
```python
def run_quant_benchmark(
    models: list[str],              # Pre-trained + fine-tuned
    methods: list[str],             # ["fp8", "kivi", "kvquant", "awq_fp8", "turboquant", "rotorquant"]
    quality_tasks: list[str],
    num_runs: int = 5,
    prompts_per_run: int = 500,
) -> QuantBenchmarkMatrix
```
- Quality: WikiText-2 PPL, C4 PPL, LongBench, Needle-in-Haystack, Tool-call F1
- Performance: peak VRAM, KV cache size, throughput, latency (TTFT/TPOT/ITL p50/p95/p99), max concurrent batch at 4096 context
- Report: mean ± std over 3–5 runs

## Quantization Quality — Standalone Evals
- `perplexity.py`: WikiText-2, C4 PPL measurement
- `longbench.py`: 15-task LongBench evaluation
- `needle_haystack.py`: Needle-in-a-Haystack at 2K–32K context lengths

## Checklist
- [x] Implement state_accuracy.py with StateMachineMetrics
- [x] Implement tool_call_f1.py with AST matching
- [x] Implement graph_extraction_eval.py with GED (networkx)
- [x] Implement composite_score.py (weighted workflow score + full success rate)
- [x] Implement quant_benchmark.py full matrix runner
- [x] Implement perplexity.py for WikiText-2 and C4
- [x] Implement longbench.py 15-task evaluation
- [x] Implement needle_haystack.py at multiple context lengths
- [x] Write test_eval_metrics.py with known-answer tests
- [x] Write test_composite_score.py (normalization, weight application, ranking stability)
