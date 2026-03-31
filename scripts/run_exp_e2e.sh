#!/usr/bin/env bash
# Run the full E2E integration experiment.
#
# Launches the multi-agent orchestrator, runs concurrency benchmarks,
# computes Pareto frontier, and generates visualization plots.
# Results are saved to results/e2e/.
#
# Usage:
#   ./scripts/run_exp_e2e.sh [--kv-cache-dtype <dtype>] [--skip-benchmark] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/e2e"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"

KV_CACHE_DTYPE="turboquant"
SKIP_BENCHMARK=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR/plots"

echo "=== E2E Integration Experiment ==="
echo "KV cache dtype: $KV_CACHE_DTYPE"
echo "Results dir:    $RESULTS_DIR"
echo "=================================="

# Best Exp A model: Qwen3-32B (highest weighted score)
ORCHESTRATOR_CONFIG="$PROJECT_ROOT/configs/models_exp_a/qwen3_32b.yaml"
# Best Exp B model: Qwen2.5-3B (best tool-call F1)
SPECIALIST_CONFIG="$PROJECT_ROOT/configs/models_exp_bc/qwen25_3b.yaml"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run E2E with orchestrator Qwen3-32B + specialist Qwen2.5-3B"
    exit 0
fi

# --- Step 1: Launch orchestrator vLLM server ---
echo ""
echo "--- Step 1: Launching orchestrator vLLM (Qwen3-32B + $KV_CACHE_DTYPE) ---"
bash "$LAUNCH_SCRIPT" "$ORCHESTRATOR_CONFIG" --kv-cache-dtype "$KV_CACHE_DTYPE" &
ORCHESTRATOR_PID=$!

echo "Waiting for orchestrator server (PID $ORCHESTRATOR_PID)..."
for i in $(seq 1 90); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Orchestrator ready after ${i}s"
        break
    fi
    sleep 5
done

# --- Step 2: Run E2E workflow test ---
echo ""
echo "--- Step 2: Running E2E workflow benchmark ---"
python3 -c "
import asyncio, json, sys
sys.path.insert(0, 'src')
from llm_workflow_agents.serving.orchestrator import MultiAgentOrchestrator

orchestrator_config = {
    'model_name': 'Qwen/Qwen3-32B-Instruct',
    'base_url': 'http://localhost:8000/v1',
}
specialist_configs = []

orch = MultiAgentOrchestrator(orchestrator_config, specialist_configs, kv_cache_dtype='$KV_CACHE_DTYPE')
sample_conversation = [
    {'role': 'system', 'content': 'You are a workflow orchestrator.'},
    {'role': 'user', 'content': 'Process a customer support ticket for a billing issue.'},
]
sample_graph = {
    'initial_state': 'INTAKE',
    'terminal_states': ['RESOLVED'],
    'nodes': [
        {'id': 'INTAKE', 'name': 'Ticket Intake'},
        {'id': 'CLASSIFY', 'name': 'Classify Issue'},
        {'id': 'RESOLVE', 'name': 'Resolve'},
        {'id': 'RESOLVED', 'name': 'Resolved'},
    ],
    'edges': [
        {'from_state': 'INTAKE', 'to_state': 'CLASSIFY'},
        {'from_state': 'CLASSIFY', 'to_state': 'RESOLVE'},
        {'from_state': 'RESOLVE', 'to_state': 'RESOLVED'},
    ],
}
result = asyncio.run(orch.run_workflow(sample_conversation, sample_graph))
with open('results/e2e/workflow_result.json', 'w') as f:
    json.dump(result.to_dict(), f, indent=2)
print(f'Workflow complete: success={result.success}, turns={len(result.turns)}')
" 2>&1 | tee "$RESULTS_DIR/workflow.log" || true

# --- Step 3: Concurrency benchmark ---
if [ "$SKIP_BENCHMARK" = false ]; then
    echo ""
    echo "--- Step 3: Running concurrency benchmark ---"
    python3 -c "
import json, sys
sys.path.insert(0, 'src')
from llm_workflow_agents.serving.benchmark_e2e import benchmark_concurrency, compute_pareto_frontier

# Benchmark Qwen3-32B with selected quant methods
configs = [
    ('Qwen/Qwen3-32B-Instruct', 'auto'),
    ('Qwen/Qwen3-32B-Instruct', 'fp8'),
    ('Qwen/Qwen3-32B-Instruct', '$KV_CACHE_DTYPE'),
]

results = []
for model_name, dtype in configs:
    result = benchmark_concurrency(model_name, dtype, context_length=4096)
    results.append(result.to_dict())

# Pareto analysis
from llm_workflow_agents.serving.benchmark_e2e import BenchmarkResult
br_list = [BenchmarkResult(**r) for r in results]
pareto = compute_pareto_frontier(br_list)
pareto_names = [f\"{r.model}_{r.kv_cache_dtype}_4096\" for r in pareto]

with open('results/e2e/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('results/e2e/pareto_configs.json', 'w') as f:
    json.dump(pareto_names, f, indent=2)
print(f'Benchmark complete: {len(results)} configs, {len(pareto)} Pareto-optimal')
" 2>&1 | tee "$RESULTS_DIR/benchmark.log" || true
fi

# --- Step 4: Generate plots ---
echo ""
echo "--- Step 4: Generating Pareto frontier plot ---"
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from llm_workflow_agents.analysis.plot_results import plot_pareto_frontier

try:
    with open('results/e2e/benchmark_results.json') as f:
        results = json.load(f)
    with open('results/e2e/pareto_configs.json') as f:
        pareto_names = json.load(f)

    for r in results:
        r['config_name'] = f\"{r['model']}_{r['kv_cache_dtype']}_{r['context_length']}\"

    plot_pareto_frontier(results, 'results/e2e/plots/pareto_frontier.png', pareto_names)
    print('Pareto frontier plot saved.')
except FileNotFoundError:
    print('No benchmark results found, skipping plot.')
" 2>&1 | tee "$RESULTS_DIR/plots.log" || true

# --- Cleanup ---
kill "$ORCHESTRATOR_PID" 2>/dev/null || true
wait "$ORCHESTRATOR_PID" 2>/dev/null || true
echo "Orchestrator server stopped."

echo ""
echo "=== E2E experiment complete. Results in $RESULTS_DIR ==="
