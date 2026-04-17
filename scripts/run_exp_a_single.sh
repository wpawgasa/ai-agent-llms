#!/usr/bin/env bash
# Run Experiment A on a single model.
#
# Usage:
#   ./scripts/run_exp_a_single.sh <config> [OPTIONS]
#
# Arguments:
#   <config>          Path to model YAML config (relative to project root or absolute).
#                     e.g. configs/models_exp_a/gemma4_31b.yaml
#
# Options:
#   --kv-cache-dtype  KV cache quantization dtype (default: auto)
#   --data <path>     Benchmark data directory (default: data/output/benchmark/task_a)
#   --max-samples <n> Limit to first N samples per level, 0=all (default: 0)
#   --dry-run         Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_a"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <config> [--kv-cache-dtype auto] [--data <path>] [--max-samples <n>] [--dry-run]"
    exit 1
fi

# First positional argument is the config path
CONFIG_ARG="$1"
shift

KV_CACHE_DTYPE="auto"
DATA_DIR="$PROJECT_ROOT/data/output/benchmark/task_a"
MAX_SAMPLES=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --data)           DATA_DIR="$2";        shift 2 ;;
        --max-samples)    MAX_SAMPLES="$2";     shift 2 ;;
        --dry-run)        DRY_RUN=true;         shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Resolve config path (support both relative-to-project-root and absolute)
if [[ "$CONFIG_ARG" = /* ]]; then
    MODEL_CONFIG="$CONFIG_ARG"
else
    MODEL_CONFIG="$PROJECT_ROOT/$CONFIG_ARG"
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: Config not found: $MODEL_CONFIG" >&2
    exit 1
fi

MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")

mkdir -p "$RESULTS_DIR"

echo "=== Experiment A (single model): Prompt-Encoded Business Logic ==="
echo "Config:         $MODEL_CONFIG"
echo "Model:          $MODEL_NAME"
echo "KV cache dtype: $KV_CACHE_DTYPE"
echo "Data dir:       $DATA_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max samples:    ${MAX_SAMPLES} (0=all)"
echo "=================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG --kv-cache-dtype $KV_CACHE_DTYPE"
    echo "[DRY RUN] Would run eval for $MODEL_NAME"
    exit 0
fi

# Launch vLLM server in background
bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" --kv-cache-dtype "$KV_CACHE_DTYPE" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "Server stopped."
}
trap cleanup EXIT

# Wait for server to be ready (poll health endpoint)
echo "Waiting for vLLM server (PID $VLLM_PID)..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 5
done
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 300s for $MODEL_NAME" >&2
    exit 1
fi

# Run evaluation
RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${KV_CACHE_DTYPE}.json"
python3 -m llm_workflow_agents.eval.agent_benchmark \
    --model           "$MODEL_NAME" \
    --kv-cache-dtype  "$KV_CACHE_DTYPE" \
    --output          "$RESULT_FILE" \
    --data            "$DATA_DIR" \
    --max-samples     "$MAX_SAMPLES" \
    --log-level       DEBUG \
    2>&1 | tee "${RESULT_FILE%.json}.log" || true

echo ""
echo "=== Done. Results in $RESULT_FILE ==="
