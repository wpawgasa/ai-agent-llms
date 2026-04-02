#!/usr/bin/env bash
# Run Experiment A: Prompt-encoded business logic evaluation.
#
# Launches each 15-30B model with vLLM, runs workflow quality evaluation,
# and saves results to results/exp_a/.
#
# Usage:
#   ./scripts/run_exp_a.sh [OPTIONS]
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

mkdir -p "$RESULTS_DIR"

MODEL_CONFIGS=(
    "configs/models_exp_a/gemma3_27b.yaml"
    "configs/models_exp_a/qwen3_32b.yaml"
    "configs/models_exp_a/qwen35_35b_a3b.yaml"
    "configs/models_exp_a/mistral_24b.yaml"
    "configs/models_exp_a/nemotron_30b.yaml"
    "configs/models_exp_a/glm47_flash.yaml"
)

echo "=== Experiment A: Prompt-Encoded Business Logic ==="
echo "KV cache dtype: $KV_CACHE_DTYPE"
echo "Data dir:       $DATA_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max samples:    ${MAX_SAMPLES} (0=all)"
echo "=================================================="

for CONFIG in "${MODEL_CONFIGS[@]}"; do
    MODEL_CONFIG="$PROJECT_ROOT/$CONFIG"
    MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")

    echo ""
    echo "--- Model: $MODEL_NAME ---"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG --kv-cache-dtype $KV_CACHE_DTYPE"
        echo "[DRY RUN] Would run eval for $MODEL_NAME"
        continue
    fi

    # Launch vLLM server in background
    bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" --kv-cache-dtype "$KV_CACHE_DTYPE" &
    VLLM_PID=$!

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
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    # Run evaluation
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${KV_CACHE_DTYPE}.json"
    python3 -m llm_workflow_agents.eval.agent_benchmark \
        --model       "$MODEL_NAME" \
        --output      "$RESULT_FILE" \
        --data        "$DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        2>&1 | tee "${RESULT_FILE%.json}.log" || true

    # Shut down vLLM
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "Server stopped."
done

echo ""
echo "=== Experiment A complete. Results in $RESULTS_DIR ==="
