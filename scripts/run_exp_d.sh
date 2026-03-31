#!/usr/bin/env bash
# Run Experiment D: KV cache quantization benchmark.
#
# For each (model, quantization method) pair, launches vLLM with the specified
# KV cache dtype, runs perplexity (WikiText-2, C4), LongBench, and
# Needle-in-Haystack evaluations. Saves results to results/exp_d/.
#
# Usage:
#   ./scripts/run_exp_d.sh [--models-only exp_a|exp_bc|all] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_d"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"

MODEL_SET="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-only)
            MODEL_SET="$2"
            shift 2
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

mkdir -p "$RESULTS_DIR"

EXP_A_CONFIGS=(
    "configs/models_exp_a/gemma3_27b.yaml"
    "configs/models_exp_a/qwen3_32b.yaml"
    "configs/models_exp_a/mistral_24b.yaml"
    "configs/models_exp_a/nemotron_30b.yaml"
    "configs/models_exp_a/glm47_flash.yaml"
)

EXP_BC_CONFIGS=(
    "configs/models_exp_bc/qwen25_3b.yaml"
    "configs/models_exp_bc/qwen35_4b.yaml"
    "configs/models_exp_bc/glm47_flash.yaml"
    "configs/models_exp_bc/gemma_2b.yaml"
    "configs/models_exp_bc/gemma3_4b.yaml"
)

case "$MODEL_SET" in
    exp_a)   MODEL_CONFIGS=("${EXP_A_CONFIGS[@]}") ;;
    exp_bc)  MODEL_CONFIGS=("${EXP_BC_CONFIGS[@]}") ;;
    all)     MODEL_CONFIGS=("${EXP_A_CONFIGS[@]}" "${EXP_BC_CONFIGS[@]}") ;;
    *)       echo "Unknown model set: $MODEL_SET (use exp_a, exp_bc, or all)"; exit 1 ;;
esac

QUANT_METHODS=("auto" "fp8" "kivi_2bit" "kivi_4bit" "kvquant" "turboquant" "rotorquant")

echo "=== Experiment D: KV Cache Quantization Benchmark ==="
echo "Model set:  $MODEL_SET (${#MODEL_CONFIGS[@]} models)"
echo "Quant methods: ${QUANT_METHODS[*]}"
echo "Results dir: $RESULTS_DIR"
echo "====================================================="

for CONFIG in "${MODEL_CONFIGS[@]}"; do
    MODEL_CONFIG="$PROJECT_ROOT/$CONFIG"
    MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")

    for KV_DTYPE in "${QUANT_METHODS[@]}"; do
        echo ""
        echo "--- Model: $MODEL_NAME | Quant: $KV_DTYPE ---"

        RUN_DIR="$RESULTS_DIR/${MODEL_NAME//\//_}/${KV_DTYPE}"
        mkdir -p "$RUN_DIR"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would eval $MODEL_NAME with $KV_DTYPE"
            continue
        fi

        # Launch vLLM with quantization
        bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" --kv-cache-dtype "$KV_DTYPE" &
        VLLM_PID=$!

        echo "Waiting for vLLM server (PID $VLLM_PID)..."
        for i in $(seq 1 60); do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
                echo "Server ready after ${i}s"
                break
            fi
            sleep 5
        done
        if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "ERROR: vLLM server failed to start within 300s for $MODEL_NAME / $KV_DTYPE" >&2
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi

        # Perplexity evaluation
        python3 -m llm_workflow_agents.eval.perplexity \
            --model "$MODEL_NAME" \
            --datasets wikitext2 c4 \
            --kv-cache-dtype "$KV_DTYPE" \
            --output "$RUN_DIR/perplexity.json" 2>&1 | tee "$RUN_DIR/perplexity.log" || true

        # LongBench evaluation
        python3 -m llm_workflow_agents.eval.longbench \
            --model "$MODEL_NAME" \
            --output "$RUN_DIR/longbench.json" 2>&1 | tee "$RUN_DIR/longbench.log" || true

        # Needle-in-Haystack evaluation
        python3 -m llm_workflow_agents.eval.needle_haystack \
            --model "$MODEL_NAME" \
            --output "$RUN_DIR/needle.json" 2>&1 | tee "$RUN_DIR/needle.log" || true

        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "Server stopped."
    done
done

echo ""
echo "=== Experiment D complete. Results in $RESULTS_DIR ==="
