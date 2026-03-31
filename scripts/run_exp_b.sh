#!/usr/bin/env bash
# Run Experiment B: Fine-tuned specialist subagent evaluation.
#
# Trains each 2-5B specialist model with LoRA, launches with vLLM,
# runs tool-call F1 evaluation, and saves results to results/exp_b/.
#
# Usage:
#   ./scripts/run_exp_b.sh [--skip-training] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_b"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"

SKIP_TRAINING=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-training)
            SKIP_TRAINING=true
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

mkdir -p "$RESULTS_DIR"

MODEL_CONFIGS=(
    "configs/models_exp_bc/qwen25_3b.yaml"
    "configs/models_exp_bc/qwen35_4b.yaml"
    "configs/models_exp_bc/glm47_flash.yaml"
    "configs/models_exp_bc/gemma_2b.yaml"
    "configs/models_exp_bc/gemma3_4b.yaml"
)

echo "=== Experiment B: Fine-Tuned Specialist Subagents ==="
echo "Skip training: $SKIP_TRAINING"
echo "Results dir:   $RESULTS_DIR"
echo "====================================================="

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
        echo "[DRY RUN] Would train and eval $MODEL_NAME"
        continue
    fi

    # Train specialist
    if [ "$SKIP_TRAINING" = false ]; then
        echo "Training specialist $MODEL_NAME..."
        python3 -m llm_workflow_agents.training.train_specialist \
            --config "$MODEL_CONFIG" \
            --output-dir "checkpoints/exp_b/${MODEL_NAME//\//_}" 2>&1 | \
            tee "$RESULTS_DIR/${MODEL_NAME//\//_}_train.log" || true
    fi

    # Launch vLLM with LoRA
    bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" &
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
        echo "ERROR: vLLM server failed to start within 300s for $MODEL_NAME" >&2
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    # Run tool-call F1 evaluation
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}.json"
    python3 -m llm_workflow_agents.eval.tool_call_f1 \
        --model "$MODEL_NAME" \
        --output "$RESULT_FILE" 2>&1 | tee "${RESULT_FILE%.json}.log" || true

    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "Server stopped."
done

echo ""
echo "=== Experiment B complete. Results in $RESULTS_DIR ==="
