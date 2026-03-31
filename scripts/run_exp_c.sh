#!/usr/bin/env bash
# Run Experiment C: Graph extraction evaluation.
#
# Trains each 2-5B model as a graph extractor with LoRA, launches with vLLM
# using constrained JSON decoding, runs graph extraction evaluation, and
# saves results to results/exp_c/.
#
# Usage:
#   ./scripts/run_exp_c.sh [--skip-training] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_c"
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

echo "=== Experiment C: Graph Extraction ==="
echo "Skip training: $SKIP_TRAINING"
echo "Results dir:   $RESULTS_DIR"
echo "======================================"

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
        echo "[DRY RUN] Would train graph extractor and eval $MODEL_NAME"
        continue
    fi

    # Train graph extractor
    if [ "$SKIP_TRAINING" = false ]; then
        echo "Training graph extractor $MODEL_NAME..."
        python3 -m llm_workflow_agents.training.train_graph_extractor \
            --config "$MODEL_CONFIG" \
            --output-dir "checkpoints/exp_c/${MODEL_NAME//\//_}" 2>&1 | \
            tee "$RESULTS_DIR/${MODEL_NAME//\//_}_train.log" || true
    fi

    # Launch vLLM server
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

    # Run graph extraction evaluation (with constrained JSON decoding)
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}.json"
    python3 -m llm_workflow_agents.eval.graph_extraction_eval \
        --model "$MODEL_NAME" \
        --use-constrained-decoding \
        --output "$RESULT_FILE" 2>&1 | tee "${RESULT_FILE%.json}.log" || true

    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "Server stopped."
done

echo ""
echo "=== Experiment C complete. Results in $RESULTS_DIR ==="
