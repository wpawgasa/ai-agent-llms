#!/usr/bin/env bash
# Run Experiment A: Prompt-encoded business logic evaluation.
#
# Launches each 15-30B model via the engine dispatcher (serving/launch.sh),
# runs workflow quality evaluation, and saves results to results/exp_a/.
#
# Usage:
#   ./scripts/run_exp_a.sh [OPTIONS]
#
# Options:
#   --kv-cache-dtype  KV cache quantization dtype (default: auto)
#   --backend <b>     Serving backend: vllm (default), sglang, tensorrt_llm.
#                     When set, uses <model>_<backend>.yaml sibling configs.
#   --data <path>     Benchmark data directory (default: data/output/benchmark/task_a)
#   --max-samples <n> Limit to first N samples per level, 0=all (default: 0)
#   --dry-run         Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_a"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch.sh"

KV_CACHE_DTYPE="auto"
BACKEND=""
DATA_DIR="$PROJECT_ROOT/data/output/benchmark/task_a"
MAX_SAMPLES=0
DRY_RUN=false
SPEC_METHOD=""
SPEC_DRAFT_MODEL=""
SPEC_NUM_TOKENS=""
NO_SPECULATIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-cache-dtype)          KV_CACHE_DTYPE="$2";    shift 2 ;;
        --backend)                 BACKEND="$2";            shift 2 ;;
        --data)                    DATA_DIR="$2";           shift 2 ;;
        --max-samples)             MAX_SAMPLES="$2";        shift 2 ;;
        --dry-run)                 DRY_RUN=true;            shift   ;;
        --speculative-method)      SPEC_METHOD="$2";        shift 2 ;;
        --speculative-draft-model) SPEC_DRAFT_MODEL="$2";  shift 2 ;;
        --speculative-num-tokens)  SPEC_NUM_TOKENS="$2";   shift 2 ;;
        --no-speculative)          NO_SPECULATIVE=true;     shift   ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build forwarding args array and result-file suffix for A/B base-vs-spec comparisons.
SPEC_ARGS=()
SPEC_SUFFIX=""
if [ "$NO_SPECULATIVE" = "true" ]; then
    SPEC_ARGS+=(--no-speculative)
    SPEC_SUFFIX="_spec-off"
else
    if [ -n "$SPEC_METHOD" ]; then
        SPEC_ARGS+=(--speculative-method "$SPEC_METHOD")
        SPEC_SUFFIX="_spec-${SPEC_METHOD}"
    fi
    [ -n "$SPEC_DRAFT_MODEL" ] && SPEC_ARGS+=(--speculative-draft-model "$SPEC_DRAFT_MODEL")
    [ -n "$SPEC_NUM_TOKENS" ]  && SPEC_ARGS+=(--speculative-num-tokens  "$SPEC_NUM_TOKENS")
fi

mkdir -p "$RESULTS_DIR"

# Base model config list (vLLM default, unchanged)
MODEL_CONFIGS=(
    "configs/models_exp_a/gemma3_27b.yaml"
    "configs/models_exp_a/qwen3_32b.yaml"
    "configs/models_exp_a/qwen35_35b_a3b.yaml"
    "configs/models_exp_a/mistral_24b.yaml"
    "configs/models_exp_a/nemotron_30b.yaml"
    "configs/models_exp_a/glm47_flash.yaml"
    "configs/models_exp_a/gemma4_26b_a4b.yaml"
    "configs/models_exp_a/gemma4_31b.yaml"
    "configs/models_exp_a/qwen36_27b.yaml"
    "configs/models_exp_a/qwen36_27b_fp8.yaml"
)

# When --backend is set, swap each config to its <model>_<backend>.yaml sibling.
if [[ -n "$BACKEND" ]]; then
    case "$BACKEND" in
        vllm|sglang|tensorrt_llm) ;;
        *)
            echo "ERROR: --backend must be one of: vllm, sglang, tensorrt_llm" >&2
            exit 1
            ;;
    esac
    if [[ "$BACKEND" != "vllm" ]]; then
        SWAPPED=()
        for cfg in "${MODEL_CONFIGS[@]}"; do
            sibling="${cfg%.yaml}_${BACKEND}.yaml"
            if [[ -f "$PROJECT_ROOT/$sibling" ]]; then
                SWAPPED+=("$sibling")
            else
                echo "WARN: sibling config not found, skipping: $sibling" >&2
            fi
        done
        MODEL_CONFIGS=("${SWAPPED[@]}")
    fi
fi

echo "=== Experiment A: Prompt-Encoded Business Logic ==="
echo "KV cache dtype: $KV_CACHE_DTYPE"
echo "Backend:        ${BACKEND:-vllm (default)}"
echo "Spec Decoding:  ${SPEC_SUFFIX:-none}"
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

    SERVING_ENGINE=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('serving', {}).get('engine', 'vllm'))
" "$MODEL_CONFIG")

    SKIP_REASON=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('serving', {}).get('skip_reason', ''))
" "$MODEL_CONFIG")

    STOCHASTIC_TRIALS=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('inference', {}).get('stochastic_trials', 5))
" "$MODEL_CONFIG")

    echo ""
    echo "--- Model: $MODEL_NAME (engine: $SERVING_ENGINE) ---"

    if [[ -n "$SKIP_REASON" ]]; then
        echo "SKIP: $SKIP_REASON"
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG --kv-cache-dtype $KV_CACHE_DTYPE ${SPEC_ARGS[*]}"
        echo "[DRY RUN] Would run eval for $MODEL_NAME (engine=$SERVING_ENGINE)"
        continue
    fi

    # Launch server in background (dispatch to the correct backend via launch.sh)
    bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" --kv-cache-dtype "$KV_CACHE_DTYPE" "${SPEC_ARGS[@]}" &
    SERVER_PID=$!

    # TRT-LLM JIT-from-HF can take 5-15 min on first launch; allow 30 min.
    if [[ "$SERVING_ENGINE" == "tensorrt_llm" ]]; then
        HEALTH_ITERS=360
    else
        HEALTH_ITERS=60
    fi

    echo "Waiting for $SERVING_ENGINE server (PID $SERVER_PID)..."
    for i in $(seq 1 $HEALTH_ITERS); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after $((i * 5))s"
            break
        fi
        sleep 5
    done
    if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "ERROR: $SERVING_ENGINE server failed to start for $MODEL_NAME" >&2
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

    # Run evaluation
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${SERVING_ENGINE}_${KV_CACHE_DTYPE}${SPEC_SUFFIX}.json"
    python3 -m llm_workflow_agents.eval.agent_benchmark \
        --model             "$MODEL_NAME" \
        --engine            "$SERVING_ENGINE" \
        --output            "$RESULT_FILE" \
        --data              "$DATA_DIR" \
        --max-samples       "$MAX_SAMPLES" \
        --stochastic-trials "$STOCHASTIC_TRIALS" \
        --log-level         DEBUG \
        2>&1 | tee "${RESULT_FILE%.json}.log" || true

    # Shut down server (process-group kill handles multi-process backends)
    kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Server stopped."
done

echo ""
echo "=== Experiment A complete. Results in $RESULTS_DIR ==="
