#!/usr/bin/env bash
# Run Experiment A on a single model.
#
# Usage (vLLM mode — spawns a local vLLM server):
#   ./scripts/run_exp_a_single.sh <config> [OPTIONS]
#
# Usage (frontier mode — routes through the BiFrost gateway):
#   ./scripts/run_exp_a_single.sh --frontier-model <provider/name> [OPTIONS]
#
# Arguments:
#   <config>                       Path to model YAML config (relative to project root or absolute).
#                                  e.g. configs/models_exp_a/gemma4_31b.yaml  [vLLM mode only]
#   --frontier-model <provider/name>  provider/model_name from bifrost/config.json,
#                                  e.g. openai/gpt-5, anthropic/claude-sonnet-4-6,
#                                       gemini/gemini-2.5-flash
#
# Options:
#   --kv-cache-dtype <dtype>  KV cache quantization dtype (vLLM mode, default: auto)
#   --data <path>             Benchmark data directory (default: data/output/benchmark/task_a)
#   --max-samples <n>         Limit to first N samples (0=all; frontier default: 50)
#   --stochastic-trials <n>   Override stochastic trial count from YAML
#   --max-model-len <n>       Override serving.max_model_len from YAML [vLLM mode only]
#   --max-num-seqs <n>        Cap max concurrent requests [vLLM mode only]
#   --dry-run                 Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_a"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"
BIFROST_CONFIG="$PROJECT_ROOT/deployments/local/data/bifrost/config.json"
FRONTIER_CONFIG="$PROJECT_ROOT/configs/models_exp_a/frontier.yaml"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

FRONTIER_MODEL=""
CONFIG_ARG=""
KV_CACHE_DTYPE="auto"
DATA_DIR="$PROJECT_ROOT/data/output/benchmark/task_a"
MAX_SAMPLES=0
MAX_SAMPLES_SET=false
MAX_MODEL_LEN=""
MAX_NUM_SEQS=""
STOCHASTIC_TRIALS_OVERRIDE=""
DRY_RUN=false

# First positional arg (if not a flag) is the vLLM config path
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG_ARG="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frontier-model)    FRONTIER_MODEL="$2";                    shift 2 ;;
        --kv-cache-dtype)    KV_CACHE_DTYPE="$2";                    shift 2 ;;
        --data)              DATA_DIR="$2";                           shift 2 ;;
        --max-samples)       MAX_SAMPLES="$2"; MAX_SAMPLES_SET=true;  shift 2 ;;
        --stochastic-trials) STOCHASTIC_TRIALS_OVERRIDE="$2";         shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2";                      shift 2 ;;
        --max-num-seqs)      MAX_NUM_SEQS="$2";                       shift 2 ;;
        --dry-run)           DRY_RUN=true;                            shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Validate: exactly one of positional config or --frontier-model must be given
if [[ -n "$FRONTIER_MODEL" && -n "$CONFIG_ARG" ]]; then
    echo "ERROR: --frontier-model and a positional config are mutually exclusive." >&2
    exit 1
fi
if [[ -z "$FRONTIER_MODEL" && -z "$CONFIG_ARG" ]]; then
    echo "Usage: $0 <config> [OPTIONS]" >&2
    echo "       $0 --frontier-model <name> [OPTIONS]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve config, engine, endpoint, and model name
# ---------------------------------------------------------------------------

if [[ -n "$FRONTIER_MODEL" ]]; then
    MODEL_CONFIG="$FRONTIER_CONFIG"
else
    if [[ "$CONFIG_ARG" = /* ]]; then
        MODEL_CONFIG="$CONFIG_ARG"
    else
        MODEL_CONFIG="$PROJECT_ROOT/$CONFIG_ARG"
    fi
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: Config not found: $MODEL_CONFIG" >&2
    exit 1
fi

SERVING_ENGINE=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('serving', {}).get('engine', 'vllm'))
" "$MODEL_CONFIG")

SERVING_ENDPOINT=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('serving', {}).get('endpoint', 'http://localhost:8000'))
" "$MODEL_CONFIG")

if [[ -n "$FRONTIER_MODEL" ]]; then
    # Validate the requested model is declared in the BiFrost config
    if [[ ! -f "$BIFROST_CONFIG" ]]; then
        echo "ERROR: BiFrost config not found: $BIFROST_CONFIG" >&2
        exit 1
    fi
    ALLOWED_MODELS=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    c = json.load(f)
models = []
for provider_name, provider in c.get('providers', {}).items():
    for key_entry in provider.get('keys', []):
        models.extend(f'{provider_name}/{m}' for m in key_entry.get('models', []))
print('\n'.join(sorted(set(models))))
" "$BIFROST_CONFIG")

    if ! printf '%s\n' "$ALLOWED_MODELS" | grep -qx "$FRONTIER_MODEL"; then
        echo "ERROR: Model '$FRONTIER_MODEL' is not configured in $BIFROST_CONFIG." >&2
        echo "Available frontier models:" >&2
        printf '%s\n' "$ALLOWED_MODELS" | sed 's/^/  /' >&2
        exit 1
    fi
    MODEL_NAME="$FRONTIER_MODEL"
else
    MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")
fi

# Stochastic trial count: CLI override → YAML → fallback
if [[ -n "$STOCHASTIC_TRIALS_OVERRIDE" ]]; then
    STOCHASTIC_TRIALS="$STOCHASTIC_TRIALS_OVERRIDE"
else
    STOCHASTIC_TRIALS=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('inference', {}).get('stochastic_trials', 5))
" "$MODEL_CONFIG")
fi

# Frontier cost guardrail: apply max_samples_default when user didn't pass --max-samples
if [[ "$SERVING_ENGINE" == "bifrost" && "$MAX_SAMPLES_SET" == "false" ]]; then
    MAX_SAMPLES=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('inference', {}).get('max_samples_default', 50))
" "$MODEL_CONFIG")
fi

mkdir -p "$RESULTS_DIR"

echo "=== Experiment A (single model): Prompt-Encoded Business Logic ==="
echo "Config:         $MODEL_CONFIG"
echo "Model:          $MODEL_NAME"
echo "Engine:         $SERVING_ENGINE"
echo "Endpoint:       $SERVING_ENDPOINT"
if [[ "$SERVING_ENGINE" != "bifrost" ]]; then
    echo "KV cache dtype: $KV_CACHE_DTYPE"
fi
echo "Data dir:       $DATA_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max samples:    ${MAX_SAMPLES} (0=all)"
echo "Stoch trials:   $STOCHASTIC_TRIALS"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Result file path
# ---------------------------------------------------------------------------

if [[ "$SERVING_ENGINE" == "bifrost" ]]; then
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_frontier.json"
else
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${KV_CACHE_DTYPE}.json"
fi

# ---------------------------------------------------------------------------
# Dry-run exit point
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = true ]; then
    if [[ "$SERVING_ENGINE" == "bifrost" ]]; then
        echo "[DRY RUN] Would run frontier benchmark: model=$MODEL_NAME endpoint=$SERVING_ENDPOINT"
    else
        DRY_LAUNCH_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
        [ -n "$MAX_MODEL_LEN" ] && DRY_LAUNCH_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
        [ -n "$MAX_NUM_SEQS" ]  && DRY_LAUNCH_ARGS+=(--max-num-seqs  "$MAX_NUM_SEQS")
        echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG ${DRY_LAUNCH_ARGS[*]}"
    fi
    echo "[DRY RUN] Would write results to $RESULT_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Server lifecycle (vLLM only — skipped for frontier/bifrost)
# ---------------------------------------------------------------------------

if [[ "$SERVING_ENGINE" != "bifrost" ]]; then
    LAUNCH_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [ -n "$MAX_MODEL_LEN" ] && LAUNCH_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    [ -n "$MAX_NUM_SEQS" ]  && LAUNCH_ARGS+=(--max-num-seqs  "$MAX_NUM_SEQS")

    bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" "${LAUNCH_ARGS[@]}" &
    VLLM_PID=$!

    cleanup() {
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "Server stopped."
    }
    trap cleanup EXIT

    # Wait for server to be ready (poll health endpoint).
    # 900s covers hybrid MoE cold start: weight load + torch.compile + optional JIT.
    echo "Waiting for vLLM server (PID $VLLM_PID)..."
    for i in $(seq 1 180); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after $((i * 5))s"
            break
        fi
        sleep 5
    done
    if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "ERROR: vLLM server failed to start within 900s for $MODEL_NAME" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

python3 -m llm_workflow_agents.eval.agent_benchmark \
    --model             "$MODEL_NAME" \
    --endpoint          "$SERVING_ENDPOINT" \
    --engine            "$SERVING_ENGINE" \
    --kv-cache-dtype    "$KV_CACHE_DTYPE" \
    --output            "$RESULT_FILE" \
    --data              "$DATA_DIR" \
    --max-samples       "$MAX_SAMPLES" \
    --stochastic-trials "$STOCHASTIC_TRIALS" \
    --log-level         DEBUG \
    2>&1 | tee "${RESULT_FILE%.json}.log" || true

echo ""
echo "=== Done. Results in $RESULT_FILE ==="
