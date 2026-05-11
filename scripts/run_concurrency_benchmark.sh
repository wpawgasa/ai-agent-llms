#!/usr/bin/env bash
# Run concurrency and latency benchmark for a single model.
#
# Three local modes (vLLM, SGLang, TensorRT-LLM) plus BiFrost frontier:
#   Local:      Spawns a server from a YAML config (engine read from YAML).
#   Frontier:   Routes requests through the BiFrost LLM gateway (no local
#               server lifecycle; gateway is assumed already running).
#
# In both modes, sweeps the requested concurrency levels at each context
# length, measures TTFT / TPOT / ITL / throughput, and reports the
# maximum concurrency that stays within the degradation envelope
# (TTFT p95 ≤ 2× baseline AND failure rate ≤ 1%).
#
# Usage (local server mode — vLLM, SGLang, or TensorRT-LLM from YAML):
#   ./scripts/run_concurrency_benchmark.sh <config> [OPTIONS]
#
# Usage (frontier mode — routes through BiFrost):
#   ./scripts/run_concurrency_benchmark.sh --frontier-model <provider/name> [OPTIONS]
#
# Arguments:
#   <config>                          Path to model YAML (relative to project root
#                                     or absolute). Backend is read from serving.engine.
#                                     e.g. configs/models_exp_a/qwen3_32b_sglang.yaml
#   --frontier-model <provider/name>  provider/model_name from bifrost/config.json,
#                                     e.g. openai/gpt-5, anthropic/claude-sonnet-4-6,
#                                          gemini/gemini-2.5-flash
#
# Options:
#   --kv-cache-dtype <d>    KV cache quantization dtype (local server only;
#                           valid values are backend-specific; default: auto)
#   --context-lengths <cs>  Comma-separated context lengths to sweep
#                           (default: 2048,4096,8192)
#   --input-tokens-min <n>  Minimum input prompt tokens, inclusive (default: 512)
#   --input-tokens-max <n>  Maximum input prompt tokens, inclusive (default: 2048)
#                           Each request gets a unique prompt drawn uniformly from
#                           [min, max] to defeat provider prefix caching.
#   --output-tokens <n>     max_tokens for each request (default: 128)
#   --concurrency-levels    Comma-separated concurrency levels to sweep
#                           (default vLLM:     1,2,4,8,16,32,64,128,256,512,1024)
#                           (default frontier: 1,2,4,8,16,32 — provider rate-limit aware)
#   --requests-per-level    Requests per concurrency level
#                           (default vLLM: 64; frontier: 16 — cost guardrail)
#   --warmup-requests <n>   Discarded warm-up requests per level (default: 8)
#   --degradation-ttft-mul  TTFT p95 threshold multiplier vs. baseline (default: 2.0)
#   --max-failure-rate <f>  Maximum allowed request failure rate (default: 0.01)
#   --port <p>              Server port (local server mode only; default: 8000)
#   --results-dir <path>    Output directory (default: results/concurrency)
#   --dry-run               Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch.sh"
BIFROST_CONFIG="$PROJECT_ROOT/deployments/local/data/bifrost/config.json"
FRONTIER_CONFIG="$PROJECT_ROOT/configs/models_exp_a/frontier.yaml"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <config> [OPTIONS]"
    echo "       $0 --frontier-model <provider/name> [OPTIONS]"
    exit 1
fi

# First positional arg (if not a flag) is the vLLM config path
CONFIG_ARG=""
if [[ "$1" != --* ]]; then
    CONFIG_ARG="$1"
    shift
fi

# Defaults — frontier-mode overrides applied below if --frontier-model is set
FRONTIER_MODEL=""
KV_CACHE_DTYPE="auto"
CONTEXT_LENGTHS="2048"
INPUT_TOKENS_MIN=512
INPUT_TOKENS_MAX=2048
OUTPUT_TOKENS=128
CONCURRENCY_LEVELS=""
CONCURRENCY_LEVELS_SET=false
REQUESTS_PER_LEVEL=""
REQUESTS_PER_LEVEL_SET=false
WARMUP_REQUESTS=8
DEGRADATION_TTFT_MUL=2.0
MAX_FAILURE_RATE=0.01
PORT=8000
RESULTS_DIR="$PROJECT_ROOT/results/concurrency"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frontier-model)       FRONTIER_MODEL="$2";        shift 2 ;;
        --kv-cache-dtype)       KV_CACHE_DTYPE="$2";        shift 2 ;;
        --context-lengths)      CONTEXT_LENGTHS="$2";       shift 2 ;;
        --input-tokens-min)     INPUT_TOKENS_MIN="$2";      shift 2 ;;
        --input-tokens-max)     INPUT_TOKENS_MAX="$2";      shift 2 ;;
        --output-tokens)        OUTPUT_TOKENS="$2";         shift 2 ;;
        --concurrency-levels)   CONCURRENCY_LEVELS="$2"; CONCURRENCY_LEVELS_SET=true; shift 2 ;;
        --requests-per-level)   REQUESTS_PER_LEVEL="$2"; REQUESTS_PER_LEVEL_SET=true; shift 2 ;;
        --warmup-requests)      WARMUP_REQUESTS="$2";       shift 2 ;;
        --degradation-ttft-mul) DEGRADATION_TTFT_MUL="$2"; shift 2 ;;
        --max-failure-rate)     MAX_FAILURE_RATE="$2";      shift 2 ;;
        --port)                 PORT="$2";                  shift 2 ;;
        --results-dir)          RESULTS_DIR="$2";           shift 2 ;;
        --dry-run)              DRY_RUN=true;               shift   ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Mutual exclusion: --frontier-model XOR positional config
if [[ -n "$FRONTIER_MODEL" && -n "$CONFIG_ARG" ]]; then
    echo "ERROR: --frontier-model and a positional config are mutually exclusive." >&2
    exit 1
fi
if [[ -z "$FRONTIER_MODEL" && -z "$CONFIG_ARG" ]]; then
    echo "Usage: $0 <config> [OPTIONS]" >&2
    echo "       $0 --frontier-model <provider/name> [OPTIONS]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve config + serving engine + model name
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
    BASE_URL="$SERVING_ENDPOINT"
    KV_CACHE_DTYPE="remote"
else
    MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")
    BASE_URL="http://localhost:${PORT}"
fi

# ---------------------------------------------------------------------------
# Mode-specific defaults (only applied when user didn't override)
# ---------------------------------------------------------------------------

case "$SERVING_ENGINE" in
    bifrost)
        # Frontier defaults: cap concurrency to provider-friendly range and reduce
        # request volume to control cost. ~3 ctx × 6 levels × 16 reqs ≈ 288 calls/model.
        [[ "$CONCURRENCY_LEVELS_SET" == "false" ]] && CONCURRENCY_LEVELS="1,2,4,8,16,32"
        [[ "$REQUESTS_PER_LEVEL_SET" == "false" ]] && REQUESTS_PER_LEVEL=16
        ;;
    vllm|sglang|tensorrt_llm)
        [[ "$CONCURRENCY_LEVELS_SET" == "false" ]] && CONCURRENCY_LEVELS="1,2,4,8,16,32,64,128,256,512,1024"
        [[ "$REQUESTS_PER_LEVEL_SET" == "false" ]] && REQUESTS_PER_LEVEL=64
        ;;
    *)
        echo "ERROR: Unknown serving.engine '$SERVING_ENGINE'" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Result file path + summary banner
# ---------------------------------------------------------------------------

if [[ "$SERVING_ENGINE" == "bifrost" ]]; then
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_frontier.json"
else
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${SERVING_ENGINE}_${KV_CACHE_DTYPE}.json"
fi
LOG_FILE="${RESULT_FILE%.json}.log"

mkdir -p "$RESULTS_DIR"

echo "=== Concurrency Benchmark: Prompt-Encoded Business Logic ==="
echo "Config:              $MODEL_CONFIG"
echo "Model:               $MODEL_NAME"
echo "Engine:              $SERVING_ENGINE"
echo "Endpoint:            $BASE_URL"
if [[ "$SERVING_ENGINE" != "bifrost" ]]; then
    echo "KV cache dtype:      $KV_CACHE_DTYPE"
fi
echo "Context lengths:     $CONTEXT_LENGTHS"
echo "Input tokens:        ${INPUT_TOKENS_MIN}..${INPUT_TOKENS_MAX} (varied per request)"
echo "Output tokens:       $OUTPUT_TOKENS"
echo "Concurrency levels:  $CONCURRENCY_LEVELS"
echo "Requests per level:  $REQUESTS_PER_LEVEL  (+$WARMUP_REQUESTS warmup)"
echo "Degradation rule:    TTFT p95 <= ${DEGRADATION_TTFT_MUL}x baseline  OR  failure rate > $MAX_FAILURE_RATE"
echo "Results dir:         $RESULTS_DIR"
echo "==========================================================="

# vLLM-only: compute required max_model_len
if [[ "$SERVING_ENGINE" != "bifrost" ]]; then
    MAX_CTX=$(python3 -c "
levels = [int(x) for x in '${CONTEXT_LENGTHS}'.split(',')]
print(max(levels) + ${OUTPUT_TOKENS} + 64)
")
    echo "Min max_model_len:   $MAX_CTX"
    echo "Port:                $PORT"
fi

if [ "$DRY_RUN" = true ]; then
    if [[ "$SERVING_ENGINE" == "bifrost" ]]; then
        echo "[DRY RUN] Would run frontier sweep against: $BASE_URL"
    else
        echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG --kv-cache-dtype $KV_CACHE_DTYPE --port $PORT --max-model-len $MAX_CTX (engine=$SERVING_ENGINE)"
    fi
    echo "[DRY RUN] Output: $RESULT_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Server lifecycle (vLLM only — BiFrost gateway is managed externally)
# ---------------------------------------------------------------------------

if [[ "$SERVING_ENGINE" != "bifrost" ]]; then
    bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --port "$PORT" \
        --max-model-len "$MAX_CTX" &
    SERVER_PID=$!

    cleanup() {
        # Process-group kill handles multi-process backends (SGLang router+workers,
        # TRT-LLM mpirun children).
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    }
    trap cleanup EXIT

    # TRT-LLM JIT-from-HF compiles the engine on first launch (5-15 min for
    # 30B-class models). Allow up to 30 min; vLLM/SGLang keep the 450s budget.
    if [[ "$SERVING_ENGINE" == "tensorrt_llm" ]]; then
        HEALTH_ITERS=360
        WARMUP_REQUESTS=32   # absorb TRT-LLM prefill warm-up bias
    else
        HEALTH_ITERS=90
    fi

    echo "Waiting for $SERVING_ENGINE server (PID $SERVER_PID) at $BASE_URL ..."
    for i in $(seq 1 $HEALTH_ITERS); do
        if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
            echo "Server ready after $((i * 5))s"
            break
        fi
        sleep 5
    done
    if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "ERROR: $SERVING_ENGINE server failed to start for $MODEL_NAME" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Run concurrency sweep
# ---------------------------------------------------------------------------

python3 -m llm_workflow_agents.eval.concurrency_benchmark \
    --model                    "$MODEL_NAME" \
    --engine                   "$SERVING_ENGINE" \
    --kv-cache-dtype           "$KV_CACHE_DTYPE" \
    --base-url                 "${BASE_URL}" \
    --context-lengths          "$CONTEXT_LENGTHS" \
    --input-tokens-min         "$INPUT_TOKENS_MIN" \
    --input-tokens-max         "$INPUT_TOKENS_MAX" \
    --output-tokens            "$OUTPUT_TOKENS" \
    --concurrency-levels       "$CONCURRENCY_LEVELS" \
    --requests-per-level       "$REQUESTS_PER_LEVEL" \
    --warmup-requests          "$WARMUP_REQUESTS" \
    --degradation-ttft-multiplier "$DEGRADATION_TTFT_MUL" \
    --max-failure-rate         "$MAX_FAILURE_RATE" \
    --output                   "$RESULT_FILE" \
    2>&1 | tee "$LOG_FILE" || true

echo ""
echo "=== Done. Results in $RESULT_FILE ==="
