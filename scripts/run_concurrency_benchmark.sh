#!/usr/bin/env bash
# Run concurrency and latency benchmark for a single Category A model.
#
# Launches a vLLM server for the given config, then sweeps the requested
# concurrency levels at each context length, measures TTFT / TPOT / ITL /
# throughput, and reports the maximum concurrency that stays within the
# degradation envelope (TTFT p95 ≤ 2× baseline AND failure rate ≤ 1%).
#
# Usage:
#   ./scripts/run_concurrency_benchmark.sh <config> [OPTIONS]
#
# Arguments:
#   <config>                Path to Cat A model YAML (relative to project root
#                           or absolute).
#                           e.g. configs/models/cat_a/qwen3_32b.yaml
#
# Options:
#   --kv-cache-dtype <d>    KV cache quantization dtype (default: auto)
#   --context-lengths <cs>  Comma-separated context lengths to sweep
#                           (default: 2048,4096,8192)
#   --input-tokens <n>      Approximate input prompt tokens (default: 512)
#   --output-tokens <n>     max_tokens for each request (default: 128)
#   --concurrency-levels    Comma-separated concurrency levels to sweep
#                           (default: 1,2,4,8,16,32,64,128,256,512,1024)
#   --requests-per-level    Number of requests issued per concurrency level
#                           (default: 64; warmup requests are additional)
#   --warmup-requests <n>   Discarded warm-up requests per level (default: 8)
#   --degradation-ttft-mul  TTFT p95 threshold multiplier vs. baseline
#                           (default: 2.0; set higher for more lenient)
#   --max-failure-rate <f>  Maximum allowed request failure rate 0.0–1.0
#                           (default: 0.01 = 1%)
#   --port <p>              vLLM server port (default: 8000)
#   --results-dir <path>    Output directory (default: results/concurrency)
#   --dry-run               Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_SCRIPT="$PROJECT_ROOT/serving/launch_vllm.sh"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <config> [OPTIONS]"
    exit 1
fi

CONFIG_ARG="$1"
shift

# Defaults
KV_CACHE_DTYPE="auto"
CONTEXT_LENGTHS="2048,4096,8192"
INPUT_TOKENS=512
OUTPUT_TOKENS=128
CONCURRENCY_LEVELS="1,2,4,8,16,32,64,128,256,512,1024"
REQUESTS_PER_LEVEL=64
WARMUP_REQUESTS=8
DEGRADATION_TTFT_MUL=2.0
MAX_FAILURE_RATE=0.01
PORT=8000
RESULTS_DIR="$PROJECT_ROOT/results/concurrency"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-cache-dtype)       KV_CACHE_DTYPE="$2";        shift 2 ;;
        --context-lengths)      CONTEXT_LENGTHS="$2";       shift 2 ;;
        --input-tokens)         INPUT_TOKENS="$2";          shift 2 ;;
        --output-tokens)        OUTPUT_TOKENS="$2";         shift 2 ;;
        --concurrency-levels)   CONCURRENCY_LEVELS="$2";    shift 2 ;;
        --requests-per-level)   REQUESTS_PER_LEVEL="$2";    shift 2 ;;
        --warmup-requests)      WARMUP_REQUESTS="$2";       shift 2 ;;
        --degradation-ttft-mul) DEGRADATION_TTFT_MUL="$2"; shift 2 ;;
        --max-failure-rate)     MAX_FAILURE_RATE="$2";      shift 2 ;;
        --port)                 PORT="$2";                  shift 2 ;;
        --results-dir)          RESULTS_DIR="$2";           shift 2 ;;
        --dry-run)              DRY_RUN=true;               shift   ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Resolve config path
if [[ "$CONFIG_ARG" = /* ]]; then
    MODEL_CONFIG="$CONFIG_ARG"
else
    MODEL_CONFIG="$PROJECT_ROOT/$CONFIG_ARG"
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: Config not found: $MODEL_CONFIG" >&2
    exit 1
fi

# Extract model name from YAML
MODEL_NAME=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c['model']['name'])
" "$MODEL_CONFIG")

# Compute required max_model_len: max context length + output tokens + small buffer
MAX_CTX=$(python3 -c "
levels = [int(x) for x in '${CONTEXT_LENGTHS}'.split(',')]
print(max(levels) + ${OUTPUT_TOKENS} + 64)
")

RESULT_FILE="$RESULTS_DIR/${MODEL_NAME//\//_}_${KV_CACHE_DTYPE}.json"
LOG_FILE="${RESULT_FILE%.json}.log"

mkdir -p "$RESULTS_DIR"

echo "=== Concurrency Benchmark: Prompt-Encoded Business Logic ==="
echo "Config:              $MODEL_CONFIG"
echo "Model:               $MODEL_NAME"
echo "KV cache dtype:      $KV_CACHE_DTYPE"
echo "Context lengths:     $CONTEXT_LENGTHS"
echo "Input tokens:        $INPUT_TOKENS"
echo "Output tokens:       $OUTPUT_TOKENS"
echo "Concurrency levels:  $CONCURRENCY_LEVELS"
echo "Requests per level:  $REQUESTS_PER_LEVEL  (+$WARMUP_REQUESTS warmup)"
echo "Degradation rule:    TTFT p95 <= ${DEGRADATION_TTFT_MUL}x baseline  OR  failure rate > $MAX_FAILURE_RATE"
echo "Min max_model_len:   $MAX_CTX"
echo "Port:                $PORT"
echo "Results dir:         $RESULTS_DIR"
echo "==========================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would launch: bash $LAUNCH_SCRIPT $MODEL_CONFIG --kv-cache-dtype $KV_CACHE_DTYPE --port $PORT --max-model-len $MAX_CTX"
    echo "[DRY RUN] Would run concurrency sweep for $MODEL_NAME"
    echo "[DRY RUN] Output: $RESULT_FILE"
    exit 0
fi

# Launch vLLM server in background
bash "$LAUNCH_SCRIPT" "$MODEL_CONFIG" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --port "$PORT" \
    --max-model-len "$MAX_CTX" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "Server stopped."
}
trap cleanup EXIT

# Poll health endpoint (90 × 5s = 450s budget for large models)
BASE_URL="http://localhost:${PORT}"
echo "Waiting for vLLM server (PID $VLLM_PID) at $BASE_URL ..."
for i in $(seq 1 90); do
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        break
    fi
    sleep 5
done
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 450s for $MODEL_NAME" >&2
    exit 1
fi

# Run concurrency sweep
python3 -m llm_workflow_agents.eval.concurrency_benchmark \
    --model                    "$MODEL_NAME" \
    --kv-cache-dtype           "$KV_CACHE_DTYPE" \
    --base-url                 "${BASE_URL}/v1" \
    --context-lengths          "$CONTEXT_LENGTHS" \
    --input-tokens             "$INPUT_TOKENS" \
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
