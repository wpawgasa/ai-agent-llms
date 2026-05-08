#!/usr/bin/env bash
# Launch an SGLang OpenAI-compatible API server from a model config YAML.
#
# Usage (via dispatcher — preferred):
#   ./serving/launch.sh <model_config_sglang.yaml> [--kv-cache-dtype <dtype>]
#                       [--port <port>] [--max-model-len <n>]
#
# Direct usage:
#   ./serving/launch_sglang.sh <model_config_sglang.yaml> [OPTIONS]
#
# Reads from YAML config (serving.*):
#   model.name              → --model-path
#   serving.tool_call_parser → --tool-call-parser
#   serving.chat_template   → --chat-template
#   serving.mem_fraction_static (default 0.90) → --mem-fraction-static
#   serving.context_length  (default 8192)     → --context-length
#   serving.max_running_requests (default 256) → --max-running-requests
#   serving.attention_backend (default triton)  → --attention-backend
#   serving.enable_torch_compile (default false) → --enable-torch-compile
#   serving.port            (default 8000)     → --port (overridable via CLI)
#   serving.kv_cache_dtype  (default auto)     → --kv-cache-dtype
#
# SGLang tool-call parser vocabulary differs from vLLM. Mapping for reference:
#   vLLM: hermes       → SGLang: qwen25
#   vLLM: qwen3_coder  → SGLang: qwen25
#   vLLM: mistral      → SGLang: mistral
#   vLLM: gemma/gemma4 → SGLang: pythonic  (no dedicated Gemma parser)
#   vLLM: glm4         → SGLang: pythonic  (no dedicated GLM parser)
#   vLLM: nemotron     → SGLang: pythonic  (no dedicated Nemotron parser)
#   vLLM: pythonic     → SGLang: pythonic  (same)
#
# KV-cache dtype vocabulary (SGLang):
#   auto, fp8_e5m2, fp8_e4m3
#   "fp8" is treated as an alias for "fp8_e5m2".
#   vLLM-specific dtypes (turboquant_*, rotorquant_*, kivi_*, kvquant*) are
#   rejected with a clear error — use the vLLM path for those.

set -euo pipefail

LAUNCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_PROJECT_ROOT="$(cd "$LAUNCH_SCRIPT_DIR/.." && pwd)"

# Activate the SGLang venv if no venv is currently active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$LAUNCH_PROJECT_ROOT/.venv-sglang/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LAUNCH_PROJECT_ROOT/.venv-sglang/bin/activate"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_config_sglang.yaml> [--kv-cache-dtype <dtype>] [--port <port>]"
    exit 1
fi

CONFIG_FILE="$1"
shift

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck source=serving/_yaml_helper.sh
source "$LAUNCH_SCRIPT_DIR/_yaml_helper.sh"

MODEL_NAME=$(parse_yaml "model.name" "")
TOOL_PARSER=$(parse_yaml "serving.tool_call_parser" "pythonic")
CHAT_TEMPLATE=$(parse_yaml "serving.chat_template" "")
MEM_FRACTION=$(parse_yaml "serving.mem_fraction_static" "0.90")
CONTEXT_LEN=$(parse_yaml "serving.context_length" "8192")
MAX_RUNNING=$(parse_yaml "serving.max_running_requests" "256")
ATTN_BACKEND=$(parse_yaml "serving.attention_backend" "triton")
TORCH_COMPILE=$(parse_yaml "serving.enable_torch_compile" "false")
PORT=$(parse_yaml "serving.port" "8000")
KV_CACHE_DTYPE=$(parse_yaml "serving.kv_cache_dtype" "auto")

if [ -z "$MODEL_NAME" ]; then
    echo "Error: model.name not found in config" >&2
    exit 1
fi

MAX_MODEL_LEN_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --kv-cache-dtype) KV_CACHE_DTYPE="$2";         shift 2 ;;
        --port)           PORT="$2";                   shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN_OVERRIDE="$2"; shift 2 ;;
        --max-num-seqs)   MAX_RUNNING="$2";            shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -n "$MAX_MODEL_LEN_OVERRIDE" ]; then
    CONTEXT_LEN="$MAX_MODEL_LEN_OVERRIDE"
fi

# KV-cache dtype guard: reject vLLM-only dtypes with a clear error.
case "$KV_CACHE_DTYPE" in
    ""|auto)
        SGLANG_KV_ARG="auto"
        ;;
    fp8_e5m2|fp8_e4m3)
        SGLANG_KV_ARG="$KV_CACHE_DTYPE"
        ;;
    fp8)
        SGLANG_KV_ARG="fp8_e5m2"
        ;;
    turboquant*|rotorquant*|kivi_*|kvquant*)
        echo "ERROR: kv-cache-dtype '$KV_CACHE_DTYPE' is vLLM-only." >&2
        echo "       SGLang supports: auto, fp8_e5m2, fp8_e4m3" >&2
        echo "       Use serving/launch_vllm.sh for TurboQuant / KIVI / KVQuant." >&2
        exit 1
        ;;
    *)
        echo "WARN: Unrecognised SGLang kv-cache-dtype '$KV_CACHE_DTYPE', passing through." >&2
        SGLANG_KV_ARG="$KV_CACHE_DTYPE"
        ;;
esac

SGLANG_ARGS=(
    --model-path            "$MODEL_NAME"
    --port                  "$PORT"
    --mem-fraction-static   "$MEM_FRACTION"
    --context-length        "$CONTEXT_LEN"
    --max-running-requests  "$MAX_RUNNING"
    --attention-backend     "$ATTN_BACKEND"
    --kv-cache-dtype        "$SGLANG_KV_ARG"
)

if [ -n "$TOOL_PARSER" ] && [ "$TOOL_PARSER" != "null" ]; then
    SGLANG_ARGS+=(--tool-call-parser "$TOOL_PARSER")
fi

if [ -n "$CHAT_TEMPLATE" ]; then
    SGLANG_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

if [ "$TORCH_COMPILE" = "true" ] || [ "$TORCH_COMPILE" = "True" ]; then
    SGLANG_ARGS+=(--enable-torch-compile)
fi

echo "=== Launching SGLang Server ==="
echo "Model:           $MODEL_NAME"
echo "Tool Parser:     $TOOL_PARSER"
echo "Context Length:  $CONTEXT_LEN"
echo "Mem Fraction:    $MEM_FRACTION"
echo "Attn Backend:    $ATTN_BACKEND"
echo "Port:            $PORT"
echo "KV Cache:        $SGLANG_KV_ARG"
echo "==============================="

exec python -m sglang.launch_server "${SGLANG_ARGS[@]}"
