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
PP_SIZE=$(parse_yaml "serving.pp_size" "1")
ENABLE_DP_ATTN=$(parse_yaml "serving.enable_dp_attention" "false")
SPEC_ENABLED=$(parse_yaml "serving.speculative.enabled" "false")
SPEC_ALGORITHM=$(parse_yaml "serving.speculative.algorithm" "")
SPEC_DRAFT_MODEL_PATH=$(parse_yaml "serving.speculative.draft_model_path" "")
SPEC_NUM_DRAFT_TOKENS=$(parse_yaml "serving.speculative.num_draft_tokens" "")
SPEC_DFLASH_BLOCK_SIZE=$(parse_yaml "serving.speculative.dflash_block_size" "")
SPEC_DFLASH_WINDOW_SIZE=$(parse_yaml "serving.speculative.dflash_draft_window_size" "")

if [ -z "$MODEL_NAME" ]; then
    echo "Error: model.name not found in config" >&2
    exit 1
fi

MAX_MODEL_LEN_OVERRIDE=""
NO_SPECULATIVE=false
SPEC_METHOD_OVERRIDE=""
SPEC_DRAFT_MODEL_OVERRIDE=""
SPEC_NUM_TOKENS_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --kv-cache-dtype)          KV_CACHE_DTYPE="$2";            shift 2 ;;
        --port)                    PORT="$2";                       shift 2 ;;
        --max-model-len)           MAX_MODEL_LEN_OVERRIDE="$2";    shift 2 ;;
        --max-num-seqs)            MAX_RUNNING="$2";               shift 2 ;;
        --speculative-method)      SPEC_METHOD_OVERRIDE="$2";      shift 2 ;;
        --speculative-draft-model) SPEC_DRAFT_MODEL_OVERRIDE="$2"; shift 2 ;;
        --speculative-num-tokens)  SPEC_NUM_TOKENS_OVERRIDE="$2";  shift 2 ;;
        --no-speculative)          NO_SPECULATIVE=true;             shift   ;;
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

# Apply CLI overrides for speculative settings (CLI > YAML); then validate + append flags.
if [ "$NO_SPECULATIVE" = "true" ]; then
    SPEC_ENABLED="false"
fi
if [ -n "$SPEC_METHOD_OVERRIDE" ] && [ "$NO_SPECULATIVE" = "false" ]; then
    SPEC_ALGORITHM="${SPEC_METHOD_OVERRIDE^^}"
    SPEC_ENABLED="true"
fi
if [ -n "$SPEC_DRAFT_MODEL_OVERRIDE" ] && [ "$NO_SPECULATIVE" = "false" ]; then
    SPEC_DRAFT_MODEL_PATH="$SPEC_DRAFT_MODEL_OVERRIDE"
    SPEC_ENABLED="true"
fi
if [ -n "$SPEC_NUM_TOKENS_OVERRIDE" ]; then
    SPEC_NUM_DRAFT_TOKENS="$SPEC_NUM_TOKENS_OVERRIDE"
fi

if [ "$SPEC_ENABLED" = "true" ] || [ "$SPEC_ENABLED" = "True" ]; then
    if [ "$ENABLE_DP_ATTN" = "true" ] || [ "$ENABLE_DP_ATTN" = "True" ]; then
        echo "ERROR: DFLASH speculative decoding is incompatible with serving.enable_dp_attention=true." >&2
        exit 1
    fi
    if [ -n "$PP_SIZE" ] && [ "$PP_SIZE" != "1" ]; then
        echo "ERROR: DFLASH speculative decoding requires serving.pp_size=1 (got pp_size=$PP_SIZE)." >&2
        exit 1
    fi
    [ -n "$SPEC_ALGORITHM" ]          && SGLANG_ARGS+=(--speculative-algorithm           "$SPEC_ALGORITHM")
    [ -n "$SPEC_DRAFT_MODEL_PATH" ]   && SGLANG_ARGS+=(--speculative-draft-model-path    "$SPEC_DRAFT_MODEL_PATH")
    [ -n "$SPEC_NUM_DRAFT_TOKENS" ]   && SGLANG_ARGS+=(--speculative-num-draft-tokens    "$SPEC_NUM_DRAFT_TOKENS")
    [ -n "$SPEC_DFLASH_BLOCK_SIZE" ]  && SGLANG_ARGS+=(--speculative-dflash-block-size   "$SPEC_DFLASH_BLOCK_SIZE")
    [ -n "$SPEC_DFLASH_WINDOW_SIZE" ] && SGLANG_ARGS+=(--speculative-dflash-draft-window-size "$SPEC_DFLASH_WINDOW_SIZE")
    echo "WARN: DFLASH causes SGLang to disable its overlap scheduler and mixed-chunked-prefill." >&2
fi

echo "=== Launching SGLang Server ==="
echo "Model:           $MODEL_NAME"
echo "Tool Parser:     $TOOL_PARSER"
echo "Context Length:  $CONTEXT_LEN"
echo "Mem Fraction:    $MEM_FRACTION"
echo "Attn Backend:    $ATTN_BACKEND"
echo "Port:            $PORT"
echo "KV Cache:        $SGLANG_KV_ARG"
if [ "$SPEC_ENABLED" = "true" ] || [ "$SPEC_ENABLED" = "True" ]; then
    echo "Spec Decoding:   ${SPEC_ALGORITHM} (draft=${SPEC_DRAFT_MODEL_PATH:-<none>})"
fi
echo "==============================="

# LAUNCH_PRINT_ONLY=1: print resolved SGLANG_ARGS and exit without launching (used by tests).
if [ "${LAUNCH_PRINT_ONLY:-}" = "1" ]; then
    echo "${SGLANG_ARGS[@]}"
    exit 0
fi

exec python -m sglang.launch_server "${SGLANG_ARGS[@]}"
