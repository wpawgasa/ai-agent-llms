#!/usr/bin/env bash
# Launch a TensorRT-LLM OpenAI-compatible API server from a model config YAML.
#
# Usage (via dispatcher — preferred):
#   ./serving/launch.sh <model_config_trtllm.yaml> [--kv-cache-dtype <dtype>]
#                       [--port <port>] [--max-model-len <n>]
#
# Direct usage:
#   ./serving/launch_trtllm.sh <model_config_trtllm.yaml> [OPTIONS]
#
# Reads from YAML config (serving.*):
#   model.name               → positional model arg (when source: "hf")
#   serving.source           → "hf" (default) or "engine_dir"
#   serving.engine_dir       → path to pre-built TRT engine (source: "engine_dir")
#   serving.max_batch_size   (default 256) → --max_batch_size
#   serving.max_num_tokens   (default 8192) → --max_num_tokens
#   serving.tp_size          (default 1) → --tp_size
#   serving.port             (default 8000) → --port (overridable via CLI)
#
# KV-cache dtype:
#   TRT-LLM KV-cache quantization (fp8) must be set at engine-build time
#   via `trtllm-build --kv_cache_type fp8`. Runtime overrides via
#   --kv-cache-dtype are logged as a warning and ignored in JIT-from-HF mode.
#   Only "auto" is accepted silently at runtime.
#
# Two engine modes:
#   source: "hf"          — JIT compile on first launch (5-15 min for 30B).
#                           Recommended for first-run smoke tests.
#   source: "engine_dir"  — Fast launch from a pre-built engine (see
#                           scripts/build_trtllm_engines.sh for the build step).

set -euo pipefail

LAUNCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_PROJECT_ROOT="$(cd "$LAUNCH_SCRIPT_DIR/.." && pwd)"

# Activate the TRT-LLM venv if no venv is currently active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$LAUNCH_PROJECT_ROOT/.venv-trtllm/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LAUNCH_PROJECT_ROOT/.venv-trtllm/bin/activate"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_config_trtllm.yaml> [--kv-cache-dtype <dtype>] [--port <port>]"
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
SOURCE=$(parse_yaml "serving.source" "hf")
ENGINE_DIR=$(parse_yaml "serving.engine_dir" "")
MAX_BATCH=$(parse_yaml "serving.max_batch_size" "256")
MAX_TOKENS=$(parse_yaml "serving.max_num_tokens" "8192")
TP_SIZE=$(parse_yaml "serving.tp_size" "1")
PORT=$(parse_yaml "serving.port" "8000")
KV_CACHE_DTYPE="auto"
MAX_NUM_TOKENS_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --kv-cache-dtype) KV_CACHE_DTYPE="$2";          shift 2 ;;
        --port)           PORT="$2";                    shift 2 ;;
        --max-model-len)  MAX_NUM_TOKENS_OVERRIDE="$2"; shift 2 ;;
        --max-num-seqs)   MAX_BATCH="$2";               shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -n "$MAX_NUM_TOKENS_OVERRIDE" ]; then
    MAX_TOKENS="$MAX_NUM_TOKENS_OVERRIDE"
fi

# KV-cache dtype handling: TRT-LLM is build-time only.
case "$KV_CACHE_DTYPE" in
    ""|auto)
        ;;  # default — no flag needed
    fp8)
        if [[ "$SOURCE" == "hf" ]]; then
            echo "WARN: --kv-cache-dtype=fp8 is ignored in JIT-from-HF mode." >&2
            echo "      Pre-build the engine with: trtllm-build --kv_cache_type fp8" >&2
            echo "      Then use source: engine_dir in your YAML config." >&2
        fi
        ;;
    turboquant*|rotorquant*|kivi_*|kvquant*)
        echo "ERROR: kv-cache-dtype '$KV_CACHE_DTYPE' is vLLM-only." >&2
        echo "       TRT-LLM supports: auto (runtime), fp8 (build-time via trtllm-build)." >&2
        exit 1
        ;;
    *)
        echo "WARN: Unknown TRT-LLM kv-cache-dtype '$KV_CACHE_DTYPE'. Ignoring." >&2
        ;;
esac

TRT_ARGS=(
    --port          "$PORT"
    --max_batch_size "$MAX_BATCH"
    --max_num_tokens "$MAX_TOKENS"
    --tp_size        "$TP_SIZE"
)

# Determine model source
if [[ "$SOURCE" == "engine_dir" ]]; then
    if [ -z "$ENGINE_DIR" ]; then
        echo "ERROR: serving.source=engine_dir but serving.engine_dir is not set in $CONFIG_FILE" >&2
        exit 1
    fi
    MODEL_ARG="$ENGINE_DIR"
else
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: model.name not found in config" >&2
        exit 1
    fi
    MODEL_ARG="$MODEL_NAME"
fi

echo "=== Launching TensorRT-LLM Server ==="
echo "Model:       $MODEL_ARG"
echo "Source:      $SOURCE"
echo "Max Batch:   $MAX_BATCH"
echo "Max Tokens:  $MAX_TOKENS"
echo "TP Size:     $TP_SIZE"
echo "Port:        $PORT"
if [[ "$SOURCE" == "hf" ]]; then
    echo "NOTE: First launch compiles TRT engine (5-15 min for large models)."
fi
echo "======================================"

exec trtllm-serve "$MODEL_ARG" "${TRT_ARGS[@]}"
