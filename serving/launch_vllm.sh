#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible API server from a model config YAML.
#
# Usage:
#   ./serving/launch_vllm.sh <model_config.yaml> [--kv-cache-dtype <dtype>] [--port <port>]
#
# Examples:
#   ./serving/launch_vllm.sh configs/models_exp_a/gemma3_27b.yaml
#   ./serving/launch_vllm.sh configs/models_exp_a/qwen3_32b.yaml --kv-cache-dtype fp8
#   ./serving/launch_vllm.sh configs/models_exp_a/qwen3_32b.yaml --port 8001
#
# Reads from YAML config:
#   model.name             → --model
#   serving.tool_call_parser → --tool-call-parser
#   serving.gpu_memory_utilization → --gpu-memory-utilization
#   serving.max_model_len  → --max-model-len
#   serving.enforce_eager  → --enforce-eager (if true)
#   serving.port           → --port (default: 8000, overridable via --port CLI arg)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_config.yaml> [--kv-cache-dtype <dtype>]"
    exit 1
fi

CONFIG_FILE="$1"
shift

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse YAML values using python (avoids yq dependency).
# Values are passed as CLI arguments to avoid shell injection via CONFIG_FILE or key names.
parse_yaml() {
    python3 - "$CONFIG_FILE" "$1" "$2" <<'EOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
# Navigate nested keys (e.g., 'model.name')
keys = sys.argv[2].split('.')
default = sys.argv[3] if len(sys.argv) > 3 else ''
val = cfg
for k in keys:
    if isinstance(val, dict) and k in val:
        val = val[k]
    else:
        val = default
        break
print(val)
EOF
}

# Read config values
MODEL_NAME=$(parse_yaml "model.name" "")
TOOL_PARSER=$(parse_yaml "serving.tool_call_parser" "hermes")
GPU_UTIL=$(parse_yaml "serving.gpu_memory_utilization" "0.90")
MAX_LEN=$(parse_yaml "serving.max_model_len" "8192")
ENFORCE_EAGER=$(parse_yaml "serving.enforce_eager" "false")
PORT=$(parse_yaml "serving.port" "8000")

if [ -z "$MODEL_NAME" ]; then
    echo "Error: model.name not found in config"
    exit 1
fi

# Build vLLM argument array (array avoids word-splitting on values with spaces)
KV_CACHE_DTYPE=""
MODEL_OVERRIDE=""   # optional local path to use instead of MODEL_NAME (e.g. patched config)

# Parse CLI overrides
while [ $# -gt 0 ]; do
    case "$1" in
        --kv-cache-dtype) KV_CACHE_DTYPE="$2";  shift 2 ;;
        --port)           PORT="$2";             shift 2 ;;
        --model-override) MODEL_OVERRIDE="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Use patched local path if provided, otherwise use model name from config
EFFECTIVE_MODEL="${MODEL_OVERRIDE:-$MODEL_NAME}"

VLLM_ARGS=(
    --model                  "$EFFECTIVE_MODEL"
    --dtype                  bfloat16
    --tool-call-parser       "$TOOL_PARSER"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len          "$MAX_LEN"
    --enable-auto-tool-choice
    --port                   "$PORT"
)

if [ "$ENFORCE_EAGER" = "true" ] || [ "$ENFORCE_EAGER" = "True" ]; then
    VLLM_ARGS+=(--enforce-eager)
fi

if [ -n "$KV_CACHE_DTYPE" ]; then
    VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

echo "=== Launching vLLM Server ==="
echo "Model:       $EFFECTIVE_MODEL"
echo "Tool Parser: $TOOL_PARSER"
echo "GPU Util:    $GPU_UTIL"
echo "Max Len:     $MAX_LEN"
echo "Port:        $PORT"
echo "KV Cache:    ${KV_CACHE_DTYPE:-auto}"
echo "============================="

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
