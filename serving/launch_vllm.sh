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
#   serving.max_num_seqs   → --max-num-seqs (optional)
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
MAX_NUM_SEQS=$(parse_yaml "serving.max_num_seqs" "")
ENFORCE_EAGER=$(parse_yaml "serving.enforce_eager" "false")
PORT=$(parse_yaml "serving.port" "8000")

# Read serving.hf_overrides as a JSON string (empty if not set).
# This is the vLLM 0.11.1+ replacement for --rope-scaling / --rope-theta CLI
# flags, which were removed in PR #28006. Use it to pass per-model config
# overrides (e.g. rope_scaling: null) without patching config.json on disk.
HF_OVERRIDES=$(python3 - "$CONFIG_FILE" <<'PYEOF'
import yaml, json, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
val = cfg.get("serving", {}).get("hf_overrides")
print(json.dumps(val) if val is not None else "", end="")
PYEOF
)

if [ -z "$MODEL_NAME" ]; then
    echo "Error: model.name not found in config"
    exit 1
fi

# Build vLLM argument array (array avoids word-splitting on values with spaces)
KV_CACHE_DTYPE=""
MAX_MODEL_LEN_OVERRIDE=""
MAX_NUM_SEQS_OVERRIDE=""

# Parse CLI overrides
while [ $# -gt 0 ]; do
    case "$1" in
        --kv-cache-dtype)  KV_CACHE_DTYPE="$2";          shift 2 ;;
        --port)            PORT="$2";                     shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN_OVERRIDE="$2";  shift 2 ;;
        --max-num-seqs)    MAX_NUM_SEQS_OVERRIDE="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# CLI --max-model-len takes precedence over YAML serving.max_model_len
if [ -n "$MAX_MODEL_LEN_OVERRIDE" ]; then
    MAX_LEN="$MAX_MODEL_LEN_OVERRIDE"
fi

VLLM_ARGS=(
    --model                  "$MODEL_NAME"
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

if [ -n "$MAX_NUM_SEQS_OVERRIDE" ]; then
    VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS_OVERRIDE")
elif [ -n "$MAX_NUM_SEQS" ]; then
    VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

# NOTE: Do NOT force --attention-backend TURBOQUANT for turboquant_* dtypes.
# vLLM's CUDA platform auto-routes per-layer (cuda.py:260): non-boundary
# layers get TURBOQUANT, boundary-skip layers (first/last 2, auto-added by
# TurboQuantConfig.get_boundary_skip_layers) use kv_cache_dtype=auto on
# FlashAttention 2 for quality protection. Forcing TURBOQUANT globally
# breaks the boundary path because TURBOQUANT rejects "auto".

if [ -n "$HF_OVERRIDES" ]; then
    VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES")
fi

echo "=== Launching vLLM Server ==="
echo "Model:        $MODEL_NAME"
echo "Tool Parser:  $TOOL_PARSER"
echo "GPU Util:     $GPU_UTIL"
echo "Max Len:      $MAX_LEN"
echo "Port:         $PORT"
echo "KV Cache:     ${KV_CACHE_DTYPE:-auto}"
echo "HF Overrides: ${HF_OVERRIDES:-(none)}"
echo "============================="

# Route any turboquant*/rotorquant* dtype through the project's custom launcher.
# Bare "turboquant"/"rotorquant" need the argparse + Pydantic patches to be
# accepted at all. Upstream variants (turboquant_3bit_nc, turboquant_k8v4,
# etc.) are first-class vLLM dtypes but still need the launcher because our
# engine-config hook (_install_turboquant_engine_config_hook) patches the
# hybrid-model guard and auto-injects --enforce-eager on Gemma-4. The custom
# scaffolding Pydantic patch is gated internally to bare "turboquant" only.
case "$KV_CACHE_DTYPE" in
    turboquant|turboquant_*)
        exec python -m llm_workflow_agents.serving.launch_vllm_turboquant "${VLLM_ARGS[@]}"
        ;;
    rotorquant|rotorquant_*)
        exec python -m llm_workflow_agents.serving.launch_vllm_rotorquant "${VLLM_ARGS[@]}"
        ;;
esac

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
