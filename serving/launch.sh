#!/usr/bin/env bash
# Engine-aware server dispatcher.
#
# Reads serving.engine from a model YAML config and exec's the matching
# backend launcher script, forwarding all remaining arguments unchanged.
# This is the single source of truth for backend dispatch — runner shells
# call this instead of launch_vllm.sh directly so adding a new backend
# only requires: (1) a new launch_<engine>.sh, (2) one new case arm here.
#
# Usage (identical CLI surface to launch_vllm.sh):
#   ./serving/launch.sh <model_config.yaml> [--kv-cache-dtype <dtype>]
#                       [--port <port>] [--max-model-len <n>] [--max-num-seqs <n>]
#
# Supported engines (serving.engine in YAML):
#   vllm           → serving/launch_vllm.sh
#   sglang         → serving/launch_sglang.sh
#   tensorrt_llm   → serving/launch_trtllm.sh
#   bifrost        → error (gateway is externally managed; no local launcher)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_config.yaml> [OPTIONS]"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINE=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('serving', {}).get('engine', 'vllm'))
" "$CONFIG_FILE")

case "$ENGINE" in
    vllm)
        exec bash "$SCRIPT_DIR/launch_vllm.sh" "$@"
        ;;
    sglang)
        exec bash "$SCRIPT_DIR/launch_sglang.sh" "$@"
        ;;
    tensorrt_llm)
        exec bash "$SCRIPT_DIR/launch_trtllm.sh" "$@"
        ;;
    bifrost)
        echo "ERROR: serving.engine='bifrost' has no local launcher." >&2
        echo "       The BiFrost gateway is managed externally." >&2
        echo "       Use --frontier-model with run_concurrency_benchmark.sh instead." >&2
        exit 1
        ;;
    *)
        echo "ERROR: Unknown serving.engine='$ENGINE' in $CONFIG_FILE" >&2
        echo "       Supported values: vllm, sglang, tensorrt_llm" >&2
        exit 1
        ;;
esac
