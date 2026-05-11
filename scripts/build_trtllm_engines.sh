#!/usr/bin/env bash
# Build a TensorRT-LLM engine from a model config YAML.
#
# Usage:
#   ./scripts/build_trtllm_engines.sh <model_config_trtllm.yaml> \
#       --output-dir <path> [--recipe dflash]
#
# This script is a STUB. NVIDIA ModelOpt PR #1211 (Dflash support) must be
# merged and installed before this script can build a functional engine.
#
# When ready, it will:
#   1. Load the model with `torch_dtype=bfloat16` via AutoModelForCausalLM.
#   2. Apply the Dflash ModelOpt recipe (training/fine-tuning the draft heads).
#   3. Export the HF-compatible draft checkpoint via DFlashExporter.
#   4. Convert + build the TRT-LLM engine via `trtllm-build`.
#   5. Write the engine to --output-dir (set as serving.engine_dir in YAML).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_config_trtllm.yaml> --output-dir <path> [--recipe dflash]" >&2
    exit 1
fi

CONFIG_FILE="$1"; shift

if [[ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]] && [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config not found: $CONFIG_FILE" >&2
    exit 1
fi

# Resolve absolute path
[[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"

OUTPUT_DIR=""
RECIPE="dflash"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --recipe)     RECIPE="$2";     shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output-dir is required." >&2
    exit 1
fi

# Extract build params from the YAML speculative block
BUILD_INFO=$(python3 - "$CONFIG_FILE" <<'PYEOF'
import yaml, json, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
model_name = cfg.get("model", {}).get("name", "<unknown>")
spec = cfg.get("serving", {}).get("speculative") or {}
recipe = spec.get("build_recipe", "modelopt_recipes/general/speculative_decoding/dflash.yaml")
params = spec.get("build_params") or {}
print(json.dumps({"model_name": model_name, "recipe": recipe, "params": params}))
PYEOF
)

MODEL_NAME=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['model_name'])" "$BUILD_INFO")
BUILD_RECIPE=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['recipe'])" "$BUILD_INFO")
BUILD_PARAMS=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(json.dumps(d['params']))" "$BUILD_INFO")

echo "=== TRT-LLM Engine Builder (STUB) ==="
echo "Config:      $CONFIG_FILE"
echo "Model:       $MODEL_NAME"
echo "Recipe:      $BUILD_RECIPE"
echo "Build params:$BUILD_PARAMS"
echo "Output dir:  $OUTPUT_DIR"
echo "======================================"
echo ""
echo "The following commands would build the Dflash engine when ModelOpt PR #1211 is available:"
echo ""
echo "  # Step 1 — Apply Dflash recipe via ModelOpt"
echo "  python -m modelopt.torch.speculative.dflash.train \\"
echo "    --model $MODEL_NAME \\"
echo "    --recipe $BUILD_RECIPE \\"
echo "    --output-dir ${OUTPUT_DIR}/draft_hf"
echo ""
echo "  # Step 2 — Build TRT-LLM engine"
echo "  trtllm-build \\"
echo "    --model_dir ${OUTPUT_DIR}/draft_hf \\"
echo "    --output_dir ${OUTPUT_DIR}/engine \\"
echo "    --speculative_decoding_mode $RECIPE \\"
echo "    --dtype bfloat16 \\"
echo "    --workers 1"
echo ""
echo "  # Step 3 — Update serving.engine_dir in your YAML to: ${OUTPUT_DIR}/engine"
echo ""
echo "ERROR: Engine builder not yet wired. NVIDIA ModelOpt PR #1211 is required." >&2
echo "       Hand-build with the commands printed above and place the result at" >&2
echo "       serving.engine_dir in your YAML config." >&2
exit 1
