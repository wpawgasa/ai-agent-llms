#!/usr/bin/env bash
# Bootstrap the training virtualenv (.venv-train).
#
# Why a separate venv: Unsloth pins torch==2.10.0, transformers==4.57.6,
# trl==0.24.0 and is incompatible with vLLM 0.20.0 (which requires
# torch>=2.11). The inference venv (.venv-infer) holds the 0.20.0 stack;
# the two cannot coexist.
#
# Usage:
#   ./scripts/install_train.sh
#
# Re-running is safe (uv pip install is idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-train"

cd "$PROJECT_ROOT"

if [[ ! -d "$VENV" ]]; then
    echo "Creating $VENV ..."
    uv venv "$VENV" --python 3.12
fi

# vLLM cu130 wheel (no PyPI build for v0.19.1+cu130; pull from GitHub release).
VLLM_CU130_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl"

echo "Installing project + train extras into $VENV ..."
VIRTUAL_ENV="$VENV" uv pip install -e ".[train,dev]" --torch-backend=auto

echo "Overriding vLLM with cu130 wheel ..."
VIRTUAL_ENV="$VENV" uv pip install "$VLLM_CU130_WHEEL"

# Unsloth's released wheel pins transformers==4.57.6, but loading newer
# model classes (e.g. RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic) needs the
# v5.7+ dev branch. Override post-install.
echo "Overriding transformers with HuggingFace git HEAD ..."
VIRTUAL_ENV="$VENV" uv pip install "transformers @ git+https://github.com/huggingface/transformers.git"

echo
echo "Done. Activate with:  source .venv-train/bin/activate"
"$VENV/bin/python" -c "
import unsloth, vllm, torch, transformers, trl
print(f'  unsloth      {unsloth.__version__}')
print(f'  vllm         {vllm.__version__}')
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  trl          {trl.__version__}')
" 2>&1 | tail -10
