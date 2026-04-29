#!/usr/bin/env bash
# Bootstrap the training virtualenv (.venv-train) — CUDA 13.0 path.
#
# Pair this with the default .devcontainer/Dockerfile (CUDA 13 base).
# Use only on hosts running NVIDIA driver 575+ (native CUDA 13 support)
# or with the cuda-compat-13-0 forward-compat package installed. On a
# driver-570 host you will see CUBLAS_STATUS_INVALID_VALUE on every
# cublasGemmEx call — switch to install_train_cu128.sh + Dockerfile.unsloth
# instead.
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

# Replace the PyPI Unsloth wheel with the cu130-torch2100 build from
# Unsloth's main branch. The PyPI wheel is a generic build; Unsloth's
# auto-installer (https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py)
# explicitly recommends the per-(cuda,torch) extra for our combo:
#   pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
#   pip install "unsloth[cu130-torch2100] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
# Without this, the MoE / RoPE codepaths hit a CUBLAS_STATUS_INVALID_VALUE
# in cublasSgemm — the default wheel was tested only against CUDA 12.x cuBLAS.
echo "Reinstalling Unsloth with cu130-torch2100 extra from main ..."
VIRTUAL_ENV="$VENV" uv pip install --no-deps "git+https://github.com/unslothai/unsloth-zoo.git"
VIRTUAL_ENV="$VENV" uv pip install --no-build-isolation \
    "unsloth[cu130-torch2100] @ git+https://github.com/unslothai/unsloth.git"

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
