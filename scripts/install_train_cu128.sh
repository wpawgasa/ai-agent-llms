#!/usr/bin/env bash
# Bootstrap the training virtualenv on a CUDA 12.8 dev container (.venv-train).
#
# Pair this with .devcontainer/Dockerfile.unsloth (CUDA 12.8 base image).
# Use this on hosts running NVIDIA driver 570.x — driver 570 fully supports
# CUDA 12.8 natively, whereas CUDA 13.0 requires driver 575+ (or the
# cuda-compat-13-0 forward-compat package). Without that, every cuBLAS GEMM
# (cublasGemmEx, cublasSgemm) returns CUBLAS_STATUS_INVALID_VALUE.
#
# Differences vs install_train.sh:
#   - Pins torch / torchvision / torchaudio to the cu128 index explicitly
#     instead of relying on --torch-backend=auto (which picks cu130 because
#     nvidia-smi still reports CUDA 13 from the driver, even on a cu128
#     toolkit).
#   - Skips the cu130 vLLM wheel override — vLLM 0.19.1 publishes no cu128
#     build, so we use the PyPI default (no-suffix wheel = cu129 build).
#     cu129 user-mode runs cleanly on a cu128 toolkit + driver 570.
#   - Switches Unsloth's per-(cuda,torch) extra to `cu128-torch2100` per the
#     auto-installer's recommendation for our combo.
#
# Usage:
#   ./scripts/install_train_cu128.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-train"

cd "$PROJECT_ROOT"

if [[ ! -d "$VENV" ]]; then
    echo "Creating $VENV ..."
    uv venv "$VENV" --python 3.12
fi

CU128_INDEX="https://download.pytorch.org/whl/cu128"

echo "Installing torch / torchvision / torchaudio (cu128) ..."
VIRTUAL_ENV="$VENV" uv pip install \
    --index-url "$CU128_INDEX" \
    --index-strategy unsafe-best-match \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

echo "Installing project + train extras into $VENV ..."
# Skip --torch-backend=auto here: torch is already pinned to cu128 above,
# and uv's auto-detection would otherwise downgrade us to cu130 because
# the driver still claims CUDA 13 to the runtime.
VIRTUAL_ENV="$VENV" uv pip install -e ".[train,dev]"

# Replace the PyPI Unsloth wheel with the cu128-torch2100 build from
# Unsloth's main branch (per https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py).
# Without this, the MoE / RoPE codepaths fall back to the generic wheel
# which has known rough edges on torch 2.10.
echo "Reinstalling Unsloth with cu128-torch2100 extra from main ..."
VIRTUAL_ENV="$VENV" uv pip install --no-deps "git+https://github.com/unslothai/unsloth-zoo.git"
VIRTUAL_ENV="$VENV" uv pip install --no-build-isolation \
    "unsloth[cu128-torch2100] @ git+https://github.com/unslothai/unsloth.git"

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

echo
echo "Sanity check — bf16 GEMM on the GPU:"
"$VENV/bin/python" -c "
import torch
a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print('  bf16 GEMM OK, output norm =', c.float().norm().item())
" 2>&1 | tail -3
