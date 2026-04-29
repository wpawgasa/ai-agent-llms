#!/usr/bin/env bash
# Bootstrap the inference / serving virtualenv (.venv-infer).
#
# Why a separate venv: vLLM 0.20.0 requires torch>=2.11 and transformers
# 5.6.2; Unsloth pins torch==2.10.0 and transformers==4.57.6. The two
# cannot share a venv.
#
# Note: vLLM 0.20.0 has no published cu130 wheel. The PyPI wheel is the
# cu129 build, which runs on CUDA 13 via forward compatibility.
#
# Usage:
#   ./scripts/install_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-infer"

cd "$PROJECT_ROOT"

if [[ ! -d "$VENV" ]]; then
    echo "Creating $VENV ..."
    uv venv "$VENV" --python 3.12
fi

echo "Installing project + infer extras into $VENV ..."
VIRTUAL_ENV="$VENV" uv pip install -e ".[infer,dev]" --torch-backend=auto

echo
echo "Done. Activate with:  source .venv-infer/bin/activate"
"$VENV/bin/python" -c "
import vllm, torch, transformers
print(f'  vllm         {vllm.__version__}')
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
" 2>&1 | tail -10
