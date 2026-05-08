#!/usr/bin/env bash
# Bootstrap the TensorRT-LLM serving virtualenv (.venv-trtllm).
#
# TensorRT-LLM exposes an OpenAI-compatible HTTP server via `trtllm-serve`
# on port 8000, used as a vLLM/SGLang alternative for Phase 1 and Phase 3
# concurrency benchmarks.
#
# This script runs in two modes:
#
#   CONTAINER MODE (inside .devcontainer/Dockerfile.tensorrt):
#     torch and tensorrt_llm are pre-installed system-wide by the NGC base
#     image. The script uses --system-site-packages and skips re-downloading
#     those packages. No apt pre-flight check is needed.
#     Detected automatically via the TRTLLM_CONTAINER=1 env var set in the
#     Dockerfile.
#
#   BARE-METAL MODE (any other environment):
#     PREREQUISITES (pip cannot satisfy these — install before running):
#       sudo apt-get install -y libopenmpi-dev
#     Installs torch==2.10.0+cu130 and tensorrt_llm==1.2.1 from scratch.
#
# Why a separate venv:
#   - TRT-LLM pins torch==2.10.0, conflicting with .venv-infer (≥2.11).
#   - Cannot coexist with .venv-train (Unsloth pins same torch but different
#     CUDA stack).
#
# Two serving modes (controlled via serving.source in the YAML):
#   source: "hf"          — trtllm-serve <hf_id>; JIT builds the engine on
#                           first launch (5-15 min for 30B-class models).
#                           KV-cache fp8 is NOT available in this mode.
#   source: "engine_dir"  — trtllm-serve <path>; uses a pre-built TRT engine.
#                           Run scripts/build_trtllm_engines.sh (follow-up PR)
#                           to build engines with fp8 KV cache.
#
# Usage (bare metal):
#   sudo apt-get install -y libopenmpi-dev   # once, as root
#   ./scripts/install_trtllm.sh
#
# Usage (inside Dockerfile.tensorrt devcontainer):
#   ./scripts/install_trtllm.sh              # TRTLLM_CONTAINER=1 auto-set
#
# Re-running is safe (uv pip install is idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-trtllm"
CU130_INDEX="https://download.pytorch.org/whl/cu130"
IN_CONTAINER="${TRTLLM_CONTAINER:-0}"

cd "$PROJECT_ROOT"

if [[ "$IN_CONTAINER" == "1" ]]; then
    # ── Container mode (Dockerfile.tensorrt) ─────────────────────────────────
    # torch + tensorrt_llm are pre-installed at the system level by the NGC
    # base image. Use --system-site-packages so the venv inherits them without
    # re-downloading. Skip all apt / CUDA pre-flight checks.
    echo "Detected Dockerfile.tensorrt container — using system site-packages."

    # ── 1. Virtual environment (inherit system packages) ─────────────────────
    if [[ ! -d "$VENV" ]]; then
        echo "Creating $VENV (--system-site-packages) ..."
        uv venv "$VENV" --python 3.12 --system-site-packages
    else
        echo "Reusing existing $VENV ..."
    fi

    # Freeze all container packages as uv constraints so installing project
    # deps cannot clobber torch / trtllm / cuda-level packages.
    CONSTRAINTS_FILE="$(mktemp /tmp/trtllm_constraints_XXXXXX.txt)"
    trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
    VIRTUAL_ENV="$VENV" uv pip freeze > "$CONSTRAINTS_FILE"

    # ── 2. Project package (editable, no deps — deps already in container) ───
    echo "Installing project package (editable, --no-deps) ..."
    VIRTUAL_ENV="$VENV" uv pip install -e . --no-deps

    # ── 3. Project base deps + dev tools (pinned to container versions) ──────
    echo "Installing project dependencies under container constraints ..."
    VIRTUAL_ENV="$VENV" uv pip install \
        --constraint "$CONSTRAINTS_FILE" \
        "peft>=0.14.0" \
        "datasets>=2.19.0" \
        "networkx>=3.2" \
        "jsonschema>=4.21.0" \
        "numpy>=1.26.0" \
        "scipy>=1.12.0" \
        "matplotlib>=3.8.0" \
        "seaborn>=0.13.0" \
        "pydantic>=2.9.0" \
        "pyyaml>=6.0" \
        "structlog>=24.0.0" \
        "wandb>=0.17.0" \
        "google-genai>=1.70.0" \
        "anthropic>=0.88.0" \
        "python-dotenv>=1.2.2" \
        "pytest>=8.0" \
        "pytest-cov>=5.0" \
        "pytest-asyncio>=0.23" \
        "ruff>=0.6.0" \
        "mypy>=1.11" \
        "dvc[gs]>=3.0"
else
    # ── Bare-metal mode ───────────────────────────────────────────────────────
    echo "Bare-metal mode — running pre-flight checks ..."

    # libopenmpi-dev is required by TRT-LLM's mpirun-based multi-process model.
    # Even at tp_size=1 it spawns child processes via MPI.
    if ! dpkg -l libopenmpi-dev > /dev/null 2>&1; then
        echo "ERROR: libopenmpi-dev is not installed." >&2
        echo "       Run: sudo apt-get install -y libopenmpi-dev" >&2
        echo "       Or use .devcontainer/Dockerfile.tensorrt which provides it." >&2
        echo "       Then re-run this script." >&2
        exit 1
    fi

    # TRT-LLM 1.2.1 requires CUDA toolkit 13.1. Warn on older drivers.
    if command -v nvidia-smi > /dev/null 2>&1; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
        if [[ -n "$CUDA_VER" ]]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            if [[ "$CUDA_MAJOR" -lt 13 ]]; then
                echo "WARN: nvidia-smi reports CUDA $CUDA_VER. TRT-LLM 1.2.1 requires CUDA 13.1+" >&2
                echo "      Ensure driver ≥575 is installed for native CUDA 13 support." >&2
                echo "      Alternatively, use .devcontainer/Dockerfile.tensorrt." >&2
            fi
        fi
    fi

    # ── 1. Virtual environment ────────────────────────────────────────────────
    if [[ ! -d "$VENV" ]]; then
        echo "Creating $VENV ..."
        uv venv "$VENV" --python 3.12
    else
        echo "Reusing existing $VENV ..."
    fi

    # ── 2. Torch 2.10.0 (cu130) — pin before TRT-LLM to prevent uv from
    #       picking a newer torch that conflicts with TRT-LLM's wheel deps. ───
    echo "Installing torch==2.10.0 (cu130) ..."
    VIRTUAL_ENV="$VENV" uv pip install \
        --index-url "$CU130_INDEX" \
        --index-strategy unsafe-best-match \
        "torch==2.10.0" "torchvision==0.25.0"

    # ── 3. TensorRT-LLM ──────────────────────────────────────────────────────
    # Freeze torch pin as constraints so TRT-LLM's own deps can't bump torch.
    CONSTRAINTS_FILE="$(mktemp /tmp/trtllm_constraints_XXXXXX.txt)"
    trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
    VIRTUAL_ENV="$VENV" uv pip freeze | grep -i "^torch==" > "$CONSTRAINTS_FILE" || true

    echo "Installing tensorrt_llm==1.2.1 ..."
    VIRTUAL_ENV="$VENV" uv pip install \
        --constraint "$CONSTRAINTS_FILE" \
        "tensorrt_llm==1.2.1"

    # ── 4. Project package + dev tools ───────────────────────────────────────
    echo "Installing project package (editable) + dev tools ..."
    VIRTUAL_ENV="$VENV" uv pip install -e . --no-deps
    VIRTUAL_ENV="$VENV" uv pip install \
        "pytest>=8.0" \
        "pytest-cov>=5.0" \
        "pytest-asyncio>=0.23" \
        "ruff>=0.6.0" \
        "mypy>=1.11" \
        "dvc[gs]>=3.0" \
        "pyyaml>=6.0"
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
echo
echo "Done.  Activate with:  source .venv-trtllm/bin/activate"
echo
"$VENV/bin/python" -c "
import importlib.metadata as m
for pkg in ('tensorrt_llm', 'torch'):
    try:
        print(f'  {pkg:<18} {m.version(pkg)}')
    except Exception:
        print(f'  {pkg:<18} (not found)')
" 2>&1 | tail -10

echo
echo "Launch a model (JIT-from-HF) with:"
echo "  ./serving/launch.sh configs/models_exp_a/qwen3_32b_trtllm.yaml"
echo
echo "NOTE: First launch compiles the TRT engine (5-15 min for 30B models)."
echo "      Use scripts/build_trtllm_engines.sh + source: engine_dir for"
echo "      pre-built engines with fp8 KV cache support."
