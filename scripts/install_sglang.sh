#!/usr/bin/env bash
# Bootstrap the SGLang serving virtualenv (.venv-sglang).
#
# SGLang exposes an OpenAI-compatible HTTP server on port 8000 (forced) and
# serves as a vLLM alternative for Phase 1 (run_exp_a.sh) and Phase 3
# concurrency benchmarks (run_concurrency_benchmark.sh).
#
# Why a separate venv: SGLang requires torch ≥2.11 and its own attention
# kernels (FlashInfer). This conflicts with .venv-train (torch==2.10.0 /
# Unsloth) and is kept separate from .venv-infer to avoid wheel clashes.
#
# What's in .venv-sglang:
#   sglang ≥0.5.11  — FlashInfer + flashattention auto-installed as deps
#   torch ≥2.11+cu130 from the project's pytorch-cu130 index
#   transformers ≥5.6.0 — required for Gemma-4 + Qwen3.6 model support
#   project package (editable, --no-deps) + dev tools
#
# NOTE: The default attention_backend in all *_sglang.yaml configs is
# "triton" (safer, avoids FlashInfer build flakiness on cu130). Switch to
# "flashinfer" per-YAML when you confirm the FlashInfer wheel built
# correctly for your CUDA toolkit.
#
# Usage:
#   ./scripts/install_sglang.sh
#
# Re-running is safe (uv pip install is idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-sglang"
CU130_INDEX="https://download.pytorch.org/whl/cu130"

cd "$PROJECT_ROOT"

# ── 0. System library: libnuma ────────────────────────────────────────────────
# sgl_kernel's prebuilt SM90 wheel links against libnuma.so.1. Without it the
# server crashes at import with "libnuma.so.1: cannot open shared object file".
# Baked into .devcontainer/Dockerfile{,.cuda128}; this block self-heals older
# images. apt-get is a no-op once the package is present.
if ! ldconfig -p | grep -q 'libnuma\.so\.1'; then
    echo "Installing libnuma1 (required by sgl_kernel) ..."
    if command -v apt-get >/dev/null 2>&1; then
        if [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qq && apt-get install -y --no-install-recommends libnuma1
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends libnuma1
        else
            echo "  WARNING: not root and sudo not available — install libnuma1 manually."
        fi
    else
        echo "  WARNING: apt-get not available — install libnuma manually for your distro."
    fi
fi

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "Creating $VENV ..."
    uv venv "$VENV" --python 3.12
else
    echo "Reusing existing $VENV ..."
fi

# ── 2. Torch (cu130) ─────────────────────────────────────────────────────────
echo "Installing torch ≥2.11.0 (cu130) ..."
VIRTUAL_ENV="$VENV" uv pip install \
    --index-url "$CU130_INDEX" \
    --index-strategy unsafe-best-match \
    "torch>=2.11.0"

# ── 3. SGLang ─────────────────────────────────────────────────────────────────
# FlashInfer is a dep of sglang[all] but its cu130 wheel may not exist on
# all platforms. We install the base sglang wheel first; FlashInfer is pulled
# automatically. If FlashInfer fails to build, set attention_backend: triton
# in your model YAMLs (the default for *_sglang.yaml configs).
#
# --prerelease=allow: sglang ≥0.5.11 depends on flash-attn-4>=4.0.0b9 (a
# pre-release wheel). uv refuses pre-releases by default, so the flag is
# required for the resolver to consider it.
echo "Installing sglang ≥0.5.11 ..."
VIRTUAL_ENV="$VENV" uv pip install --prerelease=allow "sglang>=0.5.11"

# ── 4. Transformers ≥5.6.0 ────────────────────────────────────────────────────
echo "Installing transformers ≥5.6.0 ..."
VIRTUAL_ENV="$VENV" uv pip install "transformers>=5.6.0"

# ── 5. Project package + dev tools ───────────────────────────────────────────
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

# ── Smoke test ────────────────────────────────────────────────────────────────
echo
echo "Done.  Activate with:  source .venv-sglang/bin/activate"
echo
"$VENV/bin/python" -c "
import importlib.metadata as m
for pkg in ('sglang', 'torch', 'transformers'):
    try:
        print(f'  {pkg:<16} {m.version(pkg)}')
    except Exception:
        print(f'  {pkg:<16} (not found)')
try:
    import flashinfer
    print(f'  flashinfer       {m.version(\"flashinfer\")} (FlashInfer available)')
except ImportError:
    print('  flashinfer       (not installed — triton backend will be used)')
" 2>&1 | tail -10

echo
echo "Launch a model with: ./serving/launch.sh configs/models_exp_a/qwen3_32b_sglang.yaml"
