#!/usr/bin/env bash
# Bootstrap the training virtualenv inside the Unsloth pre-built container
# (.devcontainer/Dockerfile.unsloth / unsloth/unsloth base image).
#
# The base image ships a ready-to-use stack (CUDA, torch, Unsloth, vLLM) at
# the system Python level.  This script layers the project package and dev
# tooling on top WITHOUT re-resolving or clobbering those pre-installed
# versions, with one intentional override: transformers is upgraded to
# >=5.6.0 (required for Gemma-4 mm_prefix mask and Qwen3.6 hybrid hooks in
# launch_vllm_turboquant.py / vllm_plugin.py).
#
# Strategy:
#   1. Create .venv-train with --system-site-packages so the container's
#      torch / Unsloth / vLLM / trl are visible without redundant downloads.
#   2. Freeze the system site-packages as a uv constraint file; strip the
#      transformers pin so step 5 can freely upgrade it.
#   3. Install the project package in editable mode (--no-deps) to avoid
#      re-resolving the training stack pins in pyproject.toml [train].
#   4. Install dev tools (pytest, ruff, mypy) under the same constraints.
#   5. Upgrade transformers to >=5.6.0 inside the venv (overrides the system
#      pin only within the venv; the system-level install is untouched).
#   6. Install dvc[gs] (not present in the base image).
#
# Differences vs install_train.sh / install_train_cu128.sh:
#   - Targets unsloth/unsloth Docker image (not bare CUDA base).
#   - Uses --system-site-packages: no torch/vLLM/Unsloth download needed.
#   - Skips -e ".[train,dev]" to avoid re-resolving torch==2.10.0 etc.
#   - Explicitly upgrades transformers past the Unsloth-pinned version.
#
# Usage:
#   ./scripts/install_train_unsloth.sh
#
# Re-running is safe (uv pip install is idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-train"

cd "$PROJECT_ROOT"

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "Creating $VENV (--system-site-packages) ..."
    uv venv "$VENV" --python 3.12 --system-site-packages
else
    echo "Reusing existing $VENV ..."
fi

# The unsloth/unsloth image keeps its training stack in /opt/venv (a venv
# whose python is a symlink to /usr/bin/python3.12).  --system-site-packages
# inherits the *base interpreter's* site-packages — /usr/lib/python3.12/...,
# not /opt/venv/lib/python3.12/site-packages — so unsloth/trl/vllm/torch
# would be invisible and uv would re-download ~3GB of CUDA wheels.
# A .pth file in our venv's site-packages restores /opt/venv on sys.path
# before uv runs its resolver, so uv detects torch/peft/etc. as already
# installed and skips them.
PTH_FILE="$VENV/lib/python3.12/site-packages/_opt_venv.pth"
if [[ -d /opt/venv/lib/python3.12/site-packages && ! -f "$PTH_FILE" ]]; then
    echo "Linking /opt/venv site-packages into venv via _opt_venv.pth ..."
    echo "/opt/venv/lib/python3.12/site-packages" > "$PTH_FILE"
fi

# ── 2. Freeze container packages as constraints (minus transformers) ───────────
# We use the system interpreter's pip so we capture packages pre-installed
# by the Docker image, not just what's in the (currently empty) venv.
CONSTRAINTS_FILE="$(mktemp /tmp/unsloth_constraints_XXXXXX.txt)"
OVERRIDES_FILE="$(mktemp /tmp/unsloth_overrides_XXXXXX.txt)"
trap 'rm -f "$CONSTRAINTS_FILE" "$OVERRIDES_FILE"' EXIT

echo "Capturing container package versions as constraints ..."
# Exclusions:
#   transformers           — upgraded in step 5 (Gemma-4 / Qwen3.6 hooks need >=5.6)
#   huggingface-hub        — transformers 5.x requires hub >=1.0; the /opt/venv
#                            pin (0.36.x) would drag transformers back to 4.x
#                            on every re-run, causing install thrash
#   tokenizers             — travels with transformers; 5.x needs tokenizers
#                            from the same major as the matched transformers
#   llm-workflow-agents    — project src, installed editable in step 3
# Note: pip freeze emits the dist's declared name verbatim, so e.g.
# huggingface-hub is reported as "huggingface_hub==" (underscore).  Match
# both separators to be safe.
python3 -m pip freeze 2>/dev/null \
    | grep -ivE "^transformers==" \
    | grep -ivE "^huggingface[-_]hub==" \
    | grep -ivE "^tokenizers==" \
    | grep -ivE "^llm[-_]workflow[-_]agents" \
    > "$CONSTRAINTS_FILE" || true
echo "  -> $(wc -l < "$CONSTRAINTS_FILE") constraints captured (transformers/hf-hub/tokenizers excluded)"

# Overrides: torch's +cu128 wheel pins an exact cuda-bindings build in its
# metadata, and the unsloth container may ship a different build than that
# pin (historically metadata wanted 12.9.4 while the image had 12.8.0).  A
# hardcoded pin here goes stale the moment the base image is rebuilt: if it
# names a version the image no longer has, the resolver sees torch pinned to
# BOTH the override and its real metadata pin and declares the graph
# unsatisfiable.  Instead, derive the override from the cuda-bindings that
# is ACTUALLY installed in the container.  That always matches the
# system-site-packages install; when it also equals torch's metadata pin
# (the current 12.9.4 == 12.9.4 case) the override is a harmless no-op, and
# when they diverge it forces the resolver onto the container's working
# build without a manual edit.
CUDA_BINDINGS_VER="$(python3 -c 'import importlib.metadata as m; print(m.version("cuda-bindings"))' 2>/dev/null || true)"
if [[ -n "$CUDA_BINDINGS_VER" ]]; then
    echo "cuda-bindings==$CUDA_BINDINGS_VER" > "$OVERRIDES_FILE"
    echo "  -> pinning cuda-bindings override to container version $CUDA_BINDINGS_VER"
else
    echo "  -> cuda-bindings not found in container; no override written"
fi

# ── 3. Install project src (editable, no dep re-resolution) ──────────────────
# --no-deps registers the src/ package as importable without touching the
# training stack (torch / unsloth / vllm / trl) already in the container.
echo "Installing project src (editable, --no-deps) ..."
VIRTUAL_ENV="$VENV" uv pip install -e . --no-deps

# ── 4. Install project base dependencies + dev tools under constraints ────────
# Any package already present in the container is pinned to its existing
# version by the constraints file; packages absent from the container
# (outlines, anthropic, google-genai, python-docx, …) are freshly installed.
# [train] extras are intentionally excluded — torch / vllm / unsloth are
# inherited from the container via --system-site-packages.
echo "Installing project dependencies + dev tools under container constraints ..."
# --extra-index-url is required so uv can resolve torch==2.10.0+cu128 (a
# transitive constraint via peft==0.18.1).  The +cu128 local-version wheel
# is hosted on the PyTorch index, not PyPI.  No actual download happens:
# torch is already present via --system-site-packages, so uv recognizes it
# as installed and skips the wheel.
VIRTUAL_ENV="$VENV" uv pip install \
    --extra-index-url "https://download.pytorch.org/whl/cu128" \
    --constraint "$CONSTRAINTS_FILE" \
    --override "$OVERRIDES_FILE" \
    "peft>=0.14.0" \
    "accelerate>=0.34.0" \
    "triton>=3.0.0" \
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
    "outlines>=0.0.40" \
    "google-genai>=1.70.0" \
    "anthropic>=0.88.0" \
    "python-dotenv>=1.2.2" \
    "python-docx>=1.1" \
    "pytest>=8.0" \
    "pytest-cov>=5.0" \
    "pytest-asyncio>=0.23" \
    "ruff>=0.6.0" \
    "mypy>=1.11"

# ── 5. Upgrade transformers >=5.6.0 ──────────────────────────────────────────
# The unsloth/unsloth base image pins transformers to an Unsloth-tested version
# (typically 4.x–5.5.x).  Gemma-4 + Qwen3.6 support requires >=5.6.0.
# With --system-site-packages the new wheel is installed into the venv's own
# site-packages and takes precedence over the system pin for any process that
# activates this venv.
echo "Upgrading transformers to >=5.6.0 ..."
VIRTUAL_ENV="$VENV" uv pip install "transformers>=5.6.0"

# ── 6. Install dvc[gs] ────────────────────────────────────────────────────────
echo "Installing dvc[gs] ..."
VIRTUAL_ENV="$VENV" uv pip install "dvc[gs]"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "Done.  Activate with:  source .venv-train/bin/activate"
echo
"$VENV/bin/python" -c "
packages = {}
for name in ('unsloth', 'torch', 'transformers', 'trl', 'vllm', 'dvc'):
    try:
        import importlib.metadata as m
        packages[name] = m.version(name)
    except Exception:
        packages[name] = '(not found)'
for k, v in packages.items():
    print(f'  {k:<14} {v}')
" 2>&1 | tail -10
