#!/usr/bin/env python3
"""Patch a HuggingFace model config.json to satisfy vLLM ModelConfig validation.

Downloads only config.json for the specified model into a local cache directory,
applies the patches defined below, and prints the path to use as --model.

Usage:
    python scripts/patch_model_config.py google/gemma-3-27b-it
    python scripts/patch_model_config.py google/gemma-3-27b-it --cache-dir /mnt/data/hf_patches
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Patches keyed by model id (or prefix).
# Each entry is a dict of top-level config keys to set / merge.
# Use a callable for a key if you need read-modify-write on the existing value.
_PATCHES: dict[str, dict] = {
    "google/gemma-3-27b-it": {
        # vLLM 0.11 requires rope_scaling to carry a rope_type key.
        # Gemma 3 uses local/global interleaved attention with standard RoPE
        # for local windows — "default" satisfies the validator without
        # altering the model's actual position encoding behaviour.
        "rope_scaling": lambda v: {**(v or {}), "rope_type": "default"},
    },
}


def _find_patch(model_id: str) -> dict:
    """Return the patch dict for model_id, checking exact match then prefix."""
    if model_id in _PATCHES:
        return _PATCHES[model_id]
    for key, patch in _PATCHES.items():
        if model_id.startswith(key):
            return patch
    return {}


def patch_config(model_id: str, cache_dir: Path) -> Path:
    """Download config.json, apply patches, write to cache_dir / <safe_name>.

    Returns the local directory that should be passed as --model to vLLM.
    """
    from huggingface_hub import hf_hub_download

    safe_name = model_id.replace("/", "--")
    local_dir = cache_dir / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    config_path = local_dir / "config.json"

    # Re-use cached copy if it already exists
    if not config_path.exists():
        print(f"Downloading config.json for {model_id} ...")
        downloaded = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=str(local_dir),
        )
        if Path(downloaded) != config_path:
            shutil.copy(downloaded, config_path)

    with open(config_path) as f:
        cfg: dict = json.load(f)

    patch = _find_patch(model_id)
    if not patch:
        print(f"No patch defined for {model_id!r} — config unchanged.")
        return local_dir

    changed = False
    for key, value in patch.items():
        new_val = value(cfg.get(key)) if callable(value) else value
        if cfg.get(key) != new_val:
            print(f"  Patching {key}: {cfg.get(key)!r} → {new_val!r}")
            cfg[key] = new_val
            changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Patched config written to {config_path}")
    else:
        print("Config already satisfies all patches — no changes written.")

    return local_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch HF model config for vLLM compatibility")
    parser.add_argument("model_id", help="HuggingFace model id, e.g. google/gemma-3-27b-it")
    parser.add_argument(
        "--cache-dir",
        default=".hf_patches",
        help="Local directory for patched configs (default: .hf_patches)",
    )
    args = parser.parse_args()

    out_dir = patch_config(args.model_id, Path(args.cache_dir))
    # Print the local dir so shell scripts can capture it with $(...)
    print(f"LOCAL_MODEL_DIR={out_dir}")
