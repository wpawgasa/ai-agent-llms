"""Environment-driven settings for the chatbot test UI.

Exposed as functions (not module constants) so tests can override via
monkeypatch.setenv without re-importing.
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    # webui/config.py -> webui -> llm_workflow_agents -> src -> repo root
    return Path(__file__).resolve().parents[3]


def bifrost_endpoint() -> str:
    return os.environ.get("BIFROST_ENDPOINT", "http://localhost:23040")


def bifrost_config_path() -> Path:
    default = _project_root() / "deployments/local/data/bifrost/config.json"
    return Path(os.environ.get("BIFROST_CONFIG", str(default)))


def benchmark_data_dir() -> Path:
    default = _project_root() / "data/output/benchmark/task_a"
    return Path(os.environ.get("BENCHMARK_DATA_DIR", str(default)))


def served_vllm_model() -> str | None:
    """Resolve the wildcard (``"*"``) model in the BiFrost vllm-local provider.

    The deployment serves a single, configurable vLLM model whose name lives in
    ``VLLM_MODEL`` (deployments/models/.env), so BiFrost's config uses ``["*"]``
    as a pass-through. BiFrost cannot enumerate it (``list_models`` unsupported
    for the vllm provider), so the UI substitutes this name for the ``*`` entry.
    Returns ``None`` when unset, in which case the unusable wildcard is dropped.
    """
    return os.environ.get("VLLM_MODEL") or None
