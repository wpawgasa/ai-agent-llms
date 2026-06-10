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

    Resolution order (mirrors ``scripts/run_chat_ui.sh``):
    1. ``VLLM_MODEL`` from the environment, if non-empty.
    2. Otherwise, when ``VLLM_ENV_FILE`` points at a model-serving env file
       (the docker service mounts ``deployments/models/.env`` there), read the
       ``VLLM_MODEL=`` line from it. This is how the container resolves the
       served model without threading ``VLLM_MODEL`` through the environment.

    Returns ``None`` when unresolved, in which case the unusable wildcard is
    dropped. The file fallback is opt-in via ``VLLM_ENV_FILE`` so local and test
    runs (no such var) keep returning ``None`` when ``VLLM_MODEL`` is unset.
    """
    model = os.environ.get("VLLM_MODEL")
    if model:
        return model
    return _vllm_model_from_env_file()


def _vllm_model_from_env_file() -> str | None:
    """Extract the ``VLLM_MODEL`` value from the file named by ``VLLM_ENV_FILE``.

    The file holds JSON values like ``{"rope_scaling": null}``, so it is parsed
    line-wise (not sourced): the last ``VLLM_MODEL=`` assignment wins, with any
    trailing inline comment and surrounding quotes stripped. Commented example
    lines (``#   VLLM_MODEL=...``) are ignored. Returns ``None`` if the var is
    unset, the path is missing, or no usable value is found.
    """
    env_file = os.environ.get("VLLM_ENV_FILE")
    if not env_file:
        return None
    path = Path(env_file)
    if not path.exists():
        return None
    value: str | None = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("VLLM_MODEL="):
            raw = stripped[len("VLLM_MODEL=") :].split("#", 1)[0].strip()
            value = raw.strip("'\"") or None
    return value
