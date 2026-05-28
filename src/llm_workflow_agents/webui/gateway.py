"""BiFrost gateway helpers: model discovery, request building, SSE proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from llm_workflow_agents.eval.agent_benchmark import _downgrade_tool_turns_to_text


def list_models(config_path: Path) -> list[str]:
    """Return sorted ``provider/model`` strings from a BiFrost config.json."""
    if not config_path.exists():
        return []
    with open(config_path) as f:
        cfg = json.load(f)
    models: list[str] = []
    for provider_name, provider in cfg.get("providers", {}).items():
        for key_entry in provider.get("keys", []):
            for model in key_entry.get("models", []):
                models.append(f"{provider_name}/{model}")
    return sorted(set(models))
