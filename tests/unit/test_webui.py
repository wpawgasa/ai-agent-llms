"""Unit tests for the BiFrost chatbot test UI backend."""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.webui import config


def test_config_defaults_point_at_repo_paths():
    # config path and data dir default to known repo-relative locations
    assert config.bifrost_endpoint() == "http://localhost:23040"
    assert config.bifrost_config_path().name == "config.json"
    assert config.benchmark_data_dir().name == "task_a"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("BIFROST_ENDPOINT", "http://example:9999")
    monkeypatch.setenv("BIFROST_CONFIG", "/tmp/cfg.json")
    monkeypatch.setenv("BENCHMARK_DATA_DIR", "/tmp/data")
    assert config.bifrost_endpoint() == "http://example:9999"
    assert config.bifrost_config_path() == Path("/tmp/cfg.json")
    assert config.benchmark_data_dir() == Path("/tmp/data")


from llm_workflow_agents.webui import gateway


def _write_bifrost_config(path: Path) -> None:
    cfg = {
        "providers": {
            "openai": {"keys": [{"models": ["gpt-x", "gpt-y"]}]},
            "anthropic": {"keys": [{"models": ["claude-z"]}]},
        }
    }
    path.write_text(json.dumps(cfg))


def test_list_models_parses_provider_slash_model(tmp_path):
    cfg = tmp_path / "config.json"
    _write_bifrost_config(cfg)
    models = gateway.list_models(cfg)
    assert models == ["anthropic/claude-z", "openai/gpt-x", "openai/gpt-y"]


def test_list_models_missing_config_returns_empty(tmp_path):
    assert gateway.list_models(tmp_path / "nope.json") == []
