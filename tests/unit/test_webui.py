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
