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


def test_build_chat_request_downgrades_tool_turns_and_sets_stream():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result-text"},
        {"role": "user", "content": "thanks"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    body = gateway.build_chat_request(
        "openai/gpt-x", messages, temperature=0.3, max_tokens=256, tools=tools
    )

    assert body["model"] == "openai/gpt-x"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 256
    assert body["stream"] is True
    assert body["tools"] == tools
    # benchmark bifrost path: no vLLM-only field
    assert "chat_template_kwargs" not in body
    # past tool turns are textualized: no structured tool roles/fields survive
    roles = [m["role"] for m in body["messages"]]
    assert "tool" not in roles
    assert all("tool_calls" not in m for m in body["messages"])
    joined = " ".join(m.get("content", "") for m in body["messages"])
    assert "<tool_call>" in joined
    assert "result-text" in joined


def test_build_chat_request_omits_tools_when_none():
    body = gateway.build_chat_request("m", [{"role": "user", "content": "hi"}])
    assert "tools" not in body


from llm_workflow_agents.webui import samples


def _write_sample(path: Path, conv_id: str, domain: str) -> None:
    sample = {
        "conversation_id": conv_id,
        "domain": domain,
        "num_states": 4,
        "num_tools": 1,
        "workflow_graph": {"initial": "GREETING", "terminal": ["TERMINAL"]},
        "tool_schemas": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": f"hello from {conv_id}"},
        ],
    }
    with open(path, "a") as f:
        f.write(json.dumps(sample) + "\n")


def test_list_samples_filters_by_level(tmp_path):
    _write_sample(tmp_path / "l1_a.jsonl", "L1_001", "faq")
    _write_sample(tmp_path / "l2_b.jsonl", "L2_001", "booking")
    result = samples.list_samples(tmp_path, "L2")
    assert len(result) == 1
    assert result[0]["conversation_id"] == "L2_001"
    assert result[0]["domain"] == "booking"
    assert result[0]["preview"] == "hello from L2_001"


def test_list_samples_unknown_level_returns_empty(tmp_path):
    assert samples.list_samples(tmp_path, "L9") == []


def test_get_sample_finds_across_files(tmp_path):
    _write_sample(tmp_path / "l1_a.jsonl", "L1_001", "faq")
    _write_sample(tmp_path / "l2_b.jsonl", "L2_001", "booking")
    found = samples.get_sample(tmp_path, "L2_001")
    assert found is not None
    assert found["conversation_id"] == "L2_001"
    assert samples.get_sample(tmp_path, "NOPE") is None
