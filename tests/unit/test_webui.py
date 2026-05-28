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


def test_build_sample_prompt_uses_enriched_builder():
    sample = {
        "conversation_id": "L1_TEST",
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
            {"role": "user", "content": "hi there"},
        ],
    }
    result = samples.build_sample_prompt(sample)
    # enriched prompt keeps the original persona and appends FORMAT_RULES + tool schemas
    assert "You are a helpful agent." in result["system_prompt"]
    assert "Rules:" in result["system_prompt"]
    assert "Tool schemas" in result["system_prompt"]
    assert result["tools"] == sample["tool_schemas"]
    assert result["seed_user"] == "hi there"


from starlette.testclient import TestClient


def test_api_models_endpoint(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write_bifrost_config(cfg)
    monkeypatch.setenv("BIFROST_CONFIG", str(cfg))
    from llm_workflow_agents.webui.app import app

    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == [
        "anthropic/claude-z",
        "openai/gpt-x",
        "openai/gpt-y",
    ]


def test_api_samples_endpoint(tmp_path, monkeypatch):
    _write_sample(tmp_path / "l1_a.jsonl", "L1_001", "faq")
    monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path))
    from llm_workflow_agents.webui.app import app

    client = TestClient(app)
    resp = client.get("/api/samples", params={"level": "L1"})
    assert resp.status_code == 200
    ids = [s["conversation_id"] for s in resp.json()["samples"]]
    assert ids == ["L1_001"]


def test_api_sample_detail_and_404(tmp_path, monkeypatch):
    _write_sample(tmp_path / "l1_a.jsonl", "L1_001", "faq")
    monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path))
    from llm_workflow_agents.webui.app import app

    client = TestClient(app)
    ok = client.get("/api/samples/L1_001")
    assert ok.status_code == 200
    assert "Rules:" in ok.json()["system_prompt"]
    assert ok.json()["seed_user"] == "hello from L1_001"

    missing = client.get("/api/samples/NOPE")
    assert missing.status_code == 404


def test_api_chat_unreachable_gateway_streams_error_event(monkeypatch):
    # Point the gateway at a closed port so stream_chat hits httpx.HTTPError
    # and emits a synthetic SSE error event instead of hanging.
    monkeypatch.setenv("BIFROST_ENDPOINT", "http://127.0.0.1:1")
    from llm_workflow_agents.webui.app import app

    client = TestClient(app)
    payload = {
        "model": "openai/gpt-x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    body = resp.text
    assert "error" in body
    assert "[DONE]" in body


def _graph_sample() -> dict:
    return {
        "conversation_id": "GRAPH_TEST",
        "workflow_graph": {
            "states": ["GREETING", "VERIFY", "TERMINAL"],
            "transitions": [
                {"from": "GREETING", "to": "VERIFY", "condition": "proceed"},
                {"from": "VERIFY", "to": "TERMINAL", "condition": "verified"},
            ],
            "initial": "GREETING",
            "terminal": ["TERMINAL"],
        },
        "tool_schemas": [],
        "messages": [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "hi"},
        ],
    }


def test_build_workflow_mermaid_renders_states_edges_and_highlights():
    markup = samples.build_workflow_mermaid(_graph_sample())
    assert markup.startswith("graph TD")
    # one node line per state
    for state in ("GREETING", "VERIFY", "TERMINAL"):
        assert f"{state}[{state}]" in markup
    # edges carry the condition label
    assert "GREETING -->|proceed| VERIFY" in markup
    assert "VERIFY -->|verified| TERMINAL" in markup
    # highlight styling for initial + terminal
    assert "classDef initial" in markup
    assert "classDef terminal" in markup
    assert "class GREETING initial" in markup
    assert "class TERMINAL terminal" in markup


def test_build_workflow_mermaid_empty_graph_returns_empty():
    assert samples.build_workflow_mermaid({"conversation_id": "X", "messages": []}) == ""


def test_build_sample_prompt_includes_mermaid():
    result = samples.build_sample_prompt(_graph_sample())
    assert "mermaid" in result
    assert result["mermaid"].startswith("graph TD")


def test_api_sample_detail_includes_mermaid(tmp_path, monkeypatch):
    path = tmp_path / "l2_a.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(_graph_sample()) + "\n")
    monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path))
    from llm_workflow_agents.webui.app import app

    client = TestClient(app)
    resp = client.get("/api/samples/GRAPH_TEST")
    assert resp.status_code == 200
    assert resp.json()["mermaid"].startswith("graph TD")


def test_static_mount_serves_vendor_dir(tmp_path, monkeypatch):
    # The /static mount must exist so the vendored mermaid lib is reachable.
    from llm_workflow_agents.webui.app import app

    routes = [getattr(r, "path", "") for r in app.routes]
    assert any(p == "/static" or p.startswith("/static") for p in routes)
