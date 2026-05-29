# BiFrost Chatbot Test UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A localhost browser chatbot that tests any BiFrost-gateway LLM using system prompts built byte-identically to the Phase 1 Task A benchmark.

**Architecture:** A thin FastAPI backend serves a self-contained single-page HTML/JS frontend and proxies streaming chat to the BiFrost gateway. The backend reuses the package's own `build_enriched_system_prompt()` and `_downgrade_tool_turns_to_text()` so prompts and the BiFrost tool-turn quirk match the benchmark exactly. The browser never talks to the gateway directly.

**Tech Stack:** FastAPI, httpx (async streaming), Starlette TestClient, pytest, vanilla HTML/CSS/JS. All already in `.venv`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/llm_workflow_agents/webui/__init__.py` | Package marker |
| `src/llm_workflow_agents/webui/config.py` | Env-driven settings (endpoint, config path, data dir) as functions |
| `src/llm_workflow_agents/webui/gateway.py` | Parse BiFrost config for model list; build chat request body; async SSE stream proxy |
| `src/llm_workflow_agents/webui/samples.py` | List/load Task A benchmark samples; build enriched prompt + tools |
| `src/llm_workflow_agents/webui/app.py` | FastAPI app + 5 routes + index serving |
| `src/llm_workflow_agents/webui/static/index.html` | Self-contained single-page UI |
| `scripts/run_chat_ui.sh` | `fastapi dev` launcher |
| `tests/unit/test_webui.py` | Backend unit tests |

**Conventions observed from the codebase:**
- Modules use `from __future__ import annotations`.
- Tests live in `tests/unit/`, import from `llm_workflow_agents...`, pytest configured with `asyncio_mode = "auto"`.
- Memory: activate the venv before running python (`source .venv/bin/activate && ...`).
- `tool_schemas` in samples are already OpenAI-tools format (`{"type":"function","function":{...}}`) — pass through untouched.
- Run all test commands from the repo root `/workspaces/ai-agent-llms` with the venv active.

---

## Task 1: Package scaffold + config

**Files:**
- Create: `src/llm_workflow_agents/webui/__init__.py`
- Create: `src/llm_workflow_agents/webui/config.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_workflow_agents.webui'`

- [ ] **Step 3: Create the package marker**

Create `src/llm_workflow_agents/webui/__init__.py` (empty):

```python
```

- [ ] **Step 4: Implement config**

Create `src/llm_workflow_agents/webui/config.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/webui/__init__.py src/llm_workflow_agents/webui/config.py tests/unit/test_webui.py
git commit -m "feat(webui): add package scaffold and env-driven config"
```

---

## Task 2: Gateway — list_models

**Files:**
- Create: `src/llm_workflow_agents/webui/gateway.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -k list_models -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_workflow_agents.webui.gateway'`

- [ ] **Step 3: Implement list_models**

Create `src/llm_workflow_agents/webui/gateway.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -k list_models -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/webui/gateway.py tests/unit/test_webui.py
git commit -m "feat(webui): parse BiFrost config into provider/model list"
```

---

## Task 3: Gateway — build_chat_request

**Files:**
- Modify: `src/llm_workflow_agents/webui/gateway.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -k build_chat_request -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_chat_request'`

- [ ] **Step 3: Implement build_chat_request**

Append to `src/llm_workflow_agents/webui/gateway.py`:

```python
def build_chat_request(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat body matching the benchmark's bifrost path.

    Mirrors ``eval.agent_benchmark._call_vllm`` with ``engine='bifrost'``:
    past structured tool turns are rewritten to plain text and the
    vLLM/SGLang-only ``chat_template_kwargs`` field is omitted.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": _downgrade_tool_turns_to_text(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    return body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -k build_chat_request -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/webui/gateway.py tests/unit/test_webui.py
git commit -m "feat(webui): build bifrost chat request mirroring benchmark path"
```

---

## Task 4: Gateway — stream_chat (SSE proxy)

**Files:**
- Modify: `src/llm_workflow_agents/webui/gateway.py`

No unit test (requires a live gateway); covered by the manual verification in Task 10. Keep the function tiny so review is sufficient.

- [ ] **Step 1: Implement stream_chat**

Append to `src/llm_workflow_agents/webui/gateway.py`:

```python
async def stream_chat(endpoint: str, body: dict[str, Any]) -> AsyncIterator[bytes]:
    """Proxy a streaming chat completion from BiFrost, forwarding raw SSE bytes.

    On a gateway error (>=400) a single synthetic SSE ``error`` event is
    emitted followed by ``[DONE]`` so the browser can surface it without the
    stream hanging. No auth header is required by the gateway; a dummy bearer
    is sent because the OpenAI-compatible surface expects the header to exist.
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed",
    }
    timeout = httpx.Timeout(600.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    detail = raw.decode("utf-8", errors="replace")[:1000]
                    err = json.dumps(
                        {"error": {"status": resp.status_code, "body": detail}}
                    )
                    yield f"data: {err}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
    except httpx.HTTPError as exc:
        err = json.dumps({"error": {"status": 0, "body": f"gateway unreachable: {exc}"}})
        yield f"data: {err}\n\n".encode()
        yield b"data: [DONE]\n\n"
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from llm_workflow_agents.webui import gateway; print('ok', hasattr(gateway, 'stream_chat'))"`
Expected: `ok True` (confirms the `agent_benchmark` import chain has no heavy/optional deps)

- [ ] **Step 3: Commit**

```bash
git add src/llm_workflow_agents/webui/gateway.py
git commit -m "feat(webui): add async SSE proxy to the BiFrost gateway"
```

---

## Task 5: Samples — list and load

**Files:**
- Create: `src/llm_workflow_agents/webui/samples.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -k "samples or get_sample" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_workflow_agents.webui.samples'`

- [ ] **Step 3: Implement list_samples and get_sample**

Create `src/llm_workflow_agents/webui/samples.py`:

```python
"""Task A benchmark sample loading and prompt building for the chatbot UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

_LEVELS = {"L1", "L2", "L3", "L4", "L5"}


def _level_files(data_dir: Path, level: str) -> list[Path]:
    return sorted(data_dir.glob(f"{level.lower()}_*.jsonl"))


def _iter_samples(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def list_samples(data_dir: Path, level: str) -> list[dict[str, Any]]:
    """Return lightweight summaries of all samples at a complexity level."""
    if level not in _LEVELS or not data_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for s in _iter_samples(_level_files(data_dir, level)):
        first_user = next(
            (m.get("content", "") for m in s.get("messages", []) if m.get("role") == "user"),
            "",
        )
        out.append(
            {
                "conversation_id": s.get("conversation_id", ""),
                "domain": s.get("domain", ""),
                "num_states": s.get("num_states", 0),
                "num_tools": s.get("num_tools", 0),
                "preview": first_user[:120],
            }
        )
    return out


def get_sample(data_dir: Path, conversation_id: str) -> dict[str, Any] | None:
    """Find a full sample by conversation_id across all level files."""
    if not data_dir.exists():
        return None
    for s in _iter_samples(sorted(data_dir.glob("*.jsonl"))):
        if s.get("conversation_id") == conversation_id:
            return s
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -k "samples or get_sample" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/webui/samples.py tests/unit/test_webui.py
git commit -m "feat(webui): list and load Task A benchmark samples"
```

---

## Task 6: Samples — build_sample_prompt

**Files:**
- Modify: `src/llm_workflow_agents/webui/samples.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -k build_sample_prompt -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_sample_prompt'`

- [ ] **Step 3: Implement build_sample_prompt**

Append to `src/llm_workflow_agents/webui/samples.py`:

```python
def build_sample_prompt(sample: dict[str, Any]) -> dict[str, Any]:
    """Build the benchmark's enriched system prompt + tools + seed user message.

    Reuses ``build_enriched_system_prompt`` so the prompt is identical to what
    ``eval.agent_benchmark`` sends for this sample.
    """
    system_msg = next(
        (m for m in sample.get("messages", []) if m.get("role") == "system"),
        None,
    )
    original = system_msg.get("content", "") if system_msg else ""
    enriched = build_enriched_system_prompt(sample, original)
    seed_user = next(
        (m.get("content", "") for m in sample.get("messages", []) if m.get("role") == "user"),
        "",
    )
    return {
        "system_prompt": enriched,
        "tools": sample.get("tool_schemas", []),
        "seed_user": seed_user,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -k build_sample_prompt -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/webui/samples.py tests/unit/test_webui.py
git commit -m "feat(webui): build enriched prompt and tools from a sample"
```

---

## Task 7: FastAPI app + endpoint tests

**Files:**
- Create: `src/llm_workflow_agents/webui/app.py`
- Test: `tests/unit/test_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_webui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_webui.py -k api_ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_workflow_agents.webui.app'`

- [ ] **Step 3: Implement the FastAPI app**

Create `src/llm_workflow_agents/webui/app.py`:

```python
"""FastAPI app for the BiFrost chatbot test UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from llm_workflow_agents.webui import config, gateway, samples

app = FastAPI(title="BiFrost Chatbot Test UI")

_STATIC = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 1024
    tools: list[dict[str, Any]] | None = None


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    return {"models": gateway.list_models(config.bifrost_config_path())}


@app.get("/api/samples")
def api_samples(level: str = "L1") -> dict[str, Any]:
    return {"samples": samples.list_samples(config.benchmark_data_dir(), level)}


@app.get("/api/samples/{conversation_id}")
def api_sample(conversation_id: str) -> JSONResponse:
    sample = samples.get_sample(config.benchmark_data_dir(), conversation_id)
    if sample is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(samples.build_sample_prompt(sample))


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    body = gateway.build_chat_request(
        req.model, req.messages, req.temperature, req.max_tokens, req.tools
    )
    return StreamingResponse(
        gateway.stream_chat(config.bifrost_endpoint(), body),
        media_type="text/event-stream",
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_webui.py -k api_ -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full webui suite**

Run: `pytest tests/unit/test_webui.py -v`
Expected: PASS (all tests green)

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/webui/app.py tests/unit/test_webui.py
git commit -m "feat(webui): add FastAPI routes for models, samples, and chat proxy"
```

---

## Task 8: Frontend — index.html

**Files:**
- Create: `src/llm_workflow_agents/webui/static/index.html`

- [ ] **Step 1: Apply the frontend-design skill**

Invoke the `frontend-design:frontend-design` skill to guide the visual treatment (typography, spacing, color, distinctive non-generic styling). Apply its aesthetic guidance on top of the functional reference implementation in Step 2 — keep all element IDs and JS behavior intact so the wiring still works.

- [ ] **Step 2: Implement the page**

Create `src/llm_workflow_agents/webui/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BiFrost Chatbot Test UI</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --line:#262b38; --txt:#e6e8ee; --muted:#8a93a6; --accent:#5b8cff; --user:#1f2a44; --asst:#1b2230; --tool:#2a2118; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--txt); display:flex; height:100vh; }
  #side { width:340px; min-width:340px; border-right:1px solid var(--line); padding:16px; overflow-y:auto; background:var(--panel); }
  #main { flex:1; display:flex; flex-direction:column; }
  h1 { font-size:15px; margin:0 0 12px; letter-spacing:.3px; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  select, input, textarea, button { width:100%; background:#0d0f15; color:var(--txt); border:1px solid var(--line); border-radius:8px; padding:8px; font:inherit; }
  button { cursor:pointer; background:var(--accent); color:#fff; border:none; font-weight:600; }
  button.secondary { background:#222838; color:var(--txt); }
  .row { display:flex; gap:8px; }
  details { margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:8px; }
  summary { cursor:pointer; color:var(--muted); font-size:12px; }
  textarea#system { height:180px; resize:vertical; }
  pre#tools { white-space:pre-wrap; word-break:break-word; max-height:200px; overflow:auto; font-size:12px; color:var(--muted); }
  #transcript { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg { padding:10px 12px; border-radius:10px; max-width:80%; white-space:pre-wrap; word-break:break-word; }
  .msg.user { background:var(--user); align-self:flex-end; }
  .msg.assistant { background:var(--asst); align-self:flex-start; }
  .msg.error { background:#3a1d22; align-self:flex-start; color:#ffb4bc; }
  .role { font-size:11px; color:var(--muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:.5px; }
  .toolcall { margin-top:8px; border:1px solid var(--line); border-left:3px solid #c79a4b; background:var(--tool); border-radius:6px; padding:8px; font-family:ui-monospace,monospace; font-size:12px; white-space:pre-wrap; }
  .meta { font-size:11px; color:var(--muted); margin-top:4px; }
  #composer { border-top:1px solid var(--line); padding:12px 16px; display:flex; gap:8px; background:var(--panel); }
  #input { flex:1; height:46px; resize:none; }
  #send { width:auto; padding:0 22px; }
</style>
</head>
<body>
  <aside id="side">
    <h1>BiFrost Chatbot Test UI</h1>

    <label for="model">Model</label>
    <select id="model"></select>

    <label for="temperature">Temperature: <span id="tempVal">0.0</span></label>
    <input id="temperature" type="range" min="0" max="2" step="0.1" value="0" />

    <label for="maxTokens">Max tokens</label>
    <input id="maxTokens" type="number" value="1024" min="1" max="8192" />

    <label for="level">Benchmark level</label>
    <select id="level">
      <option>L1</option><option>L2</option><option>L3</option><option>L4</option><option>L5</option>
    </select>

    <label for="sample">Sample</label>
    <select id="sample"></select>

    <div class="row" style="margin-top:10px;">
      <button id="load">Load sample</button>
      <button id="reset" class="secondary">Reset chat</button>
    </div>

    <details open>
      <summary>System prompt (editable)</summary>
      <textarea id="system" placeholder="Load a sample to build the benchmark system prompt, or type your own."></textarea>
    </details>

    <details>
      <summary>Tools sent (read-only)</summary>
      <pre id="tools">[]</pre>
    </details>
  </aside>

  <main id="main">
    <div id="transcript"></div>
    <div id="composer">
      <textarea id="input" placeholder="Type a message and press Enter…"></textarea>
      <button id="send">Send</button>
    </div>
  </main>

<script>
const $ = (id) => document.getElementById(id);
let tools = [];
let history = [];  // assistant/user/tool turns AFTER the system message

async function loadModels() {
  const r = await fetch('/api/models');
  const { models } = await r.json();
  $('model').innerHTML = models.length
    ? models.map(m => `<option>${m}</option>`).join('')
    : '<option value="">(no models in bifrost config)</option>';
}

async function loadSamples() {
  const level = $('level').value;
  const r = await fetch('/api/samples?level=' + encodeURIComponent(level));
  const { samples } = await r.json();
  $('sample').innerHTML = samples.length
    ? samples.map(s => `<option value="${s.conversation_id}">${s.conversation_id} · ${s.domain} · ${s.num_states} states</option>`).join('')
    : '<option value="">(no samples — generate benchmark data)</option>';
}

async function loadSample() {
  const id = $('sample').value;
  if (!id) return;
  const r = await fetch('/api/samples/' + encodeURIComponent(id));
  if (!r.ok) { addMsg('error', 'Failed to load sample ' + id); return; }
  const data = await r.json();
  $('system').value = data.system_prompt || '';
  tools = data.tools || [];
  $('tools').textContent = JSON.stringify(tools, null, 2);
  if (data.seed_user) $('input').value = data.seed_user;
  resetChat(false);
}

function resetChat(clearSystem) {
  history = [];
  $('transcript').innerHTML = '';
  if (clearSystem) { $('system').value = ''; tools = []; $('tools').textContent = '[]'; }
}

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = `<div class="role">${role}</div>`;
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text || '';
  div.appendChild(body);
  $('transcript').appendChild(div);
  $('transcript').scrollTop = $('transcript').scrollHeight;
  return div;
}

function renderToolCalls(container, toolCalls) {
  for (const tc of toolCalls) {
    const fn = tc.function || {};
    let args = fn.arguments;
    try { args = JSON.parse(args); } catch (e) {}
    const card = document.createElement('div');
    card.className = 'toolcall';
    card.textContent = '<tool_call> ' + JSON.stringify({ name: fn.name, arguments: args });
    container.appendChild(card);
  }
}

async function send() {
  const text = $('input').value.trim();
  if (!text) return;
  $('input').value = '';
  addMsg('user', text);
  history.push({ role: 'user', content: text });

  const messages = [{ role: 'system', content: $('system').value }, ...history];
  const payload = {
    model: $('model').value,
    messages,
    temperature: parseFloat($('temperature').value),
    max_tokens: parseInt($('maxTokens').value, 10),
    tools: tools.length ? tools : null,
  };

  const bubble = addMsg('assistant', '');
  const body = bubble.querySelector('.body');
  const t0 = performance.now();
  let ttft = null;
  let content = '';
  const toolAccum = {};

  let resp;
  try {
    resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    body.textContent = 'Request failed: ' + e;
    bubble.className = 'msg error';
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith('data:')) continue;
      const dataStr = t.slice(5).trim();
      if (dataStr === '[DONE]') continue;
      let chunk;
      try { chunk = JSON.parse(dataStr); } catch (e) { continue; }
      if (chunk.error) {
        body.textContent = 'Gateway error ' + chunk.error.status + ': ' + chunk.error.body;
        bubble.className = 'msg error';
        return;
      }
      const delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) || {};
      if (ttft === null && (delta.content || delta.tool_calls)) ttft = performance.now() - t0;
      if (delta.content) { content += delta.content; body.textContent = content; }
      for (const tcd of delta.tool_calls || []) {
        const i = tcd.index || 0;
        toolAccum[i] = toolAccum[i] || { id: '', type: 'function', function: { name: '', arguments: '' } };
        if (tcd.id) toolAccum[i].id = tcd.id;
        const f = tcd.function || {};
        if (f.name) toolAccum[i].function.name += f.name;
        if (f.arguments) toolAccum[i].function.arguments += f.arguments;
      }
      $('transcript').scrollTop = $('transcript').scrollHeight;
    }
  }

  const toolCalls = Object.keys(toolAccum).sort((a, b) => a - b).map(k => toolAccum[k]);
  if (toolCalls.length) renderToolCalls(bubble, toolCalls);

  const meta = document.createElement('div');
  meta.className = 'meta';
  const total = (performance.now() - t0).toFixed(0);
  meta.textContent = `latency ${total} ms` + (ttft !== null ? ` · ttft ${ttft.toFixed(0)} ms` : '');
  bubble.appendChild(meta);

  const asstMsg = { role: 'assistant', content };
  if (toolCalls.length) asstMsg.tool_calls = toolCalls;
  history.push(asstMsg);
}

$('temperature').addEventListener('input', () => $('tempVal').textContent = $('temperature').value);
$('level').addEventListener('change', loadSamples);
$('load').addEventListener('click', loadSample);
$('reset').addEventListener('click', () => resetChat(false));
$('send').addEventListener('click', send);
$('input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

loadModels();
loadSamples();
</script>
</body>
</html>
```

- [ ] **Step 3: Verify the page is served**

Run: `source .venv/bin/activate && python -c "from starlette.testclient import TestClient; from llm_workflow_agents.webui.app import app; r=TestClient(app).get('/'); print(r.status_code, 'text/html' in r.headers['content-type'])"`
Expected: `200 True`

- [ ] **Step 4: Commit**

```bash
git add src/llm_workflow_agents/webui/static/index.html
git commit -m "feat(webui): add single-page chatbot frontend"
```

---

## Task 9: Launcher script

**Files:**
- Create: `scripts/run_chat_ui.sh`

- [ ] **Step 1: Write the launcher**

Create `scripts/run_chat_ui.sh`:

```bash
#!/usr/bin/env bash
# Launch the BiFrost chatbot test UI (FastAPI dev server).
#
# Prereqs:
#   - The BiFrost gateway is running (deployments/local/docker-compose.yml)
#     and reachable at $BIFROST_ENDPOINT (default http://localhost:23040).
#   - Benchmark data exists at $BENCHMARK_DATA_DIR
#     (default data/output/benchmark/task_a). Generate it via
#     scripts/generate_benchmark_data.sh if missing.
#
# Then open http://127.0.0.1:8100
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

exec fastapi dev src/llm_workflow_agents/webui/app.py --host 127.0.0.1 --port 8100
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/run_chat_ui.sh && echo ok`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_chat_ui.sh
git commit -m "feat(webui): add fastapi dev launcher script"
```

---

## Task 10: Manual verification (golden path)

**Files:** none (verification only)

- [ ] **Step 1: Confirm the full unit suite passes**

Run: `source .venv/bin/activate && pytest tests/unit/test_webui.py -v`
Expected: all tests PASS

- [ ] **Step 2: Start the gateway if not already running**

Run: `cd deployments/local && docker compose up -d llm-gateway && cd ../..`
Then check: `curl -s http://localhost:23040/health || echo "gateway not healthy"`
Expected: a health response (the gateway needs at least one provider key set in `deployments/local/.env`).

- [ ] **Step 3: Launch the UI**

Run: `./scripts/run_chat_ui.sh` (leave running in a terminal)
Open `http://127.0.0.1:8100` in a browser.

- [ ] **Step 4: Walk the golden path**

Verify, observing the browser:
1. Model dropdown is populated from the bifrost config.
2. Selecting level L2 repopulates the sample dropdown.
3. "Load sample" fills the System prompt textarea (contains `Rules:` and `Tool schemas`) and the Tools panel shows the schema JSON; the input pre-fills with the seed user message.
4. Pressing Send streams an assistant reply token-by-token; latency + ttft appear.
5. If the model emits a tool call, it renders as a `<tool_call>` card.
6. A follow-up message continues the multi-turn conversation without error (past tool turns downgraded server-side).

- [ ] **Step 5: Check one edge case**

Stop the gateway (`cd deployments/local && docker compose stop llm-gateway && cd ../..`), send a message, and confirm the UI shows a gateway-unreachable error bubble instead of hanging. Restart it afterward if needed.

- [ ] **Step 6: Report**

State explicitly which steps were verified in the browser and which (if any) could not be checked (e.g., no provider key available). Do not claim success for unverified steps.

---

## Self-Review Notes

- **Spec coverage:** architecture (Tasks 1,7), `/api/models` (Task 2,7), `/api/samples` + detail (Tasks 5,6,7), `/api/chat` SSE proxy with bifrost downgrade (Tasks 3,4,7), frontend controls/system panel/tool rendering/latency (Task 8), error handling (Task 4 + Task 8 client + Task 10 edge case), launcher (Task 9), testing (Tasks 1-7 unit + Task 10 manual), out-of-scope honored (no tool-result injection, no auth/DB). All covered.
- **No placeholders:** every code step contains complete code; commands have expected output.
- **Type consistency:** `bifrost_config_path`/`benchmark_data_dir`/`bifrost_endpoint` used consistently across config/app; `list_models`, `build_chat_request`, `stream_chat`, `list_samples`, `get_sample`, `build_sample_prompt` signatures match between definition, tests, and `app.py` call sites; frontend element IDs match the JS.
