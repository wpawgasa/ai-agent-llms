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
