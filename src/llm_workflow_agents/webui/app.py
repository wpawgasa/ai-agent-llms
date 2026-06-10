"""FastAPI app for the Workflow Debugging Interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_workflow_agents.webui import config, gateway, samples

app = FastAPI(title="Workflow Debugging Interface")

_STATIC = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 1024
    tools: list[dict[str, Any]] | None = None


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    return {
        "models": gateway.list_models(
            config.bifrost_config_path(), config.served_vllm_model()
        )
    }


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


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
