# BiFrost Chatbot Test UI — Design

**Date:** 2026-05-28
**Status:** Approved (pending implementation plan)

## Goal

A simple browser chatbot for manually testing any LLM served through the BiFrost
gateway. System prompts are built **byte-identically** to the Phase 1 Task A
benchmark by reusing the project's own prompt-construction code, so what you see
in the chat is what the benchmark sends.

This is a localhost developer/test tool, not a production or packaged feature.

## Context

- **Gateway:** BiFrost runs at `http://localhost:23040`, OpenAI-compatible at
  `/v1/chat/completions`. Models are named `provider/model` (e.g.
  `openai/gpt-5.4-2026-03-05`, `anthropic/claude-sonnet-4-6`,
  `gemini/gemini-3.1-pro-preview`). The authoritative model list is in
  `deployments/local/data/bifrost/config.json`. API keys are server-side; the
  client sends no auth header.
- **Prompt setup (mimic target):**
  `src/llm_workflow_agents/data/system_prompt.py::build_enriched_system_prompt()`
  assembles the system prompt as **persona + auto-generated workflow script +
  structured reference (initial/terminal states + tool schemas as OpenAI-tools
  JSON) + FORMAT_RULES**.
- **Benchmark request behavior (mimic target):**
  `src/llm_workflow_agents/eval/agent_benchmark.py` calls the endpoint with
  `messages`, `temperature`, `tools=[...]`, `stream:true`, and renders tool calls
  as `<tool_call>{...}</tool_call>`. For the `bifrost` engine it (a) omits the
  `chat_template_kwargs` field and (b) rewrites past structured tool turns to
  plain text via `_downgrade_tool_turns_to_text()` (Gemini-via-BiFrost rejects
  re-sent `functionCall` parts that lack a `thought_signature`).
- **Stack already available in `.venv`:** `fastapi` (0.135.2), `fastapi-cli`,
  `openai` SDK (2.30.0), `python-dotenv`, `structlog`.
- **Benchmark data present:** `data/output/benchmark/task_a/l{1..5}_*.jsonl`.
  Each sample has keys: `conversation_id`, `complexity_level`, `domain`,
  `num_states`, `num_tools`, `chain_depth`, `workflow_graph`, `workflow_script`,
  `tool_schemas`, `messages`, `user_behavior`, `language`, `ground_truth`.

## Architecture

Thin **FastAPI** backend serves a single-page **HTML/JS** frontend and proxies to
BiFrost. The backend imports the package's own `build_enriched_system_prompt()`
and `_downgrade_tool_turns_to_text()` so prompt construction and the BiFrost
message-downgrade quirk match the benchmark exactly. The browser never talks to
the gateway directly — no CORS or key handling in JS.

### Files

- `src/llm_workflow_agents/webui/__init__.py`
- `src/llm_workflow_agents/webui/app.py` — FastAPI app and endpoints
- `src/llm_workflow_agents/webui/static/index.html` — single-page UI
  (HTML + CSS + JS; the frontend-design skill drives the visual treatment)
- `scripts/run_chat_ui.sh` — `fastapi dev` launcher

### Configuration (env, with defaults)

- `BIFROST_ENDPOINT` (default `http://localhost:23040`)
- `BIFROST_CONFIG` (default `deployments/local/data/bifrost/config.json`)
- `BENCHMARK_DATA_DIR` (default `data/output/benchmark/task_a`)

## Backend Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve `index.html` |
| GET | `/api/models` | Parse `bifrost config.json` → list of `provider/model` strings (grouped by provider) |
| GET | `/api/samples?level=L1..L5` | List samples from `l{n}_*.jsonl`: `conversation_id`, `domain`, `num_states`, `num_tools`, first-user-message preview |
| GET | `/api/samples/{conversation_id}` | Return the enriched system prompt (via `build_enriched_system_prompt`), the `tool_schemas`, and the seed user message |
| POST | `/api/chat` | SSE stream. Body `{model, messages, temperature, max_tokens, tools}`. Forwards to BiFrost `/v1/chat/completions` with `stream:true`, the bifrost path (no `chat_template_kwargs`, past tool turns downgraded), streaming content + tool-call deltas back to the browser |

## Frontend (single page)

- **Controls bar:** model dropdown (`/api/models`), temperature slider,
  max_tokens, level selector (L1–L5), sample dropdown (`/api/samples`), "Load
  sample" button.
- **System prompt panel** (collapsible, editable): filled by Load sample with the
  built enriched prompt; the tool schemas are shown read-only beneath it (these
  get sent as `tools`). Edits to the textarea are honored on send.
- **Transcript:** user/assistant bubbles, streaming tokens, `<tool_call>` blocks
  rendered as distinct cards. Per-response latency + TTFT displayed (on-theme with
  the benchmark's metrics).
- **Input box + Send:** multi-turn, client-side message history. **Reset** clears
  the chat but keeps the system prompt.

## Data Flow

1. Page loads → fetch models + L1 samples.
2. User picks a sample → "Load sample" → backend builds the enriched system prompt
   and extracts `tool_schemas` → UI fills the system panel and stores the tools.
3. User types → `POST /api/chat` with full `messages` (system + history + new
   user) + `tools` + `model` + `temperature` → SSE stream → render bubbles and
   tool cards.
4. Continue multi-turn. Past structured tool turns are downgraded to text on each
   request, mirroring the benchmark's BiFrost path.

## Decisions

- **Tool-result injection is out of scope.** The user continues the conversation
  by typing as the user; there is no UI to paste a `tool` role result back. Past
  tool turns are textualized via `_downgrade_tool_turns_to_text`, matching the
  benchmark. (Confirmed with user.)
- **Backend proxy, not direct browser→gateway.** Chosen so the real Python prompt
  builder is reused (no JS reimplementation that could drift) and no keys/CORS
  live in the browser.

## Error Handling

- Gateway down, non-2xx, or missing provider key → surface the gateway's error
  message as an error bubble in the transcript; do not crash the stream. Mirrors
  the benchmark's tolerance (it logs and continues rather than aborting).
- Missing `bifrost config.json` or benchmark data dir → friendly empty states from
  the relevant endpoints.

## Out of Scope (YAGNI)

- No authentication, no database/persistence, no conversation export.
- Single-user, localhost only.
- No automated scoring or metric computation (that is the benchmark's job) — only
  live latency/TTFT display.
- Not registered as a packaged console entrypoint; run via `fastapi dev`.

## Testing

- **Backend unit tests (light):** `/api/models` config parsing; `/api/samples/{id}`
  prompt building (gateway mocked). It is a dev tool, so coverage is intentionally
  minimal.
- **Manual:** run the app via `scripts/run_chat_ui.sh`, click through the golden
  path (load sample → send → stream → tool-call render → multi-turn) and a couple
  of edge cases (gateway down, empty data dir).
