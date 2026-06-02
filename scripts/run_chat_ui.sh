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
# The served vLLM model is exposed in BiFrost as a "*" wildcard, which the UI
# cannot enumerate. The model name is read from VLLM_MODEL (the same var that
# drives deployments/models/.env), extracted below so the dropdown lists the
# real model instead of a broken "vllm-local/*" entry.
#
# Then open http://127.0.0.1:8100
set -euo pipefail

cd "$(dirname "$0")/.."

# Resolve the served model so the UI can expand the BiFrost "*" wildcard to a
# real model id. Read VLLM_MODEL from the model-serving .env if not already set.
# That file is a docker-compose env_file (values may contain JSON like
# {"rope_scaling": null}), so it is NOT safe to `source` — extract the one line.
if [[ -z "${VLLM_MODEL:-}" && -f deployments/models/.env ]]; then
  VLLM_MODEL="$(grep -E '^[[:space:]]*VLLM_MODEL=' deployments/models/.env \
    | tail -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//')"
  export VLLM_MODEL
fi

# Activate .venv-infer if it exists; otherwise assume the current environment
# already has fastapi installed (e.g. an active venv or the container image).
if [[ -f .venv-infer/bin/activate ]]; then
  source .venv-infer/bin/activate
elif ! python3 -c "import fastapi" &>/dev/null; then
  echo "Error: .venv-infer/ not found and 'fastapi' is not importable in the current environment." >&2
  echo "       Run ./scripts/install_infer.sh, or activate the venv that has fastapi installed." >&2
  exit 1
fi

exec fastapi dev src/llm_workflow_agents/webui/app.py --host 127.0.0.1 --port 8100
