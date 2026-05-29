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
