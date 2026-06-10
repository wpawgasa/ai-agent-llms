# Workflow Debugging Interface — lightweight CPU image.
#
# The UI's import chain (webui + eval.graph_extraction_eval, eval.agent_benchmark,
# data.system_prompt) does NOT touch torch/vllm, so we install only the runtime
# deps it actually needs and add the package itself with --no-deps. This keeps
# the image small instead of pulling the multi-GB torch/peft/accelerate stack
# that `pip install .` (core deps) would drag in.
#
# Build context is the repo root (see docker-compose.yml `build.context`).
FROM python:3.12-slim

WORKDIR /app

# Runtime deps for the webui import chain only (no torch/vllm).
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "httpx>=0.27.0" \
    "pydantic>=2.9.0" \
    "pyyaml>=6.0" \
    "structlog>=24.0.0" \
    "python-dotenv>=1.2.2" \
    "networkx>=3.2" \
    "jsonschema>=4.21.0" \
    "email-validator>=2.0" \
    "python-multipart>=0.0.9"

# Install the project itself without dependencies (they're provided above).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8100

# The served vLLM model is resolved by webui.config.served_vllm_model() from
# VLLM_ENV_FILE (the mounted deployments/models/.env), so no custom entrypoint
# is needed. Bind to 0.0.0.0 so the port is reachable outside the container.
CMD ["uvicorn", "llm_workflow_agents.webui.app:app", "--host", "0.0.0.0", "--port", "8100"]
