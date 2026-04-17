# Serving Module

## Overview
`serving/` provides vLLM serving utilities used across all phases. Multi-agent orchestration and E2E benchmarking live in `integration/` (see `10-integration.md`).

## Files
- `launch_vllm.sh` — Model-aware vLLM server launch script
- `vllm_utils.py` — Health checks, adapter loading helpers

## launch_vllm.sh
Reads from YAML config and launches vLLM OpenAI-compatible API server:
- `model.name` → `--model`
- `serving.tool_call_parser` → `--tool-call-parser`
- `serving.gpu_memory_utilization` → `--gpu-memory-utilization`
- `serving.max_model_len` → `--max-model-len`
- Optional: `quantization.kv_cache_dtype` → `--kv-cache-dtype`
- Always: `--dtype bfloat16`, `--enable-auto-tool-choice`, `--port 8000`
- Nemotron fallback (Risk R6): if `model.family == "nemotron"`, use HF `generate()` path instead of vLLM

## vllm_utils.py
```python
def wait_for_server(endpoint: str, timeout_s: int = 120) -> bool
def load_lora_adapter(endpoint: str, adapter_name: str, adapter_path: str) -> None
def list_loaded_adapters(endpoint: str) -> list[str]
```

## Checklist
- [ ] Implement launch_vllm.sh with YAML config parsing
- [ ] Add Nemotron vLLM→HF fallback detection (Risk R6)
- [ ] Implement vllm_utils.py with health check and adapter loading
