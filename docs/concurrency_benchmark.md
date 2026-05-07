# Concurrency Benchmark — How It Works

This document describes how `scripts/run_concurrency_benchmark.sh` and `src/llm_workflow_agents/eval/concurrency_benchmark.py` measure the latency / throughput / max-sustainable-concurrency surface of a serving stack — both **local vLLM** and **frontier models routed through the BiFrost LLM gateway**.

It mirrors the code as of the commit that added frontier-model support. Quality of generated content is **not** measured here; that is the job of `eval/agent_benchmark.py` (see `metrics.md`).

---

## 1. What this benchmark answers

For one (model, serving stack) pair, at each context length, find:

- **TTFT, TPOT, ITL, end-to-end latency** — p50 / p95 / p99 percentiles per concurrency level.
- **Throughput** (output tok/s) and **goodput** (successful req/s).
- **Peak VRAM** under load (vLLM only).
- **Max sustainable concurrency** — the highest concurrency level whose TTFT and failure rate stay within a defined degradation envelope.

The output JSON is consumed by `scripts/build_concurrency_report.py` to compose the cross-model / cross-quantization report.

---

## 2. Two modes

### vLLM mode (local server)

```bash
bash scripts/run_concurrency_benchmark.sh configs/models/cat_a/qwen3_32b.yaml \
    --kv-cache-dtype turboquant_3bit_nc
```

The runner:

1. Reads `model.name` and serving knobs from the YAML.
2. Computes `max_model_len = max(context_lengths) + output_tokens + 64`.
3. Launches `serving/launch_vllm.sh` in the background with the requested `--kv-cache-dtype`, `--port`, and `--max-model-len`.
4. Polls `http://localhost:<port>/health` for up to 450 s (90 × 5 s).
5. Runs the Python sweep harness against `http://localhost:<port>`.
6. On exit, kills the vLLM process via a `trap cleanup`.

### Frontier mode (BiFrost gateway)

```bash
bash scripts/run_concurrency_benchmark.sh \
    --frontier-model anthropic/claude-sonnet-4-6
```

The runner:

1. Loads the placeholder `configs/models_exp_a/frontier.yaml`, which declares `serving.engine: bifrost` and `serving.endpoint: http://localhost:23040`.
2. Validates `<provider/model>` against `deployments/local/data/bifrost/config.json`. If unknown, lists the allowed set and exits.
3. **Skips** the vLLM launch, the health poll, and the `max_model_len` computation. The BiFrost gateway is assumed to be already running and managed externally.
4. Sends `kv_cache_dtype="remote"` as metadata only — there is no local KV cache to quantize.
5. Runs the same sweep harness against the BiFrost endpoint.

The two modes are **mutually exclusive**: passing both a positional config and `--frontier-model` is rejected.

---

## 3. The sweep loop

Implemented by `run_concurrency_sweep` in `concurrency_benchmark.py`.

For each `context_length` in the requested list, and within that, for each `concurrency` level:

1. Build a synthetic prompt sized to approximately `input_tokens`:
   ```python
   snippet = "Describe a step in a business workflow that involves state transitions and tool calls. "
   reps   = max(1, (input_tokens * 4) // len(snippet))   # 4 chars/token rule of thumb
   prompt = snippet * reps
   ```
   Every request in every level uses the *same* prompt and `max_tokens=output_tokens` (default 128) at `temperature=0.0`. This isolates serving-stack variance from content variance — see *Caveats* below.
2. Issue `effective_total = max(requests_per_level, concurrency + warmup_requests)` streaming requests, with concurrency capped by `asyncio.Semaphore(concurrency)`. Requests are fired as a rolling window, not as strict synchronous batches, to model real-world concurrent load.
3. **In vLLM mode only**, a background `_PeakVramSampler` polls `nvidia-smi --query-gpu=memory.used` every 200 ms; the maximum observed value is recorded as `peak_vram_gb`. Frontier mode reports `peak_vram_gb=0.0` (no local GPU).
4. Each request is parsed as an SSE stream (`/v1/chat/completions` with `"stream": true`). Per-request fields recorded:
   - `ttft_ms` — wall-clock from request send to first content/tool delta.
   - `itl_values_ms` — gaps between successive content tokens.
   - `e2e_ms` — total wall-clock to `[DONE]` (or to error).
   - `output_tokens` — count of content deltas.
   - `success` — whether the response stream completed without exception.
5. The first `warmup_requests` results are discarded.
6. Aggregate the remaining metrics for the level (`_compute_level_result`):
   - `ttft_ms`, `e2e_ms`, `itl_ms`: p50 / p95 / p99 over successful requests.
   - `tpot_ms`: p50 / p95 / p99 of `(e2e − ttft) / (output_tokens − 1)` per request.
   - `throughput_output_tok_s`: `Σ output_tokens / wall_time` across the level.
   - `goodput_req_s`: `n_success / wall_time`.
   - `success_rate`: `n_success / n_total`.

### Default sweep parameters

| Parameter | vLLM default | Frontier default | Why frontier differs |
|-----------|--------------|------------------|----------------------|
| `context_lengths` | `2048,4096,8192` | same | — |
| `input_tokens` | 512 | same | — |
| `output_tokens` | 128 | same | — |
| `concurrency_levels` | `1,2,4,8,16,32,64,128,256,512,1024` | `1,2,4,8,16,32` | Provider-side rate limits; tail levels would just produce 429s. |
| `requests_per_level` | 64 | 16 | Cost guardrail (~288 calls / model across 3 contexts vs ~2 100 for vLLM). |
| `warmup_requests` | 8 | 8 | — |
| `degradation_ttft_multiplier` | 2.0 | 2.0 | — |
| `max_failure_rate` | 0.01 | 0.01 | — |

All defaults are overridable via the corresponding `--…` flags.

### Early-stop

After computing each level, the harness evaluates whether it violated the degradation envelope (defined below). **Two consecutive violations** end the sweep for that context length. A single noisy violation does not, since the knee is often jagged.

---

## 4. The degradation envelope

The headline metric is `max_sustainable_concurrency`. Its definition (`concurrency_benchmark.py:7-9`):

> The highest level **N** for which **both**:
> 1. `ttft_p95[N] ≤ degradation_ttft_multiplier × ttft_p95[level=1]`
> 2. `1 − success_rate[N] ≤ max_failure_rate`

`level=1` is always prepended to the sweep regardless of the user's list, so the baseline TTFT is always measured. After the sweep completes, every recorded level is rechecked against the rule and the highest passing concurrency is reported.

Defaults: `2× baseline TTFT` and `1% failure rate`. Tighten with `--degradation-ttft-mul` for stricter SLA targets, or loosen for "best-effort" deployments.

---

## 5. Output schema

JSON written to `results/concurrency/<model>_<tag>.json` where `<tag>` is the `kv_cache_dtype` for vLLM or the literal `frontier` for BiFrost. Shape:

```json
{
  "model": "anthropic/claude-sonnet-4-6",
  "engine": "vllm" | "bifrost",
  "endpoint": "http://localhost:23040",
  "kv_cache_dtype": "remote",
  "input_tokens": 512,
  "output_tokens": 128,
  "degradation_policy": {
    "ttft_multiplier": 2.0,
    "max_failure_rate": 0.01
  },
  "by_context_length": [
    {
      "context_length": 4096,
      "baseline_ttft_p95_ms": 812.3,
      "max_sustainable_concurrency": 32,
      "levels": [
        {
          "concurrency": 1,
          "ttft_ms":  {"p50": ..., "p95": ..., "p99": ...},
          "tpot_ms":  {"p50": ..., "p95": ..., "p99": ...},
          "itl_ms":   {"p50": ..., "p95": ..., "p99": ...},
          "e2e_ms":   {"p50": ..., "p95": ..., "p99": ...},
          "throughput_output_tok_s": ...,
          "goodput_req_s":           ...,
          "success_rate":            1.0,
          "peak_vram_gb":            0.0
        },
        ...
      ]
    },
    ...
  ]
}
```

`engine` and `endpoint` distinguish frontier runs from vLLM runs at parse time — important when aggregating across the matrix.

---

## 6. Caveats

- **Synthetic prompt — provider caching.** Every request sends the same repeated sentence. vLLM automatic prefix caching and provider-side caching (Anthropic, OpenAI, Gemini) will hit on every request after the first, so frontier TTFT numbers reflect cached prefill, not cold prefill. To defeat caching, pass `--input-tokens` along with code changes to vary the prompt per request — the current harness does not vary it.
- **Token counting is approximate.** The 4-chars-per-token rule is rough; the resulting prompt may be ±10-20 % of the requested `input_tokens` after real tokenization.
- **`context_length` does not vary the prompt.** It only sizes `max_model_len` for the launcher and is recorded for grouping in the JSON. All three context-length sweeps use the same `input_tokens`-sized prompt.
- **Streaming is mandatory.** TTFT and ITL require SSE deltas. A backend that does not stream will record `ttft_ms = e2e_ms` and empty ITL.
- **Peak VRAM is GPU-0 only.** The sampler reads the first nvidia-smi line. Multi-GPU deployments need the harness extended.
- **Cost.** Frontier mode at default settings issues ~288 paid API calls per model. Increase `--requests-per-level` and `--concurrency-levels` deliberately.
- **Frontier rate limits.** Provider 429s show up as `success: false` and inflate `failure_rate`, which then trips the degradation rule. If you see early-stop at concurrency=4 or 8 on a frontier model, the limiting factor is the API tier, not the model.

---

## 7. CLI reference

```
scripts/run_concurrency_benchmark.sh <config> [OPTIONS]
scripts/run_concurrency_benchmark.sh --frontier-model <provider/name> [OPTIONS]
```

| Flag | Default (vLLM / frontier) | Notes |
|------|---------------------------|-------|
| `<config>` | — | Cat A model YAML. Mutually exclusive with `--frontier-model`. |
| `--frontier-model` | — | `<provider>/<model>` from `bifrost/config.json`. |
| `--kv-cache-dtype` | `auto` / forced to `remote` | vLLM-only knob. |
| `--context-lengths` | `2048,4096,8192` | Comma-separated. |
| `--input-tokens` | `512` | Approximate prompt size. |
| `--output-tokens` | `128` | `max_tokens` per request. |
| `--concurrency-levels` | `1..1024` / `1..32` | Comma-separated. |
| `--requests-per-level` | `64` / `16` | Excluding warmup. |
| `--warmup-requests` | `8` | Discarded. |
| `--degradation-ttft-mul` | `2.0` | TTFT-p95 envelope multiplier. |
| `--max-failure-rate` | `0.01` | Envelope failure-rate cap. |
| `--port` | `8000` | vLLM-only. |
| `--results-dir` | `results/concurrency` | — |
| `--dry-run` | off | Prints commands without executing. |
