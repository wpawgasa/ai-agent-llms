"""Concurrency and latency benchmark for vLLM-served Cat A models.

Sweeps a configurable list of concurrency levels and context lengths,
measures TTFT / TPOT / ITL via streaming SSE, and reports the maximum
sustainable concurrency based on a degradation envelope:

    max_sustainable_concurrency = max N such that:
        ttft_p95[N]  <=  degradation_ttft_multiplier * ttft_p95[baseline]
        AND  failure_rate[N]  <=  max_failure_rate

The sweep always includes concurrency=1 as the baseline regardless of
the user-supplied list, and runs every requested level so that the full
curve is captured for plotting.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PercentileStats:
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99}


def _percentiles(values: list[float]) -> PercentileStats:
    if not values:
        return PercentileStats()
    s = sorted(values)
    n = len(s)
    return PercentileStats(
        p50=s[max(0, int(n * 0.50) - 1)],
        p95=s[max(0, int(n * 0.95) - 1)],
        p99=s[max(0, int(n * 0.99) - 1)],
    )


@dataclass
class LevelResult:
    concurrency: int
    ttft_ms: PercentileStats = field(default_factory=PercentileStats)
    tpot_ms: PercentileStats = field(default_factory=PercentileStats)
    itl_ms: PercentileStats = field(default_factory=PercentileStats)
    e2e_ms: PercentileStats = field(default_factory=PercentileStats)
    throughput_output_tok_s: float = 0.0
    goodput_req_s: float = 0.0
    success_rate: float = 0.0
    peak_vram_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "itl_ms": self.itl_ms.to_dict(),
            "e2e_ms": self.e2e_ms.to_dict(),
            "throughput_output_tok_s": self.throughput_output_tok_s,
            "goodput_req_s": self.goodput_req_s,
            "success_rate": self.success_rate,
            "peak_vram_gb": self.peak_vram_gb,
        }


@dataclass
class ContextSweepResult:
    context_length: int
    baseline_ttft_p95_ms: float
    max_sustainable_concurrency: int
    levels: list[LevelResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "baseline_ttft_p95_ms": self.baseline_ttft_p95_ms,
            "max_sustainable_concurrency": self.max_sustainable_concurrency,
            "levels": [lv.to_dict() for lv in self.levels],
        }


@dataclass
class ConcurrencySweepResult:
    model: str
    kv_cache_dtype: str
    input_tokens: int
    output_tokens: int
    degradation_ttft_multiplier: float
    max_failure_rate: float
    by_context_length: list[ContextSweepResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "kv_cache_dtype": self.kv_cache_dtype,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "degradation_policy": {
                "ttft_multiplier": self.degradation_ttft_multiplier,
                "max_failure_rate": self.max_failure_rate,
            },
            "by_context_length": [c.to_dict() for c in self.by_context_length],
        }


# ---------------------------------------------------------------------------
# Peak VRAM sampler (background thread)
# ---------------------------------------------------------------------------


class _PeakVramSampler:
    """Sample nvidia-smi memory.used every interval_ms and track the maximum."""

    def __init__(self, interval_ms: int = 200) -> None:
        self._interval = interval_ms / 1000.0
        self._peak: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PeakVramSampler":
        self._peak = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _sample(self) -> float:
        import subprocess

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=3,
            )
            mib = float(out.decode().strip().splitlines()[0])
            return mib / 1024.0
        except Exception:
            return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            v = self._sample()
            if v > self._peak:
                self._peak = v
            self._stop.wait(self._interval)

    @property
    def peak_gb(self) -> float:
        return self._peak


# ---------------------------------------------------------------------------
# Per-request streaming measurement
# ---------------------------------------------------------------------------


@dataclass
class _RequestMetrics:
    ttft_ms: float = 0.0
    itl_values_ms: list[float] = field(default_factory=list)
    e2e_ms: float = 0.0
    output_tokens: int = 0
    success: bool = True


async def _stream_request(
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> _RequestMetrics:
    """Issue one streaming request via urllib (sync inside thread-pool) and record latency."""

    def _do_request() -> _RequestMetrics:
        metrics = _RequestMetrics()
        body = json.dumps(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t_start = time.monotonic()
        last_token_t = t_start
        first_token = False
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if not delta.get("content"):
                        continue
                    now = time.monotonic()
                    if not first_token:
                        metrics.ttft_ms = (now - t_start) * 1000.0
                        first_token = True
                    else:
                        metrics.itl_values_ms.append((now - last_token_t) * 1000.0)
                    last_token_t = now
                    metrics.output_tokens += 1
            metrics.e2e_ms = (time.monotonic() - t_start) * 1000.0
        except Exception as exc:
            logger.debug("request_failed", error=str(exc))
            metrics.success = False
            metrics.e2e_ms = (time.monotonic() - t_start) * 1000.0
            metrics.ttft_ms = metrics.e2e_ms
        return metrics

    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _do_request)


# ---------------------------------------------------------------------------
# Level runner
# ---------------------------------------------------------------------------


async def _run_level(
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    total_requests: int,
    warmup_requests: int,
) -> tuple[list[_RequestMetrics], float]:
    """Run `total_requests` requests in batches of `concurrency`, return (metrics, wall_s)."""
    # We fire requests in rolling windows of `concurrency` to model real-world
    # concurrent load rather than strict synchronous batches. A semaphore limits
    # the maximum number of in-flight requests at any time.
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            _stream_request(endpoint, model_name, prompt, max_tokens, sem)
        )
        for _ in range(total_requests)
    ]
    t0 = time.monotonic()
    all_metrics: list[_RequestMetrics] = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - t0
    # Discard warmup results
    return list(all_metrics[warmup_requests:]), wall_s


def _compute_level_result(
    concurrency: int,
    metrics: list[_RequestMetrics],
    wall_s: float,
    peak_vram_gb: float,
) -> LevelResult:
    success = [m for m in metrics if m.success]
    success_rate = len(success) / max(len(metrics), 1)
    ttft_vals = [m.ttft_ms for m in success]
    e2e_vals = [m.e2e_ms for m in success]
    all_itl = [v for m in success for v in m.itl_values_ms]

    # TPOT = e2e / output_tokens (excluding TTFT, approximated as (e2e - ttft) / (n_tok - 1))
    tpot_vals = []
    for m in success:
        if m.output_tokens > 1:
            tpot_vals.append((m.e2e_ms - m.ttft_ms) / (m.output_tokens - 1))

    total_output_tokens = sum(m.output_tokens for m in success)
    throughput = total_output_tokens / wall_s if wall_s > 0 else 0.0
    goodput = len(success) / wall_s if wall_s > 0 else 0.0

    return LevelResult(
        concurrency=concurrency,
        ttft_ms=_percentiles(ttft_vals),
        tpot_ms=_percentiles(tpot_vals),
        itl_ms=_percentiles(all_itl),
        e2e_ms=_percentiles(e2e_vals),
        throughput_output_tok_s=throughput,
        goodput_req_s=goodput,
        success_rate=success_rate,
        peak_vram_gb=peak_vram_gb,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_concurrency_sweep(
    model_name: str,
    kv_cache_dtype: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    context_lengths: list[int] | tuple[int, ...] = (2048, 4096, 8192),
    input_tokens: int = 512,
    output_tokens: int = 128,
    concurrency_levels: list[int] | tuple[int, ...] = (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    ),
    requests_per_level: int = 64,
    warmup_requests: int = 8,
    degradation_ttft_multiplier: float = 2.0,
    max_failure_rate: float = 0.01,
) -> ConcurrencySweepResult:
    """Sweep concurrency levels and context lengths for a running vLLM server.

    The server must already be running and healthy at ``base_url`` before
    this function is called. Server lifecycle is managed by the caller
    (e.g. ``scripts/run_concurrency_benchmark.sh``).

    Degradation rule: ``max_sustainable_concurrency`` is the highest level N
    satisfying both:
      1. ``ttft_p95[N] <= degradation_ttft_multiplier * ttft_p95[level=1]``
      2. ``1 - success_rate[N] <= max_failure_rate``
    If level=1 is not in ``concurrency_levels``, it is prepended automatically
    to serve as the baseline.
    """
    levels_list = sorted(set([1, *concurrency_levels]))

    result = ConcurrencySweepResult(
        model=model_name,
        kv_cache_dtype=kv_cache_dtype,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        degradation_ttft_multiplier=degradation_ttft_multiplier,
        max_failure_rate=max_failure_rate,
    )

    for ctx_len in sorted(context_lengths):
        logger.info(
            "sweep_context_start",
            model=model_name,
            kv_cache_dtype=kv_cache_dtype,
            context_length=ctx_len,
        )
        # Build a representative prompt whose token count approximates input_tokens.
        # Rough rule: 1 token ≈ 4 chars in English.
        snippet = "Describe a step in a business workflow that involves state transitions and tool calls. "
        reps = max(1, (input_tokens * 4) // len(snippet))
        prompt = snippet * reps

        ctx_levels: list[LevelResult] = []
        baseline_ttft_p95: float = 0.0
        consecutive_violations = 0
        total_requests = max(requests_per_level, 1)

        for concurrency in levels_list:
            effective_total = max(total_requests, concurrency + warmup_requests)
            logger.info(
                "level_start",
                concurrency=concurrency,
                context_length=ctx_len,
                requests=effective_total,
            )

            with _PeakVramSampler() as vram_sampler:
                raw_metrics, wall_s = asyncio.run(
                    _run_level(
                        base_url,
                        model_name,
                        prompt,
                        output_tokens,
                        concurrency,
                        effective_total,
                        warmup_requests,
                    )
                )

            lv = _compute_level_result(concurrency, raw_metrics, wall_s, vram_sampler.peak_gb)
            ctx_levels.append(lv)

            if concurrency == 1:
                baseline_ttft_p95 = lv.ttft_ms.p95

            failure_rate = 1.0 - lv.success_rate
            ttft_threshold = (
                degradation_ttft_multiplier * baseline_ttft_p95
                if baseline_ttft_p95 > 0
                else float("inf")
            )
            violated = (
                lv.ttft_ms.p95 > ttft_threshold or failure_rate > max_failure_rate
            )

            logger.info(
                "level_done",
                concurrency=concurrency,
                ttft_p95_ms=round(lv.ttft_ms.p95, 1),
                success_rate=round(lv.success_rate, 4),
                peak_vram_gb=round(lv.peak_vram_gb, 2),
                violated=violated,
            )

            if violated:
                consecutive_violations += 1
                # Stop early only after two consecutive violations (avoids
                # stopping on a single noisy sample at the knee).
                if consecutive_violations >= 2:
                    logger.info(
                        "early_stop",
                        concurrency=concurrency,
                        context_length=ctx_len,
                    )
                    break
            else:
                consecutive_violations = 0

        # Determine max sustainable concurrency from recorded levels
        max_sustainable = 1
        for lv in ctx_levels:
            ttft_threshold = (
                degradation_ttft_multiplier * baseline_ttft_p95
                if baseline_ttft_p95 > 0
                else float("inf")
            )
            if (
                lv.ttft_ms.p95 <= ttft_threshold
                and (1.0 - lv.success_rate) <= max_failure_rate
            ):
                max_sustainable = lv.concurrency

        ctx_result = ContextSweepResult(
            context_length=ctx_len,
            baseline_ttft_p95_ms=baseline_ttft_p95,
            max_sustainable_concurrency=max_sustainable,
            levels=ctx_levels,
        )
        result.by_context_length.append(ctx_result)

        logger.info(
            "sweep_context_done",
            context_length=ctx_len,
            baseline_ttft_p95_ms=round(baseline_ttft_p95, 1),
            max_sustainable_concurrency=max_sustainable,
        )

    return result


# ---------------------------------------------------------------------------
# CLI entry point (invoked by the shell script via python -m)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run concurrency sweep for a vLLM-served model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--context-lengths", default="2048,4096,8192")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--concurrency-levels", default="1,2,4,8,16,32,64,128,256,512,1024"
    )
    parser.add_argument("--requests-per-level", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--degradation-ttft-multiplier", type=float, default=2.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--output", required=True, help="Path for JSON result file")
    args = parser.parse_args()

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]
    conc_levels = [int(x) for x in args.concurrency_levels.split(",")]

    sweep = run_concurrency_sweep(
        model_name=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        base_url=args.base_url,
        context_lengths=ctx_lens,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        concurrency_levels=conc_levels,
        requests_per_level=args.requests_per_level,
        warmup_requests=args.warmup_requests,
        degradation_ttft_multiplier=args.degradation_ttft_multiplier,
        max_failure_rate=args.max_failure_rate,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep.to_dict(), indent=2))
    print(f"Results written to {out_path}", file=sys.stderr)
    for ctx in sweep.by_context_length:
        print(
            f"  ctx={ctx.context_length}: max_sustainable_concurrency={ctx.max_sustainable_concurrency}"
            f"  (baseline TTFT p95={ctx.baseline_ttft_p95_ms:.1f}ms)",
            file=sys.stderr,
        )
