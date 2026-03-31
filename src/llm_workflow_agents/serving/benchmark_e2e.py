"""End-to-end latency and concurrency benchmarking.

Measures maximum concurrent requests, P95 latency, and peak VRAM usage
across model and quantization configurations. Produces Pareto-optimal
configurations across quality × memory × latency axes.

Expected concurrency at 4096-token context (from spec):
  BF16:           ~175 concurrent requests
  FP8:            ~350 concurrent requests
  TurboQuant 3-bit: ~925 concurrent requests
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import structlog

from llm_workflow_agents.analysis.pareto import ParetoPoint, find_pareto_frontier

logger = structlog.get_logger(__name__)

# Axes for Pareto computation
_PARETO_MAXIMIZE = ["task_completion_rate"]
_PARETO_MINIMIZE = ["peak_vram_gb", "p95_latency_ms"]


@dataclass
class BenchmarkResult:
    """Result of a single (model, quantization, context_length) benchmark run."""

    model: str
    kv_cache_dtype: str
    context_length: int
    max_concurrent: int
    p95_latency_ms: float
    peak_vram_gb: float
    task_completion_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "kv_cache_dtype": self.kv_cache_dtype,
            "context_length": self.context_length,
            "max_concurrent": self.max_concurrent,
            "p95_latency_ms": self.p95_latency_ms,
            "peak_vram_gb": self.peak_vram_gb,
            "task_completion_rate": self.task_completion_rate,
        }


def _read_peak_vram_gb() -> float:
    """Read current peak VRAM usage via nvidia-smi, or return 0.0 on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        mib = float(out.decode().strip().splitlines()[0])
        return mib / 1024.0
    except Exception:
        return 0.0


def _compute_p95(latencies: list[float]) -> float:
    """Compute P95 latency from a sorted or unsorted list."""
    if not latencies:
        return 0.0
    sorted_lats = sorted(latencies)
    idx = max(0, int(len(sorted_lats) * 0.95) - 1)
    return sorted_lats[idx]


async def _run_concurrent_requests(
    client: Any,
    model_name: str,
    prompt: str,
    num_concurrent: int,
) -> list[float]:
    """Send num_concurrent requests simultaneously and return per-request latencies."""

    async def _single_request() -> float:
        t0 = time.perf_counter()
        try:
            await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=64,
                temperature=0.0,
            )
        except Exception:
            pass
        return (time.perf_counter() - t0) * 1000.0

    tasks = [_single_request() for _ in range(num_concurrent)]
    return await asyncio.gather(*tasks)


def benchmark_concurrency(
    model_name: str,
    kv_cache_dtype: str,
    context_length: int = 4096,
    base_url: str = "http://localhost:8000/v1",
    latency_threshold_ms: float = 10_000.0,
    max_search_concurrency: int = 1024,
    task_completion_rate: float = 0.0,
) -> BenchmarkResult:
    """Benchmark maximum concurrency for a model + quantization combination.

    Uses binary search to find the maximum number of concurrent requests where
    P95 latency stays below ``latency_threshold_ms``.

    Args:
        model_name: vLLM model name.
        kv_cache_dtype: KV cache dtype (e.g. "fp8", "turboquant", "auto").
        context_length: Token context length for test prompts.
        base_url: vLLM OpenAI-compatible API base URL.
        latency_threshold_ms: P95 latency ceiling for concurrency search.
        max_search_concurrency: Upper bound for binary search.
        task_completion_rate: Task completion rate from eval (passed in, not measured here).

    Returns:
        BenchmarkResult with measured concurrency and latency statistics.
    """
    import openai

    client = openai.AsyncOpenAI(base_url=base_url, api_key="unused")

    # Build a representative prompt of approximately the target context length
    # (rough estimate: 1 token ≈ 4 chars)
    prompt = "Describe a workflow step. " * max(1, context_length // 25)

    logger.info(
        "benchmark_concurrency_start",
        model=model_name,
        kv_cache_dtype=kv_cache_dtype,
        context_length=context_length,
    )

    # Binary search for max concurrency
    lo, hi = 1, max_search_concurrency
    best_concurrent = 1
    best_latencies: list[float] = [0.0]

    while lo <= hi:
        mid = (lo + hi) // 2
        latencies = asyncio.run(
            _run_concurrent_requests(client, model_name, prompt, mid)
        )
        p95 = _compute_p95(latencies)
        if p95 <= latency_threshold_ms:
            best_concurrent = mid
            best_latencies = latencies
            lo = mid + 1
        else:
            hi = mid - 1

    peak_vram = _read_peak_vram_gb()
    p95_final = _compute_p95(best_latencies)

    result = BenchmarkResult(
        model=model_name,
        kv_cache_dtype=kv_cache_dtype,
        context_length=context_length,
        max_concurrent=best_concurrent,
        p95_latency_ms=p95_final,
        peak_vram_gb=peak_vram,
        task_completion_rate=task_completion_rate,
    )

    logger.info(
        "benchmark_concurrency_complete",
        **result.to_dict(),
    )
    return result


def compute_pareto_frontier(
    results: list[BenchmarkResult],
    axes: tuple[str, ...] = ("task_completion_rate", "peak_vram_gb", "p95_latency_ms"),
) -> list[BenchmarkResult]:
    """Identify Pareto-optimal benchmark configurations.

    Maps BenchmarkResult objects to ParetoPoint objects, applies Pareto
    dominance filtering, then returns the corresponding BenchmarkResult objects.

    Axis directions:
      - task_completion_rate: maximize (higher is better)
      - peak_vram_gb: minimize (lower is better)
      - p95_latency_ms: minimize (lower is better)

    Args:
        results: List of BenchmarkResult objects from benchmark runs.
        axes: Axis names to include in the Pareto computation.

    Returns:
        List of Pareto-optimal BenchmarkResult objects.
    """
    if not results:
        return []

    maximize = [a for a in axes if a in _PARETO_MAXIMIZE]
    minimize = [a for a in axes if a in _PARETO_MINIMIZE]

    points = [
        ParetoPoint(
            config_name=f"{r.model}_{r.kv_cache_dtype}_{r.context_length}",
            metrics={
                "task_completion_rate": r.task_completion_rate,
                "peak_vram_gb": r.peak_vram_gb,
                "p95_latency_ms": r.p95_latency_ms,
            },
        )
        for r in results
    ]

    frontier_points = find_pareto_frontier(points, maximize=maximize, minimize=minimize)
    frontier_names = {p.config_name for p in frontier_points}

    pareto_results = [
        r
        for r, p in zip(results, points)
        if p.config_name in frontier_names
    ]

    logger.info(
        "pareto_frontier_e2e",
        total_configs=len(results),
        pareto_optimal=len(pareto_results),
    )
    return pareto_results
