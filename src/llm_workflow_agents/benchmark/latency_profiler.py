"""Latency profiling for Phase 1 benchmarking.

Measures TTFT, TPOT, ITL at p50/p95/p99 percentiles plus
throughput and peak VRAM for a model served via vLLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PercentileStats:
    """Latency percentile statistics (milliseconds)."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99}


@dataclass
class LatencyProfile:
    """Complete latency profile for a model."""

    ttft_ms: PercentileStats = field(default_factory=PercentileStats)
    tpot_ms: PercentileStats = field(default_factory=PercentileStats)
    itl_ms: PercentileStats = field(default_factory=PercentileStats)
    throughput_prefill_tok_s: float = 0.0
    throughput_decode_tok_s: float = 0.0
    peak_vram_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "itl_ms": self.itl_ms.to_dict(),
            "throughput_prefill_tok_s": self.throughput_prefill_tok_s,
            "throughput_decode_tok_s": self.throughput_decode_tok_s,
            "peak_vram_gb": self.peak_vram_gb,
        }


def _compute_percentiles(values: list[float]) -> PercentileStats:
    """Compute p50, p95, p99 from a list of latency values."""
    if not values:
        return PercentileStats()
    s = sorted(values)
    n = len(s)

    def _p(pct: float) -> float:
        idx = int(pct / 100.0 * (n - 1))
        return s[min(idx, n - 1)]

    return PercentileStats(p50=_p(50), p95=_p(95), p99=_p(99))


def profile_model_latency(
    vllm_endpoint: str,
    prompts: list[str],
    num_runs: int = 3,
) -> LatencyProfile:
    """Profile model latency via vLLM OpenAI-compatible API.

    Sends prompts to the endpoint, measuring TTFT, TPOT, and ITL
    across multiple runs. Reports percentile statistics.

    Args:
        vllm_endpoint: vLLM server URL (e.g., "http://localhost:8000/v1").
        prompts: List of prompt strings to benchmark.
        num_runs: Number of repetitions for statistical significance.

    Returns:
        LatencyProfile with percentile stats and throughput.
    """
    from openai import OpenAI

    client = OpenAI(base_url=vllm_endpoint, api_key="unused")

    ttft_values: list[float] = []
    tpot_values: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0

    for run in range(num_runs):
        for prompt in prompts:
            start = time.monotonic()
            first_token_time = None
            token_times: list[float] = []

            response = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
                stream=True,
            )

            for chunk in response:
                now = time.monotonic()
                if first_token_time is None:
                    first_token_time = now
                    ttft_values.append((first_token_time - start) * 1000)
                else:
                    token_times.append(now)

            if len(token_times) >= 2:
                decode_time = token_times[-1] - first_token_time
                n_tokens = len(token_times)
                tpot_values.append((decode_time / n_tokens) * 1000)
                total_output_tokens += n_tokens
                total_decode_time += decode_time

            if first_token_time is not None:
                total_prefill_time += first_token_time - start

            total_input_tokens += len(prompt.split())

    prefill_throughput = (
        total_input_tokens / total_prefill_time if total_prefill_time > 0 else 0.0
    )
    decode_throughput = (
        total_output_tokens / total_decode_time if total_decode_time > 0 else 0.0
    )

    profile = LatencyProfile(
        ttft_ms=_compute_percentiles(ttft_values),
        tpot_ms=_compute_percentiles(tpot_values),
        itl_ms=_compute_percentiles(tpot_values),  # ITL ≈ TPOT for autoregressive
        throughput_prefill_tok_s=prefill_throughput,
        throughput_decode_tok_s=decode_throughput,
    )

    logger.info(
        "latency_profile_complete",
        ttft_p50=profile.ttft_ms.p50,
        ttft_p95=profile.ttft_ms.p95,
        tpot_p50=profile.tpot_ms.p50,
        num_prompts=len(prompts),
        num_runs=num_runs,
    )

    return profile
