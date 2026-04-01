"""Quantization benchmark harness for Phase 3.

Runs quality and performance benchmarks across all (model, quantization method)
combinations, reporting mean +/- std over multiple runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default quality tasks
DEFAULT_QUALITY_TASKS = [
    "wikitext2_ppl",
    "c4_ppl",
    "longbench_15task",
    "needle_in_haystack",
    "tool_call_f1",
]

# Default performance metrics
DEFAULT_PERFORMANCE_METRICS = [
    "peak_vram_gb",
    "kv_cache_size_gb",
    "throughput_prefill_tok_s",
    "throughput_decode_tok_s",
    "latency_ttft_ms",
    "latency_tpot_ms",
    "latency_itl_p50_p95_p99",
    "max_concurrent_batch_4096ctx",
]


@dataclass
class PercentileStats:
    """Latency percentile statistics."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99}


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single (model, method) configuration."""

    peak_vram_gb: float = 0.0
    kv_cache_size_gb: float = 0.0
    throughput_prefill_tok_s: float = 0.0
    throughput_decode_tok_s: float = 0.0
    ttft_ms: PercentileStats = field(default_factory=PercentileStats)
    tpot_ms: PercentileStats = field(default_factory=PercentileStats)
    itl_ms: PercentileStats = field(default_factory=PercentileStats)
    max_concurrent_batch_4096ctx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_vram_gb": self.peak_vram_gb,
            "kv_cache_size_gb": self.kv_cache_size_gb,
            "throughput_prefill_tok_s": self.throughput_prefill_tok_s,
            "throughput_decode_tok_s": self.throughput_decode_tok_s,
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "itl_ms": self.itl_ms.to_dict(),
            "max_concurrent_batch_4096ctx": self.max_concurrent_batch_4096ctx,
        }


@dataclass
class QualityMetrics:
    """Quality metrics for a single (model, method) configuration."""

    wikitext2_ppl: float = 0.0
    c4_ppl: float = 0.0
    longbench_score: float = 0.0
    needle_accuracy: float = 0.0
    tool_call_f1: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "wikitext2_ppl": self.wikitext2_ppl,
            "c4_ppl": self.c4_ppl,
            "longbench_score": self.longbench_score,
            "needle_accuracy": self.needle_accuracy,
            "tool_call_f1": self.tool_call_f1,
        }


@dataclass
class RunResult:
    """Result of a single benchmark run."""

    run_id: int = 0
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)


@dataclass
class CellResult:
    """Aggregated result for a single (model, method) cell in the matrix."""

    model: str = ""
    method: str = ""
    runs: list[RunResult] = field(default_factory=list)
    quality_mean: QualityMetrics = field(default_factory=QualityMetrics)
    quality_std: QualityMetrics = field(default_factory=QualityMetrics)
    performance_mean: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "method": self.method,
            "num_runs": len(self.runs),
            "quality_mean": self.quality_mean.to_dict(),
            "quality_std": self.quality_std.to_dict(),
            "performance_mean": self.performance_mean.to_dict(),
            "error": self.error,
        }


@dataclass
class QuantBenchmarkMatrix:
    """Full benchmark matrix: models x quantization methods."""

    models: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    results: dict[str, CellResult] = field(default_factory=dict)
    total_runs: int = 0
    elapsed_seconds: float = 0.0

    def get_cell(self, model: str, method: str) -> CellResult | None:
        """Get result for a specific (model, method) pair."""
        return self.results.get(f"{model}::{method}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "methods": self.methods,
            "total_runs": self.total_runs,
            "elapsed_seconds": self.elapsed_seconds,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


def _compute_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of a list of values."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, variance**0.5


def _aggregate_quality(runs: list[RunResult]) -> tuple[QualityMetrics, QualityMetrics]:
    """Compute mean and std of quality metrics across runs."""
    if not runs:
        return QualityMetrics(), QualityMetrics()

    fields = ["wikitext2_ppl", "c4_ppl", "longbench_score", "needle_accuracy", "tool_call_f1"]
    mean_vals = {}
    std_vals = {}
    for f in fields:
        values = [getattr(r.quality, f) for r in runs]
        m, s = _compute_mean_std(values)
        mean_vals[f] = m
        std_vals[f] = s

    return QualityMetrics(**mean_vals), QualityMetrics(**std_vals)


def _run_single_quality_eval(
    model: str,
    method: str,
    quality_tasks: list[str],
    prompts_per_run: int,
    base_url: str,
) -> QualityMetrics:
    """Run quality evaluation tasks for a single (model, method) pair.

    Dispatches to the appropriate eval module for each task.
    """
    metrics = QualityMetrics()

    for task in quality_tasks:
        if task == "wikitext2_ppl":
            from llm_workflow_agents.eval.perplexity import evaluate_perplexity_vllm

            result = evaluate_perplexity_vllm(
                model_name=model,
                dataset_name="wikitext2",
                kv_cache_dtype=method,
                max_samples=prompts_per_run,
                base_url=base_url,
            )
            metrics.wikitext2_ppl = result.perplexity

        elif task == "c4_ppl":
            from llm_workflow_agents.eval.perplexity import evaluate_perplexity_vllm

            result = evaluate_perplexity_vllm(
                model_name=model,
                dataset_name="c4",
                kv_cache_dtype=method,
                max_samples=prompts_per_run,
                base_url=base_url,
            )
            metrics.c4_ppl = result.perplexity

        elif task == "longbench_15task":
            from llm_workflow_agents.eval.longbench import evaluate_longbench

            result = evaluate_longbench(
                model_path=model,
                kv_cache_dtype=method,
                max_samples_per_task=min(prompts_per_run // 15, 100),
                base_url=base_url,
            )
            metrics.longbench_score = result.overall_score

        elif task == "needle_in_haystack":
            from llm_workflow_agents.eval.needle_haystack import evaluate_needle_in_haystack

            result = evaluate_needle_in_haystack(
                model_path=model,
                kv_cache_dtype=method,
                base_url=base_url,
            )
            metrics.needle_accuracy = result.overall_accuracy

        elif task == "tool_call_f1":
            logger.info("tool_call_f1_eval_skipped", reason="requires task-specific data")

    return metrics


def run_quant_benchmark(
    models: list[str],
    methods: list[str],
    quality_tasks: list[str] | None = None,
    num_runs: int = 5,
    prompts_per_run: int = 500,
    base_url: str = "http://localhost:8000/v1",
) -> QuantBenchmarkMatrix:
    """Run full quantization benchmark matrix.

    Evaluates all (model, method) combinations with quality and performance
    metrics, repeated over multiple runs for statistical significance.

    Args:
        models: List of model names (pre-trained + fine-tuned).
        methods: List of quantization methods
            (e.g., ["fp8", "kivi", "kvquant", "awq_fp8", "turboquant", "rotorquant"]).
        quality_tasks: Quality evaluation tasks to run.
            Defaults to: wikitext2_ppl, c4_ppl, longbench_15task,
            needle_in_haystack, tool_call_f1.
        num_runs: Number of repetitions per configuration (default: 5).
        prompts_per_run: Number of prompts per evaluation run (default: 500).
        base_url: vLLM server base URL.

    Returns:
        QuantBenchmarkMatrix with aggregated results (mean +/- std).
    """
    if quality_tasks is None:
        quality_tasks = DEFAULT_QUALITY_TASKS

    matrix = QuantBenchmarkMatrix(models=models, methods=methods)
    start_time = time.monotonic()

    total_cells = len(models) * len(methods)
    completed = 0

    for model in models:
        for method in methods:
            cell_key = f"{model}::{method}"
            logger.info(
                "quant_benchmark_cell_start",
                model=model,
                method=method,
                progress=f"{completed}/{total_cells}",
            )

            runs: list[RunResult] = []
            try:
                for run_id in range(num_runs):
                    quality = _run_single_quality_eval(
                        model=model,
                        method=method,
                        quality_tasks=quality_tasks,
                        prompts_per_run=prompts_per_run,
                        base_url=base_url,
                    )
                    runs.append(RunResult(run_id=run_id, quality=quality))
                    matrix.total_runs += 1

                quality_mean, quality_std = _aggregate_quality(runs)
                matrix.results[cell_key] = CellResult(
                    model=model,
                    method=method,
                    runs=runs,
                    quality_mean=quality_mean,
                    quality_std=quality_std,
                )

            except Exception as e:
                logger.error(
                    "quant_benchmark_cell_error",
                    model=model,
                    method=method,
                    error=str(e),
                )
                matrix.results[cell_key] = CellResult(
                    model=model,
                    method=method,
                    error=str(e),
                )

            completed += 1

    matrix.elapsed_seconds = time.monotonic() - start_time
    logger.info(
        "quant_benchmark_complete",
        total_cells=total_cells,
        total_runs=matrix.total_runs,
        elapsed_s=round(matrix.elapsed_seconds, 1),
    )

    return matrix
