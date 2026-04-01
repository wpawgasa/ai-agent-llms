"""Composite scoring and winner selection for Phase 1.

Computes normalized composite scores per category using configurable
weights, then selects the top model per category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from llm_workflow_agents.benchmark.task_runner import TaskResult

logger = structlog.get_logger(__name__)

# Default selection weights (from configs/benchmark/selection_weights.yaml)
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "A": {"quality": 0.40, "latency_p95": 0.25, "throughput": 0.20, "memory": 0.15},
    "B": {"quality": 0.35, "latency_p95": 0.30, "throughput": 0.20, "memory": 0.15},
    "C": {"quality": 0.40, "latency_p95": 0.20, "throughput": 0.20, "memory": 0.20},
}


@dataclass
class CompositeScore:
    """Composite score for a single model in a category."""

    model_name: str
    category: str
    quality_score: float = 0.0
    latency_score: float = 0.0
    throughput_score: float = 0.0
    memory_score: float = 0.0
    weighted_composite: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "category": self.category,
            "quality_score": self.quality_score,
            "latency_score": self.latency_score,
            "throughput_score": self.throughput_score,
            "memory_score": self.memory_score,
            "weighted_composite": self.weighted_composite,
        }


def _normalize(values: list[float], higher_is_better: bool = True) -> list[float]:
    """Min-max normalize values to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    if not higher_is_better:
        normed = [1.0 - n for n in normed]
    return normed


def compute_composite_scores(
    task_results: list[TaskResult],
    weights: dict[str, float] | None = None,
    category: str = "A",
) -> list[CompositeScore]:
    """Compute composite scores for all models in a category.

    Normalizes quality, latency, throughput, and memory metrics to [0, 1],
    then applies category-specific weights.

    Args:
        task_results: List of TaskResult objects for models in this category.
        weights: Category-specific weights. Defaults to DEFAULT_WEIGHTS[category].
        category: Category identifier ("A", "B", or "C").

    Returns:
        List of CompositeScore sorted by weighted_composite (descending).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.get(category, DEFAULT_WEIGHTS["A"])

    if not task_results:
        return []

    qualities = [r.quality_score for r in task_results]
    latencies = [r.latency.ttft_ms.p95 for r in task_results]
    throughputs = [r.latency.throughput_decode_tok_s for r in task_results]
    memories = [r.latency.peak_vram_gb for r in task_results]

    norm_quality = _normalize(qualities, higher_is_better=True)
    norm_latency = _normalize(latencies, higher_is_better=False)
    norm_throughput = _normalize(throughputs, higher_is_better=True)
    norm_memory = _normalize(memories, higher_is_better=False)

    scores: list[CompositeScore] = []
    for i, result in enumerate(task_results):
        weighted = (
            weights["quality"] * norm_quality[i]
            + weights["latency_p95"] * norm_latency[i]
            + weights["throughput"] * norm_throughput[i]
            + weights["memory"] * norm_memory[i]
        )
        scores.append(CompositeScore(
            model_name=result.model_name,
            category=category,
            quality_score=norm_quality[i],
            latency_score=norm_latency[i],
            throughput_score=norm_throughput[i],
            memory_score=norm_memory[i],
            weighted_composite=weighted,
        ))

    scores.sort(key=lambda s: s.weighted_composite, reverse=True)
    return scores


def select_winners(
    all_scores: dict[str, list[CompositeScore]],
) -> dict[str, str]:
    """Select the winning model per category.

    If the same model wins Cat B and Cat C (Risk R2), both entries
    point to the same model (shared SFT base, diverge at GRPO).

    Args:
        all_scores: Dict mapping category → sorted CompositeScore list.

    Returns:
        Dict mapping category → winning model name.
    """
    winners: dict[str, str] = {}
    for category, scores in all_scores.items():
        if scores:
            winners[category] = scores[0].model_name
            logger.info(
                "category_winner",
                category=category,
                model=scores[0].model_name,
                composite=scores[0].weighted_composite,
            )

    # Log Risk R2 if same model wins B and C
    if winners.get("B") and winners.get("C") and winners["B"] == winners["C"]:
        logger.info(
            "shared_bc_winner",
            model=winners["B"],
            note="Risk R2: share SFT base, diverge at GRPO",
        )

    return winners
