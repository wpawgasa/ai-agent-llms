"""Pareto frontier computation for multi-objective optimization.

Identifies Pareto-optimal configurations across quality, memory, and
latency axes from the benchmark matrix results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ParetoPoint:
    """A single point in the multi-objective optimization space."""

    config_name: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "metrics": dict(self.metrics),
        }


def _dominates(
    a: ParetoPoint,
    b: ParetoPoint,
    maximize: list[str],
    minimize: list[str],
) -> bool:
    """Return True if point a dominates point b.

    Point a dominates b if:
      - a is >= b on all maximize axes
      - a is <= b on all minimize axes
      - a is strictly better on at least one axis
    """
    better_on_any = False

    for axis in maximize:
        val_a = a.metrics.get(axis, 0.0)
        val_b = b.metrics.get(axis, 0.0)
        if val_a < val_b:
            return False
        if val_a > val_b:
            better_on_any = True

    for axis in minimize:
        val_a = a.metrics.get(axis, 0.0)
        val_b = b.metrics.get(axis, 0.0)
        if val_a > val_b:
            return False
        if val_a < val_b:
            better_on_any = True

    return better_on_any


def find_pareto_frontier(
    points: list[ParetoPoint],
    maximize: list[str],
    minimize: list[str],
) -> list[ParetoPoint]:
    """Find Pareto-optimal points from a set of configurations.

    A point is Pareto-optimal if no other point dominates it on all axes.
    Uses O(n^2) pairwise dominance check (suitable for the expected
    result set size of tens to low hundreds of configs).

    Args:
        points: List of ParetoPoint objects to evaluate.
        maximize: List of metric names where higher is better.
        minimize: List of metric names where lower is better.

    Returns:
        List of Pareto-optimal ParetoPoint objects.
    """
    if not points:
        return []

    frontier: list[ParetoPoint] = []

    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            if _dominates(other, candidate, maximize, minimize):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    logger.info(
        "pareto_frontier_computed",
        total_points=len(points),
        frontier_size=len(frontier),
        maximize=maximize,
        minimize=minimize,
    )
    return frontier
