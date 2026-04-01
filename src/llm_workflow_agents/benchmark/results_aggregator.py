"""Results aggregation and ranking table generation for Phase 1.

Collects TaskResult objects from all (model, task) pairs and produces
per-category ranking tables and JSON output for handoff to Phase 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.benchmark.model_selector import CompositeScore
from llm_workflow_agents.benchmark.task_runner import TaskResult

logger = structlog.get_logger(__name__)


@dataclass
class RankingTable:
    """Per-category ranking table."""

    category: str
    rankings: list[CompositeScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "rankings": [r.to_dict() for r in self.rankings],
        }


@dataclass
class Phase1Results:
    """Complete Phase 1 benchmark results."""

    ranking_tables: dict[str, RankingTable] = field(default_factory=dict)
    winners: dict[str, str] = field(default_factory=dict)
    all_task_results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_tables": {k: v.to_dict() for k, v in self.ranking_tables.items()},
            "winners": self.winners,
            "num_evaluations": len(self.all_task_results),
        }


def aggregate_results(
    task_results: list[TaskResult],
    composite_scores: dict[str, list[CompositeScore]],
    winners: dict[str, str],
) -> Phase1Results:
    """Aggregate all Phase 1 results into ranking tables.

    Args:
        task_results: All (model, task) evaluation results.
        composite_scores: Per-category sorted composite scores.
        winners: Category → winning model name.

    Returns:
        Phase1Results with ranking tables and winners.
    """
    ranking_tables: dict[str, RankingTable] = {}
    for category, scores in composite_scores.items():
        ranking_tables[category] = RankingTable(
            category=category,
            rankings=scores,
        )

    results = Phase1Results(
        ranking_tables=ranking_tables,
        winners=winners,
        all_task_results=task_results,
    )

    logger.info(
        "phase1_aggregation_complete",
        categories=list(ranking_tables.keys()),
        winners=winners,
        total_evaluations=len(task_results),
    )

    return results


def save_results(results: Phase1Results, output_path: Path) -> Path:
    """Save Phase 1 results to JSON for handoff to Phase 2.

    Args:
        results: Aggregated Phase 1 results.
        output_path: Output JSON file path.

    Returns:
        Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info("phase1_results_saved", path=str(output_path))
    return output_path
