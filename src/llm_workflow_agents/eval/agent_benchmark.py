"""Combined workflow quality benchmark for Experiment A.

Composes state machine, tool-calling, and chain propagation metrics
into a single weighted workflow quality score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.eval.state_accuracy import StateMachineMetrics
from llm_workflow_agents.eval.tool_call_f1 import ToolCallMetrics
from llm_workflow_agents.eval.tool_chain_propagation import ChainPropagationMetrics

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowQualityMetrics:
    """Combined workflow quality metrics."""

    full_workflow_success: float = 0.0  # Target: >=55%
    weighted_workflow_score: float = 0.0  # Target: >=0.75
    latency_per_turn_median_ms: float = 0.0  # Target: <=2000 (L1-L3), <=5000 (L4-L5)

    state_metrics: StateMachineMetrics = field(default_factory=StateMachineMetrics)
    tool_metrics: ToolCallMetrics = field(default_factory=ToolCallMetrics)
    chain_metrics: ChainPropagationMetrics = field(default_factory=ChainPropagationMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_workflow_success": self.full_workflow_success,
            "weighted_workflow_score": self.weighted_workflow_score,
            "latency_per_turn_median_ms": self.latency_per_turn_median_ms,
            "state_metrics": self.state_metrics.to_dict(),
            "tool_metrics": self.tool_metrics.to_dict(),
            "chain_metrics": self.chain_metrics.to_dict(),
        }


def compute_weighted_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    completion: float,
) -> float:
    """Compute weighted workflow quality score.

    Formula: 0.4 * StateTransAcc + 0.4 * ToolCallF1 + 0.2 * TaskCompletion

    Args:
        state: State machine metrics.
        tool: Tool-calling metrics.
        completion: Task completion rate (0.0-1.0).

    Returns:
        Weighted score between 0.0 and 1.0.
    """
    return (
        0.4 * state.state_transition_accuracy
        + 0.4 * tool.tool_call_f1
        + 0.2 * completion
    )


def compute_full_workflow_success(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    chain: ChainPropagationMetrics,
) -> float:
    """Compute full workflow success rate.

    A workflow is fully successful if:
      - Task was completed (reached terminal state)
      - All tool calls were correct (F1 >= 0.8)
      - Chain propagation was correct (accuracy >= 0.7)

    Returns approximate rate based on component metrics.
    """
    # Estimate: multiply independent success probabilities
    completion_factor = state.task_completion_rate
    tool_factor = min(tool.tool_call_f1 / 0.8, 1.0) if tool.tool_call_f1 > 0 else 0.0
    chain_factor = (
        min(chain.chain_propagation_accuracy / 0.7, 1.0)
        if chain.total_chains > 0
        else 1.0  # No chains present = not a failure
    )

    return completion_factor * tool_factor * chain_factor


def compute_latency_median(latencies_ms: list[float]) -> float:
    """Compute median latency from a list of per-turn latencies."""
    if not latencies_ms:
        return 0.0
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    if n % 2 == 0:
        return (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2
    return sorted_lat[n // 2]


def evaluate_workflow_quality(
    state_metrics: StateMachineMetrics,
    tool_metrics: ToolCallMetrics,
    chain_metrics: ChainPropagationMetrics,
    latencies_ms: list[float] | None = None,
) -> WorkflowQualityMetrics:
    """Compute combined workflow quality metrics.

    Args:
        state_metrics: State machine adherence results.
        tool_metrics: Tool-calling accuracy results.
        chain_metrics: Tool chain propagation results.
        latencies_ms: Optional per-turn latency measurements.

    Returns:
        WorkflowQualityMetrics with combined scores.
    """
    weighted = compute_weighted_score(
        state_metrics,
        tool_metrics,
        state_metrics.task_completion_rate,
    )

    full_success = compute_full_workflow_success(
        state_metrics,
        tool_metrics,
        chain_metrics,
    )

    median_latency = compute_latency_median(latencies_ms or [])

    metrics = WorkflowQualityMetrics(
        full_workflow_success=full_success,
        weighted_workflow_score=weighted,
        latency_per_turn_median_ms=median_latency,
        state_metrics=state_metrics,
        tool_metrics=tool_metrics,
        chain_metrics=chain_metrics,
    )

    logger.info(
        "workflow_quality_eval_complete",
        weighted_score=weighted,
        full_success=full_success,
        median_latency_ms=median_latency,
    )

    return metrics
