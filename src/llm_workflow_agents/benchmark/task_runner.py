"""Single (model, task) evaluation runner for Phase 1.

Handles launching evaluation for one model on one task category,
including Nemotron vLLM fallback (Risk R6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.benchmark.latency_profiler import (
    LatencyProfile,
    profile_model_latency,
)

logger = structlog.get_logger(__name__)


@dataclass
class TaskResult:
    """Result of evaluating a single model on a single task."""

    model_name: str = ""
    task: str = ""
    quality_score: float = 0.0
    quality_breakdown: dict[str, float] = field(default_factory=dict)
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    num_samples: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "quality_score": self.quality_score,
            "quality_breakdown": self.quality_breakdown,
            "latency": self.latency.to_dict(),
            "num_samples": self.num_samples,
            "error": self.error,
        }


def _is_nemotron(model_name: str) -> bool:
    """Check if model is Nemotron (Risk R6: vLLM incompatibility)."""
    return "nemotron" in model_name.lower()


def _run_task_a(
    model_name: str,
    vllm_endpoint: str,
    prompts: list[str],
    ground_truths: list[dict],
) -> dict[str, float]:
    """Evaluate Task A: state machine adherence + tool-call F1."""
    from llm_workflow_agents.eval.state_accuracy import (
        StateMachineMetrics,
        evaluate_state_machine,
    )
    from llm_workflow_agents.eval.tool_call_f1 import evaluate_tool_calls
    from openai import OpenAI

    client = OpenAI(base_url=vllm_endpoint, api_key="unused")
    predictions = []
    gt_objects = []

    for prompt, gt in zip(prompts, ground_truths):
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        predictions.append({"conversation_id": gt.get("conversation_id", ""), "messages": [{"role": "assistant", "content": content}]})
        gt_objects.append(gt)

    from llm_workflow_agents.eval.state_accuracy import ConversationPrediction, ConversationGroundTruth
    pred_convs = [ConversationPrediction(conversation_id=p["conversation_id"], messages=p["messages"]) for p in predictions]
    gt_convs = [ConversationGroundTruth(conversation_id=g.get("conversation_id", ""), messages=g.get("messages", []), terminal_states=g.get("terminal_states", [])) for g in gt_objects]

    state_metrics = evaluate_state_machine(pred_convs, gt_convs)
    quality = 0.5 * state_metrics.state_transition_accuracy + 0.5 * state_metrics.task_completion_rate

    return {
        "quality_score": quality,
        "state_transition_accuracy": state_metrics.state_transition_accuracy,
        "task_completion_rate": state_metrics.task_completion_rate,
    }


def _run_task_b(
    model_name: str,
    vllm_endpoint: str,
    prompts: list[str],
    ground_truths: list[dict],
) -> dict[str, float]:
    """Evaluate Task B: tool-call F1 + slot accuracy."""
    from llm_workflow_agents.eval.tool_call_f1 import (
        compute_ast_f1,
        compute_name_accuracy,
        parse_tool_calls,
    )
    from openai import OpenAI

    client = OpenAI(base_url=vllm_endpoint, api_key="unused")
    all_f1 = []
    all_name_acc = []

    for prompt, gt in zip(prompts, ground_truths):
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        pred_tools = parse_tool_calls(content)
        gt_tools = gt.get("tool_calls", [])
        all_f1.append(compute_ast_f1(pred_tools, gt_tools))
        all_name_acc.append(compute_name_accuracy(pred_tools, gt_tools))

    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    avg_name = sum(all_name_acc) / len(all_name_acc) if all_name_acc else 0.0
    quality = 0.6 * avg_f1 + 0.4 * avg_name

    return {
        "quality_score": quality,
        "tool_call_f1": avg_f1,
        "slot_accuracy": avg_name,
    }


def _run_task_c(
    model_name: str,
    vllm_endpoint: str,
    prompts: list[str],
    ground_truths: list[dict],
) -> dict[str, float]:
    """Evaluate Task C: graph extraction (node/edge F1)."""
    from llm_workflow_agents.eval.graph_extraction_eval import (
        compute_edge_f1,
        compute_node_f1,
        parse_graph_json,
        WorkflowGraph,
    )
    from openai import OpenAI

    client = OpenAI(base_url=vllm_endpoint, api_key="unused")
    all_node_f1 = []
    all_edge_f1 = []

    for prompt, gt in zip(prompts, ground_truths):
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        pred_graph, valid = parse_graph_json(content)
        gold = WorkflowGraph(
            nodes=gt.get("nodes", []),
            edges=gt.get("edges", []),
            initial_state=gt.get("initial_state", ""),
            terminal_states=gt.get("terminal_states", []),
        )
        if pred_graph and valid:
            all_node_f1.append(compute_node_f1(pred_graph, gold))
            all_edge_f1.append(compute_edge_f1(pred_graph, gold))
        else:
            all_node_f1.append(0.0)
            all_edge_f1.append(0.0)

    avg_node = sum(all_node_f1) / len(all_node_f1) if all_node_f1 else 0.0
    avg_edge = sum(all_edge_f1) / len(all_edge_f1) if all_edge_f1 else 0.0
    quality = 0.5 * avg_node + 0.5 * avg_edge

    return {
        "quality_score": quality,
        "node_f1": avg_node,
        "edge_f1": avg_edge,
    }


_TASK_RUNNERS = {
    "task_a": _run_task_a,
    "task_b": _run_task_b,
    "task_c": _run_task_c,
}


def run_task(
    model_name: str,
    task: str,
    vllm_endpoint: str = "http://localhost:8000/v1",
    prompts: list[str] | None = None,
    ground_truths: list[dict] | None = None,
) -> TaskResult:
    """Run evaluation for a single (model, task) pair.

    Args:
        model_name: HF model ID.
        task: Task identifier ("task_a", "task_b", or "task_c").
        vllm_endpoint: vLLM server URL.
        prompts: Evaluation prompts.
        ground_truths: Ground truth annotations.

    Returns:
        TaskResult with quality score and latency profile.
    """
    if prompts is None:
        prompts = []
    if ground_truths is None:
        ground_truths = []

    if _is_nemotron(model_name):
        logger.warning("nemotron_vllm_fallback", model=model_name, hint="Risk R6: may need HF generate() path")

    runner = _TASK_RUNNERS.get(task)
    if runner is None:
        return TaskResult(model_name=model_name, task=task, error=f"Unknown task: {task}")

    try:
        breakdown = runner(model_name, vllm_endpoint, prompts, ground_truths)
        quality = breakdown.pop("quality_score", 0.0)

        latency = profile_model_latency(
            vllm_endpoint=vllm_endpoint,
            prompts=prompts[:10],  # Profile on subset
            num_runs=1,
        )

        return TaskResult(
            model_name=model_name,
            task=task,
            quality_score=quality,
            quality_breakdown=breakdown,
            latency=latency,
            num_samples=len(prompts),
        )
    except Exception as e:
        logger.error("task_runner_error", model=model_name, task=task, error=str(e))
        return TaskResult(model_name=model_name, task=task, error=str(e))
