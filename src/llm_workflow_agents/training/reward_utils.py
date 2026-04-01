"""Shared reward computation helpers.

Thin wrappers around eval module functions, adapted for the
(prompts, completions, ground_truths) -> list[float] reward interface.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def extract_state_annotations(text: str) -> list[tuple[str, str]]:
    """Extract [STATE: X -> Y] annotations from completion text."""
    from llm_workflow_agents.eval.state_accuracy import parse_state_transitions

    messages = [{"role": "assistant", "content": text}]
    return parse_state_transitions(messages)


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from completion text."""
    from llm_workflow_agents.eval.tool_call_f1 import parse_tool_calls

    return parse_tool_calls(text)


def state_sequence_match(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> float:
    """Compute state transition accuracy between predicted and ground truth."""
    from llm_workflow_agents.eval.state_accuracy import compute_transition_accuracy

    accuracy, _ = compute_transition_accuracy(predicted, ground_truth)
    return accuracy


def tool_call_f1(predicted: list[dict], ground_truth: list[dict]) -> float:
    """Compute BFCL-style AST-match F1 score."""
    from llm_workflow_agents.eval.tool_call_f1 import compute_ast_f1

    return compute_ast_f1(predicted, ground_truth)


def chain_propagation_score(
    pred_messages: list[dict],
    gt_messages: list[dict],
) -> float:
    """Compute chain propagation accuracy for a single conversation."""
    from llm_workflow_agents.eval.tool_chain_propagation import (
        check_value_propagation,
        extract_tool_chains,
    )

    pred_links = extract_tool_chains(pred_messages)
    gt_links = extract_tool_chains(gt_messages)
    n_pairs = min(len(pred_links), len(gt_links))
    if n_pairs <= 1:
        return 1.0  # No chain to evaluate
    correct = 0
    for i in range(1, n_pairs):
        if check_value_propagation(
            gt_links[i - 1].response, pred_links[i].arguments
        ):
            correct += 1
    return correct / (n_pairs - 1)


def format_compliance_check(text: str) -> float:
    """Check format compliance of completion text.

    Penalizes:
      - Mismatched <tool_call>/</ tool_call> tags
      - Raw stack traces or code blocks in output
    """
    score = 1.0
    opens = text.count("<tool_call>")
    closes = text.count("</tool_call>")
    if opens != closes:
        score -= 0.5
    if "Traceback" in text or "```python" in text:
        score -= 0.3
    return max(score, 0.0)


def reached_terminal(text: str, expected_terminal: str) -> bool:
    """Check if completion reaches the expected terminal state."""
    transitions = extract_state_annotations(text)
    if not transitions:
        return False
    return transitions[-1][1] == expected_terminal


def node_f1(predicted_nodes: list[dict], gold_nodes: list[dict]) -> float:
    """Compute node F1 score between predicted and gold graphs."""
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph,
        compute_node_f1,
    )

    pred = WorkflowGraph(nodes=predicted_nodes)
    gold = WorkflowGraph(nodes=gold_nodes)
    return compute_node_f1(pred, gold)


def edge_f1(predicted_edges: list[dict], gold_edges: list[dict]) -> float:
    """Compute edge F1 score between predicted and gold graphs."""
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph,
        compute_edge_f1,
    )

    pred = WorkflowGraph(edges=predicted_edges)
    gold = WorkflowGraph(edges=gold_edges)
    return compute_edge_f1(pred, gold)


def structural_validity(graph_dict: dict) -> float:
    """Check structural validity of a workflow graph. Returns 1.0 or 0.0."""
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph,
        check_structural_validity,
    )

    graph = WorkflowGraph(
        nodes=graph_dict.get("nodes", []),
        edges=graph_dict.get("edges", []),
        initial_state=graph_dict.get("initial_state", ""),
        terminal_states=graph_dict.get("terminal_states", []),
    )
    return 1.0 if check_structural_validity(graph) else 0.0


def normalized_graph_edit_distance(pred_dict: dict, gold_dict: dict) -> float:
    """Compute normalized graph edit distance."""
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph,
        compute_graph_edit_distance,
    )

    pred = WorkflowGraph(
        nodes=pred_dict.get("nodes", []),
        edges=pred_dict.get("edges", []),
        initial_state=pred_dict.get("initial_state", ""),
        terminal_states=pred_dict.get("terminal_states", []),
    )
    gold = WorkflowGraph(
        nodes=gold_dict.get("nodes", []),
        edges=gold_dict.get("edges", []),
        initial_state=gold_dict.get("initial_state", ""),
        terminal_states=gold_dict.get("terminal_states", []),
    )
    return compute_graph_edit_distance(pred, gold, normalize=True)
