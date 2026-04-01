"""Cat C reward function — Graph Extraction.

Five weighted components with early exit on invalid JSON:
  json_validity_bonus     0.10  (early exit → 0.0 if invalid JSON)
  node_f1                 0.35
  edge_f1                 0.35
  structural_validity     0.10
  1 - normalized_GED      0.10
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from llm_workflow_agents.training.reward_utils import (
    edge_f1,
    node_f1,
    normalized_graph_edit_distance,
    structural_validity,
)

logger = structlog.get_logger(__name__)

W_JSON_VALIDITY = 0.10
W_NODE_F1 = 0.35
W_EDGE_F1 = 0.35
W_STRUCTURAL = 0.10
W_GED_COMPLEMENT = 0.10


def _parse_json_graph(text: str) -> dict | None:
    """Try to parse a JSON graph from completion text.

    Returns the parsed dict or None if invalid.
    """
    from llm_workflow_agents.eval.graph_extraction_eval import parse_graph_json

    graph, is_valid = parse_graph_json(text)
    if not is_valid or graph is None:
        return None
    return {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "initial_state": graph.initial_state,
        "terminal_states": graph.terminal_states,
    }


def reward_graph_extraction(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute Cat C reward for a batch of completions.

    Early exit: if the completion does not contain valid JSON,
    the reward is 0.0 immediately.

    Args:
        prompts: Input prompts (unused but required by GRPOTrainer interface).
        completions: Model completions to score.
        ground_truths: Expected graph dicts with keys: ``nodes``, ``edges``,
            ``initial_state``, ``terminal_states``.

    Returns:
        List of scalar rewards in [0.0, 1.0].
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        pred_graph = _parse_json_graph(completion)
        if pred_graph is None:
            rewards.append(0.0)
            continue

        # JSON is valid — score all components
        r_json = 1.0

        pred_nodes = pred_graph.get("nodes", [])
        gt_nodes = gt.get("nodes", [])
        r_node = node_f1(pred_nodes, gt_nodes)

        pred_edges = pred_graph.get("edges", [])
        gt_edges = gt.get("edges", [])
        r_edge = edge_f1(pred_edges, gt_edges)

        r_structural = structural_validity(pred_graph)

        ged = normalized_graph_edit_distance(pred_graph, gt)
        r_ged_complement = max(0.0, 1.0 - ged)

        score = (
            W_JSON_VALIDITY * r_json
            + W_NODE_F1 * r_node
            + W_EDGE_F1 * r_edge
            + W_STRUCTURAL * r_structural
            + W_GED_COMPLEMENT * r_ged_complement
        )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
