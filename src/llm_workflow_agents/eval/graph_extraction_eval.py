"""Graph extraction evaluation for Experiment C.

Evaluates the quality of workflow graphs extracted from natural-language
prompts. Metrics include Node F1, Edge F1, Graph Edit Distance (GED),
JSON validity, structural validity, and Mermaid renderability.

Uses networkx for GED computation.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GraphExtractionMetrics:
    """Metrics for graph extraction evaluation."""

    node_f1: float = 0.0  # Target: >=85%
    edge_f1: float = 0.0  # Target: >=75%
    graph_edit_distance: float = 0.0  # Target: <=0.20 (normalized)
    json_validity: float = 0.0  # Target: >=95%
    structural_validity: float = 0.0  # Target: >=90%
    mermaid_renderability: float = 0.0  # Target: >=90%

    def to_dict(self) -> dict[str, float]:
        return {
            "node_f1": self.node_f1,
            "edge_f1": self.edge_f1,
            "graph_edit_distance": self.graph_edit_distance,
            "json_validity": self.json_validity,
            "structural_validity": self.structural_validity,
            "mermaid_renderability": self.mermaid_renderability,
        }


@dataclass
class WorkflowGraph:
    """Parsed workflow graph for evaluation."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    initial_state: str = ""
    terminal_states: list[str] = field(default_factory=list)


def parse_graph_json(text: str) -> tuple[WorkflowGraph | None, bool]:
    """Parse a JSON string into a WorkflowGraph.

    Args:
        text: Raw text that should contain a JSON graph.

    Returns:
        (WorkflowGraph or None, json_is_valid).
    """
    # Try to extract JSON from text (may have surrounding text)
    json_str = _extract_json(text)
    if json_str is None:
        return None, False

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, False

    if not isinstance(data, dict):
        return None, False

    graph = WorkflowGraph(
        nodes=data.get("nodes", []),
        edges=data.get("edges", []),
        initial_state=data.get("initial_state", ""),
        terminal_states=data.get("terminal_states", []),
    )
    return graph, True


def _extract_json(text: str) -> str | None:
    """Extract the first JSON object from text.

    Uses json.JSONDecoder.raw_decode so that braces inside string values
    are handled correctly by the JSON parser rather than a naive counter.
    """
    start = text.find("{")
    if start == -1:
        return None

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text, start)
        return text[start:end]
    except json.JSONDecodeError:
        return None


def compute_node_f1(
    predicted: WorkflowGraph,
    gold: WorkflowGraph,
) -> float:
    """Compute F1 score for node prediction.

    Matches nodes by ID. A predicted node matches if its ID exists
    in the gold graph.
    """
    pred_ids = {n.get("id", n.get("name", "")) for n in predicted.nodes}
    gold_ids = {n.get("id", n.get("name", "")) for n in gold.nodes}

    if not pred_ids and not gold_ids:
        return 1.0
    if not pred_ids or not gold_ids:
        return 0.0

    tp = len(pred_ids & gold_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(gold_ids)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_edge_f1(
    predicted: WorkflowGraph,
    gold: WorkflowGraph,
) -> float:
    """Compute F1 score for edge prediction.

    Matches edges by (from_state, to_state) pair.
    """
    pred_edges = {
        (e.get("from_state", ""), e.get("to_state", ""))
        for e in predicted.edges
    }
    gold_edges = {
        (e.get("from_state", ""), e.get("to_state", ""))
        for e in gold.edges
    }

    if not pred_edges and not gold_edges:
        return 1.0
    if not pred_edges or not gold_edges:
        return 0.0

    tp = len(pred_edges & gold_edges)
    precision = tp / len(pred_edges)
    recall = tp / len(gold_edges)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_graph_edit_distance(
    predicted: WorkflowGraph,
    gold: WorkflowGraph,
    normalize: bool = True,
) -> float:
    """Compute Graph Edit Distance using networkx.

    Args:
        predicted: Predicted workflow graph.
        gold: Gold-standard workflow graph.
        normalize: If True, normalize by max(|V_pred|+|E_pred|, |V_gold|+|E_gold|).

    Returns:
        GED (normalized to [0, 1] if normalize=True).
    """
    import networkx as nx

    g_pred = _to_networkx(predicted)
    g_gold = _to_networkx(gold)

    # Use approximate GED for efficiency (exact GED is NP-hard)
    try:
        ged = nx.graph_edit_distance(
            g_pred,
            g_gold,
            node_match=lambda a, b: a.get("label") == b.get("label"),
            timeout=5,
        )
        if ged is None:
            ged = float(abs(len(g_pred) - len(g_gold)) + abs(g_pred.number_of_edges() - g_gold.number_of_edges()))
    except Exception:
        # Fallback: simple structural difference
        ged = float(
            abs(len(g_pred) - len(g_gold))
            + abs(g_pred.number_of_edges() - g_gold.number_of_edges())
        )

    if normalize:
        max_size = max(
            len(g_pred) + g_pred.number_of_edges(),
            len(g_gold) + g_gold.number_of_edges(),
            1,
        )
        return min(ged / max_size, 1.0)

    return ged


def _to_networkx(graph: WorkflowGraph) -> "nx.DiGraph":
    """Convert WorkflowGraph to networkx DiGraph."""
    import networkx as nx

    g = nx.DiGraph()
    for node in graph.nodes:
        node_id = node.get("id", node.get("name", ""))
        g.add_node(node_id, label=node.get("name", node_id))

    for edge in graph.edges:
        src = edge.get("from_state", "")
        dst = edge.get("to_state", "")
        if src and dst:
            g.add_edge(src, dst, condition=edge.get("condition", ""))

    return g


def check_structural_validity(graph: WorkflowGraph) -> bool:
    """Check structural validity of a workflow graph.

    Validity requires:
      1. Valid initial state exists in nodes
      2. All terminal states exist in nodes
      3. Terminal states are reachable from initial state
      4. No orphan nodes (all reachable from initial or can reach terminal)
    """
    if not graph.nodes:
        return False

    node_ids = {n.get("id", n.get("name", "")) for n in graph.nodes}

    # Check 1: initial state exists
    if not graph.initial_state or graph.initial_state not in node_ids:
        return False

    # Check 2: terminal states exist
    if not graph.terminal_states:
        return False
    for ts in graph.terminal_states:
        if ts not in node_ids:
            return False

    # Check 3 & 4: reachability via BFS
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in graph.edges:
        src = edge.get("from_state", "")
        dst = edge.get("to_state", "")
        if src in adjacency and dst in node_ids:
            adjacency[src].append(dst)

    # Forward reachability from initial
    reachable = _bfs(graph.initial_state, adjacency)

    # All terminal states must be reachable
    for ts in graph.terminal_states:
        if ts not in reachable:
            return False

    # Check for orphan nodes: build reverse adjacency
    rev_adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in graph.edges:
        src = edge.get("from_state", "")
        dst = edge.get("to_state", "")
        if dst in rev_adjacency and src in node_ids:
            rev_adjacency[dst].append(src)

    # Compute backward reachability from all terminals once (multi-source BFS)
    backward_reach: set[str] = set()
    for ts in graph.terminal_states:
        backward_reach |= _bfs(ts, rev_adjacency)

    # Every node should be reachable from initial OR be able to reach a terminal
    for nid in node_ids:
        if nid not in reachable and nid not in backward_reach:
            return False

    return True


def _bfs(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    """BFS from start node, return set of reachable nodes."""
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


def check_mermaid_renderability(graph: WorkflowGraph) -> bool:
    """Check if the graph can be rendered as a valid Mermaid flowchart.

    Validates that the graph can produce syntactically valid Mermaid
    markup (does not actually render — checks structure).
    """
    if not graph.nodes:
        return False

    node_ids = {n.get("id", n.get("name", "")) for n in graph.nodes}

    # All edge endpoints must reference valid nodes
    for edge in graph.edges:
        src = edge.get("from_state", "")
        dst = edge.get("to_state", "")
        if not src or not dst:
            return False
        if src not in node_ids or dst not in node_ids:
            return False

    # Node IDs must be valid Mermaid identifiers (alphanumeric + underscore)
    for nid in node_ids:
        if not nid or not all(c.isalnum() or c == "_" for c in nid):
            return False

    return True


def graph_to_mermaid(graph: WorkflowGraph) -> str:
    """Convert a WorkflowGraph to Mermaid flowchart syntax."""
    lines = ["graph TD"]

    for node in graph.nodes:
        nid = node.get("id", node.get("name", ""))
        name = node.get("name", nid).replace("]", "&#93;")
        lines.append(f"    {nid}[{name}]")

    for edge in graph.edges:
        src = edge.get("from_state", "")
        dst = edge.get("to_state", "")
        condition = edge.get("condition", "")
        if condition:
            escaped_condition = condition.replace("|", "&#124;")
            lines.append(f"    {src} -->|{escaped_condition}| {dst}")
        else:
            lines.append(f"    {src} --> {dst}")

    return "\n".join(lines)


def evaluate_graph_extraction(
    predicted_graphs: list[dict[str, Any] | str],
    gold_graphs: list[WorkflowGraph],
) -> GraphExtractionMetrics:
    """Evaluate graph extraction quality across samples.

    Args:
        predicted_graphs: Model-predicted graphs (as dicts or JSON strings).
        gold_graphs: Gold-standard WorkflowGraph objects.

    Returns:
        GraphExtractionMetrics with all computed metrics.
    """
    n = len(gold_graphs)
    if n == 0:
        return GraphExtractionMetrics()

    total_node_f1 = 0.0
    total_edge_f1 = 0.0
    total_ged = 0.0
    valid_json_count = 0
    valid_struct_count = 0
    mermaid_count = 0

    for i, gold in enumerate(gold_graphs):
        # Parse prediction
        if i < len(predicted_graphs):
            pred_raw = predicted_graphs[i]
        else:
            pred_raw = ""

        if isinstance(pred_raw, str):
            pred_graph, json_valid = parse_graph_json(pred_raw)
        elif isinstance(pred_raw, dict):
            json_valid = True
            pred_graph = WorkflowGraph(
                nodes=pred_raw.get("nodes", []),
                edges=pred_raw.get("edges", []),
                initial_state=pred_raw.get("initial_state", ""),
                terminal_states=pred_raw.get("terminal_states", []),
            )
        else:
            pred_graph = None
            json_valid = False

        if json_valid:
            valid_json_count += 1

        if pred_graph is None:
            # Failed predictions count as worst case for GED
            total_ged += 1.0
            continue

        # Node F1
        total_node_f1 += compute_node_f1(pred_graph, gold)

        # Edge F1
        total_edge_f1 += compute_edge_f1(pred_graph, gold)

        # GED — failures count as 1.0 to keep denominator consistent with n
        try:
            total_ged += compute_graph_edit_distance(pred_graph, gold)
        except Exception:
            total_ged += 1.0

        # Structural validity
        if check_structural_validity(pred_graph):
            valid_struct_count += 1

        # Mermaid renderability
        if check_mermaid_renderability(pred_graph):
            mermaid_count += 1

    metrics = GraphExtractionMetrics(
        node_f1=total_node_f1 / n,
        edge_f1=total_edge_f1 / n,
        graph_edit_distance=total_ged / n,
        json_validity=valid_json_count / n,
        structural_validity=valid_struct_count / n,
        mermaid_renderability=mermaid_count / n,
    )

    logger.info("graph_extraction_eval_complete", n=n, **metrics.to_dict())
    return metrics
