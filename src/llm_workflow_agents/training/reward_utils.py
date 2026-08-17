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
    """Compute BFCL-style AST-match F1 score (strict; used by benchmark eval)."""
    from llm_workflow_agents.eval.tool_call_f1 import compute_ast_f1

    return compute_ast_f1(predicted, ground_truth)


def graded_tool_call_f1(predicted: list[dict], ground_truth: list[dict]) -> float:
    """Argument-graded F1 over tool calls — continuous variant for GRPO reward.

    Per (pred, gt) pair: 0.4 for matching name + up to 0.6 prorated across
    correctly-matched arguments. Aggregated as a soft F1 with greedy assignment.
    Use this for training reward (avoids the 0/1 cliff of subtree-match);
    keep ``tool_call_f1`` for benchmark eval to preserve the strict metric.
    """
    from llm_workflow_agents.eval.tool_call_f1 import compute_argument_graded_f1

    return compute_argument_graded_f1(predicted, ground_truth)


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


def transition_legality_score(
    predicted: list[tuple[str, str]],
    valid_transitions: list,
) -> float:
    """Fraction of predicted [STATE: X → Y] transitions that are legal edges.

    A transition is legal when ``(from, to)`` appears in the workflow graph's
    edge list. Unlike state-correctness scoring, this is independent of which
    transition is *expected* for the turn — it only asks whether the emitted
    transition exists in the graph at all, directly penalizing hallucinated
    states. A self-loop ``(X, X)`` is always legal — it denotes remaining in the
    current state across turns — even when the graph does not declare an explicit
    self-edge. Empty prediction → 0.0 (the model must emit a transition).
    """
    if not predicted:
        return 0.0
    valid = {
        (t[0], t[1])
        for t in valid_transitions
        if isinstance(t, (list, tuple)) and len(t) == 2
    }
    return sum(
        1 for tr in predicted if tr[0] == tr[1] or tuple(tr) in valid
    ) / len(predicted)


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


def heldout_composite_score(
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> float:
    """Deployment-aligned held-out quality score, computed with STRICT metrics.

    Mirrors ``eval.composite_score.compute_weighted_workflow_score``
    (``0.4 * state_transition_acc + 0.4 * strict_tool_f1 + 0.2 * task_completion``),
    but scored **per-turn-fair**: rows are single user->assistant turns, so each
    term is included only when it is applicable to the turn and the score is
    renormalized over the included weights — the tool term always applies, the
    state term only when GT expects a transition, the task (reached-terminal)
    term only on the terminal turn. Averaged over held-out rows. This avoids
    charging the policy on terms it cannot satisfy on an intermediate/abstention
    turn; see the ckpt-1000 audit (runs/preflight/heldout_composite_audit_*) and
    scripts/perturn_fair_composite.py.

    Deliberately uses the *strict* scorers (``state_sequence_match``,
    ``tool_call_f1`` = ``compute_ast_f1``, ``reached_terminal``) rather than any
    graded components a training reward might optimize (``graded_tool_call_f1``,
    argument-graded matching). Keeping the held-out metric numerically
    independent of whatever objective is training (GRPO reward or a DPO/ORPO
    preference loss) is what makes a reward-vs-quality divergence (Risk R5)
    detectable — see docs/grpo_diagnosis_gemma4_26b.md. Pure/CPU-only, shared by
    ``grpo.py`` and ``dpo.py``'s held-out guardrail callbacks.
    """
    if not completions:
        return 0.0

    scores: list[float] = []
    for comp, gt in zip(completions, ground_truths):
        gt = gt or {}
        gt_seq = gt.get("state_sequence") or []
        gt_trans = [
            (s.get("from", ""), s.get("to", ""))
            if isinstance(s, dict)
            else tuple(s)
            if isinstance(s, (list, tuple)) and len(s) == 2
            else ("", "")
            for s in gt_seq
        ]
        tool_f1 = tool_call_f1(extract_tool_calls(comp), gt.get("tool_calls") or [])

        terminal = gt.get("terminal_state") or ""
        task = 1.0 if terminal and reached_terminal(comp, terminal) else 0.0

        # Per-turn-fair: rows are SINGLE user->assistant turns, so only charge
        # the terms applicable to THIS turn, then renormalize over the included
        # weights (whole-conv weights 0.4/0.4/0.2 preserved). See the ckpt-1000
        # audit (runs/preflight/heldout_composite_audit_*): applying all three
        # terms per-turn punished the policy on terms it cannot satisfy on an
        # intermediate/abstention turn (task=0 off the terminal turn; state=0
        # when the trained always-annotate habit meets an empty GT transition),
        # depressing the metric ~0.19 below its honest level.
        num, den = 0.4 * tool_f1, 0.4  # tool term always applies
        if gt_trans:  # state only when a transition is expected this turn
            state_acc = state_sequence_match(extract_state_annotations(comp), gt_trans)
            num += 0.4 * state_acc
            den += 0.4
        _to = gt_trans[-1][1] if gt_trans else ""
        if terminal and _to == terminal:  # task only on the terminal turn
            num += 0.2 * task
            den += 0.2
        scores.append(num / den if den else 0.0)

    return sum(scores) / len(scores)


def is_reward_hacking(
    reward_history: list[float],
    held_out_history: list[float],
    lookback: int = 5,
) -> bool:
    """Reward-hacking test: training reward rising while held-out quality falls.

    Returns True only when there is enough history AND the latest training
    reward is above the reward ``lookback`` logs ago AND the latest held-out
    composite is below the previous one. Pure/unit-tested — the callback wires
    it to ``control.should_training_stop``. "Reward" here is whatever scalar
    the training loop reports per log step (a GRPO reward or a DPO/ORPO
    accuracy/margin metric) — the function itself is objective-agnostic.
    """
    if len(reward_history) < lookback or len(held_out_history) < 2:
        return False
    return (
        reward_history[-1] > reward_history[-lookback]
        and held_out_history[-1] < held_out_history[-2]
    )


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
