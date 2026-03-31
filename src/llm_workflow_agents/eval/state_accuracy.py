"""State machine adherence evaluation for Experiment A.

Parses [STATE: X → Y] annotations from model output and compares
against ground-truth transition sequences. Supports pass^5 consistency
metric (all 5 temperature=0.7 trials must reach correct terminal state).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Pattern to match state transition annotations in model output
_STATE_PATTERN = re.compile(
    r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:→|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)


@dataclass
class ConversationPrediction:
    """Model's predicted conversation with state annotations."""

    conversation_id: str
    messages: list[dict[str, Any]]
    stochastic_trials: list[list[dict[str, Any]]] = field(default_factory=list)


@dataclass
class ConversationGroundTruth:
    """Ground-truth conversation with expected transitions."""

    conversation_id: str
    messages: list[dict[str, Any]]
    terminal_states: list[str] = field(default_factory=list)


@dataclass
class StateMachineMetrics:
    """Metrics for state machine adherence evaluation."""

    state_transition_accuracy: float = 0.0  # Target: >=85%
    task_completion_rate: float = 0.0  # Target: >=70%
    invalid_transition_rate: float = 0.0  # Target: <=5%
    recovery_rate: float = 0.0  # Target: >=60%
    consistency_pass5: float = 0.0  # Target: >=0.40

    def to_dict(self) -> dict[str, float]:
        return {
            "state_transition_accuracy": self.state_transition_accuracy,
            "task_completion_rate": self.task_completion_rate,
            "invalid_transition_rate": self.invalid_transition_rate,
            "recovery_rate": self.recovery_rate,
            "consistency_pass5": self.consistency_pass5,
        }


def parse_state_transitions(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract state transitions from assistant messages.

    Parses [STATE: X → Y] or [STATE: X -> Y] annotations.

    Returns:
        List of (from_state, to_state) tuples in order of appearance.
    """
    transitions: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for match in _STATE_PATTERN.finditer(content):
            transitions.append((match.group(1), match.group(2)))
    return transitions


def extract_ground_truth_transitions(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Extract ground-truth transitions from annotated messages.

    Looks for 'annotations.state_transition' in assistant messages.
    """
    transitions: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        annotations = msg.get("annotations", {})
        transition = annotations.get("state_transition", {})
        if transition:
            from_s = transition.get("from", "")
            to_s = transition.get("to", "")
            if from_s and to_s:
                transitions.append((from_s, to_s))
    return transitions


def compute_transition_accuracy(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> tuple[float, int]:
    """Compute accuracy of predicted transitions vs ground truth.

    Returns:
        (accuracy, num_invalid) — accuracy as fraction, count of invalid transitions.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0, 0

    correct = 0
    invalid = 0
    gt_set = set(ground_truth)

    for i, pred in enumerate(predicted):
        if i < len(ground_truth) and pred == ground_truth[i]:
            correct += 1
        elif pred not in gt_set:
            invalid += 1

    accuracy = correct / len(ground_truth) if ground_truth else 0.0
    return accuracy, invalid


def check_task_completion(
    predicted: list[tuple[str, str]],
    terminal_states: list[str],
) -> bool:
    """Check if the predicted transitions reach a terminal state."""
    if not predicted:
        return False
    final_state = predicted[-1][1]
    return final_state in terminal_states


def check_recovery(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Count error recovery attempts and successes.

    An error is detected when a tool response contains an error payload.
    Recovery is successful if the next assistant message continues the
    workflow (has a valid state transition).

    Returns:
        (recovery_successes, total_errors)
    """
    total_errors = 0
    recoveries = 0
    prev_was_error = False

    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if '"error"' in content or '"Error"' in content:
                total_errors += 1
                prev_was_error = True
            else:
                prev_was_error = False
        elif msg.get("role") == "assistant":
            if prev_was_error:
                # Check if assistant recovered (has a state transition)
                content = msg.get("content", "")
                if _STATE_PATTERN.search(content):
                    recoveries += 1
            prev_was_error = False
        # user/system messages do not reset prev_was_error — an error from a tool
        # followed by a user message and then an assistant message is still a
        # recovery opportunity.

    return recoveries, total_errors


def compute_pass5_consistency(
    stochastic_trials: list[list[dict[str, Any]]],
    terminal_states: list[str],
) -> bool:
    """Check if all 5 stochastic trials reach the correct terminal state.

    For pass^5: all trials at temperature=0.7 must reach a valid terminal.
    """
    if not stochastic_trials:
        return False

    for trial_messages in stochastic_trials:
        transitions = parse_state_transitions(trial_messages)
        if not check_task_completion(transitions, terminal_states):
            return False
    return True


def evaluate_state_machine(
    predictions: list[ConversationPrediction],
    ground_truth: list[ConversationGroundTruth],
    num_stochastic_trials: int = 5,
) -> StateMachineMetrics:
    """Evaluate state machine adherence across conversations.

    Args:
        predictions: Model predictions with state annotations.
        ground_truth: Ground-truth conversations with expected transitions.
        num_stochastic_trials: Expected number of stochastic trials for pass^5.

    Returns:
        StateMachineMetrics with all computed metrics.
    """
    gt_map = {gt.conversation_id: gt for gt in ground_truth}

    total_accuracy = 0.0
    total_completions = 0
    total_invalid = 0
    total_transitions = 0
    total_recoveries = 0
    total_errors = 0
    total_pass5 = 0
    n = 0

    for pred in predictions:
        gt = gt_map.get(pred.conversation_id)
        if gt is None:
            logger.warning("missing_ground_truth", conversation_id=pred.conversation_id)
            continue

        n += 1
        pred_transitions = parse_state_transitions(pred.messages)
        gt_transitions = extract_ground_truth_transitions(gt.messages)

        # Transition accuracy
        accuracy, invalid = compute_transition_accuracy(pred_transitions, gt_transitions)
        total_accuracy += accuracy
        total_invalid += invalid
        total_transitions += max(len(gt_transitions), len(pred_transitions))

        # Task completion
        if check_task_completion(pred_transitions, gt.terminal_states):
            total_completions += 1

        # Recovery
        recoveries, errors = check_recovery(pred.messages)
        total_recoveries += recoveries
        total_errors += errors

        # pass^5 consistency
        if pred.stochastic_trials:
            if compute_pass5_consistency(pred.stochastic_trials, gt.terminal_states):
                total_pass5 += 1

    if n == 0:
        return StateMachineMetrics()

    metrics = StateMachineMetrics(
        state_transition_accuracy=total_accuracy / n,
        task_completion_rate=total_completions / n,
        invalid_transition_rate=total_invalid / max(total_transitions, 1),
        recovery_rate=total_recoveries / max(total_errors, 1),
        consistency_pass5=total_pass5 / n,
    )

    logger.info("state_machine_eval_complete", n=n, **metrics.to_dict())
    return metrics
