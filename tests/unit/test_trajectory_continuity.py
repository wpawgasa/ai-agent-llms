"""Task completion must also be reported for trajectories the model walked.

`check_task_completion` reads only the final `[STATE: ...]` annotation, so a
model that jumps to the terminal state from a state it was never in scores a
completion. On the Phase 1 text benchmark the untrained gemma-4-12B made 164
such jumps against the fine-tuned model's 21, and out-scored it on completion
because of them (CLAUDE.md R25).

The continuous variant is reported BESIDE the original, never in place of it:
the Phase 1 composite keeps the old definition, so rankings stay comparable.
"""

from __future__ import annotations

from llm_workflow_agents.eval.agent_benchmark import compute_weighted_score
from llm_workflow_agents.eval.state_accuracy import (
    ConversationGroundTruth,
    ConversationPrediction,
    StateMachineMetrics,
    check_task_completion,
    check_task_completion_continuous,
    count_skip_ahead_transitions,
    evaluate_state_machine,
    trajectory_is_continuous,
)
from llm_workflow_agents.eval.tool_call_f1 import ToolCallMetrics

WALKED = [("GREETING", "VERIFY"), ("VERIFY", "PAY"), ("PAY", "TERMINAL")]
TELEPORTED = [("GREETING", "VERIFY"), ("PAY", "TERMINAL")]


class TestContinuity:
    def test_a_walked_trajectory_is_continuous(self):
        assert trajectory_is_continuous(WALKED)
        assert count_skip_ahead_transitions(WALKED) == 0

    def test_a_jump_is_counted(self):
        assert not trajectory_is_continuous(TELEPORTED)
        assert count_skip_ahead_transitions(TELEPORTED) == 1

    def test_self_loops_are_continuous(self):
        assert trajectory_is_continuous([("A", "A"), ("A", "A"), ("A", "B")])

    def test_the_first_annotation_is_free(self):
        """It establishes where the conversation starts; nothing precedes it."""
        assert trajectory_is_continuous([("ANYWHERE", "TERMINAL")])

    def test_no_annotations_is_vacuously_continuous_but_not_a_completion(self):
        assert trajectory_is_continuous([])
        assert not check_task_completion_continuous([], ["TERMINAL"])


class TestContinuousCompletion:
    def test_walking_to_the_terminal_state_counts_under_both(self):
        assert check_task_completion(WALKED, ["TERMINAL"])
        assert check_task_completion_continuous(WALKED, ["TERMINAL"])

    def test_jumping_counts_only_under_the_original(self):
        assert check_task_completion(TELEPORTED, ["TERMINAL"])
        assert not check_task_completion_continuous(TELEPORTED, ["TERMINAL"])

    def test_a_continuous_trajectory_that_stops_short_counts_under_neither(self):
        short = [("GREETING", "VERIFY"), ("VERIFY", "PAY")]
        assert not check_task_completion(short, ["TERMINAL"])
        assert not check_task_completion_continuous(short, ["TERMINAL"])


def _run(trajectories):
    preds, gts = [], []
    for i, transitions in enumerate(trajectories):
        messages = [
            {"role": "assistant", "content": f"[STATE: {a} → {b}]"} for a, b in transitions
        ]
        preds.append(ConversationPrediction(conversation_id=str(i), messages=messages))
        gts.append(
            ConversationGroundTruth(
                conversation_id=str(i),
                messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "annotations": {"state_transition": {"from": a, "to": b}},
                    }
                    for a, b in WALKED
                ],
                terminal_states=["TERMINAL"],
            )
        )
    return evaluate_state_machine(preds, gts)


class TestReportedTogether:
    def test_both_rates_are_reported(self):
        metrics = _run([WALKED, TELEPORTED])
        assert metrics.task_completion_rate == 1.0
        assert metrics.task_completion_rate_continuous == 0.5
        assert metrics.trajectory_continuity_rate == 0.5

    def test_skip_ahead_rate_is_per_transition(self):
        # WALKED: 2 linked pairs, 0 jumps. TELEPORTED: 1 pair, 1 jump.
        metrics = _run([WALKED, TELEPORTED])
        assert metrics.skip_ahead_transition_rate == 1 / 3

    def test_continuous_never_exceeds_the_original(self):
        metrics = _run([WALKED, TELEPORTED, []])
        assert metrics.task_completion_rate_continuous <= metrics.task_completion_rate

    def test_both_appear_in_the_result_dict(self):
        keys = _run([WALKED]).to_dict()
        assert keys["task_completion_rate"] == 1.0
        assert keys["task_completion_rate_continuous"] == 1.0
        assert "trajectory_continuity_rate" in keys and "skip_ahead_transition_rate" in keys


class TestRankingIsUnchanged:
    def test_the_composite_still_uses_the_original_completion(self):
        """Inertness: adding the metric must not move any Phase 1 score."""
        tool = ToolCallMetrics(tool_call_f1=0.5)
        teleporting = StateMachineMetrics(
            state_sequence_accuracy=0.9,
            task_completion_rate=1.0,
            task_completion_rate_continuous=0.0,
        )
        assert compute_weighted_score(teleporting, tool, teleporting.task_completion_rate) == (
            0.4 * 0.9 + 0.4 * 0.5 + 0.2 * 1.0
        )

    def test_defaults_keep_existing_constructions_working(self):
        metrics = StateMachineMetrics(task_completion_rate=0.7)
        assert metrics.task_completion_rate_continuous == 0.0
        assert metrics.trajectory_continuity_rate == 0.0
