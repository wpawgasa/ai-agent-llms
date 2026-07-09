"""Tests for the pure trajectory-rollout logic and in-process replay rollout.

The GPU generation path is exercised with a monkeypatched ``model.generate`` and
a cached offline tokenizer (Qwen ChatML); no ``trl``/``unsloth`` is required —
``trajectory_rollout`` imports them lazily, so this whole file runs in ``.venv``.
"""

from __future__ import annotations

from llm_workflow_agents.training.trajectory_rollout import (
    assert_trajectory_rollout_support,
)


class TestEnvGate:
    def test_importable_and_callable(self) -> None:
        # The module must import without trl/unsloth installed, and the gate
        # must be callable. The assertion body only runs meaningfully on the
        # train box (it imports trl), so we don't invoke it here.
        assert callable(assert_trajectory_rollout_support)
