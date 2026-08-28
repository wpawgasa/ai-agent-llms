"""Tests for the out-of-process DPO guardrail decision helper.

The in-process R5 guardrail cannot run on this model: `load_in_4bit` leaves the
MoE experts in bf16, so the training process already holds ~46 GiB and a second
model copy does not fit (CLAUDE.md R19, docs/dpo_memory_ceiling_investigation.md
§8). The chunked runner instead trains and scores in *separate* processes and
asks this helper whether to continue.

The helper owns no policy of its own — `is_reward_hacking` remains the single
stop rule — so these tests pin the two readers and the exit contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _write_trainer_state(path: Path, log_history: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"log_history": log_history}))
    return path


def _write_audit(path: Path, composite: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"mean_composite": composite}}))
    return path


# --------------------------------------------------------------------------- #
# Reading the training metric
# --------------------------------------------------------------------------- #


def test_training_metric_prefers_rewards_accuracies(tmp_path):
    from dpo_guardrail_decide import read_training_metric

    state = _write_trainer_state(
        tmp_path / "trainer_state.json",
        [
            {"step": 1, "loss": 0.9, "rewards/accuracies": 0.5},
            {"step": 2, "loss": 0.8, "rewards/accuracies": 0.6},
        ],
    )
    assert read_training_metric(state) == [0.5, 0.6]


def test_training_metric_falls_back_to_loss(tmp_path):
    """A log line without rewards/accuracies still contributes its loss."""
    from dpo_guardrail_decide import read_training_metric

    state = _write_trainer_state(
        tmp_path / "trainer_state.json",
        [{"step": 1, "loss": 0.9}, {"step": 2, "loss": 0.7}],
    )
    assert read_training_metric(state) == [0.9, 0.7]


def test_training_metric_skips_lines_carrying_neither(tmp_path):
    """Checkpoint-save and eval lines carry neither key and must not appear."""
    from dpo_guardrail_decide import read_training_metric

    state = _write_trainer_state(
        tmp_path / "trainer_state.json",
        [
            {"step": 1, "loss": 0.9},
            {"step": 1, "eval_runtime": 3.2},
            {"step": 2, "loss": 0.7},
        ],
    )
    assert read_training_metric(state) == [0.9, 0.7]


# --------------------------------------------------------------------------- #
# Reading the held-out scores
# --------------------------------------------------------------------------- #


def test_heldout_scores_are_ordered_by_step_number_not_filename(tmp_path):
    """step-100 must follow step-20. A lexical sort would invert them."""
    from dpo_guardrail_decide import read_heldout_scores

    _write_audit(tmp_path / "step-20.json", 0.70)
    _write_audit(tmp_path / "step-100.json", 0.60)
    assert read_heldout_scores(tmp_path) == [0.70, 0.60]


def test_heldout_scores_empty_when_no_audits_yet(tmp_path):
    from dpo_guardrail_decide import read_heldout_scores

    assert read_heldout_scores(tmp_path) == []


# --------------------------------------------------------------------------- #
# The exit contract
# --------------------------------------------------------------------------- #


def _run(tmp_path, metric: list[float], heldout: list[float]) -> int:
    from dpo_guardrail_decide import main

    state = _write_trainer_state(
        tmp_path / "ckpt" / "trainer_state.json", [{"loss": m} for m in metric]
    )
    audit_dir = tmp_path / "audit"
    for i, score in enumerate(heldout):
        _write_audit(audit_dir / f"step-{(i + 1) * 10}.json", score)
    return main(["--trainer-state", str(state), "--audit-dir", str(audit_dir)])


def test_stops_when_metric_rises_and_heldout_falls(tmp_path):
    """The reward-hacking signal: training improves, held-out quality drops."""
    assert _run(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.9], [0.70, 0.60]) == 10


def test_continues_when_heldout_also_rises(tmp_path):
    assert _run(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.9], [0.60, 0.70]) == 0


def test_continues_when_history_is_too_short(tmp_path):
    """One held-out score cannot show a direction, so the loop continues."""
    assert _run(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.9], [0.70]) == 0


def test_missing_trainer_state_is_an_error_not_a_silent_continue(tmp_path):
    """A guardrail that cannot read its input must not report 'all clear'."""
    from dpo_guardrail_decide import main

    with pytest.raises(SystemExit):
        main(["--trainer-state", str(tmp_path / "nope.json"),
              "--audit-dir", str(tmp_path)])
