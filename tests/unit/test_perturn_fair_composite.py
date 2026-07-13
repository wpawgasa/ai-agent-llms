"""Unit tests for scripts/perturn_fair_composite.py (pure re-scorer)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from perturn_fair_composite import (  # noqa: E402
    _terminal_turn,
    perturn_fair_composite_from_components,
)


def test_terminal_turn_detection():
    assert _terminal_turn([{"from": "A", "to": "TERMINAL"}], "TERMINAL") is True
    assert _terminal_turn([{"from": "A", "to": "B"}], "TERMINAL") is False
    assert _terminal_turn([], "TERMINAL") is False
    assert _terminal_turn([{"from": "A", "to": "TERMINAL"}], "") is False
    # tuple form
    assert _terminal_turn([("A", "TERMINAL")], "TERMINAL") is True


def test_all_three_terms_apply_matches_whole_conv_weights():
    # terminal turn with a transition -> all three terms, weights 0.4/0.4/0.2
    s, appl = perturn_fair_composite_from_components(
        state_acc=1.0, tool_f1=0.5, task=1.0,
        n_gt_trans=1, gt_state_sequence=[{"from": "X", "to": "T"}], gt_terminal="T",
    )
    assert appl == {"state": True, "tool": True, "task": True}
    assert abs(s - (0.4 * 1.0 + 0.4 * 0.5 + 0.2 * 1.0)) < 1e-9


def test_abstention_turn_drops_state_term():
    # GT has no transition -> state term excluded; score = tool only (task not terminal)
    s, appl = perturn_fair_composite_from_components(
        state_acc=0.0, tool_f1=1.0, task=0.0,
        n_gt_trans=0, gt_state_sequence=[], gt_terminal="",
    )
    assert appl["state"] is False and appl["task"] is False
    assert abs(s - 1.0) < 1e-9  # renormalized over tool weight only


def test_intermediate_turn_drops_task_term():
    # transition present but NOT terminal -> state+tool, task excluded
    s, appl = perturn_fair_composite_from_components(
        state_acc=1.0, tool_f1=0.0, task=0.0,
        n_gt_trans=1, gt_state_sequence=[{"from": "A", "to": "B"}], gt_terminal="T",
    )
    assert appl == {"state": True, "tool": True, "task": False}
    # renormalized over 0.4 (state) + 0.4 (tool) = 0.8
    assert abs(s - (0.4 * 1.0 + 0.4 * 0.0) / 0.8) < 1e-9


def test_never_gives_unearned_credit_on_applicable_term():
    # a genuine tool miss on a turn that expects a tool still scores 0 on tool
    s, _ = perturn_fair_composite_from_components(
        state_acc=1.0, tool_f1=0.0, task=0.0,
        n_gt_trans=1, gt_state_sequence=[{"from": "A", "to": "B"}], gt_terminal="T",
    )
    assert s < 1.0  # tool=0 drags the (state,tool) average to 0.5
    assert abs(s - 0.5) < 1e-9
