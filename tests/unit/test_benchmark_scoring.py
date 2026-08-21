"""Blending the two modality strata into one Phase 1 quality number.

The blend must be a weighted mean of per-stratum means, never a mean over
pooled rows: a pooled mean takes its weight from row counts, so the weighting
drifts silently whenever anyone regenerates the data.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.composite_score import (
    DEFAULT_VOICE_WEIGHT,
    blend_modality_scores,
)


def test_default_weight_is_030():
    assert DEFAULT_VOICE_WEIGHT == 0.30


def test_blend_at_weight_zero_is_the_text_score():
    assert blend_modality_scores(0.8, 0.2, voice_weight=0.0) == 0.8


def test_blend_at_weight_one_is_the_voice_score():
    assert blend_modality_scores(0.8, 0.2, voice_weight=1.0) == 0.2


def test_blend_at_default_weight():
    assert blend_modality_scores(0.8, 0.2) == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)


def test_no_voice_stratum_returns_the_text_score_exactly():
    """Float identity, not approximate.

    This is what makes the change safe to merge: results move when a person
    adds the voice corpus, not because someone merged a branch.
    """
    assert blend_modality_scores(0.7595, None) == 0.7595


def test_no_text_stratum_returns_the_voice_score_exactly():
    assert blend_modality_scores(None, 0.4242) == 0.4242


def test_both_absent_is_zero():
    assert blend_modality_scores(None, None) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_weight_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="voice_weight"):
        blend_modality_scores(0.5, 0.5, voice_weight=bad)
