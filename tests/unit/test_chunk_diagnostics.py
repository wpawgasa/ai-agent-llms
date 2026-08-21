"""Reference-free chunk diagnostics.

Not measured against gold: the benchmark scores free generation, so the model's
words differ from the gold words, and once the words differ the boundaries
cannot be aligned. A "chunk F1 against gold" would compare boundaries in two
different sentences and return noise.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.chunk_diagnostics import chunk_diagnostics


def test_first_chunk_length_is_the_first_chunk_not_the_shortest():
    out = chunk_diagnostics(["<S>" + "a" * 40 + "</S><S>bb</S>"])
    assert out["first_chunk_p50"] == 40


def test_chunk_length_percentiles():
    """Three chunks of 10 / 50 / 90 characters.

    _pct uses nearest-rank: index int(q * n), clamped. So p50 of three values
    is the middle one and p90 is the largest. Two values would make p50 the
    UPPER of the pair, which is why this fixture uses three.
    """
    turn = "<S>" + "a" * 10 + "</S><S>" + "b" * 50 + "</S><S>" + "c" * 90 + "</S>"
    out = chunk_diagnostics([turn])
    assert out["chunk_len_p50"] == 50
    assert out["chunk_len_p90"] == 90


def test_chunks_per_turn():
    out = chunk_diagnostics(["<S>a</S><S>b</S><S>c</S>"])
    assert out["chunks_per_turn_p50"] == 3


def test_boundary_quality_english_terminal_punctuation():
    good = chunk_diagnostics(["<S>Hello there.</S><S>How can I help?</S>"], "en")
    bad = chunk_diagnostics(["<S>Hello there and</S><S>how can I</S>"], "en")
    assert good["boundary_quality"] == 1.0
    assert bad["boundary_quality"] == 0.0


def test_boundary_quality_thai_final_particles():
    out = chunk_diagnostics(["<S>สวัสดีค่ะ</S><S>ยินดีให้บริการค่ะ</S>"], "th")
    assert out["boundary_quality"] == 1.0


def test_code_switch_accepts_either_convention():
    out = chunk_diagnostics(["<S>Hello there.</S><S>ยินดีให้บริการค่ะ</S>"], "code_switch")
    assert out["boundary_quality"] == 1.0


def test_turn_with_no_chunks_is_excluded_not_counted_as_bad():
    """A silent tool-call turn is legal and carries no chunk."""
    out = chunk_diagnostics(["[STATE: A → A]\n<tool_call>{}</tool_call>"])
    assert out["n_turns_with_chunks"] == 0
    assert out["boundary_quality"] == 0.0


def test_empty_input_does_not_raise():
    out = chunk_diagnostics([])
    assert out["n_turns_with_chunks"] == 0


class TestAttachChunkDiagnostics:
    """Guardrail wiring in agent_benchmark.py: quality must never move."""

    def test_quality_unchanged_when_diagnostics_block_is_present(self):
        from llm_workflow_agents.eval.agent_benchmark import attach_chunk_diagnostics

        base_summary = {
            "quality_text": 0.6123,
            "quality_voice": 0.7456,
            "quality": 0.6584,
            "n_text": 120,
            "n_voice": 40,
            "voice_weight": 0.3,
        }

        # A pathological voice completion (one giant chunk, no terminal
        # punctuation) would tank boundary_quality and chunk_len_p90 if these
        # diagnostics were ever folded into the composite.
        voice_completions = ["<S>" + "x" * 500 + "</S>"]

        out = attach_chunk_diagnostics(dict(base_summary), voice_completions, "en")

        assert out["quality"] == base_summary["quality"]
        assert out["quality_text"] == base_summary["quality_text"]
        assert out["quality_voice"] == base_summary["quality_voice"]
        assert "chunk_diagnostics" in out
        assert out["chunk_diagnostics"]["boundary_quality"] == 0.0

    def test_original_summary_dict_is_not_mutated(self):
        from llm_workflow_agents.eval.agent_benchmark import attach_chunk_diagnostics

        base_summary = {"quality": 0.5}
        attach_chunk_diagnostics(base_summary, ["<S>hi.</S>"], "en")

        assert "chunk_diagnostics" not in base_summary


class TestVoiceStratumHelpers:
    """Diagnostics must be scored over VOICE-modality rows only."""

    def test_voice_stratum_completions_excludes_text_rows(self):
        from llm_workflow_agents.eval.agent_benchmark import _voice_stratum_completions
        from llm_workflow_agents.eval.tool_call_f1 import TurnPrediction

        samples = [
            {"modality": "text"},
            {"modality": "voice"},
            {},  # no modality field -> text stratum, per compute_modality_quality_summary
        ]
        conv_tool_preds = [
            [TurnPrediction(turn_id=0, content="plain text, no chunks")],
            [TurnPrediction(turn_id=0, content="<S>voice chunk.</S>")],
            [TurnPrediction(turn_id=0, content="more plain text")],
        ]

        out = _voice_stratum_completions(samples, conv_tool_preds)
        assert out == ["<S>voice chunk.</S>"]

    def test_voice_stratum_language_defaults_to_en_with_no_voice_rows(self):
        from llm_workflow_agents.eval.agent_benchmark import _voice_stratum_language

        assert _voice_stratum_language([{"modality": "text", "language": "th"}]) == "en"

    def test_voice_stratum_language_is_the_majority_language(self):
        from llm_workflow_agents.eval.agent_benchmark import _voice_stratum_language

        samples = [
            {"modality": "voice", "language": "th"},
            {"modality": "voice", "language": "th"},
            {"modality": "voice", "language": "en"},
            {"modality": "text", "language": "en"},
        ]
        assert _voice_stratum_language(samples) == "th"
