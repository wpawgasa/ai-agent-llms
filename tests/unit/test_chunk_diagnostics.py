"""Reference-free chunk diagnostics.

Not measured against gold: the benchmark scores free generation, so the model's
words differ from the gold words, and once the words differ the boundaries
cannot be aligned. A "chunk F1 against gold" would compare boundaries in two
different sentences and return noise.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.chunk_diagnostics import (
    chunk_diagnostics,
    chunk_diagnostics_by_language,
)


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


class TestChunkDiagnosticsByLanguage:
    """Pooled, mixed-language scoring — the fix for the majority-vote bug.

    Majority vote would collapse a mixed en/th stratum to whichever language
    has more conversations and score every chunk under that one convention.
    These assert the actual requirement: each language is scored under its
    OWN convention, and the underlying measurements are pooled (not the
    already-computed percentiles) before any percentile is taken.
    """

    def test_mixed_en_th_stratum_scores_high_boundary_quality_for_both(self):
        """3 English conversations end in full stops, 1 Thai ends in a particle.

        Under majority vote (english wins 3:1) the Thai conversation's chunks
        would be checked against English terminal punctuation and score 0 —
        this is exactly the failure this fix removes.
        """
        completions_by_language = {
            "en": [
                "<S>Hello there.</S><S>How can I help?</S>",
                "<S>Sure thing.</S>",
                "<S>Have a nice day!</S>",
            ],
            "th": [
                "<S>สวัสดีค่ะ</S><S>ยินดีให้บริการค่ะ</S>",
            ],
        }
        out = chunk_diagnostics_by_language(completions_by_language)
        assert out["boundary_quality"] == 1.0
        assert out["n_turns_with_chunks"] == 4
        assert out["languages"] == ["en", "th"]

    def test_each_language_scored_under_its_own_convention_not_majority(self):
        """Minority-language (1 th vs 3 en) chunks must not be zeroed out.

        Same shape as the mixed test above but the Thai side is deliberately
        WRONG under the Thai convention (ends mid-word, no particle) while
        every English chunk is well-formed — proves the Thai group is really
        being checked with Thai rules (some, not all, well-ended) rather than
        silently dropped or scored under English rules (which would also make
        it 0, so this fixture is the one that distinguishes the two).
        """
        completions_by_language = {
            "en": ["<S>Hello there.</S>", "<S>Sure thing.</S>", "<S>Great!</S>"],
            "th": ["<S>สวัสดี</S>"],  # no Thai final particle -> not well-ended
        }
        out = chunk_diagnostics_by_language(completions_by_language)
        # 3 well-ended English chunks out of 4 total chunks.
        assert out["boundary_quality"] == pytest.approx(0.75)
        assert out["languages"] == ["en", "th"]

    def test_pools_raw_measurements_not_average_of_percentiles(self):
        """p90 of the pooled set, not an average of two per-language p90s.

        en group chunk lengths: 10, 90. th group: 50. Pooled and sorted:
        10, 50, 90 -> nearest-rank p90 (index int(0.9*3)=2) is 90, matching
        the plain chunk_diagnostics percentile fixture. An average of
        per-language p90s (90 and 50) would give 70, which is not a
        percentile of anything.
        """
        completions_by_language = {
            "en": ["<S>" + "a" * 10 + "</S><S>" + "c" * 90 + "</S>"],
            "th": ["<S>" + "b" * 50 + "ค่ะ</S>"],
        }
        out = chunk_diagnostics_by_language(completions_by_language)
        assert out["chunk_len_p90"] == 90

    def test_empty_mapping_does_not_raise(self):
        out = chunk_diagnostics_by_language({})
        assert out["n_turns_with_chunks"] == 0
        assert out["languages"] == []


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
        voice_completions_by_language = {"en": ["<S>" + "x" * 500 + "</S>"]}

        out = attach_chunk_diagnostics(dict(base_summary), voice_completions_by_language)

        assert out["quality"] == base_summary["quality"]
        assert out["quality_text"] == base_summary["quality_text"]
        assert out["quality_voice"] == base_summary["quality_voice"]
        assert "chunk_diagnostics" in out
        assert out["chunk_diagnostics"]["boundary_quality"] == 0.0

    def test_quality_unchanged_with_a_mixed_language_voice_stratum(self):
        """Same guarantee, but exercised with the mixed en/th shape this fix targets."""
        from llm_workflow_agents.eval.agent_benchmark import attach_chunk_diagnostics

        base_summary = {"quality": 0.6584, "quality_text": 0.61, "quality_voice": 0.74}
        voice_completions_by_language = {
            "en": ["<S>Hello there.</S>"],
            "th": ["<S>สวัสดีค่ะ</S>"],
        }

        out = attach_chunk_diagnostics(dict(base_summary), voice_completions_by_language)

        assert out["quality"] == base_summary["quality"]
        assert out["chunk_diagnostics"]["boundary_quality"] == 1.0
        assert out["chunk_diagnostics"]["languages"] == ["en", "th"]

    def test_original_summary_dict_is_not_mutated(self):
        from llm_workflow_agents.eval.agent_benchmark import attach_chunk_diagnostics

        base_summary = {"quality": 0.5}
        attach_chunk_diagnostics(base_summary, {"en": ["<S>hi.</S>"]})

        assert "chunk_diagnostics" not in base_summary


class TestVoiceStratumHelpers:
    """Diagnostics must be scored over VOICE-modality rows only, per-language."""

    def test_voice_stratum_completions_by_language_excludes_text_rows(self):
        from llm_workflow_agents.eval.agent_benchmark import (
            _voice_stratum_completions_by_language,
        )
        from llm_workflow_agents.eval.tool_call_f1 import TurnPrediction

        samples = [
            {"modality": "text", "language": "en"},
            {"modality": "voice", "language": "en"},
            {"modality": "voice", "language": "th"},
            {},  # no modality field -> text stratum, per compute_modality_quality_summary
        ]
        conv_tool_preds = [
            [TurnPrediction(turn_id=0, content="plain text, no chunks")],
            [TurnPrediction(turn_id=0, content="<S>voice chunk.</S>")],
            [TurnPrediction(turn_id=0, content="<S>สวัสดีค่ะ</S>")],
            [TurnPrediction(turn_id=0, content="more plain text")],
        ]

        out = _voice_stratum_completions_by_language(samples, conv_tool_preds)
        assert out == {
            "en": ["<S>voice chunk.</S>"],
            "th": ["<S>สวัสดีค่ะ</S>"],
        }

    def test_voice_sample_with_no_language_field_defaults_to_en(self):
        from llm_workflow_agents.eval.agent_benchmark import (
            _voice_stratum_completions_by_language,
        )
        from llm_workflow_agents.eval.tool_call_f1 import TurnPrediction

        samples = [{"modality": "voice"}]
        conv_tool_preds = [[TurnPrediction(turn_id=0, content="<S>hi.</S>")]]

        out = _voice_stratum_completions_by_language(samples, conv_tool_preds)
        assert out == {"en": ["<S>hi.</S>"]}

    def test_no_voice_rows_returns_empty_mapping(self):
        from llm_workflow_agents.eval.agent_benchmark import (
            _voice_stratum_completions_by_language,
        )
        from llm_workflow_agents.eval.tool_call_f1 import TurnPrediction

        samples = [{"modality": "text", "language": "th"}]
        conv_tool_preds = [[TurnPrediction(turn_id=0, content="plain text")]]

        out = _voice_stratum_completions_by_language(samples, conv_tool_preds)
        assert out == {}


class TestPerLanguageSubScores:
    """A pooled boundary quality cannot say WHICH language drags it down.

    The benchmark voice stratum draws English and Thai at even odds, so the
    only question these metrics exist to answer — is Thai chunking the
    problem? — is invisible in the pooled figure alone.
    """

    _MIXED = {
        # Every English chunk ends on terminal punctuation.
        "en": ["<S>Hello there.</S><S>How can I help?</S>"],
        # No Thai chunk ends on a sentence-final particle.
        "th": ["<S>สวัสดี</S><S>ยินดีให้บริการ</S>"],
    }

    def test_each_language_reports_its_own_boundary_quality(self):
        out = chunk_diagnostics_by_language(self._MIXED)
        assert out["boundary_quality"] == 0.5          # pooled: 2 of 4 chunks
        assert out["per_language"]["en"]["boundary_quality"] == 1.0
        assert out["per_language"]["th"]["boundary_quality"] == 0.0

    def test_per_language_holds_every_diagnostic_key(self):
        out = chunk_diagnostics_by_language(self._MIXED)
        for language in ("en", "th"):
            sub = out["per_language"][language]
            for key in (
                "first_chunk_p50", "first_chunk_p90", "chunk_len_p50",
                "chunk_len_p90", "chunks_per_turn_p50", "boundary_quality",
                "n_turns_with_chunks",
            ):
                assert key in sub, (language, key)

    def test_the_per_language_turn_counts_sum_to_the_pooled_count(self):
        out = chunk_diagnostics_by_language(self._MIXED)
        assert sum(
            sub["n_turns_with_chunks"] for sub in out["per_language"].values()
        ) == out["n_turns_with_chunks"]

    def test_a_language_with_no_chunks_still_reports_its_own_zeros(self):
        """``languages`` already lists a chunkless language; ``per_language``
        now shows that it contributed nothing, rather than leaving the reader
        to infer it."""
        out = chunk_diagnostics_by_language(
            {"en": ["<S>Hello there.</S>"], "th": ["[STATE: A → A]"]}
        )
        assert out["per_language"]["th"]["n_turns_with_chunks"] == 0
        assert out["per_language"]["th"]["boundary_quality"] == 0.0
        assert out["per_language"]["en"]["n_turns_with_chunks"] == 1

    def test_an_empty_stratum_has_an_empty_per_language_map(self):
        out = chunk_diagnostics_by_language({})
        assert out["per_language"] == {}
        assert out["languages"] == []
