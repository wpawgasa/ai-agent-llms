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


# ---------------------------------------------------------------------------
# The blend needs a reachable invocation: --data must be repeatable, because
# the text stratum and the voice stratum are SIBLING directories and the
# loader globs one level deep. Without this, blend_modality_scores could only
# ever take its "no voice stratum" identity branch.
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from llm_workflow_agents.eval.agent_benchmark import (
    DEFAULT_DATA_DIR,
    _load_samples,
    compute_modality_quality_summary,
    compute_weighted_score,
    resolve_data_paths,
)
from llm_workflow_agents.eval.state_accuracy import (
    ConversationGroundTruth,
    ConversationPrediction,
    evaluate_state_machine,
)
from llm_workflow_agents.eval.tool_call_f1 import (
    TurnGroundTruth,
    TurnPrediction,
    evaluate_tool_calls,
    evaluate_tool_calls_conversation,
)


class TestResolveDataPaths:
    def test_no_flag_falls_back_to_the_default_stratum(self):
        assert resolve_data_paths(None) == [Path(DEFAULT_DATA_DIR)]
        assert resolve_data_paths([]) == [Path(DEFAULT_DATA_DIR)]

    def test_the_default_never_leaks_into_an_explicit_run(self):
        """The argparse trap this repo has already been bitten by.

        ``action="append"`` APPENDS to a non-``None`` default instead of
        replacing it, so a default written into the flag would silently add
        the text stratum to every explicit invocation. The flag defaults to
        ``None`` and the fallback lives in ``resolve_data_paths``.
        """
        assert resolve_data_paths(["/somewhere/else"]) == [Path("/somewhere/else")]
        assert Path(DEFAULT_DATA_DIR) not in resolve_data_paths(["/somewhere/else"])

    def test_two_strata_are_both_kept(self):
        got = resolve_data_paths([DEFAULT_DATA_DIR, f"{DEFAULT_DATA_DIR}_voice"])
        assert got == [Path(DEFAULT_DATA_DIR), Path(f"{DEFAULT_DATA_DIR}_voice")]

    def test_flag_order_cannot_change_a_result(self):
        a = resolve_data_paths([DEFAULT_DATA_DIR, f"{DEFAULT_DATA_DIR}_voice"])
        b = resolve_data_paths([f"{DEFAULT_DATA_DIR}_voice", DEFAULT_DATA_DIR])
        assert a == b

    def test_the_same_path_twice_is_loaded_once(self):
        assert resolve_data_paths([DEFAULT_DATA_DIR, DEFAULT_DATA_DIR]) == [
            Path(DEFAULT_DATA_DIR)
        ]


def _write_jsonl(directory: Path, name: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


class TestLoadSamplesAcrossStrata:
    def test_one_directory_behaves_exactly_as_before(self, tmp_path):
        _write_jsonl(tmp_path / "task_a", "l1_x.jsonl", [{"conversation_id": "a"}])
        got = _load_samples([tmp_path / "task_a"])
        assert [s["conversation_id"] for s in got] == ["a"]

    def test_two_sibling_directories_both_load(self, tmp_path):
        """The whole point: one directory can never hold both strata."""
        _write_jsonl(tmp_path / "task_a", "l1_x.jsonl", [{"conversation_id": "t"}])
        _write_jsonl(
            tmp_path / "task_a_voice", "l1_x.jsonl", [{"conversation_id": "v"}]
        )
        paths = resolve_data_paths(
            [str(tmp_path / "task_a_voice"), str(tmp_path / "task_a")]
        )
        got = _load_samples(paths)
        assert sorted(s["conversation_id"] for s in got) == ["t", "v"]

    def test_pointing_at_the_parent_still_finds_nothing(self, tmp_path):
        """The glob is deliberately non-recursive; naming the parent is not a
        workaround for naming both strata."""
        _write_jsonl(tmp_path / "task_a", "l1_x.jsonl", [{"conversation_id": "t"}])
        assert _load_samples([tmp_path]) == []

    def test_max_samples_caps_each_path_not_the_concatenation(self, tmp_path):
        """Otherwise a smoke run truncates one stratum to zero while the
        console still prints a voice weight, reading as 'blend applied'."""
        _write_jsonl(
            tmp_path / "task_a", "l1_x.jsonl",
            [{"conversation_id": f"t{i}"} for i in range(5)],
        )
        _write_jsonl(
            tmp_path / "task_a_voice", "l1_x.jsonl",
            [{"conversation_id": f"v{i}"} for i in range(5)],
        )
        paths = resolve_data_paths(
            [str(tmp_path / "task_a"), str(tmp_path / "task_a_voice")]
        )
        got = _load_samples(paths, max_samples_per_path=2)
        assert [s["conversation_id"] for s in got] == ["t0", "t1", "v0", "v1"]


# ---------------------------------------------------------------------------
# compute_modality_quality_summary — the branch's central new function.
# ---------------------------------------------------------------------------

_GOOD_MESSAGES = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": (
            "[STATE: START → CHECK]\n"
            '<tool_call>{"name": "lookup", "arguments": {"id": "1"}}</tool_call>'
        ),
    },
    {"role": "user", "content": "ok"},
    {
        "role": "assistant",
        "content": (
            "[STATE: CHECK → DONE]\n"
            '<tool_call>{"name": "close", "arguments": {"id": "1"}}</tool_call>'
        ),
    },
]
_BAD_MESSAGES = [
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": (
            "[STATE: WRONG → ELSEWHERE]\n"
            '<tool_call>{"name": "nonsense", "arguments": {"z": "9"}}</tool_call>'
        ),
    },
    {"role": "user", "content": "ok"},
    {
        "role": "assistant",
        "content": (
            "[STATE: ELSEWHERE → NOWHERE]\n"
            '<tool_call>{"name": "nonsense", "arguments": {"z": "9"}}</tool_call>'
        ),
    },
]
_GT_MESSAGES = [
    _GOOD_MESSAGES[0],
    {
        **_GOOD_MESSAGES[1],
        "annotations": {"state_transition": {"from": "START", "to": "CHECK"}},
    },
    _GOOD_MESSAGES[2],
    {
        **_GOOD_MESSAGES[3],
        "annotations": {"state_transition": {"from": "CHECK", "to": "DONE"}},
    },
]
_GT_TOOL_CALLS = [
    [{"name": "lookup", "arguments": {"id": "1"}}],
    [{"name": "close", "arguments": {"id": "1"}}],
]


def _build_run(modalities: list[str | None], correct: list[bool]):
    """Build one benchmark run's prediction/ground-truth lists.

    ``modalities`` sets each conversation's ``modality`` field (``None``
    omits the field entirely, as every sample predating the voice feature
    does). ``correct`` decides whether that conversation is answered
    perfectly or answered wrongly in every component.
    """
    samples, state_preds, state_gts, conv_preds, conv_gts = [], [], [], [], []
    for i, (modality, ok) in enumerate(zip(modalities, correct)):
        cid = f"c{i}"
        sample: dict = {"conversation_id": cid}
        if modality is not None:
            sample["modality"] = modality
        samples.append(sample)
        messages = _GOOD_MESSAGES if ok else _BAD_MESSAGES
        state_preds.append(ConversationPrediction(conversation_id=cid, messages=messages))
        state_gts.append(
            ConversationGroundTruth(
                conversation_id=cid, messages=_GT_MESSAGES, terminal_states=["DONE"]
            )
        )
        conv_preds.append(
            [
                TurnPrediction(turn_id=t, content=messages[1 + 2 * t]["content"])
                for t in range(2)
            ]
        )
        conv_gts.append(
            [TurnGroundTruth(turn_id=t, tool_calls=_GT_TOOL_CALLS[t]) for t in range(2)]
        )
    return samples, state_preds, state_gts, conv_preds, conv_gts


def _pooled_score(state_preds, state_gts, conv_preds, conv_gts) -> float:
    """The whole-population number, exactly as evaluate_workflow_quality builds it."""
    state = evaluate_state_machine(state_preds, state_gts)
    turn = evaluate_tool_calls(
        [t for c in conv_preds for t in c], [t for c in conv_gts for t in c]
    )
    conv = evaluate_tool_calls_conversation(conv_preds, conv_gts)
    best = conv if conv.tool_call_f1 >= turn.tool_call_f1 else turn
    return compute_weighted_score(state, best, state.task_completion_rate)


class TestModalityQualitySummary:
    def test_text_only_run_is_the_whole_population_score(self):
        """Harness-level inertness: with no voice rows the blended number is
        the number this benchmark has always reported."""
        run = _build_run([None] * 6 + ["text"] * 4, [True] * 7 + [False] * 3)
        summary = compute_modality_quality_summary(*run)
        assert summary["n_voice"] == 0
        assert summary["quality_voice"] is None
        assert summary["quality"] == summary["quality_text"]
        assert summary["quality"] == _pooled_score(*run[1:])

    def test_a_missing_modality_field_counts_as_text(self):
        run = _build_run([None, None, "voice"], [True, True, False])
        summary = compute_modality_quality_summary(*run)
        assert (summary["n_text"], summary["n_voice"]) == (2, 1)

    def test_voice_only_run_scores_the_voice_stratum(self):
        run = _build_run(["voice"] * 4, [True, True, False, False])
        summary = compute_modality_quality_summary(*run)
        assert summary["n_text"] == 0
        assert summary["quality_text"] is None
        assert summary["quality"] == summary["quality_voice"]

    def test_counts_always_sum_to_the_sample_count(self):
        run = _build_run([None, "text", "voice", "VOICE", "audio"], [True] * 5)
        summary = compute_modality_quality_summary(*run)
        assert summary["n_text"] + summary["n_voice"] == len(run[0])

    def test_an_unknown_modality_is_counted_as_text_and_named(self):
        """Not dropped, and not silent. A dropped row makes the ranking a
        measurement of a subset that still looks like a whole number."""
        run = _build_run(["text", "voice", "VOICE", "audio"], [True] * 4)
        summary = compute_modality_quality_summary(*run)
        assert summary["n_unknown_modality"] == 2
        assert summary["unknown_modalities"] == ["VOICE", "audio"]
        assert summary["n_text"] == 3  # text + the two unknown rows
        assert summary["n_voice"] == 1

    def test_a_clean_run_names_no_unknown_modality(self):
        run = _build_run(["text", "voice", None], [True] * 3)
        summary = compute_modality_quality_summary(*run)
        assert summary["n_unknown_modality"] == 0
        assert summary["unknown_modalities"] == []


class TestMixedStrataBlendIsNotAPooledMean:
    """The proof that the blend is worth having.

    68 text conversations answered perfectly, 7 voice conversations answered
    wrongly in every component. The blend weights the two STRATUM MEANS
    (0.7 x 1.0 + 0.3 x 0.0 = 0.700000). A mean over the pooled rows instead
    takes its effective weight from the row counts (68/75 = 0.906667) and
    would drift the moment either corpus is regenerated.
    """

    RUN = staticmethod(
        lambda: _build_run(["text"] * 68 + ["voice"] * 7, [True] * 68 + [False] * 7)
    )

    def test_the_blend_is_the_weighted_mean_of_the_two_stratum_means(self):
        summary = compute_modality_quality_summary(*self.RUN())
        assert summary["quality_text"] == 1.0
        assert summary["quality_voice"] == 0.0
        assert summary["quality"] == pytest.approx(0.700000)

    def test_the_pooled_mean_differs_and_is_the_row_count_ratio(self):
        run = self.RUN()
        assert _pooled_score(*run[1:]) == pytest.approx(0.906667, abs=1e-6)

    def test_the_blend_differs_from_either_stratum_alone(self):
        summary = compute_modality_quality_summary(*self.RUN())
        assert summary["quality"] != summary["quality_text"]
        assert summary["quality"] != summary["quality_voice"]

    def test_the_blend_does_not_move_with_the_row_counts(self):
        """Same two stratum scores, very different row counts, same blend —
        which is exactly what a pooled mean cannot do."""
        few = compute_modality_quality_summary(
            *_build_run(["text"] * 68 + ["voice"] * 7, [True] * 68 + [False] * 7)
        )
        many = compute_modality_quality_summary(
            *_build_run(["text"] * 10 + ["voice"] * 65, [True] * 10 + [False] * 65)
        )
        assert few["quality"] == many["quality"] == pytest.approx(0.7)


class TestTwoStratumRunEndToEnd:
    """Loader plus scorer: the two layers that together make the blend fire."""

    @staticmethod
    def _stratum_dirs(tmp_path):
        _write_jsonl(
            tmp_path / "task_a", "l1_a.jsonl",
            [{"conversation_id": f"t{i}", "modality": "text"} for i in range(4)],
        )
        _write_jsonl(
            tmp_path / "task_a_voice", "l1_a.jsonl",
            [{"conversation_id": f"v{i}", "modality": "voice"} for i in range(4)],
        )
        return tmp_path / "task_a", tmp_path / "task_a_voice"

    def _score(self, samples):
        run = _build_run(
            [s.get("modality") for s in samples],
            [s["conversation_id"].startswith("t") for s in samples],
        )
        return compute_modality_quality_summary(samples, *run[1:])

    def test_one_data_flag_produces_no_voice_stratum(self, tmp_path):
        text_dir, _ = self._stratum_dirs(tmp_path)
        summary = self._score(_load_samples(resolve_data_paths([str(text_dir)])))
        assert summary["n_voice"] == 0
        assert summary["quality"] == summary["quality_text"] == 1.0

    def test_two_data_flags_produce_a_blend_unlike_either_stratum(self, tmp_path):
        text_dir, voice_dir = self._stratum_dirs(tmp_path)
        samples = _load_samples(resolve_data_paths([str(text_dir), str(voice_dir)]))
        summary = self._score(samples)
        assert (summary["n_text"], summary["n_voice"]) == (4, 4)
        assert summary["quality_text"] == 1.0
        assert summary["quality_voice"] == 0.0
        assert summary["quality"] == pytest.approx(0.7)
        assert summary["quality"] != summary["quality_text"]
        assert summary["quality"] != summary["quality_voice"]
