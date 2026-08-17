"""Tests for scripts/investigate_mining_yield.py.

Probe 1 (classifier effect) needs no GPU and no checkpoint: it reclassifies
completions a prior scripts/heldout_composite_audit.py run already generated,
using the exact row schema that script writes (see its `_components` /
`main()` — row_index, completion, n_gt_tools, gt_tool_calls, gt_state_sequence,
gt_terminal, composite, ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from investigate_mining_yield import (  # noqa: E402
    _gt_from_audit_row,
    classify_rate_from_audit_json,
    classify_rate_from_split,
    print_decomposition_table,
)
import investigate_mining_yield  # noqa: E402


def _audit_row(
    row_index: int,
    completion: str,
    gt_tool_calls: list[dict],
    gt_state_sequence: list[dict],
    n_gt_tools: int | None = None,
) -> dict:
    return {
        "row_index": row_index,
        "completion": completion,
        "composite": 1.0,
        "state_acc": 1.0,
        "tool_f1": 1.0,
        "task": 1.0,
        "n_gt_trans": len(gt_state_sequence),
        "n_pred_trans": 0,
        "n_gt_tools": n_gt_tools if n_gt_tools is not None else len(gt_tool_calls),
        "n_pred_tools": 0,
        "gt_state_sequence": gt_state_sequence,
        "gt_tool_calls": gt_tool_calls,
        "gt_terminal": "S_END",
        "pred_trans": [],
        "pred_tools": [],
    }


def _write_audit_json(path: Path, rows: list[dict]) -> None:
    payload = {"summary": {"n_rows": len(rows)}, "rows": rows}
    path.write_text(json.dumps(payload))


def _split_row(tools: list[dict], gold: str, states: list[dict] | None = None) -> dict:
    """One row in the shape `_select_prompts` emits for Probe 2.

    `tools` truthiness is what buckets the row into the tool-bearing or
    no-tool stratum — the identical `gt.get("tool_calls")` test
    `_select_prompts` itself uses.
    """
    return {
        "prompt_messages": [{"role": "user", "content": f"prompt for {gold}"}],
        "ground_truth": json.dumps(
            {
                "tool_calls": tools,
                "state_sequence": states or [],
                "messages": [{"content": gold}],
            }
        ),
    }


def _patch_probe2(monkeypatch, rows: list[dict], completions: list[list[str]]) -> None:
    """Stub out Probe 2's two external dependencies (row selection and GPU
    generation), following TestClassifyRateFromSplit's existing pattern."""
    import types

    monkeypatch.setattr(
        investigate_mining_yield,
        "_select_prompts",
        lambda data_dir, split, n, tool_share, seed: rows,
    )
    monkeypatch.setitem(
        sys.modules,
        "preflight_entropy_diag",
        types.SimpleNamespace(_generate_for_checkpoint=lambda **kwargs: completions),
    )


class TestGtFromAuditRow:
    def test_reconstructs_the_shape_classify_expects(self):
        row = _audit_row(
            0, "completion text",
            gt_tool_calls=[{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            gt_state_sequence=[{"from": "S1", "to": "S2"}],
        )
        gt = _gt_from_audit_row(row)
        assert gt == {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S2"}],
        }

    def test_handles_missing_fields_as_empty(self):
        row = {"completion": "x"}
        assert _gt_from_audit_row(row) == {"tool_calls": [], "state_sequence": []}


class TestClassifyRateFromAuditJson:
    def test_counts_wrong_rows_using_classify(self, tmp_path):
        perfect = _audit_row(
            0,
            '[STATE: S1 -> S1] <tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>',
            gt_tool_calls=[{"name": "x", "arguments": {"a": 1}}],
            gt_state_sequence=[{"from": "S1", "to": "S1"}],
        )
        wrong = _audit_row(
            1, "[STATE: S1 -> S1] Sure, one moment.",
            gt_tool_calls=[{"name": "x", "arguments": {"a": 1}}],
            gt_state_sequence=[{"from": "S1", "to": "S1"}],
        )
        path = tmp_path / "audit.json"
        _write_audit_json(path, [perfect, wrong])

        result = classify_rate_from_audit_json(path)
        assert result["n_rows"] == 2
        assert result["n_wrong"] == 1
        assert result["wrong_rate"] == 0.5
        assert result["by_kind"] == {"model_no_tool_call": 1}

    def test_filters_to_tool_bearing_rows_by_default(self, tmp_path):
        tool_bearing_wrong = _audit_row(
            0, "no call here",
            gt_tool_calls=[{"name": "x", "arguments": {}}],
            gt_state_sequence=[],
        )
        no_tool_row = _audit_row(
            1, "Sure, one moment.", gt_tool_calls=[], gt_state_sequence=[],
        )
        path = tmp_path / "audit.json"
        _write_audit_json(path, [tool_bearing_wrong, no_tool_row])

        result = classify_rate_from_audit_json(path, tool_bearing_only=True)
        assert result["n_rows"] == 1
        assert result["n_wrong"] == 1

        result_all = classify_rate_from_audit_json(path, tool_bearing_only=False)
        assert result_all["n_rows"] == 2
        assert result_all["n_wrong"] == 1

    def test_empty_rows_gives_zero_rate_not_a_crash(self, tmp_path):
        path = tmp_path / "audit.json"
        _write_audit_json(path, [])
        result = classify_rate_from_audit_json(path)
        assert result["n_rows"] == 0
        assert result["wrong_rate"] == 0.0


class TestPrintDecompositionTable:
    def test_prints_all_four_known_and_new_rows(self, capsys):
        print_decomposition_table(
            train_rate=0.128,
            validation_probe={"wrong_rate": 0.25},
            heldout_classify_probe={"wrong_rate": 0.30},
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert "12.8%" in out
        assert "25.0%" in out
        assert "30.0%" in out
        assert "38.0%" in out

    def test_prints_not_run_for_missing_probes(self, capsys):
        print_decomposition_table(
            train_rate=0.128,
            validation_probe=None,
            heldout_classify_probe=None,
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert out.count("not run") == 2

    def test_names_the_sample_composition_confound(self, capsys):
        """The four rows are not matched strata — the table must say so rather
        than invite a bare vertical subtraction."""
        print_decomposition_table(
            train_rate=0.128,
            validation_probe=None,
            heldout_classify_probe=None,
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert "CONFOUND" in out
        assert "NOT matched strata" in out
        # And it must point at the matched-comparison procedure.
        assert "--split train" in out and "--split validation" in out

    def test_prints_the_stratified_breakdown_when_probe2_ran(self, capsys):
        print_decomposition_table(
            train_rate=0.128,
            validation_probe={
                "split": "validation",
                "wrong_rate": 0.25,
                "n_scored": 100,
                "wrong_rate_tool_bearing": 0.40,
                "n_tool_bearing": 50,
                "wrong_rate_no_tool": 0.10,
                "n_no_tool": 50,
                "tool_share_requested": 0.75,
                "tool_share_scored": 0.50,
            },
            heldout_classify_probe=None,
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert "stratified [validation]" in out
        assert "tool-bearing 40.0% (n=50)" in out
        assert "no-tool 10.0% (n=50)" in out
        assert "requested 0.75 vs realized 0.50" in out


class TestClassifyRateFromSplit:
    def test_refuses_test_split(self):
        import pytest

        with pytest.raises(SystemExit, match="test split"):
            classify_rate_from_split(
                checkpoint="fake", data_dir=Path("unused"), split="test",
                n_prompts=1, tool_share=0.75, seed=1,
                max_new_tokens=8, max_seq_length=64, batch_size=1,
            )

    def test_scores_generations_with_classify_and_skips_empty_or_identical(
        self, monkeypatch
    ):
        fake_rows = [
            {
                "prompt_messages": [{"role": "user", "content": "q1"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer one"}],
                    }
                ),
            },
            {
                "prompt_messages": [{"role": "user", "content": "q2"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer two"}],
                    }
                ),
            },
            {
                "prompt_messages": [{"role": "user", "content": "q3"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer three"}],
                    }
                ),
            },
        ]

        def fake_select_prompts(data_dir, split, n, tool_share, seed):
            return fake_rows

        def fake_generate_for_checkpoint(**kwargs):
            # row 1: wrong (no tool call) -> counted
            # row 2: identical to gold -> skipped, not scored
            # row 3: empty -> skipped, not scored
            return [["not a tool call"], ["gold answer two"], [""]]

        monkeypatch.setattr(investigate_mining_yield, "_select_prompts", fake_select_prompts)
        import types

        fake_module = types.SimpleNamespace(
            _generate_for_checkpoint=fake_generate_for_checkpoint
        )
        monkeypatch.setitem(sys.modules, "preflight_entropy_diag", fake_module)

        result = classify_rate_from_split(
            checkpoint="fake-checkpoint",
            data_dir=Path("unused"),
            split="validation",
            n_prompts=3,
            tool_share=0.75,
            seed=1,
            max_new_tokens=8,
            max_seq_length=64,
            batch_size=1,
        )
        assert result["n_prompts_sampled"] == 3
        assert result["n_scored"] == 1
        assert result["n_wrong"] == 1
        assert result["wrong_rate"] == 1.0
        assert result["by_kind"] == {"model_no_tool_call": 1}

    def test_reports_a_stratified_tool_bearing_breakdown(self, monkeypatch):
        """The aggregate wrong_rate mixes strata of very different difficulty
        (38.0% tool-bearing vs 8.9% no-tool on the held-out audit), so it is
        not comparable to the tool-bearing-only 38.0% figure. Assert the
        stratified fields that ARE comparable."""
        fake_rows = [
            # tool-bearing, wrong (gold calls a tool, generation does not)
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold one"),
            # tool-bearing, wrong (right tool, different arguments)
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold two"),
            # tool-bearing, acceptable (matches gold on every scored axis)
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold three"),
            # no-tool, acceptable
            _split_row(tools=[], gold="gold four"),
            # no-tool, wrong (spurious call on a turn needing none)
            _split_row(tools=[], gold="gold five"),
        ]
        completions = [
            ["I will look that up for you."],
            ['<tool_call>{"name": "x", "arguments": {"a": 999}}</tool_call>'],
            ['<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'],
            ["Sure, one moment."],
            ['<tool_call>{"name": "y", "arguments": {}}</tool_call>'],
        ]
        _patch_probe2(monkeypatch, fake_rows, completions)

        result = classify_rate_from_split(
            checkpoint="fake-checkpoint", data_dir=Path("unused"),
            split="validation", n_prompts=5, tool_share=0.6, seed=1,
            max_new_tokens=8, max_seq_length=64, batch_size=1,
        )

        assert result["n_scored"] == 5
        # Aggregate: 3 of 5 wrong.
        assert result["n_wrong"] == 3
        assert result["wrong_rate"] == 0.6
        # Stratified: 2 of 3 tool-bearing wrong, 1 of 2 no-tool wrong.
        assert result["n_tool_bearing"] == 3
        assert result["n_wrong_tool_bearing"] == 2
        assert result["wrong_rate_tool_bearing"] == 2 / 3
        assert result["n_no_tool"] == 2
        assert result["n_wrong_no_tool"] == 1
        assert result["wrong_rate_no_tool"] == 0.5
        # The strata sum back to the aggregate — no row is double-counted or lost.
        assert result["n_tool_bearing"] + result["n_no_tool"] == result["n_scored"]
        assert (
            result["n_wrong_tool_bearing"] + result["n_wrong_no_tool"]
            == result["n_wrong"]
        )

    def test_reports_requested_versus_realized_tool_share(self, monkeypatch):
        """--tool-share is a target, not a guarantee: _select_prompts keeps one
        row per conversation, so validation (~289 conversations, R17) may not
        be able to supply it. A silently tool-poor sample biases the aggregate
        wrong-rate downward, so the realized share must be visible."""
        fake_rows = [
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold one"),
            _split_row(tools=[], gold="gold two"),
            _split_row(tools=[], gold="gold three"),
            _split_row(tools=[], gold="gold four"),
        ]
        completions = [["nope"], ["nope"], ["nope"], ["nope"]]
        _patch_probe2(monkeypatch, fake_rows, completions)

        result = classify_rate_from_split(
            checkpoint="fake-checkpoint", data_dir=Path("unused"),
            split="validation", n_prompts=4, tool_share=0.75, seed=1,
            max_new_tokens=8, max_seq_length=64, batch_size=1,
        )
        assert result["tool_share_requested"] == 0.75
        assert result["tool_share_scored"] == 0.25  # 1 of 4 — far short

    def test_warns_on_stderr_when_the_realized_tool_share_falls_short(
        self, monkeypatch, capsys
    ):
        fake_rows = [
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold one"),
            _split_row(tools=[], gold="gold two"),
            _split_row(tools=[], gold="gold three"),
            _split_row(tools=[], gold="gold four"),
        ]
        _patch_probe2(monkeypatch, fake_rows, [["nope"]] * 4)
        classify_rate_from_split(
            checkpoint="fake-checkpoint", data_dir=Path("unused"),
            split="validation", n_prompts=4, tool_share=0.75, seed=1,
            max_new_tokens=8, max_seq_length=64, batch_size=1,
        )
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "tool-share" in err
        assert "wrong_rate_tool_bearing" in err

    def test_no_warning_when_the_realized_tool_share_matches(
        self, monkeypatch, capsys
    ):
        fake_rows = [
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold one"),
            _split_row(tools=[{"name": "x", "arguments": {"a": 1}}], gold="gold two"),
        ]
        _patch_probe2(monkeypatch, fake_rows, [["nope"], ["nope"]])
        classify_rate_from_split(
            checkpoint="fake-checkpoint", data_dir=Path("unused"),
            split="validation", n_prompts=2, tool_share=1.0, seed=1,
            max_new_tokens=8, max_seq_length=64, batch_size=1,
        )
        assert "WARNING" not in capsys.readouterr().err

    def test_empty_sample_gives_zero_stratified_rates_not_a_crash(self, monkeypatch):
        _patch_probe2(monkeypatch, [], [])
        result = classify_rate_from_split(
            checkpoint="fake-checkpoint", data_dir=Path("unused"),
            split="validation", n_prompts=0, tool_share=0.75, seed=1,
            max_new_tokens=8, max_seq_length=64, batch_size=1,
        )
        assert result["n_scored"] == 0
        assert result["wrong_rate"] == 0.0
        assert result["wrong_rate_tool_bearing"] == 0.0
        assert result["wrong_rate_no_tool"] == 0.0
        assert result["tool_share_scored"] == 0.0
