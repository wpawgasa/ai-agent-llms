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
    print_decomposition_table,
)


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
