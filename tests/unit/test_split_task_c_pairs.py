"""Tests for scripts/split_task_c_pairs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from split_task_c_pairs import main, partition_rows, validate_partition  # noqa: E402


def _row(pair_id, graph_id, split, domain="banking", register="sop_document"):
    return {"pair_id": pair_id, "graph_id": graph_id, "split": split,
            "domain": domain, "register": register}


def test_partition_by_recorded_split():
    rows = [_row("A_r1", "A", "train"), _row("B_r1", "B", "test"), _row("C_r1", "C", "validation")]
    buckets = partition_rows(rows)
    assert [r["pair_id"] for r in buckets["train"]] == ["A_r1"]
    assert [r["pair_id"] for r in buckets["test"]] == ["B_r1"]
    assert [r["pair_id"] for r in buckets["validation"]] == ["C_r1"]


def test_validate_clean():
    rows = [_row("A_r1", "A", "train"), _row("A_r2", "A", "train"),
            _row("U_r1", "U", "test", domain="utilities")]
    assert validate_partition(rows, "graph_id", ["utilities", "surveys"], ["manager_transcript"]) == []


def test_group_integrity_violation():
    rows = [_row("A_r1", "A", "train"), _row("A_r2", "A", "test")]
    violations = validate_partition(rows, "graph_id", [], [])
    assert any("group spans splits: A" in v for v in violations)


def test_heldout_domain_and_register_violations():
    rows = [
        _row("U_r1", "U", "train", domain="utilities"),          # held-out domain in train
        _row("M_r1", "M", "train", register="manager_transcript"),  # held-out register in train
    ]
    violations = validate_partition(rows, "graph_id", ["utilities", "surveys"], ["manager_transcript"])
    assert any("held-out domain outside test: U_r1" in v for v in violations)
    assert any("held-out register in train: M_r1" in v for v in violations)


def test_missing_split_field():
    rows = [{"pair_id": "X_r1", "graph_id": "X", "domain": "banking", "register": "sop_document"}]
    violations = validate_partition(rows, "graph_id", [], [])
    assert any("missing split field: X_r1" in v for v in violations)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_group_integrity_blocks_writes(tmp_path, monkeypatch):
    _write(tmp_path / "in.jsonl", [_row("A_r1", "A", "train"), _row("A_r2", "A", "test")])
    out = tmp_path / "splits"
    monkeypatch.setattr(sys, "argv", ["x", "--input-dir", str(tmp_path), "--output-dir", str(out)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert not out.exists()


def test_output_deterministic_bytes(tmp_path, monkeypatch):
    rows = [_row(f"{g}_r{i}", g, split) for g, split in
            [("A", "train"), ("B", "train"), ("C", "validation"), ("D", "test")] for i in (1, 2)]
    _write(tmp_path / "in.jsonl", rows)

    def run(out):
        monkeypatch.setattr(sys, "argv", ["x", "--input-dir", str(tmp_path),
                                          "--output-dir", str(out), "--force"])
        main()
        return {s: (out / f"{s}.jsonl").read_bytes() for s in ("train", "validation", "test")}

    assert run(tmp_path / "a") == run(tmp_path / "b")
    # train has A,B (2 graphs x 2 renderings)
    train = [json.loads(line) for line in (tmp_path / "a" / "train.jsonl").read_text().splitlines()]
    assert len(train) == 4
    assert [r["pair_id"] for r in train] == sorted(r["pair_id"] for r in train)
