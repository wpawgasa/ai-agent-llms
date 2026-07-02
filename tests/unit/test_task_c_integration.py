"""End-to-end integration smoke test for the Task C playbook->graph pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from clean_task_c_pairs import main as clean_main  # noqa: E402
from split_task_c_pairs import main as split_main  # noqa: E402

from llm_workflow_agents.data import generate_playbook_dataset  # noqa: E402
from tests.unit._task_c_helpers import CompliantTeacher, _patch_all_teachers  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_pipeline_end_to_end(monkeypatch, tmp_path):
    _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
    raw = tmp_path / "raw"
    stats = generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=1.0,
                                      seed=142, output_dir=raw)
    assert stats.renderings_accepted > 0 and not stats.halted_legs

    monkeypatch.setattr(sys, "argv", ["clean_task_c_pairs.py", "--input-dir", str(raw),
                                      "--output-dir", str(tmp_path / "cleaned"), "--quiet"])
    clean_main()

    monkeypatch.setattr(sys, "argv", ["split_task_c_pairs.py", "--input-dir", str(tmp_path / "cleaned"),
                                      "--output-dir", str(tmp_path / "splits"), "--force"])
    split_main()

    for name in ("train", "validation", "test"):
        assert (tmp_path / "splits" / f"{name}.jsonl").exists()
    train_rows = _read_jsonl(tmp_path / "splits" / "train.jsonl")
    assert all(r["register"] != "manager_transcript" for r in train_rows)


def test_clean_record_accepts_every_generated_row(monkeypatch, tmp_path):
    # Every well-formed generated row passes clean_record; only cross-file dedupe
    # (a legitimate clean step) removes rows, so we check at the record level here.
    from clean_task_c_pairs import clean_record

    _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
    generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=0.0,
                              seed=142, output_dir=tmp_path)
    rows = [r for f in sorted(tmp_path.glob("pairs_*.jsonl")) for r in _read_jsonl(f)]
    assert rows
    for row in rows:
        _cleaned, reason = clean_record(row)
        assert reason is None, f"generated row rejected: {reason}"


def test_gold_scores_perfect_under_eval(monkeypatch, tmp_path):
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph,
        check_mermaid_renderability,
        check_structural_validity,
        compute_edge_f1,
        compute_node_f1,
        parse_graph_json,
    )

    _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
    generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=0.0,
                              seed=142, output_dir=tmp_path)
    rows = [r for f in sorted(tmp_path.glob("pairs_*.jsonl")) for r in _read_jsonl(f)]
    assert rows
    for row in rows:
        parsed, ok = parse_graph_json(row["messages"][2]["content"])
        assert ok and parsed is not None
        gold = WorkflowGraph(**row["graph"])
        assert check_structural_validity(gold) and check_mermaid_renderability(gold)
        assert compute_node_f1(parsed, gold) == 1.0 and compute_edge_f1(parsed, gold) == 1.0


def test_pipeline_determinism(monkeypatch, tmp_path):
    def run(tag):
        _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
        raw = tmp_path / tag / "raw"
        generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=1.0,
                                  seed=142, output_dir=raw)
        monkeypatch.setattr(sys, "argv", ["clean_task_c_pairs.py", "--input-dir", str(raw),
                                          "--output-dir", str(tmp_path / tag / "cleaned"), "--quiet"])
        clean_main()
        monkeypatch.setattr(sys, "argv", ["split_task_c_pairs.py", "--input-dir",
                                          str(tmp_path / tag / "cleaned"), "--output-dir",
                                          str(tmp_path / tag / "splits"), "--force"])
        split_main()
        return {s: (tmp_path / tag / "splits" / f"{s}.jsonl").read_bytes()
                for s in ("train", "validation", "test")}

    assert run("a") == run("b")
