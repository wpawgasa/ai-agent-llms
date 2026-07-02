"""Tests for scripts/clean_task_c_pairs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from clean_task_c_pairs import _structurally_valid, clean_record, main  # noqa: E402

_GRAPH = {
    "nodes": [
        {"id": "S1", "name": "START", "tools": [], "entry_actions": []},
        {"id": "S2", "name": "WORK", "tools": ["do_thing"], "entry_actions": []},
        {"id": "S3", "name": "TERMINAL", "tools": [], "entry_actions": []},
    ],
    "edges": [
        {"from_state": "S1", "to_state": "S2", "condition": "begin", "priority": 0},
        {"from_state": "S2", "to_state": "S3", "condition": "done", "priority": 0},
    ],
    "initial_state": "S1",
    "terminal_states": ["S3"],
}


def _base_record(**overrides):
    record = {
        "pair_id": "G0001_r1",
        "graph_id": "G0001",
        "source": "registry",
        "domain": "banking",
        "complexity_level": "L2",
        "register": "sop_document",
        "language": "en",
        "num_states": 3,
        "num_edges": 2,
        "distractor_count": 0,
        "paraphrase_density": "low",
        "condition_explicitness": "explicit",
        "verification": {"anchor_coverage": 1.0, "edge_ref_coverage": 1.0, "back_extraction": None},
        "graph": json.loads(json.dumps(_GRAPH)),
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "playbook text"},
            {"role": "assistant", "content": json.dumps(_GRAPH, separators=(",", ":"))},
        ],
        "split": "train",
    }
    record.update(overrides)
    return record


def test_clean_record_valid_passthrough():
    record = _base_record()
    cleaned, reason = clean_record(record)
    assert reason is None and cleaned is record


def test_drops_missing_field():
    record = _base_record()
    del record["graph"]
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason == "missing_field:graph"


def test_drops_bad_messages_shape():
    record = _base_record(messages=[{"role": "user", "content": "x"}])
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason == "bad_messages_shape"


def test_drops_assistant_graph_mismatch():
    record = _base_record()
    other = json.loads(json.dumps(_GRAPH))
    other["edges"] = other["edges"][:1]
    record["messages"][2]["content"] = json.dumps(other)
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason == "assistant_graph_mismatch"


def test_drops_unparseable_assistant():
    record = _base_record()
    record["messages"][2]["content"] = "not json"
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason == "assistant_not_json"


def test_drops_structural_invalid():
    bad_graph = json.loads(json.dumps(_GRAPH))
    bad_graph["edges"] = [{"from_state": "S1", "to_state": "S9", "condition": "x", "priority": 0}]
    record = _base_record(graph=bad_graph)
    record["messages"][2]["content"] = json.dumps(bad_graph, separators=(",", ":"))
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason.startswith("structural:")


def test_drops_bad_verification_shape():
    record = _base_record(verification={"anchor_coverage": 1.0})
    cleaned, reason = clean_record(record)
    assert cleaned is None and reason == "bad_verification_shape"


@pytest.mark.parametrize("mutate", [
    lambda g: g,  # valid
    lambda g: {**g, "edges": [{"from_state": "S1", "to_state": "S9", "condition": "x", "priority": 0}]},
    lambda g: {**g, "terminal_states": []},
    lambda g: {**g, "initial_state": "S9"},
    lambda g: {**g, "nodes": []},
])
def test_structural_mirror_matches_eval_predicate(mutate):
    from llm_workflow_agents.eval.graph_extraction_eval import WorkflowGraph, check_structural_validity

    graph = mutate(json.loads(json.dumps(_GRAPH)))
    mirror_valid = _structurally_valid(graph) is None
    eval_valid = check_structural_validity(WorkflowGraph(**graph))
    assert mirror_valid == eval_valid


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_dedupe_keeps_first_across_files(tmp_path, monkeypatch, capsys):
    a_rec = _base_record(pair_id="A")
    a_rec["messages"][1]["content"] = "SHARED PLAYBOOK"
    b_rec = _base_record(pair_id="B")
    b_rec["messages"][1]["content"] = "SHARED PLAYBOOK"
    _write(tmp_path / "a_file.jsonl", [a_rec])
    _write(tmp_path / "b_file.jsonl", [b_rec])

    out = tmp_path / "cleaned"
    monkeypatch.setattr(sys, "argv", ["clean_task_c_pairs.py", "--input-dir", str(tmp_path),
                                      "--output-dir", str(out), "--quiet"])
    main()
    kept = [json.loads(line) for f in sorted(out.glob("*.jsonl")) for line in f.read_text().splitlines()]
    assert [r["pair_id"] for r in kept] == ["A"]  # first file's row wins


def test_idempotent(tmp_path, monkeypatch):
    rec_a = _base_record(pair_id="A")
    rec_a["messages"][1]["content"] = "playbook A"
    rec_b = _base_record(pair_id="B")
    rec_b["messages"][1]["content"] = "playbook B"
    _write(tmp_path / "in.jsonl", [rec_a, rec_b])
    out1 = tmp_path / "c1"
    monkeypatch.setattr(sys, "argv", ["x", "--input-dir", str(tmp_path), "--output-dir", str(out1), "--quiet"])
    main()
    out2 = tmp_path / "c2"
    monkeypatch.setattr(sys, "argv", ["x", "--input-dir", str(out1), "--output-dir", str(out2), "--quiet"])
    main()
    assert (out1 / "in.jsonl").read_bytes() == (out2 / "in.jsonl").read_bytes()
