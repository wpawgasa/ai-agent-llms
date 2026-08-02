import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "remediate_task_a_states.py"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _record(cid, messages, transitions, terminal=("TERMINAL",), initial="A"):
    return {
        "conversation_id": cid,
        "workflow_graph": {
            "states": sorted({t for pair in transitions for t in pair} | {initial, *terminal}),
            "transitions": [{"from": a, "to": b, "condition": "", "priority": 0} for a, b in transitions],
            "initial": initial, "terminal": list(terminal),
        },
        "messages": messages,
        "ground_truth": {"terminal_reached": True},
        "tool_schemas": [],
        "conversation_initiator": "user",
    }


def _amsg(content):
    return {"role": "assistant", "content": content, "annotations": None}


def test_triage_reports_bucket_counts(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    conformant = _record("A_001", [
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ], [("A", "TERMINAL")])
    needs_relabel = _record("A_002", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll set."),
    ], [("A", "TERMINAL")])
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [conformant, needs_relabel])

    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "triage", "--input-dir", str(input_dir),
         "--report", str(report_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["totals"]["rows"] == 2
    assert report["by_move"]["none"] == 1
    assert report["by_move"]["relabel"] == 1
    assert report["records"][1]["conversation_id"] == "A_002"
    assert report["records"][1]["key"] == ["l1_merged_test", 1]


def test_triage_embeds_context_window_for_each_insert(tmp_path):
    # Task 12's agent builds its prompt ONLY from the triage report, so
    # every insert must carry enough surrounding conversation to author
    # in-register prose -- and must NOT carry the 5-7 KB system message or
    # any structured annotations.
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    needs_insert = _record("A_003", [
        {"role": "system", "content": "S" * 6000, "annotations": None},
        {"role": "user", "content": "hello there", "annotations": None},
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t1", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → TERMINAL]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll done."),
    ], [("A", "B"), ("B", "TERMINAL")])
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [needs_insert])

    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "triage", "--input-dir", str(input_dir),
         "--report", str(report_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    rec = json.loads(report_path.read_text())["records"][0]
    assert rec["move"] == "insert_handoff_turn"
    assert rec["inserts"], "an insert-bearing move must emit at least one insert"
    for ins in rec["inserts"]:
        window = ins["context_window"]
        assert window, "every insert needs a non-empty context window"
        assert all(m["index"] >= 1 for m in window), "system message must be excluded"
        assert all("annotations" not in m for m in window), "annotations must be stripped"
        assert all({"index", "role", "content"} == set(m) for m in window)
        # the window must actually bracket the insert point
        indices = [m["index"] for m in window]
        assert min(indices) <= ins["position_after_msg_index"] <= max(indices) + 1


def test_apply_writes_repaired_output_and_drops_unrepairable(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    good = _record("B_001", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll set."),
    ], [("A", "TERMINAL")])
    # `_record`'s helper hardcodes conversation_initiator="user", but this
    # fixture's messages[] opens with an assistant turn (no leading user
    # message) -- an outbound-style shape. verify_repaired's shape gate
    # (find_shape_violations) requires the initiator to match the first
    # non-system message's role, so left as "user" this record would be
    # rejected as a shape violation regardless of the repair, contradicting
    # the assertion below that it is KEPT. Mark it "agent" to match its
    # actual shape instead of prepending a user turn (which would shift
    # messages[0] away from the tool-call turn the assertion below checks).
    good["conversation_initiator"] = "agent"
    unrepairable = _record("B_002", [
        _amsg('[STATE: A → Z]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ], [("A", "TERMINAL")])
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [good, unrepairable])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), "--on-unrepairable", "drop"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out_lines = (output_dir / "l1_merged_test.jsonl").read_text().splitlines()
    assert len(out_lines) == 1
    kept = json.loads(out_lines[0])
    assert kept["conversation_id"] == "B_001"
    assert "[STATE: A → A]" in kept["messages"][0]["content"]


def test_verify_strict_exits_nonzero_on_violation(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    still_broken = _record("C_001", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ], [("A", "TERMINAL")])
    _write_jsonl(bad_dir / "l1_merged_test.jsonl", [still_broken])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--input-dir", str(bad_dir), "--strict"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
