import json
import os
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


def _v1_system_content() -> str:
    """A frozen-v1 system message, shaped like a real corpus row's.

    Persona line, then the ``Workflow script`` marker ``force_rebuild`` strips
    back to, then the v1 FORMAT_RULES -- whose rule-2 worked example shows the
    ADVANCING transition this remediation exists to remove.
    """
    from llm_workflow_agents.data.system_prompt import _format_rules

    v1_rules = _format_rules(retry_budget=1, stay_rule=False)
    assert "[STATE: VERIFY_PATIENT → TERMINAL]" in v1_rules, "fixture must carry the v1 defect"
    return (
        "You are a helpful support agent.\n\n"
        "Workflow script (follow this for conversation flow):\n"
        "[A]\n  - on success: proceed to [TERMINAL]\n\n"
        f"{v1_rules}"
    )


def _rebuildable_record(cid: str, level: str) -> dict:
    """A repairable row that opens with a v1 system message."""
    rec = _record(cid, [
        {"role": "system", "content": _v1_system_content(), "annotations": None},
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll set."),
    ], [("A", "TERMINAL")])
    # find_shape_violations skips the system message, so the first *body*
    # message is the assistant tool turn -- an outbound shape (see the note in
    # test_apply_writes_repaired_output_and_drops_unrepairable).
    rec["conversation_initiator"] = "agent"
    rec["complexity_level"] = level
    return rec


def _run_apply(tmp_path, records, *extra_args, out_name="out"):
    input_dir = tmp_path / f"in_{out_name}"
    input_dir.mkdir()
    output_dir = tmp_path / out_name
    _write_jsonl(input_dir / "l1_merged_test.jsonl", records)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), *extra_args],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = (output_dir / "l1_merged_test.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_rebuild_prompts_replaces_the_v1_worked_example(tmp_path):
    # D5: a v2 row must STATE the convention it demonstrates. Without the flag
    # the frozen v1 system prompt -- wrong worked example and all -- survives
    # byte-for-byte.
    rec = _rebuildable_record("D_001", "L3")
    original_system = rec["messages"][0]["content"]

    without = _run_apply(tmp_path, [rec], out_name="off")
    assert len(without) == 1
    assert without[0]["messages"][0]["content"] == original_system, (
        "omitting --rebuild-prompts must leave the system message byte-identical"
    )

    with_flag = _run_apply(tmp_path, [rec], "--rebuild-prompts", out_name="on")
    assert len(with_flag) == 1
    rebuilt = with_flag[0]["messages"][0]["content"]
    assert "[STATE: VERIFY_PATIENT → VERIFY_PATIENT]" in rebuilt, "corrected self-loop example"
    assert "[STATE: VERIFY_PATIENT → TERMINAL]" not in rebuilt, "old advancing example must be gone"
    # the stay rule itself, not just the example
    assert "Tool-execution turns do NOT advance" in rebuilt
    # and the rebuild must not disturb the role or the rest of the record
    assert with_flag[0]["messages"][0]["role"] == "system"
    assert with_flag[0]["conversation_id"] == "D_001"
    assert with_flag[0]["messages"][1:] == without[0]["messages"][1:]
    assert {k: v for k, v in with_flag[0].items() if k != "messages"} == \
           {k: v for k, v in without[0].items() if k != "messages"}


def test_rebuild_prompts_states_the_per_sample_retry_budget(tmp_path):
    # Task 10 made the budget per-sample off complexity_level: L5 -> 3 total
    # attempts, L1 -> 2 (raised from 1 in the final review wave). Passing the
    # whole record as the sample is what makes that flow through.
    l5 = _run_apply(tmp_path, [_rebuildable_record("D_L5", "L5")],
                    "--rebuild-prompts", out_name="l5")[0]
    l1 = _run_apply(tmp_path, [_rebuildable_record("D_L1", "L1")],
                    "--rebuild-prompts", out_name="l1")[0]
    unlabelled = _rebuildable_record("D_NONE", "L1")
    del unlabelled["complexity_level"]
    bare = _run_apply(tmp_path, [unlabelled], "--rebuild-prompts", out_name="bare")[0]

    l5_prompt = l5["messages"][0]["content"]
    l1_prompt = l1["messages"][0]["content"]
    bare_prompt = bare["messages"][0]["content"]
    assert "3 attempts at that call in total, counting the first" in l5_prompt
    assert "do NOT retry it" not in l5_prompt, "L5 must not state the no-retry policy"

    assert "2 attempts at that call in total, counting the first" in l1_prompt
    assert "do NOT retry it" not in l1_prompt, "L1 must not state the no-retry policy"
    assert "3 attempts at that call in total" not in l1_prompt

    # The budget-1 no-retry wording is now reachable only via the degradation
    # default for a row with no complexity_level.
    assert "do NOT retry it — this workflow allows one attempt per" in bare_prompt
    assert "attempts at that call in total" not in bare_prompt


def _env_with_stay_rule(value: str | None) -> dict:
    env = os.environ.copy()
    if value is None:
        env.pop("TASK_A_STAY_RULE", None)
    else:
        env["TASK_A_STAY_RULE"] = value
    return env


def test_rebuild_prompts_refuses_when_stay_rule_is_0(tmp_path):
    # TASK_A_STAY_RULE=0 selects the frozen v1 system prompt. Rebuilding under
    # it would bake that v1 prompt into what --rebuild-prompts exists to make
    # a v2 corpus -- silently, since verify --strict and the quality profiler
    # never look at system-prompt content. Must fail BEFORE any output is
    # written, not after a partial run.
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [_rebuildable_record("E_001", "L3")])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), "--rebuild-prompts"],
        capture_output=True, text=True, env=_env_with_stay_rule("0"),
    )
    assert result.returncode != 0
    assert "TASK_A_STAY_RULE" in result.stderr
    assert not output_dir.exists(), "must refuse before creating the output directory"


def test_rebuild_prompts_works_when_stay_rule_unset(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [_rebuildable_record("E_002", "L3")])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), "--rebuild-prompts"],
        capture_output=True, text=True, env=_env_with_stay_rule(None),
    )
    assert result.returncode == 0, result.stderr
    out = json.loads((output_dir / "l1_merged_test.jsonl").read_text().splitlines()[0])
    assert "Tool-execution turns do NOT advance" in out["messages"][0]["content"]


def test_apply_without_rebuild_prompts_ignores_stay_rule_env(tmp_path):
    # apply without --rebuild-prompts never touches system prompts, so the
    # guard must not fire even when TASK_A_STAY_RULE=0 is set.
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    rec = _rebuildable_record("E_003", "L3")
    original_system = rec["messages"][0]["content"]
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [rec])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, env=_env_with_stay_rule("0"),
    )
    assert result.returncode == 0, result.stderr
    out = json.loads((output_dir / "l1_merged_test.jsonl").read_text().splitlines()[0])
    assert out["messages"][0]["content"] == original_system


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


def _load_ledger_module():
    """Import the CLI module in-process so `_load_ledger` can be called directly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("remediate_task_a_states", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_ledger_fails_loudly_on_a_malformed_line(tmp_path):
    """The ledger is append-only but hand-editable, so a bad line is reachable.
    It must name the file and line rather than surfacing a bare JSONDecodeError,
    and it must not silently skip the entry -- a dropped entry drops a
    conversation from the corpus without saying so.
    """
    import pytest

    module = _load_ledger_module()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps({"insert_id": "f:0:0", "content": "ok"}) + "\n"
        + '{"insert_id": "f:0:1", "content": trunc\n'
    )
    with pytest.raises(SystemExit) as excinfo:
        module._load_ledger(ledger_dir)
    message = str(excinfo.value)
    assert "accepted.jsonl:2" in message and "not valid JSON" in message


def test_load_ledger_rejects_an_entry_without_an_insert_id(tmp_path):
    import pytest

    module = _load_ledger_module()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text('{"content": "no id here"}\n')
    with pytest.raises(SystemExit) as excinfo:
        module._load_ledger(ledger_dir)
    assert "insert_id" in str(excinfo.value)


def test_load_ledger_reads_a_well_formed_ledger(tmp_path):
    module = _load_ledger_module()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps({"insert_id": "f:0:0", "content": "a"}) + "\n\n"
        + json.dumps({"insert_id": "f:0:1", "content": "b"}) + "\n"
    )
    entries = module._load_ledger(ledger_dir)
    assert set(entries) == {"f:0:0", "f:0:1"}


# ---------------------------------------------------------------------------
# Fix round 2: explicit encodings
#
# This file's IO used the locale default. The corpus and the authored ledger are
# largely Thai, so under LC_ALL=C every read died with a bare UnicodeDecodeError
# naming neither the file nor the cause.
# ---------------------------------------------------------------------------

THAI_LEDGER_LINE = json.dumps(
    {"insert_id": "f:0:0", "content": "ขอบคุณค่ะ รับทราบแล้วนะคะ"},
    ensure_ascii=False,
)


def test_load_ledger_reads_thai_content(tmp_path):
    module = _load_ledger_module()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(THAI_LEDGER_LINE + "\n", encoding="utf-8")
    entries = module._load_ledger(ledger_dir)
    assert entries["f:0:0"]["content"] == "ขอบคุณค่ะ รับทราบแล้วนะคะ"


def test_load_ledger_tolerates_a_utf8_bom(tmp_path):
    """A BOM-prefixed ledger used to be reported as invalid JSON on line 1,
    which blames the entry rather than the encoding."""
    module = _load_ledger_module()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_bytes(
        b"\xef\xbb\xbf" + (THAI_LEDGER_LINE + "\n").encode("utf-8")
    )
    entries = module._load_ledger(ledger_dir)
    assert set(entries) == {"f:0:0"}


def test_load_ledger_reads_thai_under_a_c_locale(tmp_path):
    """The actual reproduction: a child interpreter with LC_ALL=C, where the
    locale default encoding is ASCII. Runs the real module, not a stub."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(THAI_LEDGER_LINE + "\n", encoding="utf-8")
    program = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "from pathlib import Path\n"
        f"e = m._load_ledger(Path(r'{ledger_dir}'))\n"
        "print(len(e))\n"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", program],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_iter_records_reads_thai_corpus_under_a_c_locale(tmp_path):
    """Same defect on the corpus reader, which every subcommand goes through."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "l1.jsonl").write_text(
        json.dumps({"conversation_id": "L1_1", "messages": [_amsg("สวัสดีค่ะ")]},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    program = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "from pathlib import Path\n"
        f"print(len(list(m._iter_records(Path(r'{input_dir}')))))\n"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", program],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_tool_attribution_guard_allows_losing_a_pair_but_never_gaining_one():
    """The invariant is a subset relation, not equality.

    Losing a (state, tool) pair means the repair stopped crediting a tool to a
    state it never legitimately ran from — that is the retry-after-error fix
    working. Gaining one means the repair invented a call-site, which is the
    corpus-corrupting case the guard exists to catch.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rtas", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _gains_tool_attribution = mod._gains_tool_attribution

    before = {"A": ["t1"], "B": ["t1", "t2"]}

    # unchanged
    assert _gains_tool_attribution(before, before) is False
    # narrowed: the retry stops being credited to B
    assert _gains_tool_attribution(before, {"A": ["t1"], "B": ["t2"]}) is False
    # a whole state's attribution disappears
    assert _gains_tool_attribution(before, {"A": ["t1"]}) is False
    # gained: t2 newly credited to A
    assert _gains_tool_attribution(before, {"A": ["t1", "t2"], "B": ["t1", "t2"]}) is True
    # moved: lost from B, gained at a brand-new state
    assert _gains_tool_attribution(before, {"A": ["t1"], "C": ["t2"]}) is True
