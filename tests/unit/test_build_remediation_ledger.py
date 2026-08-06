"""Tests for scripts/build_remediation_ledger.py.

The `claude` CLI is NEVER invoked here: every test that reaches the subprocess
layer patches `subprocess.run` (and `shutil.which`). A test run costs $0.

The bulk of the file exercises `validate_entry`, which is the trust boundary
between LLM-authored prose and a fine-tuning corpus. Each check in the gate has
at least one accept case and one reject case, because both directions are bugs:
a missing check bakes bad training data, an over-strict check silently drops a
conversation that was fine.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_remediation_ledger as brl

MARKER = "[STATE: VERIFY_IDENTITY → AUTHENTICATE]"


def _request(**overrides) -> dict:
    req = {
        "insert_id": "f:0:0",
        "conversation_id": "L1_009",
        "language": "en",
        "role": "assistant",
        "required_marker": MARKER,
        "position_after_msg_index": 3,
        "context_window": [],
        "key": ["f", 0],
    }
    req.update(overrides)
    return req


def _entry(**overrides) -> dict:
    entry = {
        "insert_id": "f:0:0",
        "conversation_id": "L1_009",
        "role": "assistant",
        "content": f"{MARKER}\nYour identity checks out. Let me open the account now.",
        "rationale": "bridges the tool result to the next call",
        "agent_model": "test-model",
        "schema_version": 1,
    }
    entry.update(overrides)
    return entry


def _user_request(**overrides) -> dict:
    defaults = {"role": "user", "required_marker": "", "language": "th"}
    defaults.update(overrides)
    return _request(**defaults)


def _user_entry(**overrides) -> dict:
    entry = _entry(role="user", content="รบกวนดำเนินการต่อเลยค่ะ ขอบคุณมากนะคะ")
    entry.update(overrides)
    return entry


def _paths_from_prompt(prompt: str) -> tuple[Path, Path]:
    """Recover the batch/ledger paths the driver assigned, as the agent would."""
    marker = "ledger entries, one JSON object per line, to "
    ledger = Path(prompt.split(marker)[1].split(" ")[0])
    batch = Path(prompt.split("batch request file at ")[1].split(" ")[0])
    return batch, ledger


# --------------------------------------------------------------------------
# accept cases -- the gate must not reject conforming work
# --------------------------------------------------------------------------


def test_accepts_conforming_assistant_entry():
    assert brl.validate_entry(_entry(), _request()) == []


def test_accepts_conforming_user_ack_with_empty_marker():
    """An empty-marker user ack must NOT be required to start with a marker."""
    assert brl.validate_entry(_user_entry(), _user_request()) == []


def test_accepts_ascii_arrow_marker_copied_verbatim():
    """78 of 2,229 markers carry an ASCII '->'; copying it verbatim must pass."""
    marker = "[STATE: IDENTIFY_ISSUE -> COLLECT_DIAGNOSTICS]"
    req = _request(required_marker=marker)
    entry = _entry(content=f"{marker}\nThe diagnostics service did not respond.")
    assert brl.validate_entry(entry, req) == []


def test_rejects_arrow_glyph_normalisation():
    """Rewriting '->' as '→' breaks the byte-for-byte prefix rule."""
    req = _request(required_marker="[STATE: A -> B]")
    entry = _entry(content="[STATE: A → B]\nAll set, thanks for waiting.")
    assert any("required marker" in v for v in brl.validate_entry(entry, req))


# --------------------------------------------------------------------------
# marker rules
# --------------------------------------------------------------------------


def test_rejects_wrong_marker():
    req = _request(required_marker="[STATE: A → A]")
    entry = _entry(content="[STATE: A → B]\nAll set for you now.")
    assert any("marker" in v for v in brl.validate_entry(entry, req))


def test_rejects_missing_newline_after_marker():
    entry = _entry(content=f"{MARKER} Your identity checks out, thank you.")
    violations = brl.validate_entry(entry, _request())
    assert any("newline" in v for v in violations)


def test_rejects_marker_with_no_prose():
    """A long marker alone clears the 20-char floor but carries no message."""
    entry = _entry(content=f"{MARKER}\n   ")
    assert any("prose" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_second_state_marker():
    entry = _entry(content=f"{MARKER}\nDone. [STATE: B → C] next.")
    assert any("second" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_state_marker_in_user_entry():
    entry = _user_entry(content="[STATE: A → B]\nรบกวนดำเนินการต่อเลยค่ะ")
    assert any("[STATE:" in v for v in brl.validate_entry(entry, _user_request()))


# --------------------------------------------------------------------------
# tool calls, length, identity
# --------------------------------------------------------------------------


def test_rejects_tool_call_in_content():
    entry = _entry(content=f'{MARKER}\n<tool_call>{{"name":"t","arguments":{{}}}}</tool_call>')
    assert any("tool_call" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_bare_closing_tool_call_tag():
    entry = _entry(content=f"{MARKER}\nHere you go.</tool_call>")
    assert any("tool_call" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_content_too_short():
    entry = _user_entry(content="ค่ะ")
    assert any("length" in v for v in brl.validate_entry(entry, _user_request()))


def test_rejects_content_too_long():
    entry = _entry(content=f"{MARKER}\n" + "x" * 700)
    assert any("length" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_insert_id_mismatch():
    entry = _entry(insert_id="f:0:1")
    assert any("insert_id" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_conversation_id_mismatch():
    entry = _entry(conversation_id="L9_999")
    assert any("conversation_id" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_role_mismatch():
    entry = _entry(role="user")
    assert any("role" in v for v in brl.validate_entry(entry, _request()))


def test_rejects_unexpected_schema_version():
    entry = _entry(schema_version=2)
    assert any("schema_version" in v for v in brl.validate_entry(entry, _request()))


# --------------------------------------------------------------------------
# refusals and hostile shapes -- validate_entry must never raise
# --------------------------------------------------------------------------


def test_refusal_entry_is_treated_as_rejected():
    entry = {"insert_id": "f:0:0", "conversation_id": "L1_009", "refuse": True,
             "rationale": "context insufficient", "agent_model": "test", "schema_version": 1}
    assert brl.validate_entry(entry, _request()) == ["agent refused this insert"]


def test_non_dict_entry_is_rejected_not_raised():
    assert brl.validate_entry(["not", "an", "object"], _request()) == [
        "ledger entry is not a JSON object"
    ]


def test_non_string_content_is_rejected_not_raised():
    entry = _entry(content=12345)
    violations = brl.validate_entry(entry, _request())
    assert any("expected string" in v for v in violations)


def test_missing_fields_are_reported():
    entry = {"insert_id": "f:0:0"}
    violations = brl.validate_entry(entry, _request())
    assert any("'content'" in v for v in violations)
    assert any("'rationale'" in v for v in violations)


def test_blank_agent_model_is_rejected():
    entry = _entry(agent_model="   ")
    assert any("agent_model" in v for v in brl.validate_entry(entry, _request()))


# --------------------------------------------------------------------------
# language gate (playbook Layer-1 "language" row)
# --------------------------------------------------------------------------


def test_rejects_thai_row_written_in_english():
    entry = _user_entry(content="Yes please go ahead and close the case, thank you.")
    assert any("Thai" in v for v in brl.validate_entry(entry, _user_request())), (
        "a th row with zero Thai characters is a wrong-language generation"
    )


def test_rejects_code_switch_row_written_in_pure_english():
    req = _user_request(language="code_switch")
    entry = _user_entry(content="Yes please go ahead and close the case, thank you.")
    assert any("Thai" in v for v in brl.validate_entry(entry, req))


def test_accepts_genuine_code_switch_mixed_text():
    """code_switch mixes scripts by definition -- the gate must not reject it."""
    req = _user_request(language="code_switch")
    entry = _user_entry(content="รบกวน close case ให้เลยค่ะ แล้วส่ง confirmation email มาด้วยนะคะ")
    assert brl.validate_entry(entry, req) == []


def test_rejects_english_row_written_in_thai():
    entry = _entry(content=f"{MARKER}\nยืนยันตัวตนเรียบร้อยแล้วค่ะ กำลังเปิดบัญชีให้นะคะ")
    assert any("Thai" in v for v in brl.validate_entry(entry, _request(language="en")))


def test_thai_assistant_entry_with_ascii_marker_passes_language_gate():
    """The marker is ASCII; the Thai check must look at the prose, not be fooled."""
    req = _request(language="th")
    entry = _entry(content=f"{MARKER}\nยืนยันตัวตนเรียบร้อยแล้วค่ะ กำลังเปิดบัญชีให้นะคะ")
    assert brl.validate_entry(entry, req) == []


# --------------------------------------------------------------------------
# request-integrity guards (driver bugs must fail loudly, not silently skip)
# --------------------------------------------------------------------------


def test_request_missing_language_is_a_violation():
    req = _request()
    del req["language"]
    assert any("language" in v for v in brl.validate_entry(_entry(), req))


def test_request_with_unknown_language_is_a_violation():
    assert any("language" in v for v in brl.validate_entry(_entry(), _request(language="fr")))


def test_user_request_carrying_a_marker_is_a_violation():
    req = _user_request(required_marker=MARKER)
    assert any("required_marker" in v for v in brl.validate_entry(_user_entry(), req))


def test_assistant_request_without_a_marker_is_a_violation():
    req = _request(required_marker="")
    assert any("required_marker" in v for v in brl.validate_entry(_entry(), req))


# --------------------------------------------------------------------------
# copy-guard
# --------------------------------------------------------------------------


def test_rejects_verbatim_copy_of_a_context_message():
    prose = "Your identity checks out. Let me open the account for you now."
    req = _request(context_window=[
        {"index": 3, "role": "assistant", "content": f"[STATE: A → B]\n{prose}"},
    ])
    entry = _entry(content=f"{MARKER}\n{prose}")
    assert any("copies" in v for v in brl.validate_entry(entry, req))


def test_short_repeated_ack_is_not_flagged_as_a_copy():
    """Short particles legitimately repeat; only long verbatim copies are wrong."""
    req = _user_request(context_window=[
        {"index": 3, "role": "user", "content": "ขอบคุณค่ะ"},
    ])
    entry = _user_entry(content="ขอบคุณค่ะ รบกวนดำเนินการต่อได้เลยนะคะ")
    assert brl.validate_entry(entry, req) == []


def test_state_marker_regex_matches_the_package_parser():
    """The local marker regex must not drift from the corpus parser's."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from llm_workflow_agents.data._workflow_script import _STATE_RE

    assert brl._STATE_MARKER_RE.pattern == _STATE_RE.pattern


# --------------------------------------------------------------------------
# ledger IO / resumability
# --------------------------------------------------------------------------


def test_load_accepted_ids_on_missing_file():
    assert brl.load_accepted_ids(Path("/nonexistent/ledger/dir")) == set()


def test_resume_skips_accepted_insert_ids(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps({"insert_id": "f:0:0", "role": "assistant", "content": "x",
                    "rationale": "x", "agent_model": "t", "schema_version": 1}) + "\n"
    )
    assert brl.load_accepted_ids(ledger_dir) == {"f:0:0"}


def test_load_accepted_ids_repairs_a_torn_final_line(tmp_path):
    """A crash mid-append leaves a partial JSON line; resume must repair it."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    good = json.dumps({"insert_id": "f:0:0", "content": "x"})
    path = ledger_dir / "accepted.jsonl"
    path.write_text(good + "\n" + '{"insert_id": "f:0:1", "conte')

    assert brl.load_accepted_ids(ledger_dir) == {"f:0:0"}
    # the torn tail is physically removed so the next append cannot glue onto it
    assert path.read_text() == good + "\n"

    brl.append_jsonl(path, [{"insert_id": "f:0:2", "content": "y"}])
    assert brl.load_accepted_ids(ledger_dir) == {"f:0:0", "f:0:2"}


def test_load_accepted_ids_keeps_a_valid_last_line_without_trailing_newline(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(json.dumps({"insert_id": "f:0:0"}))
    assert brl.load_accepted_ids(ledger_dir) == {"f:0:0"}


def test_load_accepted_ids_skips_a_malformed_interior_line(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps({"insert_id": "a"}) + "\n{not json}\n"
        + json.dumps({"insert_id": "b"}) + "\n"
    )
    assert brl.load_accepted_ids(ledger_dir) == {"a", "b"}


# --------------------------------------------------------------------------
# run_agent_batch -- never raises
# --------------------------------------------------------------------------


def test_run_batch_skips_on_missing_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: None)
    result = brl.run_agent_batch(tmp_path / "batch_0.json", tmp_path, timeout=10)
    assert result["error"] == "claude CLI not found"
    assert result["ledger_path"] is None


def test_run_batch_handles_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=10)

    monkeypatch.setattr(brl.subprocess, "run", _boom)
    assert "timeout" in brl.run_agent_batch(tmp_path / "b.json", tmp_path, timeout=10)["error"]


def test_run_batch_handles_filenotfound_race(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _boom(*a, **kw):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(brl.subprocess, "run", _boom)
    assert brl.run_agent_batch(tmp_path / "b.json", tmp_path, timeout=10)["error"] == (
        "claude CLI not found"
    )


def test_run_batch_handles_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _boom(*a, **kw):
        raise OSError("fork failed")

    monkeypatch.setattr(brl.subprocess, "run", _boom)
    assert "fork failed" in brl.run_agent_batch(tmp_path / "b.json", tmp_path, timeout=10)["error"]


def test_run_batch_handles_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        brl.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(["claude"], 2, stdout="", stderr="boom"),
    )
    assert "exited 2" in brl.run_agent_batch(tmp_path / "b.json", tmp_path, timeout=10)["error"]


def test_run_batch_reports_a_missing_ledger_file(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        brl.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(["claude"], 0, stdout="{}", stderr=""),
    )
    out = brl.run_agent_batch(tmp_path / "b.json", tmp_path, timeout=10)
    assert out["ledger_path"] is None
    assert "no ledger file" in out["error"]


def test_run_batch_deletes_a_stale_ledger_before_invoking(tmp_path, monkeypatch):
    """A ledger left by an earlier crashed attempt must never be read as fresh."""
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    stale = tmp_path / "batch.ledger.jsonl"
    stale.write_text(json.dumps({"insert_id": "stale:0:0"}) + "\n")
    seen = {}

    def _run(cmd, **kw):
        seen["existed"] = stale.exists()
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(brl.subprocess, "run", _run)
    brl.run_agent_batch(tmp_path / "batch.json", tmp_path, timeout=10)
    assert seen["existed"] is False


def test_run_batch_locates_the_ledger_by_assigned_path_not_the_reply(tmp_path, monkeypatch):
    """A chatty/lying reply must not redirect where the driver reads from."""
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    decoy = tmp_path / "decoy.jsonl"
    decoy.write_text("junk\n")

    def _run(cmd, **kw):
        (tmp_path / "batch.ledger.jsonl").write_text("{}\n")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"result": f"LEDGER: {decoy} 99\nAll done!"}), stderr="",
        )

    monkeypatch.setattr(brl.subprocess, "run", _run)
    out = brl.run_agent_batch(tmp_path / "batch.json", tmp_path, timeout=10)
    assert out["ledger_path"] == tmp_path / "batch.ledger.jsonl"
    assert out["reported_count"] == 99  # parsed for logging only


# --------------------------------------------------------------------------
# process_batch
# --------------------------------------------------------------------------


def _mock_agent(monkeypatch, lines: list[str] | None, returncode: int = 0):
    """Patch subprocess.run so the 'agent' writes `lines` to the assigned path."""
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _run(cmd, **kw):
        _batch, ledger_path = _paths_from_prompt(cmd[2])
        if lines is not None:
            ledger_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, returncode,
            stdout=json.dumps({"result": f"LEDGER: {ledger_path} {len(lines or [])}"}),
            stderr="err",
        )

    monkeypatch.setattr(brl.subprocess, "run", _run)


def test_process_batch_accepts_and_normalises(tmp_path, monkeypatch):
    entry = _entry()
    entry["stray_key"] = {"annotations": "smuggled"}
    _mock_agent(monkeypatch, [json.dumps(entry, ensure_ascii=False)])
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert len(out["accepted"]) == 1 and out["rejected"] == []
    assert set(out["accepted"][0]) == set(brl.LEDGER_FIELDS)
    assert "stray_key" not in out["accepted"][0]


def test_process_batch_rejects_a_bad_entry(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, [json.dumps(_entry(content=f"{MARKER}\n<tool_call></tool_call>"))])
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert out["accepted"] == []
    assert any("tool_call" in r for r in out["rejected"][0]["reasons"])


def test_process_batch_survives_malformed_json(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, ["{not json at all"])
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert out["accepted"] == []
    assert len(out["rejected"]) == 2  # the unparseable line + the unanswered request
    assert any("malformed JSON" in r for rec in out["rejected"] for r in rec["reasons"])


def test_process_batch_reports_unanswered_requests(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, [json.dumps(_entry())])
    reqs = [_request(), _request(insert_id="f:0:1")]
    out = brl.process_batch(reqs, tmp_path, timeout=10)
    assert len(out["accepted"]) == 1
    assert out["rejected"] == [{"insert_id": "f:0:1", "conversation_id": "L1_009",
                                "reasons": ["no ledger entry produced"]}]


def test_process_batch_logs_an_out_of_batch_insert_id(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, [json.dumps(_entry(insert_id="other:9:9"))])
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert out["accepted"] == []
    reasons = [r for rec in out["rejected"] for r in rec["reasons"]]
    assert any("not in this batch" in r for r in reasons)


def test_process_batch_keeps_only_the_first_entry_per_insert_id(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, [json.dumps(_entry()), json.dumps(_entry())])
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert len(out["accepted"]) == 1, "a duplicated line must not double-append"
    assert any("duplicate" in r for rec in out["rejected"] for r in rec["reasons"])


def test_process_batch_rejects_reused_prose_within_a_batch(tmp_path, monkeypatch):
    """The 'generic ack repeated across rows' defect. First occurrence stands."""
    shared = f"{MARKER}\nThank you for waiting, that is all noted and confirmed now."
    _mock_agent(monkeypatch, [
        json.dumps(_entry(insert_id="f:0:0", content=shared)),
        json.dumps(_entry(insert_id="f:0:1", content=shared)),
    ])
    out = brl.process_batch([_request(), _request(insert_id="f:0:1")], tmp_path, timeout=10)
    assert [a["insert_id"] for a in out["accepted"]] == ["f:0:0"]
    assert out["rejected"][0]["insert_id"] == "f:0:1"
    assert "duplicates the prose" in out["rejected"][0]["reasons"][0]


def test_process_batch_allows_a_repeated_short_ack(tmp_path, monkeypatch):
    req_a = _user_request(insert_id="f:0:0")
    req_b = _user_request(insert_id="f:0:1")
    short = "ขอบคุณมากค่ะ รบกวนด้วยนะคะ"
    _mock_agent(monkeypatch, [
        json.dumps(_user_entry(insert_id="f:0:0", content=short), ensure_ascii=False),
        json.dumps(_user_entry(insert_id="f:0:1", content=short), ensure_ascii=False),
    ])
    out = brl.process_batch([req_a, req_b], tmp_path, timeout=10)
    assert len(out["accepted"]) == 2 and out["rejected"] == []


def test_process_batch_rejects_everything_when_the_agent_fails(tmp_path, monkeypatch):
    _mock_agent(monkeypatch, None, returncode=1)
    reqs = [_request(), _request(insert_id="f:0:1")]
    out = brl.process_batch(reqs, tmp_path, timeout=10)
    assert out["accepted"] == []
    assert len(out["rejected"]) == 2
    assert all("exited 1" in r for rec in out["rejected"] for r in rec["reasons"])


def test_process_batch_never_raises_on_unreadable_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _run(cmd, **kw):
        prompt = cmd[2]
        _batch, p = _paths_from_prompt(prompt)
        p.mkdir()  # a directory where a file is expected
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(brl.subprocess, "run", _run)
    out = brl.process_batch([_request()], tmp_path, timeout=10)
    assert out["accepted"] == []
    assert out["rejected"]


# --------------------------------------------------------------------------
# check_ledger_file -- the agent's self-check, same code as the driver's gate
# --------------------------------------------------------------------------


def _selfcheck_files(tmp_path, requests, entries):
    batch = tmp_path / "batch.json"
    ledger = tmp_path / "batch.ledger.jsonl"
    batch.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
    ledger.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                      encoding="utf-8")
    return batch, ledger


def test_check_ledger_file_clean(tmp_path):
    batch, ledger = _selfcheck_files(tmp_path, [_request()], [_entry()])
    assert brl.check_ledger_file(batch, ledger) == []


def test_check_ledger_file_reports_violations_with_a_colon_bearing_id(tmp_path):
    """insert_ids contain colons; the report must still name them correctly."""
    req = _request(insert_id="l1_merged_20260629:101:0")
    entry = _entry(insert_id="l1_merged_20260629:101:0",
                   content=f"{MARKER}\nHere you go.</tool_call>")
    batch, ledger = _selfcheck_files(tmp_path, [req], [entry])
    problems = brl.check_ledger_file(batch, ledger)
    assert len(problems) == 1
    assert problems[0].startswith("l1_merged_20260629:101:0: ")
    assert "tool_call" in problems[0]


def test_check_ledger_file_reports_an_unwritten_entry(tmp_path):
    batch, ledger = _selfcheck_files(
        tmp_path, [_request(), _request(insert_id="f:0:1")], [_entry()])
    assert brl.check_ledger_file(batch, ledger) == ["f:0:1: no ledger entry written yet"]


def test_check_ledger_file_does_not_flag_a_deliberate_refusal(tmp_path):
    refusal = {"insert_id": "f:0:0", "conversation_id": "L1_009", "refuse": True,
               "rationale": "context insufficient", "agent_model": "t", "schema_version": 1}
    batch, ledger = _selfcheck_files(tmp_path, [_request()], [refusal])
    assert brl.check_ledger_file(batch, ledger) == []


def test_check_ledger_file_reports_duplicate_prose(tmp_path):
    shared = f"{MARKER}\nThank you for waiting, that is all noted and confirmed now."
    batch, ledger = _selfcheck_files(
        tmp_path,
        [_request(), _request(insert_id="f:0:1")],
        [_entry(content=shared), _entry(insert_id="f:0:1", content=shared)],
    )
    problems = brl.check_ledger_file(batch, ledger)
    assert len(problems) == 1 and "duplicates the prose" in problems[0]


def test_check_ledger_file_never_raises_on_missing_files(tmp_path):
    assert "cannot read batch file" in brl.check_ledger_file(
        tmp_path / "nope.json", tmp_path / "nope.jsonl")[0]
    batch, _ = _selfcheck_files(tmp_path, [_request()], [])
    assert "cannot read ledger file" in brl.check_ledger_file(batch, tmp_path / "nope.jsonl")[0]


def test_check_ledger_file_reports_malformed_json(tmp_path):
    batch, ledger = _selfcheck_files(tmp_path, [_request()], [])
    ledger.write_text("{broken\n", encoding="utf-8")
    problems = brl.check_ledger_file(batch, ledger)
    assert any("malformed JSON" in p for p in problems)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_build_batches_never_splits_a_conversation():
    reqs = (
        [{"insert_id": f"f:0:{i}", "key": ["f", 0]} for i in range(6)]
        + [{"insert_id": f"f:1:{i}", "key": ["f", 1]} for i in range(6)]
        + [{"insert_id": f"f:2:{i}", "key": ["f", 2]} for i in range(2)]
    )
    batches = brl.build_batches(reqs, batch_size=10)
    for batch in batches:
        keys = {tuple(r["key"]) for r in batch}
        for key in keys:
            in_batch = sum(1 for r in batch if tuple(r["key"]) == key)
            total = sum(1 for r in reqs if tuple(r["key"]) == key)
            assert in_batch == total, f"conversation {key} was split across batches"
    assert sum(len(b) for b in batches) == len(reqs)


def test_build_batches_gives_an_oversized_conversation_its_own_batch():
    reqs = [{"insert_id": f"f:0:{i}", "key": ["f", 0]} for i in range(14)]
    batches = brl.build_batches(reqs, batch_size=10)
    assert len(batches) == 1 and len(batches[0]) == 14


def test_build_batches_packs_up_to_the_size_limit():
    reqs = [{"insert_id": f"f:{c}:0", "key": ["f", c]} for c in range(25)]
    batches = brl.build_batches(reqs, batch_size=10)
    assert [len(b) for b in batches] == [10, 10, 5]


# --------------------------------------------------------------------------
# main() -- CLI wiring, dry-run, resume, progress
# --------------------------------------------------------------------------


def _write_report(tmp_path: Path, n_convs: int = 2) -> Path:
    records = []
    for c in range(n_convs):
        records.append({
            "key": ["f", c],
            "conversation_id": f"L1_{c:03d}",
            "language": "en",
            "move": "insert_handoff_turn",
            "inserts": [{
                "insert_id": f"f:{c}:0",
                "position_after_msg_index": 3,
                "role": "assistant",
                "required_marker": MARKER,
                "context_window": [],
            }],
        })
    records.append({"key": ["f", 99], "conversation_id": "L1_099", "language": "en",
                    "move": "none", "inserts": []})
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"input_dir": "data/output/sft/task_a", "records": records}))
    return path


def _argv(tmp_path: Path, report: Path, *extra: str) -> list[str]:
    return ["--input-dir", "data/output/sft/task_a", "--triage-report", str(report),
            "--ledger-dir", str(tmp_path / "ledger"), *extra]


def test_dry_run_makes_zero_subprocess_calls(tmp_path, capsys):
    report = _write_report(tmp_path)
    with patch.object(brl.subprocess, "run") as run:
        assert brl.main(_argv(tmp_path, report, "--dry-run")) == 0
    run.assert_not_called()
    out = capsys.readouterr().out
    assert "corpus-remediator" in out and "LEDGER:" in out, "dry-run must render the prompt"
    assert "f:0:0" in out


def test_dry_run_writes_no_ledger_files(tmp_path):
    report = _write_report(tmp_path)
    brl.main(_argv(tmp_path, report, "--dry-run"))
    assert not (tmp_path / "ledger" / "accepted.jsonl").exists()


def test_input_dir_mismatch_is_refused(tmp_path):
    report = _write_report(tmp_path)
    argv = ["--input-dir", "data/output/sft/somewhere_else", "--triage-report", str(report),
            "--ledger-dir", str(tmp_path / "ledger"), "--dry-run"]
    assert brl.main(argv) == 2, "insert_ids are file-stem keyed; a mismatched dir is unsafe"


def test_main_writes_accepted_rejected_and_progress(tmp_path, monkeypatch):
    report = _write_report(tmp_path, n_convs=2)
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")

    def _run(cmd, **kw):
        batch_path, ledger = _paths_from_prompt(cmd[2])
        reqs = json.loads(batch_path.read_text())
        lines = []
        for i, r in enumerate(reqs):
            e = _entry(insert_id=r["insert_id"], conversation_id=r["conversation_id"])
            if i == 1:  # one bad entry per batch
                e["content"] = f"{MARKER}\n<tool_call></tool_call>"
            lines.append(json.dumps(e, ensure_ascii=False))
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"result": "LEDGER: x 2"}), stderr="")

    monkeypatch.setattr(brl.subprocess, "run", _run)
    assert brl.main(_argv(tmp_path, report, "--batch-size", "10", "--max-workers", "1")) == 0

    ledger_dir = tmp_path / "ledger"
    accepted = [json.loads(x) for x in (ledger_dir / "accepted.jsonl").read_text().splitlines()]
    rejected = [json.loads(x) for x in (ledger_dir / "rejected.jsonl").read_text().splitlines()]
    assert [a["insert_id"] for a in accepted] == ["f:0:0"]
    assert [r["insert_id"] for r in rejected] == ["f:1:0"]

    progress = json.loads((ledger_dir / "progress.json").read_text())
    assert progress["accepted"] == 1 and progress["rejected"] == 1
    assert progress["batches_total"] == 1 and progress["batches_done"] == 1
    assert progress["pending_at_start"] == 2


def test_main_resumes_without_reprocessing_accepted_ids(tmp_path, monkeypatch):
    report = _write_report(tmp_path, n_convs=2)
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps(_entry(insert_id="f:0:0", conversation_id="L1_000")) + "\n"
    )
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    sent = []

    def _run(cmd, **kw):
        batch_path, ledger = _paths_from_prompt(cmd[2])
        reqs = json.loads(batch_path.read_text())
        sent.extend(r["insert_id"] for r in reqs)
        ledger.write_text("\n".join(
            json.dumps(_entry(insert_id=r["insert_id"], conversation_id=r["conversation_id"]))
            for r in reqs) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(brl.subprocess, "run", _run)
    assert brl.main(_argv(tmp_path, report, "--max-workers", "1")) == 0

    assert sent == ["f:1:0"], "an already-accepted insert must not be re-sent to the agent"
    lines = (ledger_dir / "accepted.jsonl").read_text().splitlines()
    ids = [json.loads(x)["insert_id"] for x in lines]
    assert ids == ["f:0:0", "f:1:0"], "resume must neither duplicate nor lose entries"


def test_main_limit_bounds_the_smoke_run(tmp_path, monkeypatch):
    report = _write_report(tmp_path, n_convs=5)
    monkeypatch.setattr(brl.shutil, "which", lambda _: "/usr/bin/claude")
    sent = []

    def _run(cmd, **kw):
        batch_path, _ledger = _paths_from_prompt(cmd[2])
        sent.extend(r["insert_id"] for r in json.loads(batch_path.read_text()))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(brl.subprocess, "run", _run)
    brl.main(_argv(tmp_path, report, "--limit", "2", "--max-workers", "1"))
    assert sent == ["f:0:0", "f:1:0"]


def test_main_returns_nonzero_when_nothing_could_be_accepted(tmp_path, monkeypatch):
    report = _write_report(tmp_path)
    monkeypatch.setattr(brl.shutil, "which", lambda _: None)
    assert brl.main(_argv(tmp_path, report, "--max-workers", "1")) == 1


def test_main_is_a_noop_when_everything_is_already_accepted(tmp_path, monkeypatch):
    report = _write_report(tmp_path, n_convs=1)
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps(_entry(insert_id="f:0:0", conversation_id="L1_000")) + "\n")
    with patch.object(brl.subprocess, "run") as run:
        assert brl.main(_argv(tmp_path, report, "--max-workers", "1")) == 0
    run.assert_not_called()


def test_main_rejects_a_malformed_triage_report(tmp_path):
    bad = tmp_path / "report.json"
    bad.write_text("{not json")
    assert brl.main(_argv(tmp_path, bad)) == 2


@pytest.mark.parametrize("payload", ['{"records": "nope"}', '{}'])
def test_main_rejects_a_report_without_usable_records(tmp_path, payload):
    bad = tmp_path / "report.json"
    bad.write_text(payload)
    argv = ["--input-dir", "data/output/sft/task_a", "--triage-report", str(bad),
            "--ledger-dir", str(tmp_path / "ledger")]
    assert brl.main(argv) == 2


# --------------------------------------------------------------------------
# fix round 1 -- adversarial false-accept regressions
#
# Each test here encodes an input the reviewer proved the gate ACCEPTED. They
# are the highest-value tests in the file: a regression re-opens a hole that
# bakes corrupt prose into the corpus with nothing downstream to catch it.
# --------------------------------------------------------------------------


def test_rejects_thai_row_whose_only_thai_character_is_the_baht_sign():
    """The measured hole: one ฿ let all 2,853 th/code_switch rows through."""
    entry = _entry(content=f"{MARKER}\nYour refund of ฿1,200 has been approved today.")
    violations = brl.validate_entry(entry, _request(language="th"))
    assert any("no Thai letters" in v for v in violations)


@pytest.mark.parametrize("char", ["฿", "ๆ", "๛", "๏", "ฯ", "๐", "๙"])  # noqa: RUF001
def test_non_letter_thai_block_characters_do_not_satisfy_the_language_rule(char):
    """Currency, repetition/abbreviation marks and Thai digits are not evidence
    that a sentence was written in Thai."""
    entry = _entry(content=f"{MARKER}\nAll done here {char} please confirm shortly.")
    violations = brl.validate_entry(entry, _request(language="code_switch"))
    assert any("no Thai letters" in v for v in violations)


def test_english_row_may_quote_a_baht_price():
    """The asymmetry ran backwards: `en` was rejected for the one legitimate
    English use of a Thai-block character."""
    entry = _entry(content=f"{MARKER}\nYour refund of ฿1,200 has been approved today.")
    assert brl.validate_entry(entry, _request(language="en")) == []


def test_thai_letters_still_satisfy_the_language_rule_after_narrowing():
    entry = _user_entry(content="รับทราบแล้วค่ะ รบกวนดำเนินการต่อได้เลยนะคะ")
    assert brl.validate_entry(entry, _user_request(language="th")) == []


def test_rejects_prose_padded_with_zero_width_characters():
    """U+200B is not whitespace to Python, so str.strip() left it in place and
    the prose floor was satisfied by nothing at all."""
    entry = _entry(content=f"{MARKER}\n" + "​" * 40)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations)
    assert any("control, format" in v for v in violations)


@pytest.mark.parametrize("char", ["​", "‌", "‍", "﻿", "⁠", "­"])
def test_rejects_each_zero_width_character_family(char):
    entry = _user_entry(content="ขอบคุณค่ะ รับทราบแล้วนะคะ" + char)
    violations = brl.validate_entry(entry, _user_request(language="th"))
    assert any("control, format" in v for v in violations)


def test_rejects_user_ack_that_is_one_character_of_padding():
    """1,673 of the 3,902 inserts are acks and had no prose floor at all."""
    entry = _user_entry(content="x" + " " * 19, language="en")
    violations = brl.validate_entry(entry, _user_request(language="en"))
    assert any("meaningful characters" in v for v in violations)


def test_accepts_a_short_but_real_english_user_ack():
    """The floor is 10 = len("ok, thanks"); a real ack clears it comfortably."""
    entry = _user_entry(content="ok, thanks -- please go ahead", language="en")
    assert brl.validate_entry(entry, _user_request(language="en")) == []


@pytest.mark.parametrize("token", [
    "<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|endoftext|>", "<|user|>",
    "<end_of_turn>", "<start_of_turn>", "</s>", "<bos>", "[INST]", "[gMASK]",
    "<extra_id_0>",
])
def test_rejects_chat_template_special_tokens(token):
    """A sentinel baked into the corpus is re-read as a turn boundary at
    template time, corrupting tokenisation for whichever family owns it."""
    entry = _entry(content=f"{MARKER}\nAll set on my side. {token} Anything else?")
    violations = brl.validate_entry(entry, _request())
    assert any("special token" in v for v in violations), violations


def test_state_marker_is_not_mistaken_for_a_special_token():
    """The one bracketed token the corpus legitimately contains."""
    assert brl.validate_entry(_entry(), _request()) == []


# --------------------------------------------------------------------------
# fix round 1 -- torn tail must never be glued to the next append
# --------------------------------------------------------------------------


def test_append_after_an_unterminated_tail_keeps_every_line_parseable(tmp_path):
    """`_repair_torn_tail` preserves a complete-but-unterminated last line; the
    next append used to concatenate onto it, destroying both entries."""
    path = tmp_path / "accepted.jsonl"
    path.write_text(
        json.dumps(_entry(insert_id="f:0:0")) + "\n"
        + json.dumps(_entry(insert_id="f:0:1")),  # no trailing newline
        encoding="utf-8",
    )
    seen = brl.load_accepted_ids(tmp_path)
    assert seen == {"f:0:0", "f:0:1"}

    brl.append_jsonl(path, [_entry(insert_id="f:0:2"), _entry(insert_id="f:0:3")])

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4
    ids = [json.loads(ln)["insert_id"] for ln in lines]  # must not raise
    assert ids == ["f:0:0", "f:0:1", "f:0:2", "f:0:3"]
    assert brl.load_accepted_ids(tmp_path) == {"f:0:0", "f:0:1", "f:0:2", "f:0:3"}


def test_append_jsonl_separates_a_hand_edited_unterminated_tail(tmp_path):
    """The ledger is marketed as PR-reviewable, so an editor that drops the
    final newline is a reachable state -- and `rejected.jsonl` is appended
    without ever passing through `_repair_torn_tail`."""
    path = tmp_path / "rejected.jsonl"
    path.write_text(json.dumps({"insert_id": "f:0:0"}), encoding="utf-8")
    brl.append_jsonl(path, [{"insert_id": "f:0:1"}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["insert_id"] for ln in lines] == ["f:0:0", "f:0:1"]


def test_repair_terminates_a_preserved_final_line_on_disk(tmp_path):
    path = tmp_path / "accepted.jsonl"
    path.write_text(json.dumps(_entry()), encoding="utf-8")
    brl.load_accepted_ids(tmp_path)
    assert path.read_bytes().endswith(b"\n")


# --------------------------------------------------------------------------
# fix round 1 -- never-raise: a request without insert_id must degrade
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [{}, {"conversation_id": "L1_009"},
                                 {"insert_id": None}, {"insert_id": ""}, "not-a-dict"])
def test_process_batch_degrades_on_a_request_without_insert_id(tmp_path, monkeypatch, bad):
    _mock_agent(monkeypatch, [json.dumps(_entry())])
    out = brl.process_batch([_request(), bad], tmp_path, timeout=10)
    assert [a["insert_id"] for a in out["accepted"]] == ["f:0:0"]
    assert any("no insert_id" in r for rec in out["rejected"] for r in rec["reasons"])


def test_process_batch_degrades_when_no_request_is_usable(tmp_path, monkeypatch):
    with patch.object(brl.subprocess, "run") as run:
        out = brl.process_batch([{}, {}], tmp_path, timeout=10)
    run.assert_not_called()
    assert out["accepted"] == [] and len(out["rejected"]) == 2
    assert out["error"]


def test_reject_all_path_survives_a_request_without_insert_id(tmp_path, monkeypatch):
    """The agent-failure path built its records from `r["insert_id"]` too."""
    _mock_agent(monkeypatch, None, returncode=1)
    out = brl.process_batch([_request(), {}], tmp_path, timeout=10)
    assert out["accepted"] == [] and len(out["rejected"]) == 2


def test_main_safety_net_survives_a_request_without_insert_id(tmp_path, monkeypatch, capsys):
    """main's own `except Exception` net indexed the same missing key."""
    report = _write_report(tmp_path, n_convs=1)
    monkeypatch.setattr(brl, "process_batch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(brl, "build_batches", lambda pending, size: [[{}]])
    assert brl.main(_argv(tmp_path, report, "--max-workers", "1")) == 1
    assert "driver error" in (tmp_path / "ledger" / "rejected.jsonl").read_text()


# --------------------------------------------------------------------------
# fix round 1 -- --limit must not split a conversation
# --------------------------------------------------------------------------


def _insert(conv: int, ordinal: int) -> dict:
    return {"insert_id": f"f:{conv}:{ordinal}", "key": ["f", conv],
            "conversation_id": f"L1_{conv:03d}"}


def test_limit_never_splits_a_conversation():
    """The playbook's own `--limit 20` took 2 of a conversation's 4 inserts, so
    the smoke run authored a conversation `apply_plan` would then drop."""
    pending = [_insert(0, i) for i in range(4)] + [_insert(1, i) for i in range(4)]
    trimmed = brl.limit_pending(pending, 6)
    assert [r["insert_id"] for r in trimmed] == [f"f:0:{i}" for i in range(4)]


def test_limit_takes_whole_conversations_up_to_the_budget():
    pending = [_insert(0, 0), _insert(1, 0), _insert(2, 0), _insert(3, 0)]
    assert len(brl.limit_pending(pending, 3)) == 3


def test_limit_always_takes_at_least_one_whole_conversation():
    """A conversation bigger than the limit is taken whole; a smoke run that
    produced nothing would be worse than one that overshoots by a few inserts."""
    pending = [_insert(0, i) for i in range(14)]
    assert len(brl.limit_pending(pending, 5)) == 14


@pytest.mark.parametrize("limit,expected", [(None, 8), (0, 0), (-3, 0)])
def test_limit_edge_values(limit, expected):
    pending = [_insert(c, i) for c in range(2) for i in range(4)]
    assert len(brl.limit_pending(pending, limit)) == expected


# ---------------------------------------------------------------------------
# Fix round 2: the meaningful-content rule
#
# The prose floor used to measure "visible" length -- everything not on an
# enumerated blocklist of invisible codepoints -- so the no-empty-turn invariant
# was only as good as that list, and it had holes. Each case below is a turn
# that carries no message at all and was ACCEPTED before the rule was restated
# as "the prose must contain N meaningful characters".
# ---------------------------------------------------------------------------

# All are invisible or content-free, and none is `Cf`, so none was on the old
# blocklist. Categories are noted because they are why enumeration kept losing:
# the Hangul fillers are *letters*.
INVISIBLE_BUT_NOT_FORMAT = [
    pytest.param("ㅤ", id="U+3164-hangul-filler-Lo"),
    pytest.param("⠀", id="U+2800-braille-blank-So"),
    pytest.param("️", id="U+FE0F-variation-selector-16-Mn"),
    pytest.param("\U000e0100", id="U+E0100-variation-selector-17-Mn"),
    pytest.param("ᅟ", id="U+115F-choseong-filler-Lo"),
    pytest.param("ᅠ", id="U+1160-jungseong-filler-Lo"),
    pytest.param("ﾠ", id="U+FFA0-halfwidth-filler-Lo"),
    pytest.param("឴", id="U+17B4-khmer-vowel-aq-Mn"),
    pytest.param("឵", id="U+17B5-khmer-vowel-aa-Mn"),
    pytest.param("͏", id="U+034F-combining-grapheme-joiner-Mn"),
    pytest.param("⁥", id="U+2065-unassigned-Cn"),
    pytest.param("\U000e0020", id="U+E0020-tag-space-Cf"),
]


@pytest.mark.parametrize("char", INVISIBLE_BUT_NOT_FORMAT)
def test_rejects_assistant_turn_made_entirely_of_an_invisible_character(char):
    entry = _entry(content=f"{MARKER}\n" + char * 40)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations), violations


@pytest.mark.parametrize("char", INVISIBLE_BUT_NOT_FORMAT)
def test_rejects_user_ack_made_entirely_of_an_invisible_character(char):
    entry = _user_entry(content=char * 40)
    violations = brl.validate_entry(entry, _user_request(language="th"))
    assert any("meaningful characters" in v for v in violations), violations


def test_rejects_a_body_of_dots():
    """The old floor counted any visible character, so 30 dots cleared it."""
    entry = _entry(content=f"{MARKER}\n" + "." * 30)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations)


def test_rejects_a_body_of_punctuation_and_spaces():
    entry = _entry(content=f"{MARKER}\n" + ". " * 20)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations)


def test_rejects_prose_whose_letters_are_all_outside_the_corpus_scripts():
    """Positive rule: content must contain Latin/Thai letters or digits. A body
    of Hangul syllables is real text, but it is not this corpus's text, and it
    is the same category of accident as a filler character."""
    entry = _entry(content=f"{MARKER}\n" + "가" * 30)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations)


def test_accepts_a_short_real_thai_ack():
    """The floor must not cost a legitimate short Thai acknowledgement. Thai
    spends ~20% of its length on combining marks, which are not meaningful on
    their own, so this is the case most at risk from the change."""
    entry = _user_entry(content="ขอบคุณค่ะ รับทราบนะคะ")
    assert brl.validate_entry(entry, _user_request(language="th")) == []


def test_accepts_a_short_real_english_ack():
    entry = _user_entry(content="ok, thanks -- please go ahead", language="en")
    assert brl.validate_entry(entry, _user_request(language="en")) == []


def test_accepts_prose_containing_an_emoji_alongside_real_content():
    """Only the *floor* is positive: a turn is not rejected for containing a
    non-meaningful character, it is rejected for containing nothing else."""
    entry = _entry(content=f"{MARKER}\nAll set on my side, thanks for waiting ☀")
    assert brl.validate_entry(entry, _request()) == []


def test_accepts_fullwidth_latin_because_the_count_is_nfkc_folded():
    # Fullwidth forms are the point of this test, hence the noqa.
    entry = _user_entry(content="Ｏｋａｙ, please proceed with that", language="en")  # noqa: RUF001
    assert brl.validate_entry(entry, _user_request(language="en")) == []


# ---------------------------------------------------------------------------
# Final review wave: the two residuals the Task 12 re-review left open
# ---------------------------------------------------------------------------


def test_nfkc_expansion_cannot_multiply_a_codepoint_past_the_floor():
    """U+33AF folds to "rad/s2" -- five meaningful characters from one glyph.

    Folding the whole string first let two codepoints of visible garbage clear a
    ten-character floor. Each SOURCE character now contributes at most 1.
    """
    assert brl._meaningful_len("\u33af" * 2) == 2
    entry = _entry(content=f"{MARKER}\n" + "\u33af" * 2)
    violations = brl.validate_entry(entry, _request())
    assert any("meaningful characters" in v for v in violations), violations


def test_nfkc_folding_still_credits_legitimate_compatibility_forms():
    """The cap must not undo the folding it constrains."""
    assert brl._meaningful_len("Ｏｋａｙ") == 4  # noqa: RUF001 - fullwidth is the point
    assert brl._meaningful_len("x\u00b2\u00b3") == 3  # superscript digits


@pytest.mark.parametrize(
    ("codepoint", "name"),
    [
        ("\u0e50", "THAI DIGIT ZERO"),
        ("\u0e2f", "THAI CHARACTER PAIYANNOI"),
        ("\u0e46", "THAI CHARACTER MAIYAMOK"),
    ],
)
def test_thai_script_non_letters_cannot_carry_an_english_entry(codepoint, name):
    """They count toward the prose floor (Thai script) but are not Thai letters,
    so before this an all-Thai-digit body passed as an `en` insert.

    Parametrised deliberately: the first fix enumerated `[\u0e50-\u0e59]` and left
    PAIYANNOI and MAIYAMOK open. The gate is now derived from the script, so the
    class is closed rather than three codepoints being listed.
    """
    entry = _user_entry(content=codepoint * 40, language="en")
    violations = brl.validate_entry(entry, _user_request(language="en"))
    assert any("Thai-script characters" in v for v in violations), (name, violations)


def test_no_thai_script_character_can_buy_floor_length_for_an_english_entry():
    """The structural claim, brute-forced over the whole Thai block.

    Anything `_meaningful_len` counts must be visible to the `en` guard; a gap
    between the two is exactly the hole this closed twice.
    """
    escapees = [
        f"U+{cp:04X}"
        for cp in range(0x0E00, 0x0E80)
        if brl._meaningful_len(chr(cp)) > 0
        and not brl._has_counting_thai_script(chr(cp))
    ]
    assert escapees == [], escapees


def test_thai_digits_still_do_not_satisfy_a_thai_request():
    """Rejecting them for `en` must not promote them to Thai-language evidence."""
    entry = _user_entry(content="\u0e50" * 40)
    violations = brl.validate_entry(entry, _user_request(language="th"))
    assert any("no Thai letters" in v for v in violations), violations


def test_a_baht_price_is_still_legal_in_an_english_entry():
    """The one legitimate English use of a Thai-block character stays legal."""
    entry = _entry(content=f"{MARKER}\nYour refund of \u0e3f1,200 has been approved today.")
    assert brl.validate_entry(entry, _request(language="en")) == []


# --- the guards -------------------------------------------------------------

_LONG_TH = "สวัสดีค่ะ ขอบคุณสำหรับข้อมูลนะคะ ระบบกำลังตรวจสอบรายการให้เรียบร้อยแล้วค่ะ"


@pytest.mark.parametrize("char", INVISIBLE_BUT_NOT_FORMAT[:4])
def test_copy_guard_survives_one_invisible_character(char):
    """A verbatim copy of a context turn plus one invisible character defeated
    the raw-text comparison."""
    request = _user_request(
        language="th",
        context_window=[{"index": 2, "role": "assistant", "content": _LONG_TH}],
    )
    entry = _user_entry(content=_LONG_TH[:5] + char + _LONG_TH[5:])
    violations = brl.validate_entry(entry, request)
    assert any("copies a message" in v for v in violations), violations


def test_copy_guard_survives_a_punctuation_change():
    request = _user_request(
        language="th",
        context_window=[{"index": 2, "role": "assistant", "content": _LONG_TH}],
    )
    entry = _user_entry(content=_LONG_TH + "!")
    violations = brl.validate_entry(entry, request)
    assert any("copies a message" in v for v in violations), violations


def test_copy_guard_still_allows_genuinely_different_prose():
    request = _user_request(
        language="th",
        context_window=[{"index": 2, "role": "assistant", "content": _LONG_TH}],
    )
    entry = _user_entry(content="รับทราบค่ะ รบกวนช่วยตรวจสอบยอดคงเหลือให้ด้วยนะคะ")
    assert brl.validate_entry(entry, request) == []


@pytest.mark.parametrize("char", INVISIBLE_BUT_NOT_FORMAT[:4])
def test_duplicate_guard_survives_one_invisible_character(char):
    first = {"insert_id": "f:0:0", "conversation_id": "L1_009", "content": _LONG_TH}
    second = {"insert_id": "f:0:1", "conversation_id": "L1_009",
              "content": _LONG_TH[:5] + char + _LONG_TH[5:]}
    kept, rejected = brl._reject_duplicate_content([first, second], {})
    assert [e["insert_id"] for e in kept] == ["f:0:0"]
    assert len(rejected) == 1
    assert "duplicates the prose" in rejected[0]["reasons"][0]


def test_duplicate_guard_keeps_two_genuinely_different_turns():
    first = {"insert_id": "f:0:0", "conversation_id": "L1_009", "content": _LONG_TH}
    second = {"insert_id": "f:0:1", "conversation_id": "L1_009",
              "content": "รับทราบค่ะ รบกวนช่วยตรวจสอบยอดคงเหลือในบัญชีให้ด้วยนะคะ ขอบคุณมากค่ะ"}
    kept, rejected = brl._reject_duplicate_content([first, second], {})
    assert len(kept) == 2 and rejected == []


def test_comparison_key_deletes_rather_than_spaces_invisibles():
    """Regression: substituting a space let an invisible character act as a word
    separator, so 'status' keyed as 'sta tus' and the guards missed the copy."""
    assert brl._comparison_key("status") == brl._comparison_key("staㅤtus")


# --- special tokens ---------------------------------------------------------

@pytest.mark.parametrize("token", [
    "[TOOL_CALLS]", "[AVAILABLE_TOOLS]", "[TOOL_RESULTS]",   # Mistral v3
    "<<SYS>>",                                                # Llama 2
    "<think>", "</think>",                                    # reasoning models
    "<unused0>", "<unused99>",                                # Gemma reserved
    "<reserved_special_token_0>",                             # Llama 3 reserved
    "<tool_response>", "</tool_response>",                    # Hermes/Qwen
])
def test_rejects_additional_chat_template_sentinels(token):
    entry = _entry(content=f"{MARKER}\nUnderstood, proceeding now {token} all done.")
    violations = brl.validate_entry(entry, _request())
    assert any("special token" in v for v in violations), violations


@pytest.mark.parametrize("text", [
    "Your reference is [1] and the follow-up is [2].",
    "Please review [the attached summary] before we proceed today.",
    "The status is [PENDING] until the review completes tomorrow.",
    "Bracketed prose like [see notes] must not trip the sentinel rule.",
])
def test_ordinary_bracketed_prose_is_not_a_special_token(text):
    entry = _entry(content=f"{MARKER}\n{text}")
    assert brl.validate_entry(entry, _request()) == []


def test_state_marker_is_not_a_special_token():
    """The gate's own required marker must survive the widened sentinel rule."""
    assert brl._SPECIAL_TOKEN_RE.search(MARKER) is None


def test_build_command_grants_the_tools_the_headless_agent_needs():
    """`claude -p` cannot prompt, so an ungranted Write is a silent no-op.

    The first smoke run rejected 20/20 as "wrote no ledger file" — the agent had
    authored every entry and had no way to persist it. Unit tests mock the CLI,
    so this contract only exists against the real binary; assert it explicitly.
    """
    cmd = brl.build_command("do the thing")
    assert "--allowedTools" in cmd
    granted = cmd[cmd.index("--allowedTools") + 1:]
    assert "Write" in granted, cmd
    # everything corpus-remediator.md's frontmatter declares
    for tool in ("Read", "Grep", "Glob", "Bash"):
        assert tool in granted, (tool, cmd)


def test_corpus_fingerprint_notices_a_modified_file(tmp_path):
    """The tripwire behind granting Write."""
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"y": 2}\n', encoding="utf-8")
    before = brl._corpus_fingerprint(str(tmp_path))
    assert set(before) == {"a.jsonl", "b.jsonl"}
    assert brl._corpus_fingerprint(str(tmp_path)) == before

    (tmp_path / "a.jsonl").write_text('{"x": 999999}\n', encoding="utf-8")
    assert brl._corpus_fingerprint(str(tmp_path)) != before


def test_corpus_fingerprint_is_empty_and_harmless_on_a_missing_dir(tmp_path):
    """A tripwire must never be the thing that breaks the run."""
    assert brl._corpus_fingerprint(str(tmp_path / "nope")) == {}
