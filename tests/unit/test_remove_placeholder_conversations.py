"""Tests for scripts/remove_placeholder_conversations.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from remove_placeholder_conversations import (  # noqa: E402
    DEFAULT_DROP_SOURCES,
    MISSING_SOURCE,
    filter_file,
    record_source,
    removal_reason,
    select_lines,
)


def _record(source: str | None = "teacher", *, cid: str = "C1") -> dict:
    rec: dict = {
        "conversation_id": cid,
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }
    if source is not None:
        rec["generation_source"] = source
    return rec


def test_default_drops_placeholder_fallback() -> None:
    rec = _record("placeholder_fallback")
    assert removal_reason(rec, DEFAULT_DROP_SOURCES, None) is not None


def test_default_keeps_teacher() -> None:
    rec = _record("teacher")
    assert removal_reason(rec, DEFAULT_DROP_SOURCES, None) is None


def test_missing_source_kept_in_drop_mode() -> None:
    # Drop-list mode only removes explicit matches; a missing field is kept.
    rec = _record(source=None)
    assert record_source(rec) == MISSING_SOURCE
    assert removal_reason(rec, DEFAULT_DROP_SOURCES, None) is None


def test_empty_string_source_treated_as_missing() -> None:
    rec = _record(source="")
    assert record_source(rec) == MISSING_SOURCE


def test_keep_only_removes_non_allowlisted() -> None:
    rec = _record("placeholder_fallback")
    assert removal_reason(rec, DEFAULT_DROP_SOURCES, ("teacher",)) is not None


def test_keep_only_keeps_allowlisted() -> None:
    rec = _record("teacher")
    assert removal_reason(rec, (), ("teacher",)) is None


def test_keep_only_removes_missing_source() -> None:
    # In allowlist mode a row without the field is not in the allowlist => removed.
    rec = _record(source=None)
    reason = removal_reason(rec, (), ("teacher",))
    assert reason is not None
    assert MISSING_SOURCE in reason


def test_custom_drop_list() -> None:
    rec = _record("retry_stub")
    assert removal_reason(rec, ("retry_stub",), None) is not None
    assert removal_reason(rec, ("placeholder_fallback",), None) is None


def test_select_lines_keeps_raw_bytes_and_counts() -> None:
    good = json.dumps(_record("teacher"), ensure_ascii=False)
    bad = json.dumps(_record("placeholder_fallback", cid="BAD"), ensure_ascii=False)
    kept, report = select_lines([good, bad])
    assert report.total == 2
    assert report.kept == 1
    assert report.removed_count == 1
    assert report.removed[0].conversation_id == "BAD"
    assert report.removed[0].source == "placeholder_fallback"
    assert kept == [good]  # surviving line returned verbatim


def test_malformed_line_is_kept_and_recorded() -> None:
    kept, report = select_lines(["{not valid json", json.dumps(_record("teacher"))])
    assert report.malformed_lines == [1]
    assert report.kept == 2  # malformed kept conservatively + the clean row


def test_filter_file_dry_run_writes_nothing(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    original = (
        json.dumps(_record("teacher"), ensure_ascii=False)
        + "\n"
        + json.dumps(_record("placeholder_fallback", cid="BAD"), ensure_ascii=False)
        + "\n"
    )
    f.write_text(original, encoding="utf-8")
    report = filter_file(f, dry_run=True)
    assert report.removed_count == 1
    assert report.output_path is None
    assert f.read_text(encoding="utf-8") == original  # untouched


def test_filter_file_in_place_backs_up_and_removes(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    good = json.dumps(_record("teacher"), ensure_ascii=False)
    bad = json.dumps(_record("placeholder_fallback", cid="BAD"), ensure_ascii=False)
    f.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    report = filter_file(f)
    assert report.removed_count == 1
    assert f.read_text(encoding="utf-8").splitlines() == [good]
    backup = f.with_suffix(f.suffix + ".bak")
    assert backup.exists()
    assert len(backup.read_text(encoding="utf-8").splitlines()) == 2


def test_filter_file_to_separate_output_keeps_input(tmp_path: Path) -> None:
    f = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    good = json.dumps(_record("teacher"), ensure_ascii=False)
    bad = json.dumps(_record("placeholder_fallback", cid="BAD"), ensure_ascii=False)
    f.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    filter_file(f, output_path=out)
    assert len(f.read_text(encoding="utf-8").splitlines()) == 2  # input untouched
    assert out.read_text(encoding="utf-8").splitlines() == [good]


def test_filter_file_in_place_noop_leaves_file_untouched(tmp_path: Path) -> None:
    f = tmp_path / "clean.jsonl"
    original = json.dumps(_record("teacher"), ensure_ascii=False) + "\n"
    f.write_text(original, encoding="utf-8")
    mtime_before = f.stat().st_mtime_ns
    report = filter_file(f)
    assert report.removed_count == 0
    assert report.output_path is None  # signals "not written"
    assert f.read_text(encoding="utf-8") == original
    assert f.stat().st_mtime_ns == mtime_before  # truly untouched
    assert not f.with_suffix(f.suffix + ".bak").exists()


def test_keep_only_via_filter_file(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    teacher = json.dumps(_record("teacher"), ensure_ascii=False)
    missing = json.dumps(_record(source=None, cid="NOSRC"), ensure_ascii=False)
    ph = json.dumps(_record("placeholder_fallback", cid="BAD"), ensure_ascii=False)
    f.write_text("\n".join([teacher, missing, ph]) + "\n", encoding="utf-8")
    report = filter_file(f, keep_only=("teacher",))
    assert report.removed_count == 2  # missing + placeholder both dropped
    assert f.read_text(encoding="utf-8").splitlines() == [teacher]
