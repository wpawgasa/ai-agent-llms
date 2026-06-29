"""Tests for scripts/clean_corrupted_conversations.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from clean_corrupted_conversations import (  # noqa: E402
    KNOWN_THAI_GARBLE,
    clean_file,
    clean_lines,
    detect_corruption,
    detect_garble_blocklist,
)


def _record(content: str, *, role: str = "assistant", language: str = "th", cid: str = "C1") -> dict:
    return {
        "conversation_id": cid,
        "language": language,
        "messages": [
            {"role": "system", "content": "machine-authored system prompt มัสดี"},
            {"role": role, "content": content},
        ],
    }


def test_known_garble_flags_corrupted_greeting() -> None:
    rec = _record("มัสดีค่ะ ติดต่อจากแผนกบริการ")
    reasons = detect_corruption(rec)
    assert any("มัสดี" in r for r in reasons)


def test_clean_thai_greeting_not_flagged() -> None:
    rec = _record("สวัสดีค่ะ ติดต่อจากแผนกบริการ")  # correct spelling
    assert detect_corruption(rec) == []


def test_system_message_excluded() -> None:
    # The garble in _record's system message must NOT trigger a flag on its own.
    rec = {
        "conversation_id": "C2",
        "language": "th",
        "messages": [{"role": "system", "content": "มัสดี garble in system only"}],
    }
    assert detect_garble_blocklist(rec, KNOWN_THAI_GARBLE) == []


def test_legit_code_switch_not_flagged() -> None:
    # Latin-after-Thai at a clause boundary is legitimate code-switch, not garble.
    rec = _record("ขอบคุณสำหรับเรื่องhotel booking", language="code_switch")
    assert detect_corruption(rec) == []


def test_dedup_same_garble_reported_once() -> None:
    rec = _record("มัสดี ... มัสดี ... มัสดี")
    reasons = detect_garble_blocklist(rec, KNOWN_THAI_GARBLE)
    assert reasons == ["known-garble 'มัสดี'"]


def test_clean_lines_keeps_raw_bytes_and_counts() -> None:
    good = json.dumps(_record("สวัสดีค่ะ"), ensure_ascii=False)
    bad = json.dumps(_record("มัสดีค่ะ", cid="BAD"), ensure_ascii=False)
    kept, report = clean_lines([good, bad])
    assert report.total == 2
    assert report.kept == 1
    assert report.removed_count == 1
    assert report.removed[0].conversation_id == "BAD"
    assert kept == [good]  # surviving line returned verbatim


def test_malformed_line_is_kept_and_recorded() -> None:
    kept, report = clean_lines(["{not valid json", json.dumps(_record("สวัสดี"))])
    assert report.malformed_lines == [1]
    assert report.kept == 2  # malformed kept conservatively + the clean row


def test_clean_file_dry_run_writes_nothing(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    original = (
        json.dumps(_record("สวัสดีค่ะ"), ensure_ascii=False)
        + "\n"
        + json.dumps(_record("มัสดีค่ะ", cid="BAD"), ensure_ascii=False)
        + "\n"
    )
    f.write_text(original, encoding="utf-8")
    report = clean_file(f, dry_run=True)
    assert report.removed_count == 1
    assert report.output_path is None
    assert f.read_text(encoding="utf-8") == original  # untouched


def test_clean_file_in_place_backs_up_and_removes(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    good = json.dumps(_record("สวัสดีค่ะ"), ensure_ascii=False)
    bad = json.dumps(_record("มัสดีค่ะ", cid="BAD"), ensure_ascii=False)
    f.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    report = clean_file(f)
    assert report.removed_count == 1
    out_lines = f.read_text(encoding="utf-8").splitlines()
    assert out_lines == [good]
    backup = f.with_suffix(f.suffix + ".bak")
    assert backup.exists()
    assert len(backup.read_text(encoding="utf-8").splitlines()) == 2


def test_clean_file_to_separate_output_keeps_input(tmp_path: Path) -> None:
    f = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    good = json.dumps(_record("สวัสดีค่ะ"), ensure_ascii=False)
    bad = json.dumps(_record("มัสดีค่ะ", cid="BAD"), ensure_ascii=False)
    f.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    clean_file(f, output_path=out)
    assert len(f.read_text(encoding="utf-8").splitlines()) == 2  # input untouched
    assert out.read_text(encoding="utf-8").splitlines() == [good]


def test_clean_file_in_place_noop_leaves_file_untouched(tmp_path: Path) -> None:
    f = tmp_path / "clean.jsonl"
    original = json.dumps(_record("สวัสดีค่ะ"), ensure_ascii=False) + "\n"
    f.write_text(original, encoding="utf-8")
    mtime_before = f.stat().st_mtime_ns
    report = clean_file(f)
    assert report.removed_count == 0
    assert report.output_path is None  # signals "not written"
    assert f.read_text(encoding="utf-8") == original
    assert f.stat().st_mtime_ns == mtime_before  # truly untouched
    assert not f.with_suffix(f.suffix + ".bak").exists()


def test_extra_garble_extends_detection() -> None:
    rec = _record("คำแปลกพิเศษ")  # clean by default
    assert detect_corruption(rec) == []
    reasons = detect_corruption(rec, garble=KNOWN_THAI_GARBLE + ("คำแปลกพิเศษ",))
    assert reasons == ["known-garble 'คำแปลกพิเศษ'"]
