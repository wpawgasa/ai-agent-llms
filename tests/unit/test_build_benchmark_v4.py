"""v4 = v3 minus the replaced rows, plus the accepted replacements."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from build_benchmark_v4 import REPLACEMENTS_FILE, build  # noqa: E402


def _write(directory: Path, name: str, ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("".join(json.dumps({"conversation_id": i}) + "\n" for i in ids))


def _ids(path: Path) -> list[str]:
    return [json.loads(l)["conversation_id"] for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture
def v3(tmp_path: Path) -> tuple[Path, Path]:
    text, voice = tmp_path / "task_a_v3", tmp_path / "task_a_voice_v2"
    _write(text, "l1.jsonl", ["L1_001", "L1_002", "L1_003"])
    _write(voice, "l2.jsonl", ["L2_001", "L2_002"])
    return text, voice


def test_replaced_rows_go_and_replacements_arrive(tmp_path: Path, v3: tuple[Path, Path]) -> None:
    text, voice = v3
    manifest = [
        {"replaces": f"{text}/l1.jsonl:2", "accepted": {"conversation_id": "L1_V4R01"}},
        {"replaces": f"{voice}/l2.jsonl:1", "accepted": {"conversation_id": "L2_V4R01"}},
        {"replaces": None, "accepted": {"conversation_id": "L3_V4R01"}},
    ]
    replacements = {"text": [{"conversation_id": "L1_V4R01"}, {"conversation_id": "L3_V4R01"}], "voice": [{"conversation_id": "L2_V4R01"}]}
    out_text, out_voice = tmp_path / "v4", tmp_path / "voice_v3"
    counts = build([("text", text, out_text), ("voice", voice, out_voice)], manifest, replacements)
    assert _ids(out_text / "l1.jsonl") == ["L1_001", "L1_003"]
    assert _ids(out_text / REPLACEMENTS_FILE) == ["L1_V4R01", "L3_V4R01"]
    assert _ids(out_voice / "l2.jsonl") == ["L2_002"]
    assert counts == {"text": {"kept": 2, "added": 2, "total": 4}, "voice": {"kept": 1, "added": 1, "total": 2}}


def test_an_unfilled_slot_refuses_the_build(tmp_path: Path, v3: tuple[Path, Path]) -> None:
    text, _ = v3
    with pytest.raises(ValueError, match="never filled"):
        build([("text", text, tmp_path / "o")], [{"replaces": f"{text}/l1.jsonl:1"}], {})


def test_a_missing_row_to_replace_is_an_error(tmp_path: Path, v3: tuple[Path, Path]) -> None:
    text, _ = v3
    manifest = [{"replaces": f"{text}/l1.jsonl:9", "accepted": {"conversation_id": "X"}}]
    with pytest.raises(ValueError, match="not found"):
        build([("text", text, tmp_path / "o")], manifest, {"text": [{"conversation_id": "X"}]})


def test_an_id_clash_is_an_error(tmp_path: Path, v3: tuple[Path, Path]) -> None:
    text, _ = v3
    manifest = [{"replaces": None, "accepted": {"conversation_id": "L1_001"}}]
    with pytest.raises(ValueError, match="already in use"):
        build([("text", text, tmp_path / "o")], manifest, {"text": [{"conversation_id": "L1_001"}]})
