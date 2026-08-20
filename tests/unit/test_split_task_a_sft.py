"""Tests for scripts/split_task_a_sft.py."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import directly without packaging
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from split_task_a_sft import (  # noqa: E402
    _load_rows,
    _modality_of,
    assert_unmoved,
    split_rows,
)

RATIOS = {"train": 0.85, "validation": 0.10, "test": 0.05}


def _rows(n: int, modality: str | None, prefix: str) -> list[dict]:
    out = []
    for i in range(n):
        row: dict = {
            "conversation_id": f"{prefix}_{i:04d}",
            "messages": [{"role": "user", "content": f"{prefix}-utterance-{i}"}],
        }
        if modality is not None:
            row["modality"] = modality
        out.append(row)
    return out


def _assignment(chunks: dict[str, list[dict]]) -> dict[str, str]:
    return {
        row["conversation_id"]: name for name, rows in chunks.items() for row in rows
    }


def test_load_rows_reads_two_directories(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.jsonl").write_text('{"conversation_id": "a1"}\n')
    (b / "two.jsonl").write_text('{"conversation_id": "b1"}\n')

    rows = _load_rows([a, b])
    assert {r["conversation_id"] for r in rows} == {"a1", "b1"}


def test_load_rows_reads_one_directory(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "one.jsonl").write_text('{"conversation_id": "a1"}\n')

    assert len(_load_rows([a])) == 1


def test_load_rows_directory_list_sorted(tmp_path):
    """Verify that directories are sorted before globbing.

    This ensures reproducibility regardless of argument order.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.jsonl").write_text('{"conversation_id": "a1"}\n')
    (b / "two.jsonl").write_text('{"conversation_id": "b1"}\n')

    # Verify both orderings produce same sequence of rows (deterministic)
    rows_ab = _load_rows([a, b])
    rows_ba = _load_rows([b, a])

    ids_ab = [r["conversation_id"] for r in rows_ab]
    ids_ba = [r["conversation_id"] for r in rows_ba]

    # After sorting directories, order should be deterministic
    assert ids_ab == ids_ba


# --- append stability -------------------------------------------------------
#
# The one property that matters: adding the voice batch must not move a single
# text conversation. Seven of the 278 pinned held-out candidates moved into
# train under the old positional shuffle, silently contaminating the only
# measurement linking back to cell C2's 0.7595 composite (risk R17).


def test_adding_a_voice_batch_moves_no_text_row():
    text = _rows(5543, None, "text")
    before = _assignment(split_rows(list(text), RATIOS, 42))

    voice = _rows(2400, "voice", "voice")
    after = _assignment(split_rows(list(text) + voice, RATIOS, 42))

    moved = {
        cid: (before[cid], after[cid]) for cid in before if before[cid] != after[cid]
    }
    assert moved == {}


def test_voice_batch_position_in_the_pool_is_irrelevant():
    """A differently-named voice directory sorts first; that must not matter."""
    text = _rows(500, None, "text")
    voice = _rows(200, "voice", "voice")
    appended = _assignment(split_rows(list(text) + list(voice), RATIOS, 42))
    prepended = _assignment(split_rows(list(voice) + list(text), RATIOS, 42))
    assert appended == prepended


def test_old_positional_shuffle_would_have_moved_rows():
    """Guards the test above against being vacuously true."""
    text = _rows(5543, None, "text")
    voice = _rows(2400, "voice", "voice")

    def positional(rows):
        rows = list(rows)
        random.Random(42).shuffle(rows)
        n = len(rows)
        n_train = int(n * 0.85)
        n_val = int(n * 0.10)
        return _assignment(
            {
                "train": rows[:n_train],
                "validation": rows[n_train : n_train + n_val],
                "test": rows[n_train + n_val :],
            }
        )

    before = positional(text)
    after = positional(text + voice)
    assert any(before[cid] != after[cid] for cid in before)


def test_ratios_hold_within_each_modality():
    """Spec section 5 needs a per-modality held-out score, so test needs both."""
    chunks = split_rows(_rows(1000, None, "text") + _rows(400, "voice", "v"), RATIOS, 42)
    for name, expected_text, expected_voice in (
        ("train", 850, 340),
        ("validation", 100, 40),
        ("test", 50, 20),
    ):
        counts = {"text": 0, "voice": 0}
        for row in chunks[name]:
            counts[_modality_of(row)] += 1
        assert counts["text"] == expected_text
        assert counts["voice"] == expected_voice


def test_every_row_lands_in_exactly_one_split():
    rows = _rows(317, None, "text") + _rows(83, "voice", "v")
    chunks = split_rows(rows, RATIOS, 42)
    assigned = [r["conversation_id"] for c in chunks.values() for r in c]
    assert sorted(assigned) == sorted(r["conversation_id"] for r in rows)
    assert len(assigned) == len(set(assigned))


def test_missing_modality_key_counts_as_text():
    """Every conversation predating the field is a written one."""
    assert _modality_of({}) == "text"
    assert _modality_of({"modality": None}) == "text"
    assert _modality_of({"modality": "voice"}) == "voice"


def test_split_is_deterministic():
    rows = _rows(400, None, "text") + _rows(100, "voice", "v")
    assert _assignment(split_rows(list(rows), RATIOS, 42)) == _assignment(
        split_rows(list(rows), RATIOS, 42)
    )


# --- --assert-unmoved -------------------------------------------------------


def _write_prior(directory: Path, chunks: dict[str, list[dict]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in chunks.items():
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    return directory


def test_assert_unmoved_passes_on_an_additive_batch(tmp_path):
    text = _rows(500, None, "text")
    prior = _write_prior(tmp_path / "prior", split_rows(list(text), RATIOS, 42))
    after = split_rows(list(text) + _rows(200, "voice", "v"), RATIOS, 42)
    assert assert_unmoved(after, prior) == []


def test_assert_unmoved_reports_a_moved_row(tmp_path):
    text = _rows(500, None, "text")
    chunks = split_rows(list(text), RATIOS, 42)
    prior = _write_prior(tmp_path / "prior", chunks)
    # Move one conversation from test into train by hand.
    moved_row = chunks["test"][0]
    tampered = {
        "train": chunks["train"] + [moved_row],
        "validation": chunks["validation"],
        "test": chunks["test"][1:],
    }
    moved = assert_unmoved(tampered, prior)
    assert len(moved) == 1
    assert "test -> train" in moved[0]


def test_assert_unmoved_ignores_new_conversations(tmp_path):
    prior = _write_prior(tmp_path / "prior", split_rows(_rows(100, None, "t"), RATIOS, 42))
    fresh = split_rows(_rows(50, "voice", "brand-new"), RATIOS, 42)
    assert assert_unmoved(fresh, prior) == []


def test_assert_unmoved_requires_all_three_split_files(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "train.jsonl").write_text("")
    with pytest.raises(SystemExit):
        assert_unmoved({"train": [], "validation": [], "test": []}, prior)
