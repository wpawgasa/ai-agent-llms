"""Tests for scripts/split_task_a_sft.py."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to path so we can import directly without packaging
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from split_task_a_sft import _load_rows  # noqa: E402


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
