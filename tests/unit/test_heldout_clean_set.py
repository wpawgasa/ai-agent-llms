"""Tests for src/llm_workflow_agents/data/heldout_clean_set.py."""

from __future__ import annotations


def test_fingerprint_ignores_voice_markup():
    """A no-op today, since only an assistant turn carries markup. It keeps the
    fingerprint correct if a user turn ever carries a marker."""
    from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint

    plain = {"messages": [{"role": "user", "content": "hello"}]}
    marked = {"messages": [{"role": "user", "content": "<S>hello</S>"}]}
    assert user_turn_fingerprint(plain) == user_turn_fingerprint(marked)


def test_modality_filter_keeps_only_the_named_modality():
    from llm_workflow_agents.data.heldout_clean_set import filter_by_modality

    rows = [
        {"conversation_id": "a", "modality": "text"},
        {"conversation_id": "b", "modality": "voice"},
        {"conversation_id": "c"},  # no field: a pre-existing text row
    ]
    assert [r["conversation_id"] for r in filter_by_modality(rows, "voice")] == ["b"]
    assert [r["conversation_id"] for r in filter_by_modality(rows, "text")] == ["a", "c"]
    assert len(filter_by_modality(rows, "all")) == 3


def test_missing_modality_counts_as_text():
    """Every one of the 5,549 pre-existing rows predates the field."""
    from llm_workflow_agents.data.heldout_clean_set import filter_by_modality

    assert filter_by_modality([{"conversation_id": "x"}], "text") != []
