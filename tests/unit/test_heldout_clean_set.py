"""Tests for src/llm_workflow_agents/data/heldout_clean_set.py."""

from __future__ import annotations


def test_fingerprint_ignores_voice_markup():
    """A no-op today, since only an assistant turn carries markup. It keeps the
    fingerprint correct if a user turn ever carries a marker."""
    from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint

    plain = {"messages": [{"role": "user", "content": "hello"}]}
    marked = {"messages": [{"role": "user", "content": "<S>hello</S>"}]}
    assert user_turn_fingerprint(plain) == user_turn_fingerprint(marked)
