"""Session context: facts the agent holds before the conversation begins.

Some tool arguments (an interaction id, an account on file) come from an
external system in production, never from the customer. A sample carries them
as ``session_context`` and the served prompt states them, so the model can
produce a gold call it previously had no way to produce (CLAUDE.md R28).

A sample without the field must render exactly as before; the committed
fixture test in test_system_prompt_voice.py guards that on real rows.
"""

from __future__ import annotations

from llm_workflow_agents.data.system_prompt import (
    SESSION_CONTEXT_HEADER,
    build_enriched_system_prompt,
)


def _sample(**extra: object) -> dict:
    sample = {
        "complexity_level": "L1",
        "workflow_graph": {
            "initial": "GREETING",
            "terminal": ["TERMINAL"],
            "state_details": [
                {"name": "GREETING", "tools": [], "instruction": "Greet the customer."},
                {"name": "RATE", "tools": ["collect_csat"], "instruction": "Collect a rating."},
                {"name": "TERMINAL", "tools": [], "instruction": "End."},
            ],
            "transitions": [
                {"from": "GREETING", "to": "RATE"},
                {"from": "RATE", "to": "TERMINAL"},
            ],
        },
        "tool_schemas": [
            {
                "type": "function",
                "function": {
                    "name": "collect_csat",
                    "parameters": {
                        "type": "object",
                        "properties": {"interaction_id": {"type": "string"}, "score": {"type": "integer"}},
                        "required": ["interaction_id", "score"],
                    },
                },
            }
        ],
        "messages": [{"role": "system", "content": "You are a survey agent."}],
    }
    sample.update(extra)
    return sample


def _render(sample: dict) -> str:
    return build_enriched_system_prompt(sample, sample["messages"][0]["content"])


def test_absent_and_empty_session_context_render_identically() -> None:
    assert _render(_sample()) == _render(_sample(session_context={}))
    assert SESSION_CONTEXT_HEADER not in _render(_sample())


def test_each_value_is_stated_verbatim() -> None:
    prompt = _render(_sample(session_context={"interaction_id": "INT-7302", "customer_tier": "premium"}))
    assert prompt.count(SESSION_CONTEXT_HEADER) == 1
    block = prompt[prompt.index(SESSION_CONTEXT_HEADER):]
    assert "  interaction_id: INT-7302" in block
    assert "  customer_tier: premium" in block


def test_block_sits_after_the_tool_schemas_and_before_the_rules() -> None:
    prompt = _render(_sample(session_context={"interaction_id": "INT-7302"}))
    assert prompt.index("Tool schemas") < prompt.index(SESSION_CONTEXT_HEADER) < prompt.index("Rules:")


def test_force_rebuild_regenerates_the_block_from_the_sample() -> None:
    # The block lives after the "Workflow script" marker, so a rebuild must not
    # keep a stale copy from an already-enriched system message.
    stale = _render(_sample(session_context={"interaction_id": "INT-0001"}))
    fresh = build_enriched_system_prompt(
        _sample(session_context={"interaction_id": "INT-7302"}), stale, force_rebuild=True
    )
    assert "INT-7302" in fresh
    assert "INT-0001" not in fresh
