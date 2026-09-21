"""The replacement gate accepts only conversations that are clean on every check."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from generate_benchmark_replacements import gate, parse_extra, slots_from_triage  # noqa: E402


def _render(sample: dict, original: str) -> str:
    return original


SLOT = {"modality": "text", "initiator": "user", "must_visit": "CREATE_PROPOSAL", "level": "L3", "domain": "sales"}


def _clean() -> dict:
    turns = [
        ("user", "Please send the proposal. My customer ID is CUST-4471."),
        ("assistant", "[STATE: CREATE_PROPOSAL → CREATE_PROPOSAL]\n" '<tool_call>{"name": "create_quote", "arguments": {"customer_id": "CUST-4471"}}</tool_call>'),
        ("tool", '{"quote_id": "QT-9031"}'),
        ("assistant", "[STATE: CREATE_PROPOSAL → SEND_PROPOSAL] Your quote is QT-9031."),
        ("user", "Send it."),
        ("assistant", "[STATE: SEND_PROPOSAL → SEND_PROPOSAL]\n" '<tool_call>{"name": "send_proposal", "arguments": {"quote_id": "QT-9031"}}</tool_call>'),
        ("tool", '{"status": "sent"}'),
        ("assistant", "[STATE: SEND_PROPOSAL → CLOSE_DEAL] Sent."),
    ]
    return {
        "generation_source": "teacher",
        "modality": "text",
        "conversation_initiator": "user",
        "messages": [{"role": "system", "content": "You are a sales agent."}] + [{"role": r, "content": c} for r, c in turns],
        "tool_schemas": [
            {"type": "function", "function": {"name": "create_quote", "parameters": {"required": ["customer_id"]}}},
            {"type": "function", "function": {"name": "send_proposal", "parameters": {"required": ["quote_id"]}}},
        ],
        "workflow_graph": {
            "initial": "CREATE_PROPOSAL", "terminal": ["CLOSE_DEAL"],
            "state_details": [
                {"name": "CREATE_PROPOSAL", "tools": ["create_quote"]},
                {"name": "SEND_PROPOSAL", "tools": ["send_proposal"]},
                {"name": "CLOSE_DEAL", "tools": []},
            ],
        },
    }


def test_a_clean_conversation_is_accepted() -> None:
    assert gate(_clean(), SLOT, prompt="") == []


def test_placeholder_rows_are_rejected() -> None:
    sample = _clean()
    sample["generation_source"] = "placeholder"
    assert any("not teacher" in r for r in gate(sample, SLOT, prompt=""))


def test_a_multi_tool_state_is_rejected() -> None:
    sample = _clean()
    sample["workflow_graph"]["state_details"][0]["tools"] = ["create_quote", "send_proposal"]
    assert any("multi-tool" in r for r in gate(sample, SLOT, prompt=""))


def test_an_invented_value_is_rejected() -> None:
    sample = _clean()
    sample["messages"][-1]["content"] = "[STATE: SEND_PROPOSAL → CLOSE_DEAL] Use code PREM-20-VIP."
    assert any("invented" in r for r in gate(sample, SLOT, prompt=""))


def test_a_conversation_that_skips_the_replaced_state_is_rejected() -> None:
    assert any("never visits" in r for r in gate(_clean(), {**SLOT, "must_visit": "NEGOTIATE_TERMS"}, prompt=""))


def test_a_wrong_opener_is_rejected() -> None:
    assert any("initiator" in r for r in gate(_clean(), {**SLOT, "initiator": "agent"}, prompt=""))


def test_slots_come_from_multi_tool_uses_only() -> None:
    triage = {"rows": [
        {"key": "a:1", "multi_tool_states": [{"kind": "uses", "state": "CREATE_PROPOSAL"}]},
        {"key": "a:2", "multi_tool_states": [{"kind": "offers", "state": "CREATE_PROPOSAL"}]},
    ]}
    rows = {"a:1": {"conversation_id": "L3_005", "complexity_level": "L3", "domain": "sales", "language": "en"}, "a:2": {}}
    (slot,) = slots_from_triage(triage, rows)
    assert (slot["replaces"], slot["must_visit"], slot["modality"], slot["initiator"]) == ("a:1", "CREATE_PROPOSAL", "text", "user")


def test_parse_extra() -> None:
    assert parse_extra("L3:sales:text:en:user:CREATE_PROPOSAL")["must_visit"] == "CREATE_PROPOSAL"


def test_a_tool_result_without_a_call_is_rejected() -> None:
    sample = _clean()
    sample["messages"].insert(4, {"role": "assistant", "content": "[STATE: CREATE_PROPOSAL → CREATE_PROPOSAL] Let me check."})
    sample["messages"].insert(5, {"role": "tool", "content": '{"ok": true}'})
    assert any("no tool call" in r and "result" in r for r in gate(sample, SLOT, prompt=""))
