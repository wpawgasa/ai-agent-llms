"""Plan once into a ledger, replay it deterministically, verify the result."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from repair_task_a_benchmark import apply_rows, plan_rows, verify_rows  # noqa: E402


def _render(sample: dict, original: str) -> str:
    return original


def _row() -> dict:
    state = {"from": "RATE", "to": "RATE"}
    return {
        "conversation_id": "L1_006",
        "conversation_initiator": "user",
        "modality": "text",
        "messages": [
            {"role": "system", "content": "You are a survey agent."},
            {"role": "user", "content": "I'd give it a 5. My ID is CUST-882."},
            {"role": "assistant", "content": "[STATE: RATE → RATE] Thank you, recording that now.", "annotations": {"state_transition": state}},
            {
                "role": "assistant",
                "content": '[STATE: RATE → RATE]\n<tool_call>{"name": "collect_csat", "arguments": {"interaction_id": "INT-5541", "score": 5}}</tool_call>',
                "annotations": {"state_transition": state, "tool_calls": [{"name": "collect_csat", "arguments": {"interaction_id": "INT-5541", "score": 5}}]},
            },
            {"role": "tool", "content": '{"status": "success"}'},
            {"role": "assistant", "content": "[STATE: RATE → TERMINAL] Thanks, CUST-882!", "annotations": {"state_transition": {"from": "RATE", "to": "TERMINAL"}}},
        ],
        "tool_schemas": [{"type": "function", "function": {"name": "collect_csat", "parameters": {"required": ["interaction_id", "score"]}}}],
        "workflow_graph": {"initial": "RATE", "terminal": ["TERMINAL"], "state_details": []},
        "ground_truth": {
            "state_sequence": [state, state, {"from": "RATE", "to": "TERMINAL"}],
            "tool_calls": [{"name": "collect_csat", "arguments": {"interaction_id": "INT-5541", "score": 5}}],
            "tool_chain_dependencies": [[{"name": "collect_csat", "arguments": {"interaction_id": "INT-5541", "score": 5}}]],
            "terminal_state": "TERMINAL",
        },
    }


def _plan(rows):
    return plan_rows(rows, render_prompt=_render, forbidden={"CUST-882", "INT-5541"}, seed=7)


def test_plan_records_every_repair_for_the_row() -> None:
    ledger = _plan([("text/l1.jsonl:6", _row())])
    entry = ledger["rows"]["text/l1.jsonl:6"]
    assert set(entry["id_remap"]) == {"CUST-882", "INT-5541"}
    assert entry["merge_runs"] == [[2, 3]]
    # INT-5541 is remapped first, so session context holds the NEW value.
    assert entry["session_context"] == {"interaction_id": entry["id_remap"]["INT-5541"]}


def test_plan_is_deterministic_for_a_seed() -> None:
    assert _plan([("k:1", _row())]) == _plan([("k:1", _row())])


def test_apply_replays_the_ledger_and_verifies_clean() -> None:
    rows = [("text/l1.jsonl:6", _row())]
    ledger = _plan(rows)
    (repaired,) = apply_rows(rows, ledger)
    text = json.dumps(repaired)
    assert "INT-5541" not in text and "CUST-882" not in text
    assert [m["role"] for m in repaired["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert repaired["session_context"] == ledger["rows"]["text/l1.jsonl:6"]["session_context"]
    assert verify_rows(rows, [repaired], ledger, render_prompt=_render) == []


def test_rows_without_ledger_entries_pass_through_unchanged() -> None:
    clean = _row()
    clean["messages"] = clean["messages"][:2]
    assert apply_rows([("k:1", clean)], {"rows": {}}) == [clean]


def test_verify_catches_a_missing_session_context() -> None:
    rows = [("text/l1.jsonl:6", _row())]
    ledger = _plan(rows)
    (repaired,) = apply_rows(rows, ledger)
    broken = copy.deepcopy(repaired)
    del broken["session_context"]
    problems = verify_rows(rows, [broken], ledger, render_prompt=_render)
    assert any("unsourced" in p for p in problems)


def test_verify_catches_a_surviving_old_identifier() -> None:
    rows = [("text/l1.jsonl:6", _row())]
    ledger = _plan(rows)
    (repaired,) = apply_rows(rows, ledger)
    broken = copy.deepcopy(repaired)
    broken["messages"][1]["content"] += " (was CUST-882)"
    assert any("CUST-882" in p for p in verify_rows(rows, [broken], ledger, render_prompt=_render))


def test_verify_catches_an_old_identifier_inside_a_longer_token() -> None:
    rows = [("text/l1.jsonl:6", _row())]
    ledger = _plan(rows)
    (repaired,) = apply_rows(rows, ledger)
    broken = copy.deepcopy(repaired)
    broken["messages"][1]["content"] += " ref CUST-882-OLD"
    assert any("CUST-882" in p for p in verify_rows(rows, [broken], ledger, render_prompt=_render))


def test_a_ledger_key_with_no_matching_row_is_an_error() -> None:
    with pytest.raises(ValueError, match="no row"):
        apply_rows([("k:1", _row())], {"rows": {"k:2": {}}})


def test_facts_ledger_is_applied_after_the_mechanical_one() -> None:
    rows = [("text/l1.jsonl:6", _row())]
    ledger = _plan(rows)
    (mechanical,) = apply_rows(rows, ledger)
    new_cust = ledger["rows"]["text/l1.jsonl:6"]["id_remap"]["CUST-882"]
    facts = {"edits": [{
        "key": "text/l1.jsonl:6", "fact": new_cust, "body_index": 3,
        "action": "rewrite", "old": f"Thanks, {new_cust}!", "new": "Thanks!",
    }]}
    (repaired,) = apply_rows(rows, ledger, facts)
    assert repaired["messages"][-1]["content"].endswith("Thanks!")
    assert mechanical["messages"][-1]["content"].endswith(f"Thanks, {new_cust}!")


def test_verify_reports_an_invented_fact_unless_it_was_accepted() -> None:
    row = _row()
    row["messages"][-1]["content"] = "[STATE: RATE → TERMINAL] Use code PREM-20-VIP."
    rows = [("text/l1.jsonl:6", row)]
    ledger = _plan(rows)
    (repaired,) = apply_rows(rows, ledger)
    assert any("PREM-20-VIP" in p for p in verify_rows(rows, [repaired], ledger, render_prompt=_render, facts_ledger={"edits": []}))
    accepted = {"edits": [{"key": "text/l1.jsonl:6", "fact": "PREM-20-VIP", "action": "accept_example"}]}
    (repaired,) = apply_rows(rows, ledger, accepted)
    assert verify_rows(rows, [repaired], ledger, render_prompt=_render, facts_ledger=accepted) == []


def test_a_facts_ledger_key_with_no_row_is_an_error() -> None:
    rows = [("k:1", _row())]
    with pytest.raises(ValueError, match="no row"):
        apply_rows(rows, {"rows": {}}, {"edits": [{"key": "k:9", "fact": "x", "action": "accept_example"}]})
