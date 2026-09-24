"""Stage 1 of the SFT corpus repair: the mechanical half, and only that.

The benchmark repair (R28) ran two stages: a mechanical one (identifier
remap, stay merges, unknowable values into session context) and an authored
one (31 invented facts reviewed by hand). The corpus carries 978 confident
invented facts across 903 conversations — 30x the benchmark's — so the
authored stage is deliberately deferred until the mechanical half is measured.

Three things differ from the benchmark wrapper and are tested here:

1. **The forbidden set flips.** The benchmark repair kept fresh identifiers
   away from *training* values. Repairing the corpus must keep them away from
   the *benchmark's* v4 values, or the memorization channel v4 just closed
   reopens from the other side.
2. **Orphan-result conversations are dropped.** A tool result answering a turn
   that only announces the call trains announce-but-don't-call; there is no
   mechanical repair for one, and 41 of 9,932 rows is a cheap loss.
3. **Residual invented facts do not fail the run** — they are stage 2's job —
   but they are counted in the ledger summary, never silently passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from repair_task_a_corpus import (  # noqa: E402
    orphan_row_keys,
    plan_corpus,
    residual_finding_counts,
)

CALL = '<tool_call>{"name": "check_balance", "arguments": {"account_id": "ACC-1"}}</tool_call>'


def _row(account: str = "ACC-1", orphan: bool = False) -> dict:
    messages = [
        {"role": "system", "content": "You are a banking agent."},
        {"role": "user", "content": f"balance for {account}"},
        {
            "role": "assistant",
            "content": f"[STATE: GREET → LOOKUP] Checking.\n{CALL.replace('ACC-1', account)}",
            "annotations": {
                "state_transition": {"from": "GREET", "to": "LOOKUP"},
                "tool_calls": [{"name": "check_balance", "arguments": {"account_id": account}}],
            },
        },
        {"role": "tool", "content": '{"balance": 10}'},
        {
            "role": "assistant",
            "content": "[STATE: LOOKUP → DONE] It is 10.",
            "annotations": {"state_transition": {"from": "LOOKUP", "to": "DONE"}, "tool_calls": []},
        },
    ]
    if orphan:
        # A result answering a turn that only announces the call.
        messages.insert(3, {"role": "assistant", "content": "[STATE: LOOKUP → LOOKUP] One moment."})
        messages.insert(4, {"role": "tool", "content": '{"note": "orphan"}'})
    return {
        "conversation_id": "C1",
        "messages": messages,
        "tool_schemas": [{"type": "function", "function": {"name": "check_balance", "parameters": {}}}],
        "workflow_graph": {"initial": "GREET", "terminal": ["DONE"], "state_details": []},
        "ground_truth": {
            "state_sequence": [{"from": "GREET", "to": "LOOKUP"}, {"from": "LOOKUP", "to": "DONE"}],
            "terminal_state": "DONE",
        },
    }


def test_a_fresh_identifier_avoids_the_benchmark_values():
    rows = [("train.jsonl:1", _row("ACC-1"))]
    # Pretend the v4 benchmark already uses these; the repair may take none.
    forbidden = {f"ACC-{n}" for n in range(1, 400)} - {"ACC-377"}
    ledger = plan_corpus(rows, forbidden=forbidden, seed=7)
    remap = ledger["rows"]["train.jsonl:1"].get("id_remap", {})
    assert remap.get("ACC-1") in (None, "ACC-377"), remap


def test_orphan_rows_are_found_and_recorded_for_the_drop():
    rows = [("train.jsonl:1", _row()), ("train.jsonl:2", _row(orphan=True))]
    assert orphan_row_keys(rows) == ["train.jsonl:2"]
    ledger = plan_corpus(rows, forbidden=set(), seed=7)
    assert ledger["dropped"] == {"train.jsonl:2": "orphan_tool_result"}
    # A dropped row carries no repair entry — apply removes it instead.
    assert "train.jsonl:2" not in ledger["rows"]
    assert ledger["summary"]["rows_dropped"] == 1


def test_residual_findings_are_counted_not_hidden():
    sample = _row()
    sample["messages"][4]["content"] = "[STATE: LOOKUP → DONE] Your discount code is SAVE-4471."
    counts = residual_finding_counts([("train.jsonl:1", sample)])
    assert counts["invented_facts"] >= 1
    assert set(counts) == {"invented_facts", "unsourced_arguments", "rows_with_findings"}


def test_plan_is_seed_deterministic():
    rows = [("train.jsonl:1", _row())]
    assert plan_corpus(rows, forbidden=set(), seed=11) == plan_corpus(rows, forbidden=set(), seed=11)


def test_clean_corpus_rows_are_left_alone():
    ledger = plan_corpus([("train.jsonl:1", _row())], forbidden=set(), seed=7)
    entry = ledger["rows"].get("train.jsonl:1", {})
    assert not entry.get("session_context"), "nothing unknowable here to move"
    assert ledger["dropped"] == {}
