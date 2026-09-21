"""The triage script turns the traceability checks into a per-row defect list."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from triage_task_a_quality import load_rows, triage_rows  # noqa: E402


def _row(cid: str, messages: list[dict], tools: list[str] | None = None, modality: str = "text") -> dict:
    return {
        "conversation_id": cid,
        "modality": modality,
        "messages": [{"role": "system", "content": "sys"}] + messages,
        "tool_schemas": [
            {"type": "function", "function": {"name": t, "parameters": {"required": ["id"]}}} for t in (tools or [])
        ],
        "workflow_graph": {"state_details": [{"name": "S", "tools": tools or []}]},
    }


def _render(sample: dict, original: str) -> str:
    return original


def _clean() -> dict:
    return _row("L1_001", [
        {"role": "user", "content": "my id is CUST-1"},
        {"role": "assistant", "content": '[STATE: S → S]\n<tool_call>{"name": "lookup", "arguments": {"id": "CUST-1"}}</tool_call>'},
    ], tools=["lookup"])


def _dirty() -> dict:
    return _row("L1_002", [
        {"role": "user", "content": "rate it 5"},
        {"role": "assistant", "content": '[STATE: S → S]\n<tool_call>{"name": "lookup", "arguments": {"id": "INT-5541"}}</tool_call>'},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "[STATE: S → T] Use code PREM20."},
    ], tools=["lookup"])


class TestTriageRows:

    def test_clean_rows_are_left_out_of_the_row_list(self) -> None:
        report = triage_rows([("a.jsonl:1", _clean())], render_prompt=_render)
        assert report["rows"] == []
        assert report["summary"]["rows"] == 1
        assert report["summary"]["rows_with_findings"] == 0

    def test_a_dirty_row_lists_each_finding_under_its_check(self) -> None:
        report = triage_rows([("a.jsonl:2", _dirty())], render_prompt=_render)
        (row,) = report["rows"]
        assert row["key"] == "a.jsonl:2"
        assert row["conversation_id"] == "L1_002"
        assert [f["value"] for f in row["unsourced_arguments"]] == ["INT-5541"]
        assert [f["value"] for f in row["unsourced_facts"]] == ["PREM20"]
        summary = report["summary"]
        assert summary["unsourced_arguments"] == {"confident": 1, "needs_review": 0}
        assert summary["unsourced_facts"] == {"confident": 1, "needs_review": 0}

    def test_rows_with_the_same_conversation_id_stay_separate(self) -> None:
        # Text and voice strata both number conversations L1_001 onward (R25).
        text, voice = _dirty(), _dirty()
        voice["modality"] = "voice"
        report = triage_rows([("text.jsonl:1", text), ("voice.jsonl:1", voice)], render_prompt=_render)
        assert [r["key"] for r in report["rows"]] == ["text.jsonl:1", "voice.jsonl:1"]

    def test_identifier_reuse_is_reported_across_rows(self) -> None:
        report = triage_rows([("a.jsonl:1", _clean()), ("a.jsonl:2", _clean())], render_prompt=_render)
        assert report["summary"]["reused_identifiers"] == 1
        assert {r["key"] for r in report["rows"]} == {"a.jsonl:1", "a.jsonl:2"}
        assert report["rows"][0]["reused_identifiers"] == ["CUST-1"]

    def test_overlap_with_a_reference_corpus_is_reported(self) -> None:
        report = triage_rows(
            [("bench.jsonl:1", _clean())],
            render_prompt=_render,
            reference_identifiers={"CUST-1": 117},
        )
        (row,) = report["rows"]
        assert row["identifiers_in_reference"] == {"CUST-1": 117}
        assert report["summary"]["rows_sharing_identifiers_with_reference"] == 1


class TestLoadRows:

    def test_keys_are_path_and_line_number(self, tmp_path: Path) -> None:
        corpus = tmp_path / "stratum"
        corpus.mkdir()
        (corpus / "l1.jsonl").write_text(json.dumps(_clean()) + "\n\n" + json.dumps(_dirty()) + "\n")
        rows = load_rows([corpus])
        assert [key.rsplit(":", 1)[1] for key, _ in rows] == ["1", "3"]
        assert all(key.startswith(str(corpus)) for key, _ in rows)
