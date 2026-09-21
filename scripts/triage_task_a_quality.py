#!/usr/bin/env python3
"""Triage Task A conversations for values the model could never have produced.

Runs every check in :mod:`llm_workflow_agents.data.source_traceability` over one
or more corpus directories and writes a per-row defect list plus a summary. It
repairs nothing: the report is the work list the repair steps consume.

    python scripts/triage_task_a_quality.py \\
        --data data/output/benchmark/task_a_v2 \\
        --data data/output/benchmark/task_a_voice \\
        --reference data/output/sft/task_a_splits \\
        --out runs/audit/triage_benchmark_v2.json

Rows are keyed by ``<file>:<line>``, never by ``conversation_id``: the text and
voice strata both number conversations ``L1_001`` onward, so an id-keyed report
silently merges two different conversations (CLAUDE.md R25).

``--reference`` names a second corpus -- normally the SFT training corpus --
whose identifier values are counted, so each benchmark row reports which of its
identifiers the model also saw in training. A shared value rewards memorization
on the benchmark.

The system prompt each row is checked against is the one the model is served,
rebuilt with ``build_enriched_system_prompt``; the stored system message is the
authored prompt, not what the model saw.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from llm_workflow_agents.data.source_traceability import (
    collect_identifiers,
    find_identifier_reuse,
    find_mergeable_stay_pairs,
    find_multi_tool_states,
    find_unsourced_argument_values,
    find_unsourced_facts,
)

Row = tuple[str, dict[str, Any]]
RenderPrompt = Callable[[dict[str, Any], str], str]


def load_rows(paths: Iterable[Path]) -> list[Row]:
    """Every conversation under ``paths`` as ``(<file>:<line>, sample)``."""
    files: list[Path] = []
    for path in dict.fromkeys(Path(p) for p in paths):
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    rows: list[Row] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    rows.append((f"{file}:{line_no}", json.loads(line)))
    return rows


def _default_render(sample: dict[str, Any], original: str) -> str:
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    return build_enriched_system_prompt(sample, original)


def _original_system(sample: dict[str, Any]) -> str:
    for message in sample.get("messages", []):
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""


def _body(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages after the system message; the prompt is passed separately."""
    return [m for m in sample.get("messages", []) if m.get("role") != "system"]


def triage_rows(
    rows: list[Row],
    render_prompt: RenderPrompt = _default_render,
    reference_identifiers: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run every check over ``rows``; return ``{"summary": ..., "rows": [...]}``.

    Only rows with at least one finding are listed. ``reference_identifiers``
    maps an identifier value to the number of reference-corpus rows holding it.
    """
    reference_identifiers = reference_identifiers or {}
    per_row: list[dict[str, Any]] = []
    identifiers: dict[str, set[str]] = {}
    render_failures = 0

    for key, sample in rows:
        original = _original_system(sample)
        try:
            prompt = render_prompt(sample, original)
        except Exception:  # noqa: BLE001 - a row we cannot render is still checked
            prompt = original
            render_failures += 1
        body = _body(sample)
        session = sample.get("session_context")

        identifiers[key] = collect_identifiers(body, vocabulary_text=prompt)
        per_row.append({
            "key": key,
            "conversation_id": sample.get("conversation_id"),
            "modality": sample.get("modality", "text"),
            "complexity_level": sample.get("complexity_level"),
            "unsourced_arguments": [
                vars(f) for f in find_unsourced_argument_values(body, sample.get("tool_schemas"), prompt, session)
            ],
            "unsourced_facts": [vars(f) for f in find_unsourced_facts(body, prompt, session)],
            "multi_tool_states": [
                {**vars(f), "tools": list(f.tools), "msg_indices": list(f.msg_indices)}
                for f in find_multi_tool_states(sample.get("workflow_graph"), body)
            ],
            "mergeable_stay_pairs": [list(p) for p in find_mergeable_stay_pairs(body)],
        })

    reuse = find_identifier_reuse(identifiers)
    for row in per_row:
        mine = identifiers[row["key"]]
        row["reused_identifiers"] = sorted(v for v in mine if v in reuse)
        row["identifiers_in_reference"] = {
            v: reference_identifiers[v] for v in sorted(mine) if v in reference_identifiers
        }

    def has_findings(row: dict[str, Any]) -> bool:
        return any(row[k] for k in (
            "unsourced_arguments", "unsourced_facts", "multi_tool_states",
            "mergeable_stay_pairs", "reused_identifiers", "identifiers_in_reference",
        ))

    listed = [r for r in per_row if has_findings(r)]

    def confidence_split(field: str) -> dict[str, int]:
        counts = Counter(f["confidence"] for r in per_row for f in r[field])
        return {"confident": counts["confident"], "needs_review": counts["needs_review"]}

    occurrences = sum(len(v) for v in identifiers.values())
    reused_occurrences = sum(len(keys) for keys in reuse.values())
    summary = {
        "rows": len(rows),
        "rows_with_findings": len(listed),
        "render_failures": render_failures,
        "unsourced_arguments": confidence_split("unsourced_arguments"),
        "unsourced_facts": confidence_split("unsourced_facts"),
        "rows_with_confident_unsourced": sum(
            1 for r in per_row
            if any(f["confidence"] == "confident" for f in r["unsourced_arguments"] + r["unsourced_facts"])
        ),
        "multi_tool_states": dict(Counter(f["kind"] for r in per_row for f in r["multi_tool_states"])),
        "mergeable_stay_pairs": sum(len(r["mergeable_stay_pairs"]) for r in per_row),
        "reused_identifiers": len(reuse),
        "identifier_occurrences_reused_share": (reused_occurrences / occurrences) if occurrences else 0.0,
        "rows_sharing_identifiers_with_reference": sum(1 for r in per_row if r["identifiers_in_reference"]),
    }
    return {"summary": summary, "rows": listed}


def reference_identifier_counts(rows: list[Row], render_prompt: RenderPrompt = _default_render) -> dict[str, int]:
    """Identifier value -> number of rows that hold it, prompt vocabulary excluded."""
    counts: Counter[str] = Counter()
    for _, sample in rows:
        original = _original_system(sample)
        try:
            prompt = render_prompt(sample, original)
        except Exception:  # noqa: BLE001
            prompt = original
        counts.update(collect_identifiers(_body(sample), vocabulary_text=prompt))
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="append", required=True, type=Path, help="corpus dir or file; repeatable")
    parser.add_argument("--reference", action="append", default=[], type=Path, help="corpus to check identifier overlap against")
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON report")
    args = parser.parse_args()

    rows = load_rows(args.data)
    reference = reference_identifier_counts(load_rows(args.reference)) if args.reference else None
    report = triage_rows(rows, reference_identifiers=reference)
    report["summary"]["data"] = [str(p) for p in args.data]
    report["summary"]["reference"] = [str(p) for p in args.reference]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {len(report['rows'])} rows with findings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
