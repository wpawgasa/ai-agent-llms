#!/usr/bin/env python3
"""Propose a `source` for every tool argument, with the evidence, for review.

`eval/argument_provenance.py` reads a declared source; this writes a ledger
PROPOSING one per (tool, argument) pair from how the corpus actually uses it.
It applies nothing: a human reads the ledger, edits what is wrong, and only
then does anything annotate a schema.

    python scripts/propose_argument_sources.py \
        --data data/output/benchmark/task_a_v4 \
        --data data/output/benchmark/task_a_voice_v3 \
        --out data/interim/task_a_argument_sources/proposal.json

The rules, in order, and the evidence each rests on:

``free_text``  the name is a known prose field (description, notes, …) or the
               values are long — a paraphrase is correct, so exact match
               cannot score it whoever writes it.
``system``     the value is usually NOT present anywhere the model could read
               (the conversation, the rendered prompt, session context). An
               external system supplies it; the model could only invent it.
``derived``    the schema declares an enum, so the model picks among options.
``user``       everything else: the value is in the conversation and the model
               must carry it.

The `system` rule is the one that matters and the one to check hardest: it is
an accusation that a value is unknowable, and R31 records a whole hypothesis
dying because eight such accusations turned out to be deliberate test cases.
Each proposal therefore carries `sourced_share` and sample values.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path
from typing import Any

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
from llm_workflow_agents.eval.tool_call_failures import value_is_visible

FREE_TEXT_NAMES = frozenset({
    "description", "notes", "note", "summary", "resolution_summary", "message",
    "waiver_reason", "reason", "comments", "feedback", "details", "issue_description",
    "symptoms", "complaint", "request_details",
})
LONG_VALUE_CHARS = 40
#: Below this share of occurrences visible to the model, call it system-supplied.
SYSTEM_SOURCED_SHARE = 0.40


def propose_source(evidence: dict[str, Any]) -> str:
    """The proposed owner for one (tool, argument) pair. Pure; see module docs."""
    if evidence["argument"] in FREE_TEXT_NAMES or evidence["median_length"] > LONG_VALUE_CHARS:
        return "free_text"
    if evidence["sourced_share"] < SYSTEM_SOURCED_SHARE:
        return "system"
    if evidence["has_enum"]:
        return "derived"
    return "user"


def _schema_properties(sample: dict[str, Any], tool: str) -> dict[str, Any]:
    for schema in sample.get("tool_schemas") or []:
        function = schema.get("function", schema)
        if function.get("name") == tool:
            return (function.get("parameters") or {}).get("properties") or {}
    return {}


def gather(samples: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(
        lambda: {"occurrences": 0, "sourced": 0, "lengths": [], "has_enum": False, "samples": []}
    )
    for sample in samples:
        original = next(
            (m.get("content") or "" for m in sample.get("messages", []) if m.get("role") == "system"), ""
        )
        try:
            readable = build_enriched_system_prompt(sample, original)
        except Exception:
            readable = original
        for message in sample.get("messages", []):
            if message.get("role") != "system":
                readable += "\n" + (message.get("content") or "")
            for call in (message.get("annotations") or {}).get("tool_calls") or []:
                tool = call.get("name") or ""
                properties = _schema_properties(sample, tool)
                arguments = call.get("arguments")
                if not isinstance(arguments, dict):
                    continue
                for argument, value in arguments.items():
                    entry = stats[(tool, argument)]
                    entry["occurrences"] += 1
                    entry["sourced"] += bool(value_is_visible(value, readable))
                    entry["lengths"].append(len(str(value)))
                    spec = properties.get(argument) or {}
                    if isinstance(spec, dict) and spec.get("enum"):
                        entry["has_enum"] = True
                    if len(entry["samples"]) < 4:
                        entry["samples"].append(value)
    return stats


def build_proposal(stats: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for (tool, argument), entry in sorted(stats.items()):
        evidence = {
            "tool": tool,
            "argument": argument,
            "occurrences": entry["occurrences"],
            "sourced_share": round(entry["sourced"] / max(entry["occurrences"], 1), 3),
            "median_length": int(statistics.median(entry["lengths"])) if entry["lengths"] else 0,
            "has_enum": entry["has_enum"],
            "samples": entry["samples"],
        }
        rows.append({**evidence, "proposed_source": propose_source(evidence)})
    counts = collections.Counter(row["proposed_source"] for row in rows)
    return {"version": 1, "summary": dict(counts.most_common()), "pairs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    samples: list[dict[str, Any]] = []
    for directory in sorted(dict.fromkeys(args.data)):
        for path in sorted(glob.glob(str(Path(directory) / "*.jsonl"))):
            with open(path) as handle:
                samples.extend(json.loads(line) for line in handle if line.strip())

    proposal = build_proposal(gather(samples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(proposal["pairs"])
    print(f"{total} (tool, argument) pairs over {len(samples)} conversations")
    for source, count in proposal["summary"].items():
        print(f"  {source:10} {count:4} {count / total:6.1%}")
    print(f"\nwrote {args.out} — REVIEW IT before any schema is annotated; the "
          f"`system` rows are an accusation that a value is unknowable (R31).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
