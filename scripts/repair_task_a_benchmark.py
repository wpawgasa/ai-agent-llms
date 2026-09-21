#!/usr/bin/env python3
"""Repair the Task A benchmark's untraceable values: plan into a ledger, replay it.

Two subcommands, the pattern R25 used to build the v2 text stratum:

``plan`` decides every repair once and writes it to a ledger. It is the only
step that is random (seeded) or reads the training corpus, which it needs so a
fresh identifier never repeats a value the model saw in training::

    python scripts/repair_task_a_benchmark.py plan \\
        --input-dir data/output/benchmark/task_a_v2 \\
        --input-dir data/output/benchmark/task_a_voice \\
        --reference data/output/sft/task_a_splits \\
        --ledger data/interim/task_a_benchmark_repair_ledger/ledger.json

``apply`` replays the ledger. It reads no training data, calls no model and uses
no randomness, so a frozen DVC stage can rebuild the repaired strata exactly::

    python scripts/repair_task_a_benchmark.py apply \\
        --stratum data/output/benchmark/task_a_v2 data/output/benchmark/task_a_v3 \\
        --stratum data/output/benchmark/task_a_voice data/output/benchmark/task_a_voice_v2 \\
        --ledger data/interim/task_a_benchmark_repair_ledger/ledger.json

The repairs, per row and in this order (see ``benchmark_repair``):
identifier remap, stay merges, session context. ``apply`` then verifies every
row: no new format violations, ground truth still aligned, no confident
unsourced tool argument left, no remapped value surviving. Any problem exits 1.

Rows are keyed ``<stratum dir name>/<file>:<line>``; ``conversation_id`` repeats
across the text and voice strata (CLAUDE.md R25). See CLAUDE.md R28.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from llm_workflow_agents.data._workflow_script import (
    find_continuity_violations,
    find_shape_violations,
)
from llm_workflow_agents.data.benchmark_repair import (
    apply_fact_edits,
    apply_identifier_remap,
    apply_stay_merges,
    plan_identifier_remap,
    plan_session_context,
    plan_stay_merges,
)
from llm_workflow_agents.data.source_traceability import (
    _IDENTIFIER_RE,
    find_unsourced_argument_values,
    find_unsourced_facts,
)
from llm_workflow_agents.data.state_convention import (
    find_tool_stay_violations,
    parse_assistant_turns,
)
from llm_workflow_agents.data.voice_convention import find_voice_violations

Row = tuple[str, dict[str, Any]]
RenderPrompt = Callable[[dict[str, Any], str], str]
LEDGER_VERSION = 1
DEFAULT_SEED = 20260921


# --------------------------------------------------------------------------- io


def load_stratum(directory: Path) -> list[Row]:
    rows: list[Row] = []
    for file in sorted(directory.glob("*.jsonl")):
        with file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    rows.append((f"{directory.name}/{file.name}:{line_no}", json.loads(line)))
    return rows


def write_stratum(rows: list[Row], samples: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[dict[str, Any]]] = {}
    for (key, _), sample in zip(rows, samples):
        by_file.setdefault(key.split("/", 1)[1].rsplit(":", 1)[0], []).append(sample)
    for name, items in by_file.items():
        with (directory / name).open("w", encoding="utf-8") as handle:
            for sample in items:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _default_render(sample: dict[str, Any], original: str) -> str:
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    return build_enriched_system_prompt(sample, original)


def _original_system(sample: dict[str, Any]) -> str:
    for message in sample.get("messages", []):
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""


def _body(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in sample.get("messages", []) if m.get("role") != "system"]


# --------------------------------------------------------------------------- checks


def violation_counts(sample: dict[str, Any]) -> dict[str, int]:
    """How many violations each existing format checker reports for ``sample``.

    Counts, not messages: violation text names turn numbers, which shift when
    turns are merged, so only a rising count means a repair broke something.
    """
    messages = sample.get("messages", [])
    graph = sample.get("workflow_graph") or {}
    return {
        "shape": len(find_shape_violations(messages, sample.get("conversation_initiator") or "user")),
        "continuity": len(find_continuity_violations(messages, graph.get("initial", ""), set(graph.get("terminal") or []))),
        "tool_stay": len(find_tool_stay_violations(messages)),
        "voice": len(find_voice_violations(_body(sample), sample.get("modality") or "text")),
    }


def _added_violations(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before[k] for k in before if after[k] > before[k]}


# --------------------------------------------------------------------------- plan


def plan_rows(
    rows: list[Row],
    render_prompt: RenderPrompt = _default_render,
    forbidden: set[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Decide every repair for ``rows``; return the ledger.

    ``forbidden`` holds identifiers a fresh value may not take -- every value in
    the training corpus. Every identifier already in ``rows`` is added to it, so
    a new value never collides with an existing one either.
    """
    forbidden = set(forbidden or ())
    for _, sample in rows:
        forbidden.update(_IDENTIFIER_RE.findall(json.dumps(sample, ensure_ascii=False)))
    rng = random.Random(seed)
    taken: set[str] = set()
    ledger_rows: dict[str, Any] = {}
    skipped_merges = 0

    for key, sample in rows:
        original = _original_system(sample)
        decisions = plan_identifier_remap(sample, render_prompt(sample, original), forbidden, rng, taken)
        mapping = {d.old: d.new for d in decisions if d.new}
        remapped = apply_identifier_remap(sample, mapping)

        # Keep only merges that add no format violation (a voice turn can run
        # past its chunk limit once two turns' chunks are combined).
        baseline = violation_counts(remapped)
        runs = [
            run for run in plan_stay_merges(remapped)
            if not _added_violations(baseline, violation_counts(apply_stay_merges(remapped, [run])))
        ]
        skipped_merges += len(plan_stay_merges(remapped)) - len(runs)
        merged = apply_stay_merges(remapped, runs)

        context = plan_session_context(merged, render_prompt(merged, original))

        entry: dict[str, Any] = {}
        if mapping:
            entry["id_remap"] = mapping
        skipped = {d.old: d.reason for d in decisions if not d.new}
        if skipped:
            entry["id_skipped"] = skipped
        if runs:
            entry["merge_runs"] = [list(run) for run in runs]
        if context:
            entry["session_context"] = context
        if entry:
            ledger_rows[key] = entry

    summary = {
        "rows": len(rows),
        "rows_changed": sum(1 for e in ledger_rows.values() if set(e) - {"id_skipped"}),
        "identifiers_remapped": sum(len(e.get("id_remap", {})) for e in ledger_rows.values()),
        "identifiers_skipped": _count_reasons(ledger_rows),
        "merge_runs": sum(len(e.get("merge_runs", [])) for e in ledger_rows.values()),
        "merge_runs_skipped_for_violations": skipped_merges,
        "session_context_values": sum(len(e.get("session_context", {})) for e in ledger_rows.values()),
    }
    return {"version": LEDGER_VERSION, "seed": seed, "summary": summary, "rows": ledger_rows}


def _count_reasons(ledger_rows: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in ledger_rows.values():
        for reason in entry.get("id_skipped", {}).values():
            counts[reason] = counts.get(reason, 0) + 1
    return counts


# --------------------------------------------------------------------------- apply


def repair_sample(sample: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Replay one ledger entry onto one sample."""
    out = apply_identifier_remap(sample, entry.get("id_remap", {}))
    out = apply_stay_merges(out, [tuple(run) for run in entry.get("merge_runs", [])])
    if entry.get("session_context"):
        out["session_context"] = {**(out.get("session_context") or {}), **entry["session_context"]}
    return out


def _fact_edits_by_key(facts_ledger: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edit in (facts_ledger or {}).get("edits", []):
        grouped.setdefault(edit["key"], []).append(edit)
    return grouped


def apply_rows(
    rows: list[Row],
    ledger: dict[str, Any],
    facts_ledger: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every row repaired per its ledger entries; rows without one are copied.

    The mechanical ledger is replayed first, then the authored facts ledger,
    whose values are post-remap.
    """
    keys = {key for key, _ in rows}
    entries = ledger.get("rows", {})
    fact_edits = _fact_edits_by_key(facts_ledger)
    stale = sorted((set(entries) | set(fact_edits)) - keys)
    if stale:
        raise ValueError(f"ledger names {len(stale)} keys with no row, e.g. {stale[:3]}")
    repaired: list[dict[str, Any]] = []
    for key, sample in rows:
        out = repair_sample(sample, entries[key]) if key in entries else copy.deepcopy(sample)
        if key in fact_edits:
            out = apply_fact_edits(out, fact_edits[key])
        repaired.append(out)
    return repaired


def verify_rows(
    rows: list[Row],
    repaired: list[dict[str, Any]],
    ledger: dict[str, Any],
    render_prompt: RenderPrompt = _default_render,
    facts_ledger: dict[str, Any] | None = None,
) -> list[str]:
    """Every problem with the repaired rows; an empty list means all passed.

    With a facts ledger, invented facts are checked too: none may remain
    except those the ledger accepts as format examples.
    """
    problems: list[str] = []
    entries = ledger.get("rows", {})
    accepted = {
        (e["key"], e["fact"])
        for e in (facts_ledger or {}).get("edits", [])
        if e["action"] == "accept_example"
    }
    for (key, original), sample in zip(rows, repaired):
        added = _added_violations(violation_counts(original), violation_counts(sample))
        if added:
            problems.append(f"{key}: repair added format violations {added}")

        annotated = [l for l in parse_assistant_turns(sample.get("messages", [])) if l is not None]
        sequence = sample.get("ground_truth", {}).get("state_sequence") or []
        if [(l.from_state, l.to_state) for l in annotated] != [(e.get("from"), e.get("to")) for e in sequence]:
            problems.append(f"{key}: ground_truth.state_sequence no longer matches the turns")

        prompt = render_prompt(sample, _original_system(sample))
        left = [
            f for f in find_unsourced_argument_values(
                _body(sample), sample.get("tool_schemas"), prompt, sample.get("session_context")
            )
            if f.confidence == "confident"
        ]
        if left:
            problems.append(f"{key}: {len(left)} confident unsourced argument(s) left, e.g. {left[0].describe()}")

        if facts_ledger is not None:
            invented = [
                f for f in find_unsourced_facts(_body(sample), prompt, sample.get("session_context"))
                if f.confidence == "confident" and (key, f.value) not in accepted
            ]
            if invented:
                problems.append(f"{key}: {len(invented)} invented fact(s) left, e.g. {invented[0].describe()}")

        # Plain substring, not whole-token: a remapped value surviving inside a
        # longer token is exactly the inconsistency to catch.
        text = json.dumps(
            [_body(sample), sample.get("ground_truth", {}), sample.get("session_context", {})],
            ensure_ascii=False,
        )
        for old in entries.get(key, {}).get("id_remap", {}):
            if old in text:
                problems.append(f"{key}: remapped identifier {old} still present")
    return problems


# --------------------------------------------------------------------------- cli


def _reference_identifiers(directories: Iterable[Path]) -> set[str]:
    found: set[str] = set()
    for directory in directories:
        for file in sorted(Path(directory).glob("*.jsonl")):
            found.update(_IDENTIFIER_RE.findall(file.read_text(encoding="utf-8")))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="decide every repair and write the ledger")
    plan.add_argument("--input-dir", action="append", required=True, type=Path)
    plan.add_argument("--reference", action="append", default=[], type=Path, help="training corpus; its identifiers are never reused")
    plan.add_argument("--ledger", required=True, type=Path)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)

    apply = sub.add_parser("apply", help="replay the ledger into repaired strata and verify them")
    apply.add_argument("--stratum", nargs=2, action="append", required=True, type=Path, metavar=("INPUT_DIR", "OUTPUT_DIR"))
    apply.add_argument("--ledger", required=True, type=Path)
    apply.add_argument("--facts-ledger", type=Path, help="authored edits for invented facts, replayed after --ledger")

    args = parser.parse_args()

    if args.command == "plan":
        rows = [row for directory in sorted(dict.fromkeys(args.input_dir)) for row in load_stratum(directory)]
        ledger = plan_rows(rows, forbidden=_reference_identifiers(args.reference), seed=args.seed)
        ledger["inputs"] = [str(d) for d in args.input_dir]
        ledger["reference"] = [str(d) for d in args.reference]
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(ledger, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(ledger["summary"], indent=1))
        return 0

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    all_rows: list[Row] = []
    outputs: list[tuple[list[Row], Path]] = []
    for input_dir, output_dir in args.stratum:
        rows = load_stratum(input_dir)
        all_rows.extend(rows)
        outputs.append((rows, output_dir))
    facts_ledger = json.loads(args.facts_ledger.read_text(encoding="utf-8")) if args.facts_ledger else None
    repaired_all = apply_rows(all_rows, ledger, facts_ledger)
    problems = verify_rows(all_rows, repaired_all, ledger, facts_ledger=facts_ledger)

    offset = 0
    for rows, output_dir in outputs:
        write_stratum(rows, repaired_all[offset:offset + len(rows)], output_dir)
        offset += len(rows)

    print(f"repaired {len(all_rows)} rows into {[str(o) for _, o in outputs]}")
    if problems:
        print(f"{len(problems)} verification problem(s):", file=sys.stderr)
        for problem in problems[:50]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("verification: all rows pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
