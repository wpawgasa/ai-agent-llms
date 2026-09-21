#!/usr/bin/env python3
"""Generate clean replacements for the benchmark conversations with multi-tool states.

The 19 benchmark conversations that call two different tools in one state
cannot be repaired in place (CLAUDE.md R28). This replaces each with a newly
generated conversation from the same stratum -- level, domain, modality,
language, who opens the call -- written by the same teacher model, but built
with ``single_tool_states=True`` so no state offers more than one tool.

Every candidate goes through the same repair steps as the v3 benchmark (fresh
identifiers absent from training, stay merges, session context) and is kept
only if it passes the gate in :func:`gate`: zero findings on every
traceability check, zero format violations, at least one tool call, and a walk
through the state the removed conversation was about, so the replacement
exercises the same part of the workflow in its new one-tool-per-state form.

    set -a; . ./.env; set +a
    python scripts/generate_benchmark_replacements.py \\
        --triage runs/audit/triage_benchmark_v3.json \\
        --benchmark data/output/benchmark/task_a_v3 \\
        --benchmark data/output/benchmark/task_a_voice_v2 \\
        --reference data/output/sft/task_a_splits \\
        --extra L3:sales:text:en:user:CREATE_PROPOSAL \\
        --out-dir data/interim/task_a_benchmark_v4_replacements

Writes ``replacements_text.jsonl``, ``replacements_voice.jsonl`` and a
``manifest.json`` recording, per slot, the conversation it replaces, the
teacher, every seed tried and why each rejected candidate failed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_task_a_benchmark import (  # noqa: E402
    _added_violations,
    _default_render,
    _original_system,
    _reference_identifiers,
    violation_counts,
)
from triage_task_a_quality import load_rows  # noqa: E402

from llm_workflow_agents.data.benchmark_repair import (  # noqa: E402
    apply_identifier_remap,
    apply_stay_merges,
    plan_identifier_remap,
    plan_session_context,
    plan_stay_merges,
)
from llm_workflow_agents.data.source_traceability import (  # noqa: E402
    _IDENTIFIER_RE,
    find_mergeable_stay_pairs,
    find_multi_tool_states,
    find_orphan_tool_results,
    find_unsourced_argument_values,
    find_unsourced_facts,
)
from llm_workflow_agents.data.state_convention import parse_assistant_turns  # noqa: E402

#: The teacher that wrote each stratum being replaced.
TEACHERS = {"text": "gemini-3-flash-preview", "voice": "gemini-3.7-flash"}


def _body(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in sample.get("messages", []) if m.get("role") != "system"]


# --------------------------------------------------------------------------- slots


def slots_from_triage(triage: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One slot per benchmark row whose conversation uses two tools in one state."""
    slots: list[dict[str, Any]] = []
    for row in triage["rows"]:
        uses = [m for m in row["multi_tool_states"] if m["kind"] == "uses"]
        if not uses:
            continue
        sample = rows[row["key"]]
        slots.append({
            "replaces": row["key"],
            "replaces_id": sample.get("conversation_id"),
            "level": sample["complexity_level"],
            "domain": sample["domain"],
            "modality": sample.get("modality") or "text",
            "language": sample.get("language") or "en",
            "initiator": sample.get("conversation_initiator") or "user",
            "must_visit": uses[0]["state"],
        })
    return slots


def parse_extra(spec: str) -> dict[str, Any]:
    """``LEVEL:domain:modality:language:initiator:STATE`` -> an extra slot."""
    level, domain, modality, language, initiator, state = spec.split(":")
    return {
        "replaces": None, "replaces_id": None, "level": level, "domain": domain,
        "modality": modality, "language": language, "initiator": initiator, "must_visit": state,
    }


# --------------------------------------------------------------------------- gate


def gate(sample: dict[str, Any], slot: dict[str, Any], prompt: str) -> list[str]:
    """Every reason ``sample`` may not replace ``slot``; empty means accept."""
    reasons: list[str] = []
    if sample.get("generation_source") != "teacher":
        reasons.append(f"generation_source={sample.get('generation_source')!r}, not teacher")
    if (sample.get("modality") or "text") != slot["modality"]:
        reasons.append(f"modality {sample.get('modality')!r} != {slot['modality']!r}")
    if (sample.get("conversation_initiator") or "user") != slot["initiator"]:
        reasons.append(f"initiator {sample.get('conversation_initiator')!r} != {slot['initiator']!r}")

    body = _body(sample)
    context = sample.get("session_context")
    arguments = [f for f in find_unsourced_argument_values(body, sample.get("tool_schemas"), prompt, context) if f.confidence == "confident"]
    if arguments:
        reasons.append(f"{len(arguments)} unsourced tool argument(s), e.g. {arguments[0].describe()}")
    facts = [f for f in find_unsourced_facts(body, prompt, context) if f.confidence == "confident"]
    if facts:
        reasons.append(f"{len(facts)} invented fact(s), e.g. {facts[0].describe()}")
    multi = find_multi_tool_states(sample.get("workflow_graph"), body)
    if multi:
        reasons.append(f"multi-tool state {multi[0].state} ({multi[0].kind})")
    if find_mergeable_stay_pairs(body):
        reasons.append("mergeable stay+stay pair")
    orphans = find_orphan_tool_results(body)
    if orphans:
        reasons.append(f"{len(orphans)} tool result(s) with no tool call")
    violations = {k: v for k, v in violation_counts(sample).items() if v}
    if violations:
        reasons.append(f"format violations {violations}")

    labels = [l for l in parse_assistant_turns(sample.get("messages", [])) if l is not None]
    if not any(l.tool_names for l in labels):
        reasons.append("no tool call")
    visited = {l.from_state for l in labels} | {l.to_state for l in labels}
    if slot["must_visit"] not in visited:
        reasons.append(f"never visits {slot['must_visit']}")
    return reasons


def repair_candidate(
    sample: dict[str, Any], forbidden: set[str], rng: random.Random, taken: set[str], render=_default_render
) -> dict[str, Any]:
    """The v3 repair steps, in the v3 order: remap, merges, session context."""
    original = _original_system(sample)
    decisions = plan_identifier_remap(sample, render(sample, original), forbidden, rng, taken)
    out = apply_identifier_remap(sample, {d.old: d.new for d in decisions if d.new})
    baseline = violation_counts(out)
    runs = [
        run for run in plan_stay_merges(out)
        if not _added_violations(baseline, violation_counts(apply_stay_merges(out, [run])))
    ]
    out = apply_stay_merges(out, runs)
    context = plan_session_context(out, render(out, original))
    if context:
        out["session_context"] = {**(out.get("session_context") or {}), **context}
    return out


# --------------------------------------------------------------------------- generation


def _generate(slot: dict[str, Any], seed: int, n: int, workdir: Path) -> list[dict[str, Any]]:
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    out = workdir / f"seed{seed}"
    generate_workflow_dataset(
        slot["level"],
        num_samples=n,
        teacher_model=TEACHERS[slot["modality"]],
        output_dir=out,
        seed=seed,
        domain=slot["domain"],
        language=slot["language"],
        modality_preset="voice_only" if slot["modality"] == "voice" else "default",
        initiation_preset="outbound_heavy" if slot["initiator"] == "agent" else "default",
        single_tool_states=True,
        max_workers=n,
    )
    return [json.loads(line) for f in sorted(out.glob("*.jsonl")) for line in f.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument("--benchmark", action="append", required=True, type=Path)
    parser.add_argument("--reference", action="append", default=[], type=Path)
    parser.add_argument("--extra", action="append", default=[], help="LEVEL:domain:modality:language:initiator:STATE")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--candidates-per-attempt", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--slots", help="comma-separated slot indices to run, for a smoke test")
    args = parser.parse_args()

    rows = dict(load_rows(args.benchmark))
    slots = slots_from_triage(json.loads(args.triage.read_text()), rows) + [parse_extra(e) for e in args.extra]
    if args.slots:
        keep = {int(i) for i in args.slots.split(",")}
        slots = [s for i, s in enumerate(slots) if i in keep]
    forbidden = _reference_identifiers(args.reference)
    for sample in rows.values():
        forbidden.update(_IDENTIFIER_RE.findall(json.dumps(sample, ensure_ascii=False)))
    rng = random.Random(args.seed)
    taken: set[str] = set()

    accepted: dict[str, list[dict[str, Any]]] = {"text": [], "voice": []}
    manifest: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, slot in enumerate(slots):
            record = {**slot, "teacher": TEACHERS[slot["modality"]], "attempts": []}
            for attempt in range(args.max_attempts):
                seed = args.seed + index * 100 + attempt
                try:
                    candidates = _generate(slot, seed, args.candidates_per_attempt, Path(tmp))
                except Exception as exc:  # noqa: BLE001 - one failed call must not end the run
                    record["attempts"].append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                rejected = []
                for candidate in candidates:
                    repaired = repair_candidate(candidate, forbidden, rng, taken)
                    prompt = _default_render(repaired, _original_system(repaired))
                    reasons = gate(repaired, slot, prompt)
                    if reasons:
                        rejected.append({"id": candidate.get("conversation_id"), "reasons": reasons})
                        continue
                    level_count = sum(1 for s in accepted[slot["modality"]] if s["complexity_level"] == slot["level"])
                    repaired["conversation_id"] = f"{slot['level']}_V4R{level_count + 1:02d}"
                    repaired["replaces"] = slot["replaces_id"]
                    accepted[slot["modality"]].append(repaired)
                    record["accepted"] = {"seed": seed, "conversation_id": repaired["conversation_id"]}
                    break
                record["attempts"].append({"seed": seed, "candidates": len(candidates), "rejected": rejected})
                if "accepted" in record:
                    break
            status = record.get("accepted", {}).get("conversation_id", "NOT FILLED")
            print(f"[{index + 1}/{len(slots)}] {slot['level']} {slot['domain']} {slot['modality']} {slot['language']} "
                  f"{slot['initiator']} -> {status} after {len(record['attempts'])} attempt(s)", flush=True)
            manifest.append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for modality, samples in accepted.items():
        with (args.out_dir / f"replacements_{modality}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    unfilled = [m for m in manifest if "accepted" not in m]
    print(f"filled {len(manifest) - len(unfilled)} of {len(manifest)} slots; manifest at {args.out_dir / 'manifest.json'}")
    return 1 if unfilled else 0


if __name__ == "__main__":
    raise SystemExit(main())
