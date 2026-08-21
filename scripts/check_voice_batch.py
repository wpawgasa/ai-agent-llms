#!/usr/bin/env python3
"""Quality gate for a generated voice batch. Exits non-zero on a bad batch.

Spec risk 2 (docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md
section 7): "The teacher model fails the format. Placeholder rows then fill the
voice batch." That failure is INVISIBLE without this script. The placeholder
generator is format-perfect by construction, so a run in which the teacher
model failed on every single sample produces 2,400 rows, zero format
violations and fifteen cheerful success lines — indistinguishable from
success. Merging that batch would put 2,400 deterministic, structurally
uniform rows into the corpus, which is risk R15 at scale.

The data needed to tell the two apart is already recorded per leg in the
``.stats.json`` sidecars (``generation_source_counts``, ``repair_fallbacks``,
``teacher_model``). Nothing read it. This script reads it.

Four checks:

1. **Placeholder share.** Fails when the share of conversations that came from
   the placeholder generator exceeds ``--max-placeholder-share`` (default
   0.10). Skipped for a batch generated with no teacher model at all — every
   sidecar reporting ``teacher_model: null`` is a deliberate offline run, not
   a degraded one.
2. **Format.** Re-runs ``find_voice_violations`` over every row, in the
   modality the row claims. This duplicates the generator's own repair-loop
   check on purpose: the check that matters is the one run against the
   artifact on disk, not the one run against an in-memory object.
3. **Modality label.** Every row of a voice batch must be labelled
   ``modality: voice``.
4. **Barge-in realisation.** Warns (never fails) when the TEACHER was asked
   for interruptions and its own rows wrote far fewer than requested. A
   warning, not a failure, because the request is a rate and the checker
   cannot know how many the teacher should have managed. Counted from
   ``generation_source == "teacher"`` rows only, read directly off the
   conversations on disk — not from the sidecars' aggregate
   ``barge_in_realized``, which (since Task 3 gave the placeholder generator
   the ability to realise a barge-in too) also includes ``placeholder`` and
   ``placeholder_fallback`` rows and so can no longer tell a real teacher
   delivery from a degraded run the placeholder covered for.

Usage:
    python scripts/check_voice_batch.py --input-dir data/output/sft/task_a_voice
    python scripts/check_voice_batch.py --input-dir DIR --max-placeholder-share 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_INPUT = Path("data/output/sft/task_a_voice")
DEFAULT_MAX_PLACEHOLDER_SHARE = 0.10
#: Below this many realised barge-ins per requested one, say so out loud.
BARGE_IN_WARN_RATIO = 0.5


def _read_sidecars(input_dir: Path) -> list[dict]:
    """Load every ``*.stats.json`` sidecar in ``input_dir``."""
    return [
        json.loads(path.read_text())
        for path in sorted(input_dir.glob("*.stats.json"))
    ]


def summarise(sidecars: list[dict]) -> dict:
    """Aggregate the per-leg sidecars into one batch-level summary."""
    source_counts: dict[str, int] = {}
    totals = {
        "legs": len(sidecars),
        "conversations": 0,
        "repair_fallbacks": 0,
        "repair_retries": 0,
        "teacher_call_failures": 0,
        "barge_in_requested": 0,
        "barge_in_realized": 0,
    }
    teacher_models: set[str | None] = set()
    for stats in sidecars:
        teacher_models.add(stats.get("teacher_model"))
        body = stats.get("stats", {})
        totals["conversations"] += int(stats.get("num_samples", 0) or 0)
        for key in (
            "repair_fallbacks",
            "repair_retries",
            "teacher_call_failures",
            "barge_in_requested",
            "barge_in_realized",
        ):
            totals[key] += int(body.get(key, 0) or 0)
        for source, count in (body.get("generation_source_counts") or {}).items():
            source_counts[source] = source_counts.get(source, 0) + int(count)

    scored = sum(source_counts.values())
    placeholder = source_counts.get("placeholder", 0) + source_counts.get(
        "placeholder_fallback", 0
    )
    return {
        **totals,
        "scored": scored,
        "generation_source_counts": source_counts,
        "placeholder": placeholder,
        "placeholder_share": (placeholder / scored) if scored else 0.0,
        # A batch every leg of which named no teacher model was asked to be
        # offline. Nothing degraded; the placeholder gate does not apply.
        "offline_run": teacher_models == {None},
        "teacher_models": sorted(m for m in teacher_models if m),
    }


def count_teacher_realized_barge_ins(input_dir: Path) -> int:
    """Count realised barge-ins on rows attributed to the teacher, on disk.

    The sidecars' aggregate ``barge_in_realized`` (see ``summarise``) counts
    every row that carries the marker, regardless of ``generation_source``.
    Before Task 3 that was safe: only a teacher row could realise a barge-in,
    so "realised" and "realised by the teacher" were the same number. Task 3
    gave the placeholder generator that ability too, so a degraded run — the
    teacher failing and ``placeholder``/``placeholder_fallback`` rows filling
    in — can now realise barge-ins the sidecar happily counts, even though the
    teacher delivered none. The warning this feeds exists specifically to
    catch the teacher dropping interruptions, so it must read the rows on
    disk and count only ``generation_source == "teacher"``.
    """
    count = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("barge_in") and row.get("generation_source") == "teacher":
                count += 1
    return count


def check_rows(input_dir: Path, expect_modality: str = "voice") -> list[str]:
    """Re-check every conversation on disk. Returns a list of failure lines."""
    from llm_workflow_agents.data.voice_convention import find_voice_violations

    failures: list[str] = []
    n_rows = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        for lineno, line in enumerate(path.open(), 1):
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            modality = row.get("modality") or "text"
            where = f"{path.name}:{lineno} ({row.get('conversation_id')})"
            if modality != expect_modality:
                failures.append(f"{where}: modality is {modality!r}, expected {expect_modality!r}")
                continue
            violations = find_voice_violations(row.get("messages") or [], modality)
            if violations:
                failures.append(f"{where}: {violations[0]}")
    if not n_rows:
        failures.append(f"no conversations found in {input_dir}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory of generated voice *.jsonl + *.stats.json (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--max-placeholder-share",
        type=float,
        default=DEFAULT_MAX_PLACEHOLDER_SHARE,
        help=(
            "Fail when more than this share of a teacher-model batch came from "
            f"the placeholder generator (default: {DEFAULT_MAX_PLACEHOLDER_SHARE}). "
            "A batch generated with no teacher model at all is exempt."
        ),
    )
    parser.add_argument(
        "--modality",
        default="voice",
        help="Modality every row must claim (default: voice).",
    )
    parser.add_argument(
        "--skip-format-check",
        action="store_true",
        help="Only aggregate the sidecars; do not re-check every row.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"FAIL: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    sidecars = _read_sidecars(args.input_dir)
    if not sidecars:
        print(
            f"FAIL: no *.stats.json sidecars in {args.input_dir} — cannot tell a "
            "teacher-generated batch from a placeholder one, which is the whole "
            "point of this gate",
            file=sys.stderr,
        )
        return 2

    summary = summarise(sidecars)
    teacher_realized = count_teacher_realized_barge_ins(args.input_dir)
    print("=== Voice batch gate ===")
    print(f"Input dir          : {args.input_dir}")
    print(f"Legs               : {summary['legs']}")
    print(f"Conversations      : {summary['conversations']}")
    print(f"Teacher models     : {', '.join(summary['teacher_models']) or 'none (offline run)'}")
    print(f"Generation sources : {summary['generation_source_counts']}")
    print(
        f"Placeholder share  : {summary['placeholder_share']:.4f} "
        f"({summary['placeholder']}/{summary['scored']})  "
        f"threshold {args.max_placeholder_share}"
    )
    print(
        f"Repair fallbacks   : {summary['repair_fallbacks']}   "
        f"retries {summary['repair_retries']}   "
        f"teacher call failures {summary['teacher_call_failures']}"
    )
    print(
        f"Barge-in           : {summary['barge_in_realized']} realised total "
        f"(any source) / {teacher_realized} realised by the teacher / "
        f"{summary['barge_in_requested']} requested"
    )

    failed = False

    if summary["offline_run"]:
        print(
            "[skip] every leg names no teacher model, so this batch was asked "
            "to be placeholder-only. The placeholder-share gate does not apply."
        )
    elif summary["placeholder_share"] > args.max_placeholder_share:
        print(
            f"FAIL: {summary['placeholder']} of {summary['scored']} conversations "
            f"({summary['placeholder_share']:.1%}) came from the placeholder "
            f"generator, above the {args.max_placeholder_share:.1%} threshold. "
            f"The teacher model ({', '.join(summary['teacher_models'])}) is not "
            "producing usable voice conversations. Placeholder rows are "
            "format-perfect and deterministic, so this batch would look clean "
            "and teach the model a structurally uniform habit (risk R15). Do "
            "not merge it. Check the teacher API key, the model name, and the "
            "voice rules in the teacher prompt.",
            file=sys.stderr,
        )
        failed = True
    else:
        print("[ok] placeholder share is within the threshold")

    if (
        summary["barge_in_requested"]
        and teacher_realized < BARGE_IN_WARN_RATIO * summary["barge_in_requested"]
    ):
        non_teacher = summary["barge_in_realized"] - teacher_realized
        aside = (
            f" ({non_teacher} more were realised by non-teacher rows — "
            "placeholder or placeholder_fallback — and are deliberately not "
            "counted here, since this warning exists to catch the TEACHER "
            "dropping interruptions.)"
            if non_teacher > 0
            else ""
        )
        print(
            f"[warn] the teacher was asked for {summary['barge_in_requested']} "
            f"interruptions and its own rows (generation_source == \"teacher\") "
            f"realised {teacher_realized}.{aside} The barge-in rules are the "
            "hardest part of the voice format; a teacher silently dropping "
            "them is worth knowing about before the merge.",
            file=sys.stderr,
        )

    if not args.skip_format_check:
        failures = check_rows(args.input_dir, args.modality)
        if failures:
            print(
                f"FAIL: {len(failures)} conversation(s) do not satisfy the voice "
                "format on disk. First few:",
                file=sys.stderr,
            )
            for line in failures[:10]:
                print(f"  - {line}", file=sys.stderr)
            failed = True
        else:
            print("[ok] every row on disk passes find_voice_violations")

    if failed:
        print("\n=== Voice batch gate: FAILED ===", file=sys.stderr)
        return 1
    print("\n=== Voice batch gate: PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
