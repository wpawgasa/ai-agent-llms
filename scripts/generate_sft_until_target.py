#!/usr/bin/env python3
"""Target-driven SFT data generation loop for Task A workflows.

Unlike ``generate_sft_data.sh`` (fixed curriculum: generate once, accept whatever
comes back), this script *keeps generating, verifies each batch, removes
unqualified samples, and loops until the qualified count reaches the per-level
target* — so the delivered corpus has a guaranteed number of clean samples.

Per level (L1-L5) x language leg (en/th/code_switch) it repeatedly:
  1. generates a batch via ``generate_workflow_dataset`` (seed varied each
     iteration -- generation is deterministic per seed, so a fixed seed would
     regenerate identical samples and the loop would never make progress);
  2. qualifies each sample against the bar **hard defects + cleaning fixes**:
       - ``clean_task_a_sft.clean_record`` drops truncated / role-confused rows
         (and sets the ``terminal_reached`` flag);
       - ``quality_profiler.profile_task_a`` + ``defective_conversation_ids``
         drop rows with hard structural defects;
  3. accumulates survivors, renaming duplicate ``conversation_id``s
     (``L1_001 -> L1_001_2``) the way ``concat_task_a`` does, since each batch
     restarts IDs at ``*_001``;
  4. stops when the qualified count hits target (then trims any overshoot), or
     when ``--max-iterations`` is exhausted (a shortfall WARNING is logged).

The deterministic loop runs unattended. The *hybrid* final step -- a semantic
rationality verdict from the ``dataset-verifier`` LLM agent -- is a Claude-driven
post-step: this script writes a manifest and prints the exact follow-up command.

Required env var: matches the chosen --teacher-model provider prefix
  gemini-*  -> GEMINI_API_KEY    gpt-*  -> OPENAI_API_KEY    claude-*  -> ANTHROPIC_API_KEY
Pass --teacher-model placeholder (or "") for offline placeholder generation (no API).

Usage:
    python scripts/generate_sft_until_target.py [OPTIONS]

Examples:
    # Full per-level curriculum (gemini teacher), all three language legs:
    GEMINI_API_KEY=... python scripts/generate_sft_until_target.py

    # Offline smoke loop (no API), single level/leg, 8 qualified samples:
    python scripts/generate_sft_until_target.py \\
        --languages en --levels L1 --samples-per-leg 8 \\
        --teacher-model placeholder --max-iterations 5

    # With an advisory per-batch verdict from the dataset-verifier agent
    # (logged only -- the deterministic filter still does the removal; slow + costly):
    GEMINI_API_KEY=... python scripts/generate_sft_until_target.py --verify-batches
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace as _dc_replace
from datetime import datetime
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling clean_task_a_sft
from clean_task_a_sft import clean_record  # noqa: E402

from llm_workflow_agents.data.data_validator import detect_thai_corruption  # noqa: E402
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset  # noqa: E402
from llm_workflow_agents.data.quality_profiler import (  # noqa: E402
    ProfileReport,
    defective_conversation_ids,
    profile_task_a,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Substring identifying a tool-stay defect among ``ProfileReport.defects``.
# The profiler records these as f"{cid}: {v}" where v comes from
# ``state_convention.find_tool_stay_violations``; this is the invariant slice of
# that message (the surrounding text interpolates turn index and state names).
# ``tests/unit/test_generate_sft_until_target_cli.py`` asserts the two still
# match, so a reword upstream fails loudly instead of silently making
# --no-require-tool-stay a no-op and stay_dropped a permanent zero.
_STAY_DEFECT_MARKER = "issues a <tool_call> but annotates an advancing transition"


def _select_defects(rep: ProfileReport, keep) -> set[str]:
    """Conversation ids whose defects survive the *keep* predicate.

    Routes through :func:`defective_conversation_ids` on a filtered copy rather
    than re-implementing the ``"<cid>: ..."`` / ``"file: ..."`` prefix parsing —
    that contract is owned by the profiler and documented there as the thing
    callers must not duplicate.
    """
    return defective_conversation_ids(
        _dc_replace(rep, defects=[d for d in rep.defects if keep(d)])
    )


def _disqualified_ids(rep: ProfileReport, require_tool_stay: bool) -> set[str]:
    """Conversation ids to DROP from a batch.

    With *require_tool_stay* False, a conversation whose ONLY hard defect is an
    advancing tool-call turn is kept — the v1-comparable regeneration mode. Any
    other hard defect still drops it, and a conversation carrying both a stay
    violation and some other defect still drops on the other defect.

    (The brief called this ``_qualified_ids``; it returns the disqualified set,
    so it is named for what it returns.)
    """
    if require_tool_stay:
        return defective_conversation_ids(rep)
    return _select_defects(rep, lambda d: _STAY_DEFECT_MARKER not in d)


def _resolve_retry_exhaustion(value: str | None) -> str | None:
    """Translate the CLI's ``"auto"`` sentinel into the library's ``None``.

    ``generate_workflow_dataset`` spells "keep the level's own policy" as
    ``None``; argparse ``choices`` can't express that, so the CLI spells it
    ``"auto"``. Passing the literal string through would be silently wrong —
    it is not a valid ``retry_exhaustion`` value and would raise.
    """
    return None if value in (None, "auto") else value


def _stay_violating_ids(rep: ProfileReport) -> set[str]:
    """Conversation ids with at least one advancing tool-call turn."""
    return _select_defects(rep, lambda d: _STAY_DEFECT_MARKER in d)

# Per-leg curriculum target (1/3 of the per-level total; mirrors generate_sft_data.sh).
CURRICULUM: dict[str, int] = {"L1": 1000, "L2": 1000, "L3": 834, "L4": 667, "L5": 667}
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
LANGUAGES = ["en", "th", "code_switch"]
PLACEHOLDER_ALIASES = {"", "placeholder", "none"}


def _model_tag(teacher_model: str | None) -> str:
    if not teacher_model:
        return "placeholder"
    return teacher_model.replace("/", "-").replace(".", "-")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _resolve_required_key(teacher_model: str | None) -> str | None:
    """Return the env var name required for this teacher, or None for placeholder."""
    if not teacher_model:
        return None
    if teacher_model.startswith("gemini"):
        return "GEMINI_API_KEY"
    if teacher_model.startswith("gpt"):
        return "OPENAI_API_KEY"
    if teacher_model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    sys.exit(
        f"Unsupported --teacher-model: {teacher_model} "
        "(expected prefix gemini-*, gpt-*, or claude-*, or 'placeholder')"
    )


_VERIFY_PROMPT = (
    "Verify the Task A workflow dataset at {path}. These are the kept survivors of one "
    "batch from an automated generation loop; the deterministic filter has already removed "
    "structurally-defective rows, so focus on semantic rationality across a small sample. "
    "Activate the project venv first (source .venv/bin/activate) before running any Python "
    "tools. Begin your reply with a single line exactly of the form "
    "'VERDICT: sound' | 'VERDICT: sound-with-caveats' | 'VERDICT: not-fit', then 2-3 "
    "sentences of justification. Do not edit any files."
)


def verify_batch_with_agent(batch_file: Path, timeout: int) -> dict:
    """Run the dataset-verifier agent headlessly on one batch (advisory gate).

    Returns ``{"verdict": str, "not_fit": bool, "text": str, "error": str|None}``.
    Never raises: a failed/timed-out verification is logged and the loop continues
    (the deterministic filter is the authoritative remover; this is a logged gate).
    """
    cmd = [
        "claude", "-p", _VERIFY_PROMPT.format(path=batch_file),
        "--agent", "dataset-verifier",
        "--output-format", "json",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"verdict": "unknown", "not_fit": False, "text": "", "error": f"timeout after {timeout}s"}
    except FileNotFoundError:
        return {"verdict": "unknown", "not_fit": False, "text": "", "error": "claude CLI not found"}

    if proc.returncode != 0:
        return {"verdict": "unknown", "not_fit": False, "text": "",
                "error": f"claude exited {proc.returncode}: {proc.stderr.strip()[:200]}"}

    # --output-format json wraps the final text in a {"result": "..."} envelope.
    text = proc.stdout.strip()
    try:
        env = json.loads(text)
        if isinstance(env, dict):
            text = env.get("result") or env.get("text") or text
    except json.JSONDecodeError:
        pass

    verdict = "unknown"
    for line in text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().lower()
            break
    not_fit = "not-fit" in verdict or "not fit" in verdict
    return {"verdict": verdict, "not_fit": not_fit, "text": text[:500], "error": None}


def generate_leg(
    *,
    level: str,
    language: str,
    target: int,
    teacher_model: str | None,
    out_file: Path,
    base_seed: int,
    seed_offset: int,
    batch_size: int | None,
    max_iterations: int,
    behavior_preset: str,
    intent_category: str,
    initiation: str,
    keep_intermediates: bool,
    max_workers: int = 1,
    verify_batches: bool = False,
    verify_timeout: int = 600,
    retry_budget: int | None = None,
    retry_exhaustion: str | None = None,
    require_tool_stay: bool = True,
) -> dict:
    """Run the generate -> qualify -> accumulate loop for one level/language leg.

    Returns a stats dict and writes ``out_file``. ``seed_offset`` makes each leg's
    seed stream disjoint so legs don't regenerate identical batches.

    ``retry_budget`` / ``retry_exhaustion`` are ``None`` to keep the level's own
    ``COMPLEXITY_SPECS`` values. ``require_tool_stay`` gates the convention at
    BOTH ends: the generator's teacher repair loop stops treating an advancing
    tool-call turn as a coherence violation, and the qualification filter stops
    dropping conversations for it.
    """
    accum: list[dict] = []
    seen_ids: dict[str, int] = {}
    gen = clean_dropped = corruption_dropped = defect_dropped = 0
    stay_dropped = stay_violating = 0
    iteration = 0
    batch_verdicts: list[dict] = []

    while len(accum) < target and iteration < max_iterations:
        need = target - len(accum)
        n = batch_size or need
        seed = base_seed + seed_offset + iteration
        scratch = Path(tempfile.mkdtemp(prefix=f"sftloop_{level}_{language}_"))
        try:
            meta = generate_workflow_dataset(
                complexity_level=level,  # type: ignore[arg-type]
                num_samples=n,
                teacher_model=teacher_model,
                output_dir=scratch,
                seed=seed,
                language=language,  # type: ignore[arg-type]
                behavior_preset=behavior_preset,
                intent_category_preset=intent_category,
                initiation_preset=initiation,
                max_workers=max_workers,
                retry_budget=retry_budget,
                retry_exhaustion=retry_exhaustion,  # type: ignore[arg-type]
                require_tool_stay=require_tool_stay,
            )
            batch = _read_jsonl(meta.output_files[0])
            gen += len(batch)

            # --- qualify: cleaning fixes ---
            cleaned: list[dict] = []
            for rec in batch:
                rec2, _reason = clean_record(rec)
                if rec2 is None:
                    clean_dropped += 1
                else:
                    cleaned.append(rec2)
            n_after_clean = len(cleaned)

            # --- qualify: Thai-corruption gate (teacher transcription garble
            # that is invisible to the structural validators) ---
            uncorrupted: list[dict] = []
            iter_corruption = 0
            for rec in cleaned:
                if detect_thai_corruption(rec):
                    iter_corruption += 1
                else:
                    uncorrupted.append(rec)
            corruption_dropped += iter_corruption

            # --- qualify: hard structural defects ---
            clean_file = scratch / "cleaned.jsonl"
            _write_jsonl(clean_file, uncorrupted)
            # Profile ONCE and derive every count from that single report: a
            # second profile_task_a() pass over the same file would re-parse
            # every conversation to answer a question this report already
            # answers.
            report = profile_task_a(clean_file)
            bad = _disqualified_ids(report, require_tool_stay)
            qualified = [r for r in uncorrupted if r.get("conversation_id") not in bad]
            defect_dropped += len(uncorrupted) - len(qualified)

            # Stay observability, reported in both modes:
            #   iter_stay_violating — conversations with an advancing tool-call
            #     turn, whether or not the gate dropped them (the number that
            #     tells you how often the teacher breaks the convention);
            #   iter_stay_dropped — conversations this gate ALONE removed, i.e.
            #     the marginal cost of --require-tool-stay. Zero by construction
            #     when the gate is off. A conversation that also carries another
            #     hard defect is not attributed to the stay gate, since it would
            #     have been dropped regardless.
            stay_ids = _stay_violating_ids(report)
            iter_stay_violating = len(stay_ids)
            iter_stay_dropped = (
                len(stay_ids - _select_defects(report, lambda d: _STAY_DEFECT_MARKER not in d))
                if require_tool_stay
                else 0
            )
            stay_violating += iter_stay_violating
            stay_dropped += iter_stay_dropped

            # --- optional advisory gate: dataset-verifier agent on survivors ---
            verdict = None
            if verify_batches and qualified:
                qual_file = scratch / "qualified.jsonl"
                _write_jsonl(qual_file, qualified)
                verdict = verify_batch_with_agent(qual_file, verify_timeout)
                verdict["iteration"] = iteration + 1
                batch_verdicts.append(verdict)

            # --- accumulate with dedup-by-rename (concat_task_a pattern) ---
            for r in qualified:
                cid = r.get("conversation_id", "")
                if cid in seen_ids:
                    seen_ids[cid] += 1
                    r["conversation_id"] = f"{cid}_{seen_ids[cid]}"
                else:
                    seen_ids[cid] = 1
                accum.append(r)
        finally:
            if not keep_intermediates:
                shutil.rmtree(scratch, ignore_errors=True)

        iteration += 1
        verdict_msg = ""
        if verdict is not None:
            if verdict["error"]:
                verdict_msg = f"  [verify: ERROR {verdict['error']}]"
            else:
                verdict_msg = f"  [verify: {verdict['verdict']}]"
                if verdict["not_fit"]:
                    verdict_msg += "  ⚠ NOT-FIT (advisory; rows kept)"
        print(
            f"    iter {iteration}: +{len(qualified)} qualified "
            f"(gen {len(batch)}, clean-drop {len(batch) - n_after_clean}, "
            f"corruption-drop {iter_corruption}, "
            f"defect-drop {n_after_clean - iter_corruption - len(qualified)}, "
            f"stay-drop {iter_stay_dropped}/{iter_stay_violating} violating) "
            f"-> {len(accum)}/{target}{verdict_msg}"
        )

    shortfall = max(0, target - len(accum))
    accum = accum[:target]  # trim overshoot
    _write_jsonl(out_file, accum)

    if shortfall:
        print(
            f"    WARNING: {level}/{language} hit max_iterations "
            f"({max_iterations}) with {len(accum)}/{target} "
            f"(short {shortfall}). Raise --max-iterations or --batch-size."
        )

    return {
        "level": level,
        "language": language,
        "target": target,
        "kept": len(accum),
        "shortfall": shortfall,
        "generated": gen,
        "clean_dropped": clean_dropped,
        "corruption_dropped": corruption_dropped,
        "defect_dropped": defect_dropped,
        # Subset of defect_dropped attributable solely to the stay gate.
        "stay_dropped": stay_dropped,
        "stay_violating": stay_violating,
        "retry_budget": retry_budget,
        "retry_exhaustion": retry_exhaustion,
        "require_tool_stay": require_tool_stay,
        "iterations": iteration,
        "output_file": str(out_file),
        "batch_verdicts": batch_verdicts,
        "batches_not_fit": sum(1 for v in batch_verdicts if v.get("not_fit")),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate Task A SFT data, verify+filter each batch, loop until target."
    )
    p.add_argument("--output-dir", default="data/output/sft/task_a",
                   help="Destination dir for final leg files (default: data/output/sft/task_a).")
    p.add_argument("--seed", type=int, default=42, help="Base seed (default: 42).")
    p.add_argument("--teacher-model", default="gemini-3.5-flash",
                   help="Teacher model (prefix-routed); 'placeholder' for offline (default: gemini-3.5-flash).")
    p.add_argument("--levels", default=",".join(LEVELS),
                   help="Comma-separated levels (default: L1,L2,L3,L4,L5).")
    p.add_argument("--languages", default=",".join(LANGUAGES),
                   help="Comma-separated language legs (default: en,th,code_switch).")
    p.add_argument("--samples-per-leg", type=int, default=None,
                   help="Override per-leg target for all levels (else curriculum).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Samples generated per iteration (default: remaining-to-target).")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Concurrent teacher API calls per batch (default: 8). Teacher "
                        "generation is I/O-bound, so a thread pool hides per-call latency "
                        "for a near-linear speedup up to the provider's rate limit. Output "
                        "is identical regardless of this value (per-sample seeding). "
                        "No effect on placeholder/offline runs. Set 1 to disable.")
    p.add_argument("--max-iterations", type=int, default=20,
                   help="Per-leg safety cap on iterations (default: 20).")
    p.add_argument("--behavior-preset", default="adversarial")
    p.add_argument("--intent-category", default="default")
    p.add_argument("--initiation", default="default")
    p.add_argument("--retry-budget", type=int, default=None,
                   help="Override the retry budget for ALL selected levels: TOTAL attempts "
                        "at a failing tool call, counting the first (so 1 = no retry). "
                        "Default: per-level from COMPLEXITY_SPECS (L1-L4: 2, L5: 3).")
    p.add_argument("--retry-exhaustion", choices=["auto", "error_path", "handoff_in_state", "none"],
                   default="auto",
                   help="Override what the agent does once the budget is spent. "
                        "'auto' (default) keeps the per-level policy. Note 'error_path' still "
                        "degrades per-sample to 'handoff_in_state' on subgraphs with no "
                        "tool_error arc.")
    p.add_argument("--require-tool-stay", action="store_true", default=True,
                   help="Enforce the stay convention (default: on): a turn issuing a "
                        "<tool_call> must annotate [STATE: X -> X]. Violations are repaired "
                        "during generation and dropped during qualification.")
    p.add_argument("--no-require-tool-stay", dest="require_tool_stay", action="store_false",
                   help="Disable the stay-convention gate to regenerate v1-comparable data. "
                        "Advancing tool-call turns are neither repaired nor dropped.")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep per-iteration scratch dirs for debugging.")
    p.add_argument("--verify-batches", action="store_true",
                   help="Advisory per-batch gate: run the dataset-verifier agent (claude -p) on "
                        "each batch's qualified survivors and log its verdict. Does NOT remove rows "
                        "(deterministic filter is authoritative). Spawns a Claude session per batch "
                        "-- adds token cost + latency; requires the 'claude' CLI to be authenticated.")
    p.add_argument("--verify-timeout", type=int, default=600,
                   help="Per-batch verification timeout in seconds (default: 600). A headless "
                        "dataset-verifier run is slow (runs validator+profiler, reads samples); "
                        "on timeout the verdict is logged as an error and the loop continues.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned legs/targets without generating.")
    args = p.parse_args()

    if args.verify_batches and not shutil.which("claude"):
        sys.exit("Error: --verify-batches requires the 'claude' CLI on PATH (not found).")

    if args.retry_budget is not None and args.retry_budget < 1:
        sys.exit(
            "--retry-budget must be >= 1 (it counts the first attempt), "
            f"got {args.retry_budget}"
        )
    retry_exhaustion = _resolve_retry_exhaustion(args.retry_exhaustion)

    teacher_model: str | None = args.teacher_model
    if teacher_model is not None and teacher_model.lower() in PLACEHOLDER_ALIASES:
        teacher_model = None

    levels = [lv.strip().upper() for lv in args.levels.split(",") if lv.strip()]
    for lv in levels:
        if lv not in CURRICULUM:
            sys.exit(f"Unknown level: {lv} (expected one of {LEVELS})")
    languages = [lg.strip() for lg in args.languages.split(",") if lg.strip()]
    for lg in languages:
        if lg not in LANGUAGES:
            sys.exit(f"Unknown language: {lg} (expected one of {LANGUAGES})")

    required_key = _resolve_required_key(teacher_model)
    if required_key and not args.dry_run:
        import os
        if not os.environ.get(required_key):
            sys.exit(f"Error: {required_key} is not set (required for teacher model {teacher_model})")

    out_dir = Path(args.output_dir)
    model_tag = _model_tag(teacher_model)
    preset_tag = f"_{args.behavior_preset}" if args.behavior_preset != "default" else ""
    # One run-wide timestamp so all legs of this run share a suffix (matches the
    # `_{timestamp}` convention in generate_workflow_dataset) and re-runs don't
    # clobber prior output. concat_task_a globs `l<n>_*.jsonl`, so the suffix is
    # transparent to the merge step.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def leg_fname(level: str, language: str) -> str:
        return f"{level.lower()}_conversations_{language}_{model_tag}{preset_tag}_{run_ts}.jsonl"

    def target_for(level: str) -> int:
        return args.samples_per_leg if args.samples_per_leg is not None else CURRICULUM[level]

    print("=== Target-Driven SFT Data Generation ===")
    print(f"Output dir:    {out_dir}")
    print(f"Teacher model: {teacher_model or 'placeholder (offline)'}")
    print(f"Levels:        {','.join(levels)}")
    print(f"Languages:     {','.join(languages)}")
    print(f"Behavior:      {args.behavior_preset}   Intent: {args.intent_category}   Initiation: {args.initiation}")
    print(f"Base seed:     {args.seed}   Batch size: {args.batch_size or 'remaining'}   Max iters/leg: {args.max_iterations}")
    print(
        f"Retry budget:  {args.retry_budget if args.retry_budget is not None else 'per-level'}"
        f"   Exhaustion: {args.retry_exhaustion}"
        f"   Tool-stay gate: {'on' if args.require_tool_stay else 'OFF (v1-comparable)'}"
    )
    _workers_note = (
        "serial" if args.max_workers <= 1 or teacher_model is None
        else "concurrent teacher calls"
    )
    print(f"Max workers:   {args.max_workers} ({_workers_note})")
    if args.verify_batches:
        print(f"Verify gate:   dataset-verifier agent per batch (advisory, timeout {args.verify_timeout}s)")
    grand_target = sum(target_for(lv) for lv in levels for _ in languages)
    print(f"Total target:  {grand_target} qualified samples "
          f"({len(levels)} levels x {len(languages)} legs)")
    print("==========================================")

    if args.dry_run:
        for level in levels:
            for language in languages:
                fname = leg_fname(level, language)
                print(f"  [DRY RUN] {level}/{language}: target {target_for(level)} -> {out_dir / fname}")
        return 0

    all_stats: list[dict] = []
    seed_offset = 0
    for level in levels:
        for language in languages:
            target = target_for(level)
            fname = leg_fname(level, language)
            out_file = out_dir / fname
            print(f"\n  --- {level} / {language} (target {target}) ---")
            stats = generate_leg(
                level=level,
                language=language,
                target=target,
                teacher_model=teacher_model,
                out_file=out_file,
                base_seed=args.seed,
                seed_offset=seed_offset,
                batch_size=args.batch_size,
                max_iterations=args.max_iterations,
                behavior_preset=args.behavior_preset,
                intent_category=args.intent_category,
                initiation=args.initiation,
                keep_intermediates=args.keep_intermediates,
                max_workers=args.max_workers,
                verify_batches=args.verify_batches,
                verify_timeout=args.verify_timeout,
                retry_budget=args.retry_budget,
                retry_exhaustion=retry_exhaustion,
                require_tool_stay=args.require_tool_stay,
            )
            all_stats.append(stats)
            # Reserve a disjoint seed band per leg so legs never share batches.
            seed_offset += stats["iterations"] + args.max_iterations

    # Manifest + loop stats sidecar
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_dir": str(out_dir),
        "timestamp": run_ts,
        "teacher_model": teacher_model or "placeholder",
        "behavior_preset": args.behavior_preset,
        "intent_category": args.intent_category,
        "initiation": args.initiation,
        "base_seed": args.seed,
        "retry_budget": args.retry_budget,
        "retry_exhaustion": args.retry_exhaustion,
        "require_tool_stay": args.require_tool_stay,
        "legs": all_stats,
        "total_kept": sum(s["kept"] for s in all_stats),
        "total_shortfall": sum(s["shortfall"] for s in all_stats),
        "total_stay_dropped": sum(s["stay_dropped"] for s in all_stats),
        "total_stay_violating": sum(s["stay_violating"] for s in all_stats),
        "verify_batches": args.verify_batches,
        "total_batches_not_fit": sum(s.get("batches_not_fit", 0) for s in all_stats),
    }
    stats_path = out_dir / f"loop_stats_{run_ts}.json"
    stats_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n=== Done ===")
    print(f"Kept {manifest['total_kept']} qualified samples across {len(all_stats)} legs.")
    print(
        f"Tool-stay: {manifest['total_stay_violating']} conversation(s) had an advancing "
        f"tool-call turn; {manifest['total_stay_dropped']} dropped by the gate "
        f"({'on' if args.require_tool_stay else 'OFF'})."
    )
    if manifest["total_shortfall"]:
        print(f"WARNING: total shortfall {manifest['total_shortfall']} (see warnings above).")
    if args.verify_batches and manifest["total_batches_not_fit"]:
        print(f"WARNING: {manifest['total_batches_not_fit']} batch(es) flagged NOT-FIT by the "
              f"verifier agent (advisory; rows were kept). See batch_verdicts in {stats_path}.")
    print(f"Loop stats: {stats_path}")
    print("\nHybrid final step -- run the semantic verifier on the corpus:")
    print(f"  Ask Claude Code: \"Verify the Task A datasets in {out_dir} using the dataset-verifier agent\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
