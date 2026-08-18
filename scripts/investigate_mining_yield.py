#!/usr/bin/env python3
"""Decompose the mining-yield gap into a split effect and a classifier effect.

`scripts/mine_model_negatives.py`'s first run found C2 wrong on 51 of 399
TRAIN prompts (12.8%). `docs/cat_a_c2_heldout_result.md`'s held-out audit
found C2 wrong on 27 of 71 tool-bearing TEST-derived rows (38.0%) — a
contamination-free set the checkpoint never trained on. CLAUDE.md's `dc84adb`
commit names two untested explanations for the gap:

  1. SPLIT EFFECT   — TRAIN rows were seen during SFT; TEST rows were not.
  2. CLASSIFIER EFFECT — mine_model_negatives.py's `_classify()` may be more
     lenient than the held-out audit's strict composite scorer.

This script measures each while holding the other fixed:

  PROBE 1 (classifier effect, no GPU) --from-audit-json PATH
    Reclassifies a stored scripts/heldout_composite_audit.py output's
    completions with `_classify()` — the SAME rows that produced 38.0%, so
    only the scoring function varies.

  PROBE 2 (split effect, needs GPU) --checkpoint PATH --split validation
    Runs mine_model_negatives.py's own `_select_prompts`/`_classify` code
    path, unchanged except for --split, directly comparable to the known
    12.8% TRAIN figure — only the split varies.

SAMPLE-COMPOSITION CONFOUND (read before interpreting Probe 2)
--------------------------------------------------------------
Probe 2's headline `wrong_rate` is an aggregate over a MIXED sample, and the
two historical figures it sits beside in the table are not matched strata:

  * 38.0% (held-out composite) is TOOL-BEARING ROWS ONLY (27 of 71).
  * 12.8% (train, dc84adb) is a ~75/25 tool-bearing/no-tool MIX
    (`--tool-share 0.75`), because that is what `_select_prompts` samples.

Tool-bearing turns are far harder (38.0% wrong versus 8.9% on no-tool rows on
the held-out audit), so the aggregate rate moves with the sample's tool share
independently of any real split effect. Two things follow:

  1. Always read the STRATIFIED fields `wrong_rate_tool_bearing` /
     `wrong_rate_no_tool` (with their `n_tool_bearing` / `n_no_tool` counts),
     not just `wrong_rate`. Only the tool-bearing stratum is comparable to
     the held-out 38.0%.
  2. Run Probe 2 on **both** `--split train` and `--split validation` with
     otherwise identical flags, and compare the stratified rates to each
     other rather than to 12.8%. That is the only way the train-vs-validation
     delta is attributable to the split rather than to sample composition.
     (`--split test` stays hard-refused; this recommendation is about the
     train/validation pair only.)

`--tool-share 0.75` is a TARGET, not a guarantee: `_select_prompts` keeps one
row per conversation, so on the ~289-conversation validation split (R17) the
achievable tool-bearing count can fall well short of `round(n * tool_share)`.
Probe 2 therefore reports `tool_share_requested` alongside the realized
`tool_share_scored` and warns on stderr when they diverge materially — a
silently tool-poor validation sample would bias its aggregate wrong-rate
downward and look like "validation is easier".

Never point --split at "test" here — see mine_model_negatives.py's identical
guard and CLAUDE.md R18. Probe 1 does not generate anything; it only
reclassifies completions a prior audit already produced, and this script
never writes a chosen/rejected-shaped pairs file, so its output can never be
mistaken for DPO training data.

Usage:
    # Cheap, no GPU:
    python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --output runs/audit/mining_yield_investigation.json

    # Needs .venv-train + GPU. Run BOTH splits for a matched comparison:
    .venv-train/bin/python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \\
        --split train --output runs/audit/mining_yield_investigation_train.json
    .venv-train/bin/python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \\
        --split validation \\
        --output runs/audit/mining_yield_investigation_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mine_model_negatives import _classify, _select_prompts  # noqa: E402

#: Known figures this script's new measurements get compared against.
_KNOWN_TRAIN_WRONG_RATE = 0.128  # dc84adb: 51/399
_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE = 0.380  # docs/cat_a_c2_heldout_result.md: 27/71


def _gt_from_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the {tool_calls, state_sequence} shape _classify() expects
    from one row of a scripts/heldout_composite_audit.py output JSON.

    The audit stores GT as separate `gt_tool_calls` / `gt_state_sequence`
    fields (see that script's `_components()`), not a single `ground_truth`
    dict — this bridges the two shapes.
    """
    return {
        "tool_calls": row.get("gt_tool_calls") or [],
        "state_sequence": row.get("gt_state_sequence") or [],
    }


def classify_rate_from_audit_json(
    path: Path, tool_bearing_only: bool = True
) -> dict[str, Any]:
    """Probe 1: reclassify a stored audit's completions with _classify().

    Holds the split fixed (these are the exact rows that produced the
    composite "wrong" rate) and varies only the scoring function, isolating
    the classifier effect.
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["rows"]
    if tool_bearing_only:
        rows = [r for r in rows if r.get("n_gt_tools", 0) > 0]

    wrong = 0
    by_kind: dict[str, int] = {}
    for row in rows:
        gt = _gt_from_audit_row(row)
        kind = _classify(row.get("completion", ""), gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

    n = len(rows)
    return {
        "probe": "classifier_effect",
        "source": str(path),
        "tool_bearing_only": tool_bearing_only,
        "n_rows": n,
        "n_wrong": wrong,
        "wrong_rate": wrong / n if n else 0.0,
        "by_kind": by_kind,
    }


def classify_rate_from_split(
    checkpoint: str,
    data_dir: Path,
    split: str,
    n_prompts: int,
    tool_share: float,
    seed: int,
    max_new_tokens: int,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Probe 2: mine_model_negatives.py's own selection + classify code path
    against `split`, unchanged except for which split it reads.

    Comparable to the known 12.8% TRAIN figure, which used the identical
    function with split="train" — only the split varies, isolating the split
    effect.

    **The aggregate `wrong_rate` carries a sample-composition confound** (see
    the module docstring): it mixes tool-bearing and no-tool rows in whatever
    proportion `_select_prompts` actually achieved, and tool-bearing turns are
    several times harder. So this also returns a stratified breakdown —
    `n_tool_bearing` / `wrong_rate_tool_bearing` and `n_no_tool` /
    `wrong_rate_no_tool` — computed in the same single pass, bucketed by
    ``gt.get("tool_calls")`` truthiness (the identical test `_select_prompts`
    uses to bucket rows). Only the tool-bearing stratum is comparable to the
    held-out audit's 38.0%, which is tool-bearing-only; the 12.8% TRAIN figure
    is a ~75/25 mix and so is comparable only to `wrong_rate` from an
    identically-composed sample.

    For a clean split-effect reading, call this on **both** ``split="train"``
    and ``split="validation"`` with every other argument held fixed and
    compare the stratified rates to each other. `tool_share_requested` versus
    `tool_share_scored` reveals whether the requested `tool_share` was
    actually achievable — on validation (~289 conversations, R17) one-row-per-
    conversation dedup can make it not be.

    On ``split="validation"`` this samples from the FULL split, including
    rows `reserve_guardrail_slice` would reserve for `dpo.py`'s R5 guardrail
    (unlike `mine_model_negatives.py --split validation`, which excludes
    them). That's deliberate here — a read-only diagnostic measurement, not a
    negative-mining run, so there is nothing to exclude reserved rows from;
    excluding them would only shrink the sample for no benefit, since the
    partition is a content hash and so an unbiased subsample either way.
    """
    if split == "test":
        raise SystemExit(
            "refusing to generate against the test split — it feeds the "
            "held-out audit set, the only contamination-free measuring "
            "stick for this lineage"
        )

    from preflight_entropy_diag import _generate_for_checkpoint

    rows = _select_prompts(data_dir, split, n_prompts, tool_share, seed)
    gens = _generate_for_checkpoint(
        checkpoint=checkpoint,
        prompts=rows,
        n_completions=1,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        seed=seed,
        do_sample=False,
    )

    wrong = 0
    by_kind: dict[str, int] = {}
    n_scored = 0
    # Stratified counters — the aggregate rate alone is confounded by sample
    # composition (module docstring). Bucketed by the same
    # `gt.get("tool_calls")` truthiness test `_select_prompts` uses.
    n_tool_bearing = 0
    n_wrong_tool_bearing = 0
    n_no_tool = 0
    n_wrong_no_tool = 0
    for row, comp_list in zip(rows, gens):
        completion = (comp_list[0] if comp_list else "") or ""
        gt = json.loads(row["ground_truth"])
        gold = (gt.get("messages") or [{}])[0].get("content") or ""
        if not completion.strip() or not gold.strip():
            continue
        if completion.strip() == gold.strip():
            continue
        n_scored += 1
        is_tool_bearing = bool(gt.get("tool_calls"))
        if is_tool_bearing:
            n_tool_bearing += 1
        else:
            n_no_tool += 1
        kind = _classify(completion, gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if is_tool_bearing:
                n_wrong_tool_bearing += 1
            else:
                n_wrong_no_tool += 1

    tool_share_scored = n_tool_bearing / n_scored if n_scored else 0.0
    # A tool-poor sample biases the aggregate wrong-rate downward and can be
    # mistaken for "this split is easier" — say so loudly rather than silently.
    if n_scored and abs(tool_share_scored - tool_share) > 0.05:
        print(
            f"[probe2] WARNING: requested --tool-share {tool_share:.2f} but only "
            f"{tool_share_scored:.2f} of the {n_scored} scored rows are "
            f"tool-bearing ({n_tool_bearing}/{n_scored}). _select_prompts keeps "
            "one row per conversation, so the target may be unachievable on this "
            "split. The aggregate wrong_rate is NOT comparable to a differently "
            "composed sample — use wrong_rate_tool_bearing.",
            file=sys.stderr,
        )

    return {
        "probe": "split_effect",
        "split": split,
        "n_prompts_sampled": len(rows),
        "n_scored": n_scored,
        "n_wrong": wrong,
        # Aggregate over a MIXED sample — confounded by tool share; compare
        # only against an identically composed sample.
        "wrong_rate": wrong / n_scored if n_scored else 0.0,
        "by_kind": by_kind,
        # Stratified — `wrong_rate_tool_bearing` is the figure comparable to
        # the held-out audit's tool-bearing-only 38.0%.
        "n_tool_bearing": n_tool_bearing,
        "n_wrong_tool_bearing": n_wrong_tool_bearing,
        "wrong_rate_tool_bearing": (
            n_wrong_tool_bearing / n_tool_bearing if n_tool_bearing else 0.0
        ),
        "n_no_tool": n_no_tool,
        "n_wrong_no_tool": n_wrong_no_tool,
        "wrong_rate_no_tool": n_wrong_no_tool / n_no_tool if n_no_tool else 0.0,
        "tool_share_requested": tool_share,
        "tool_share_scored": tool_share_scored,
    }


def print_decomposition_table(
    train_rate: float,
    validation_probe: dict[str, Any] | None,
    heldout_classify_probe: dict[str, Any] | None,
    heldout_composite_rate: float,
) -> None:
    """Print the 4-row comparison table. Pure formatting: no I/O, no model.

    **The rows are not matched strata** — reading a vertical delta off this
    table alone overstates what has been measured:

      * `held-out (composite, known)` = 38.0% is TOOL-BEARING ROWS ONLY.
      * `train (classify, known)` = 12.8% is a ~75/25 tool-bearing/no-tool MIX.
      * the Probe 2 row (`validation`, or `train` if that is the split run) is
        an aggregate over whatever mix `_select_prompts` actually achieved.

    Since tool-bearing turns are several times harder, part of any delta here
    is sample composition rather than the split or the classifier. Probe 2's
    result dict carries the stratified `wrong_rate_tool_bearing` /
    `wrong_rate_no_tool` breakdown, printed beneath the table when available;
    those are what to compare. For a clean split-effect reading, run Probe 2
    on **both** `--split train` and `--split validation` with identical flags
    and compare the two stratified results to each other. (`--split test`
    remains hard-refused — this concerns the train/validation pair only.)
    """
    probe_split = (validation_probe or {}).get("split", "validation")
    rows = [
        ("train      (classify, known)", f"{train_rate:.1%}"),
        (
            f"{probe_split:<10} (classify, new)",
            f"{validation_probe['wrong_rate']:.1%}" if validation_probe else "not run",
        ),
        (
            "held-out   (classify, new)",
            f"{heldout_classify_probe['wrong_rate']:.1%}" if heldout_classify_probe else "not run",
        ),
        ("held-out   (composite, known)", f"{heldout_composite_rate:.1%}"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")

    print()
    print(
        "  CONFOUND: these rows are NOT matched strata. 38.0% (held-out\n"
        "  composite) is tool-bearing rows only; 12.8% (train) is a ~75/25\n"
        "  tool/no-tool mix; the new row is whatever mix _select_prompts\n"
        "  achieved. Compare the stratified rates below, and run Probe 2 on\n"
        "  BOTH --split train and --split validation for a matched comparison."
    )
    if validation_probe and validation_probe.get("n_scored"):
        print(
            f"  stratified [{probe_split}]: "
            f"tool-bearing {validation_probe['wrong_rate_tool_bearing']:.1%} "
            f"(n={validation_probe['n_tool_bearing']})  |  "
            f"no-tool {validation_probe['wrong_rate_no_tool']:.1%} "
            f"(n={validation_probe['n_no_tool']})  |  "
            f"tool share requested {validation_probe['tool_share_requested']:.2f} "
            f"vs realized {validation_probe['tool_share_scored']:.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-audit-json", type=Path, default=None,
        help="Probe 1: reclassify a stored heldout_composite_audit.py output.",
    )
    ap.add_argument(
        "--include-non-tool-rows", action="store_true",
        help="Probe 1: score all rows, not just the tool-bearing subset.",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="Probe 2: run a fresh generation pass against --split.",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--tool-share", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-wrong-rate", type=float, default=_KNOWN_TRAIN_WRONG_RATE)
    ap.add_argument(
        "--heldout-composite-wrong-rate", type=float,
        default=_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE,
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.from_audit_json and not args.checkpoint:
        print(
            "Nothing to do: pass --from-audit-json and/or --checkpoint.",
            file=sys.stderr,
        )
        return 1

    result: dict[str, Any] = {
        "train_wrong_rate": args.train_wrong_rate,
        "heldout_composite_wrong_rate": args.heldout_composite_wrong_rate,
    }

    heldout_classify_probe = None
    if args.from_audit_json:
        heldout_classify_probe = classify_rate_from_audit_json(
            args.from_audit_json, tool_bearing_only=not args.include_non_tool_rows
        )
        result["classifier_effect_probe"] = heldout_classify_probe
        print(
            f"[probe1] classifier effect: {heldout_classify_probe['wrong_rate']:.1%} "
            f"on {heldout_classify_probe['n_rows']} rows "
            f"(composite said {args.heldout_composite_wrong_rate:.1%})"
        )

    validation_probe = None
    if args.checkpoint:
        t0 = time.time()
        validation_probe = classify_rate_from_split(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            n_prompts=args.n_prompts,
            tool_share=args.tool_share,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        result["split_effect_probe"] = validation_probe
        print(
            f"[probe2] split effect on --split {args.split}: "
            f"{validation_probe['wrong_rate']:.1%} aggregate "
            f"({time.time() - t0:.0f}s, {validation_probe['n_scored']} rows scored)"
            f"; tool-bearing {validation_probe['wrong_rate_tool_bearing']:.1%} "
            f"(n={validation_probe['n_tool_bearing']}), "
            f"no-tool {validation_probe['wrong_rate_no_tool']:.1%} "
            f"(n={validation_probe['n_no_tool']})"
        )

    print()
    print_decomposition_table(
        args.train_wrong_rate,
        validation_probe,
        heldout_classify_probe,
        args.heldout_composite_wrong_rate,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
