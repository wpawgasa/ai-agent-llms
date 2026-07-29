#!/usr/bin/env python3
"""Decompose held-out composite audits: which term is actually failing?

Answers the question docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md
§12.4 step 1 poses — "If the failing term is tool-F1 rather than state accuracy,
this factorial is aimed at the wrong component and C1/C2 are both mis-targeted."

Read-only post-hoc analysis over the JSON that
``scripts/heldout_composite_audit.py`` persists. It does NOT re-run generation and
does not modify any scoring surface (§7's confound discipline: the gate must not
move between cells).

Why this script is needed at all
-------------------------------
``heldout_composite_audit.py`` scores a FLAT ``0.4*state + 0.4*tool + 0.2*task``
and its docstring claims that "mirrors grpo._heldout_composite_score exactly".
It does not. The real gate (``grpo._heldout_composite_score``) is *per-turn-fair*:
the tool term always applies, the state term only when GT expects a transition,
the task term only on the terminal turn — then it renormalizes over the included
weights. Consequences on the current corpus:

  * The audit's ``mean_state_acc`` is a flat mean over ALL rows, including those
    with no GT transition, where it applies a ``1.0 if not pred_trans else 0.0``
    fallback. An always-annotating SFT policy fails that by construction, so the
    figure reads systematically low and is NOT the gate's state term.
  * The audit's ``mean_composite`` is not the gate's number either.

So we recompute the gate-aligned score here, reusing the already-unit-tested
``perturn_fair_composite.perturn_fair_composite_from_components`` rather than
reimplementing the weighting a third time.

The 0.5 spike
-------------
§12.3 observed a hard spike at exactly 0.5. That value is *arithmetically
impossible* on rows with no GT transition (there the gate reduces to
``composite == tool_f1``, which is 0 or 1). It can only arise on
transition-expected non-terminal rows, where ``composite = (tool_f1 + state_acc)/2``
— so 0.5 means exactly one of the two terms scored 0. This script tallies which.

Note ``tool_call_f1([], []) == 1.0``: on transition-expected rows with no GT
tools, the tool term is an *abstention* indicator, not a call-quality one.

Usage:
    .venv-train/bin/python scripts/analyze_composite_decomposition.py \\
        --audit runs/preflight/heldout_audit_C0_ckpt500.json \\
        --label C0-ckpt500 \\
        --baseline-audit runs/preflight/heldout_audit_ckpt1000_baseline.json \\
        --baseline-label ckpt1000 \\
        --verify-gate runs/preflight/heldout_composite_C0_ckpt500.json \\
        --output runs/preflight/composite_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL = 1e-9

# Spec §4.4 baselines this analysis reports against.
BASELINE_TOOL_F1 = 0.4623
BASELINE_ABSTENTION = 0.9494
GATE_TARGET = 0.80
STATE_TARGET = 0.833


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Return audit rows in stable row_index order (the file stores them sorted by score)."""
    d = json.loads(path.read_text())
    rows = d["rows"]
    return sorted(rows, key=lambda r: r["row_index"])


def _augment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the gate-aligned composite and applicability flags to each row."""
    from perturn_fair_composite import perturn_fair_composite_from_components

    out = []
    for r in rows:
        gate, appl = perturn_fair_composite_from_components(
            state_acc=r["state_acc"],
            tool_f1=r["tool_f1"],
            task=r["task"],
            n_gt_trans=r["n_gt_trans"],
            gt_state_sequence=r.get("gt_state_sequence") or [],
            gt_terminal=r.get("gt_terminal") or "",
        )
        den = 0.4 + (0.4 if appl["state"] else 0.0) + (0.2 if appl["task"] else 0.0)
        out.append({**r, "gate_composite": gate, "incl": appl, "den": round(den, 2)})
    return out


def _strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Row counts by (transition-expected, tool-expected, terminal) — the gate denominators."""
    c: Counter = Counter()
    for r in rows:
        c[(r["incl"]["state"], r["n_gt_tools"] > 0, r["incl"]["task"], r["den"])] += 1
    return [
        {"state_expected": k[0], "tool_expected": k[1], "terminal": k[2], "den": k[3], "rows": v}
        for k, v in sorted(c.items(), key=lambda x: -x[1])
    ]


def _spike(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decompose the rows scoring exactly 0.5 by which term is the zero."""
    spike = [r for r in rows if abs(r["gate_composite"] - 0.5) < 1e-6]
    breakdown: Counter = Counter()
    for r in spike:
        state_zero = r["state_acc"] == 0.0
        tool_zero = r["tool_f1"] == 0.0
        if state_zero and not tool_zero:
            kind = "state=0, tool=1"
        elif tool_zero and not state_zero:
            kind = "tool=0, state=1"
        else:  # e.g. both partial and summing to 1.0
            kind = f"other (state={r['state_acc']:.2f}, tool={r['tool_f1']:.2f})"
        breakdown[(kind, "gt_has_tools" if r["n_gt_tools"] > 0 else "gt_no_tools")] += 1
    return {
        "n_spike_rows": len(spike),
        "frac_of_all_rows": len(spike) / len(rows) if rows else 0.0,
        "breakdown": [
            {"pattern": k[0], "gt_tools": k[1], "rows": v}
            for k, v in sorted(breakdown.items(), key=lambda x: -x[1])
        ],
        "impossible_on_den_0.4_rows": sum(1 for r in rows if r["den"] == 0.4),
    }


def _summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    trans = [r for r in rows if r["incl"]["state"]]
    tool_exp = [r for r in rows if r["n_gt_tools"] > 0]
    zero_tool = [r for r in rows if r["n_gt_tools"] == 0]
    mean = statistics.fmean

    return {
        "label": label,
        "n_rows": n,
        "gate_composite_mean": mean([r["gate_composite"] for r in rows]),
        "gate_composite_median": statistics.median([r["gate_composite"] for r in rows]),
        "audit_flat_composite_mean": mean([r["composite"] for r in rows]),
        "state_term": {
            "n_transition_expected": len(trans),
            "frac_of_rows": len(trans) / n if n else 0.0,
            "mean_state_acc_gate_aligned": mean([r["state_acc"] for r in trans]) if trans else 0.0,
            "mean_state_acc_audit_flat": mean([r["state_acc"] for r in rows]),
            "target": STATE_TARGET,
        },
        "tool_term": {
            "n_tool_expected": len(tool_exp),
            "mean_tool_f1_on_tool_expected": mean([r["tool_f1"] for r in tool_exp])
            if tool_exp
            else 0.0,
            "baseline": BASELINE_TOOL_F1,
            "n_zero_tool": len(zero_tool),
            "abstention_rate": (
                sum(1 for r in zero_tool if r["tool_f1"] == 1.0) / len(zero_tool)
                if zero_tool
                else 0.0
            ),
            "abstention_baseline": BASELINE_ABSTENTION,
        },
        "strata": _strata(rows),
        "spike_at_0.5": _spike(rows),
    }


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]], la: str, lb: str) -> dict[str, Any]:
    """Paired comparison on identical rows: McNemar on binary state, Wilcoxon on composite."""
    import numpy as np
    from scipy import stats

    idx_b = {r["row_index"]: r for r in b}
    pairs = [(r, idx_b[r["row_index"]]) for r in a if r["row_index"] in idx_b]
    if not pairs:
        return {"error": "no overlapping row_index between the two audits"}

    da = np.array([p[0]["gate_composite"] for p in pairs])
    db = np.array([p[1]["gate_composite"] for p in pairs])
    diff = da - db

    # McNemar on the binary state term, restricted to rows where BOTH include it.
    both = [p for p in pairs if p[0]["incl"]["state"] and p[1]["incl"]["state"]]
    n01 = sum(1 for x, y in both if x["state_acc"] == 0.0 and y["state_acc"] == 1.0)
    n10 = sum(1 for x, y in both if x["state_acc"] == 1.0 and y["state_acc"] == 0.0)
    disc = n01 + n10
    mcnemar_p = float(stats.binomtest(n10, disc, 0.5).pvalue) if disc else 1.0

    nonzero = diff[diff != 0]
    wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
    t_p = float(stats.ttest_rel(da, db).pvalue) if np.any(diff) else 1.0

    return {
        "labels": {"a": la, "b": lb},
        "n_paired_rows": len(pairs),
        "mean_a": float(da.mean()),
        "mean_b": float(db.mean()),
        "mean_paired_delta_a_minus_b": float(diff.mean()),
        "n_rows_differing": int((diff != 0).sum()),
        "n_favoring_a": int((diff > 0).sum()),
        "n_favoring_b": int((diff < 0).sum()),
        "wilcoxon_p": wilcoxon_p,
        "paired_t_p": t_p,
        "state_mcnemar": {
            "n_rows_state_applicable_both": len(both),
            f"{la}_wrong_{lb}_right": n01,
            f"{la}_right_{lb}_wrong": n10,
            "n_discordant": disc,
            "exact_p": mcnemar_p,
        },
    }


def _verify(rows: list[dict[str, Any]], gate_json: Path) -> dict[str, Any]:
    """Cross-check recomputed gate composites against a prior gate run's per_row output."""
    d = json.loads(gate_json.read_text())
    prior = {r["row_index"]: r["composite"] for r in d["per_row"]}
    mism = [
        {
            "row_index": r["row_index"],
            "recomputed": r["gate_composite"],
            "prior": prior[r["row_index"]],
        }
        for r in rows
        if r["row_index"] in prior and abs(r["gate_composite"] - prior[r["row_index"]]) > 1e-6
    ]
    return {
        "gate_json": str(gate_json),
        "n_compared": sum(1 for r in rows if r["row_index"] in prior),
        "n_mismatched": len(mism),
        "prior_mean_composite": d["summary"]["mean_composite"],
        "recomputed_mean_composite": statistics.fmean([r["gate_composite"] for r in rows]),
        "verdict": "MATCH" if not mism else "MISMATCH",
        "examples": mism[:5],
    }


def _print(s: dict[str, Any]) -> None:
    st, tt = s["state_term"], s["tool_term"]
    print(f"\n=== {s['label']} — n={s['n_rows']} ===")
    print(
        f"  gate composite   : {s['gate_composite_mean']:.4f}   "
        f"(bar {GATE_TARGET}; audit's flat number {s['audit_flat_composite_mean']:.4f})"
    )
    print(
        f"  STATE (gate-aligned, n={st['n_transition_expected']}): "
        f"{st['mean_state_acc_gate_aligned']:.4f}  target {st['target']}"
    )
    print(
        f"    audit's flat mean over all rows : {st['mean_state_acc_audit_flat']:.4f}  "
        f"<- understates; do not use"
    )
    print(
        f"  TOOL  (tool-expected, n={tt['n_tool_expected']}): "
        f"{tt['mean_tool_f1_on_tool_expected']:.4f}  baseline {tt['baseline']}"
    )
    print(
        f"  ABSTENTION (zero-tool, n={tt['n_zero_tool']}): "
        f"{tt['abstention_rate']:.4f}  baseline {tt['abstention_baseline']}"
    )
    sp = s["spike_at_0.5"]
    print(f"  0.5 spike: {sp['n_spike_rows']} rows ({sp['frac_of_all_rows']:.1%})")
    for b in sp["breakdown"]:
        print(f"    {b['pattern']:<34} {b['gt_tools']:<13} {b['rows']:>4}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, type=Path, help="primary heldout_composite_audit.py JSON")
    ap.add_argument("--label", default="audit")
    ap.add_argument("--baseline-audit", type=Path, help="second audit JSON for a paired comparison")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--verify-gate", type=Path, help="prior gate JSON to cross-check recomputation")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    rows = _augment(_load_rows(args.audit))
    result: dict[str, Any] = {"primary": _summarize(rows, args.label)}
    _print(result["primary"])

    if args.verify_gate:
        v = _verify(rows, args.verify_gate)
        result["verification"] = v
        print(
            f"\n=== verification vs {args.verify_gate.name} ===\n"
            f"  compared {v['n_compared']} rows, mismatched {v['n_mismatched']}\n"
            f"  prior mean {v['prior_mean_composite']:.4f} vs "
            f"recomputed {v['recomputed_mean_composite']:.4f}  -> {v['verdict']}"
        )
        for e in v["examples"]:
            print(f"    row {e['row_index']}: recomputed {e['recomputed']:.4f} vs prior {e['prior']:.4f}")

    if args.baseline_audit:
        brows = _augment(_load_rows(args.baseline_audit))
        result["baseline"] = _summarize(brows, args.baseline_label)
        _print(result["baseline"])
        p = _paired(rows, brows, args.label, args.baseline_label)
        result["paired"] = p
        print(f"\n=== paired: {args.label} vs {args.baseline_label} ===")
        if "error" in p:
            print(f"  {p['error']}")
        else:
            print(
                f"  n={p['n_paired_rows']}  mean {p['mean_a']:.4f} vs {p['mean_b']:.4f}  "
                f"delta {p['mean_paired_delta_a_minus_b']:+.4f}"
            )
            print(
                f"  rows differing {p['n_rows_differing']} "
                f"({p['n_favoring_a']} favor {args.label}, {p['n_favoring_b']} favor {args.baseline_label})"
            )
            print(f"  Wilcoxon p={p['wilcoxon_p']:.4f}   paired-t p={p['paired_t_p']:.4f}")
            m = p["state_mcnemar"]
            print(f"  state McNemar: {m['n_discordant']} discordant, exact p={m['exact_p']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
