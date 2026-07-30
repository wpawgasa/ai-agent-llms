#!/usr/bin/env python3
"""Self-loop habit decomposition over a heldout_composite_audit.py JSON.

Reproduces the tables in docs/cat_a_state_annotation_convention_review.md
sections 4.1 / 4.2 / 4.3 from the persisted audit rows. Read-only post-hoc
analysis: it re-uses the `pred_trans` and `gt_state_sequence` fields the audit
already stores, so it never re-runs generation and never touches a scoring
surface.

Why this exists
---------------
`analyze_composite_decomposition.py` answers "which composite TERM fails".
It does not answer "does the policy emit self-loops at all", which is the
question that decides whether a corpus regeneration is required (section 6.5).
That decomposition was computed ad hoc for section 4; this script makes it
reproducible and lets two checkpoints be compared on identical row sets.

Usage:
    python scripts/analyze_selfloop_habit.py \
        --audit runs/preflight/heldout_audit_C0_ckpt1770_goldenval.json \
        --label ckpt-1770 \
        --baseline-audit runs/preflight/heldout_audit_C0_ckpt500_goldenval.json \
        --baseline-label ckpt-500
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    d = json.loads(path.read_text())
    return sorted(d["rows"], key=lambda r: r["row_index"])


def _gt_trans(row: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for s in row.get("gt_state_sequence") or []:
        if isinstance(s, dict):
            out.append((s.get("from", ""), s.get("to", "")))
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            out.append((s[0], s[1]))
    return out


def _pred_trans(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [tuple(t) for t in (row.get("pred_trans") or []) if len(t) == 2]


def _is_stay(t: tuple[str, str]) -> bool:
    return t[0] == t[1]


def analyze(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    # ---- section 4.1 / 4.3: rows keyed by what GOLD expects -------------------
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    gold_stay_rows, gold_adv_rows = [], []
    for r in rows:
        gt = _gt_trans(r)
        if not gt:
            continue  # no transition expected: state term not applicable
        gold_kind = "stay" if _is_stay(gt[0]) else "advance"
        tool_kind = "tool" if r["n_gt_tools"] > 0 else "no tool"
        buckets.setdefault((gold_kind, tool_kind), []).append(r)
        (gold_stay_rows if gold_kind == "stay" else gold_adv_rows).append(r)

    def says_stay(r: dict[str, Any]) -> bool:
        p = _pred_trans(r)
        return bool(p) and any(_is_stay(t) for t in p)

    def summarize(rs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rs:
            return {"rows": 0}
        return {
            "rows": len(rs),
            "state_acc": statistics.fmean(r["state_acc"] for r in rs),
            "model_says_stay": sum(says_stay(r) for r in rs) / len(rs),
        }

    # ---- section 4.2: emission rate over ALL predicted annotations ------------
    all_pred = [t for r in rows for t in _pred_trans(r)]
    n_selfloop = sum(_is_stay(t) for t in all_pred)

    n_gold_trans_rows = len(gold_stay_rows) + len(gold_adv_rows)

    # ---- section 4.1 tail: on FAILED gold-stay rows, was `from` correct? ------
    failed_stay = [r for r in gold_stay_rows if r["state_acc"] == 0.0]
    from_ok = 0
    for r in failed_stay:
        p, g = _pred_trans(r), _gt_trans(r)
        if p and g and p[0][0] == g[0][0]:
            from_ok += 1

    return {
        "label": label,
        "n_rows": len(rows),
        "gold": {
            "n_transition_expected": n_gold_trans_rows,
            "self_loop_rate": len(gold_stay_rows) / n_gold_trans_rows
            if n_gold_trans_rows
            else 0.0,
        },
        "model_emission": {
            "n_pred_annotations": len(all_pred),
            "n_self_loop": n_selfloop,
            "self_loop_rate": n_selfloop / len(all_pred) if all_pred else 0.0,
        },
        "by_gold_expectation": {
            "advance": summarize(gold_adv_rows),
            "self_loop": summarize(gold_stay_rows),
        },
        "buckets": {
            f"{k[0]} + {k[1]}": summarize(v)
            for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))
        },
        "failed_stay_rows": {
            "n": len(failed_stay),
            "from_state_correct": from_ok,
            "frac_from_correct": from_ok / len(failed_stay) if failed_stay else 0.0,
        },
    }


def _print(s: dict[str, Any]) -> None:
    print(f"\n=== self-loop habit — {s['label']} (n={s['n_rows']}) ===")
    g, m = s["gold"], s["model_emission"]
    print(f"  gold self-loop rate   : {g['self_loop_rate']:.4f} "
          f"({g['n_transition_expected']} transition-expected rows)")
    print(f"  MODEL self-loop rate  : {m['self_loop_rate']:.4f} "
          f"({m['n_self_loop']} of {m['n_pred_annotations']} annotations)")
    print("  by gold expectation:")
    for k, v in s["by_gold_expectation"].items():
        if v["rows"]:
            print(f"    {k:<10} rows={v['rows']:<4} state_acc={v['state_acc']:.4f} "
                  f"model_says_stay={v['model_says_stay']:.4f}")
    print("  buckets:")
    for k, v in s["buckets"].items():
        print(f"    {k:<20} rows={v['rows']:<4} state_acc={v['state_acc']:.4f} "
              f"model_says_stay={v['model_says_stay']:.4f}")
    f = s["failed_stay_rows"]
    print(f"  failed gold-stay rows : {f['n']}, `from` correct on "
          f"{f['from_state_correct']} ({f['frac_from_correct']:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--label", default="audit")
    ap.add_argument("--baseline-audit", type=Path)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result: dict[str, Any] = {"primary": analyze(_load(args.audit), args.label)}
    _print(result["primary"])

    if args.baseline_audit:
        result["baseline"] = analyze(_load(args.baseline_audit), args.baseline_label)
        _print(result["baseline"])
        a, b = result["primary"], result["baseline"]
        print(f"\n=== delta ({a['label']} - {b['label']}) ===")
        print(f"  model self-loop rate : "
              f"{b['model_emission']['self_loop_rate']:.4f} -> "
              f"{a['model_emission']['self_loop_rate']:.4f}  "
              f"(gold {a['gold']['self_loop_rate']:.4f})")
        for k in a["by_gold_expectation"]:
            va, vb = a["by_gold_expectation"][k], b["by_gold_expectation"][k]
            if va.get("rows") and vb.get("rows"):
                print(f"  state_acc {k:<10}: {vb['state_acc']:.4f} -> {va['state_acc']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
