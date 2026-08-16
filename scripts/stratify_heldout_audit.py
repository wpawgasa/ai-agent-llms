"""Stratify and compare held-out composite audits produced by
``scripts/heldout_composite_audit.py``.

The audit JSON reports an aggregate composite plus per-row components. The
aggregate alone is misleading on this task for two reasons documented in
``docs/cat_a_corpus_v2_heldout_regression.md`` section 4:

  * ``tool_f1`` is mostly credit for staying silent — ``tool_call_f1([], [])``
    is 1.0 and roughly two thirds of rows carry no ground-truth tool call, so
    the aggregate hides real tool-calling ability behind a large no-op majority.
  * ``task`` scores reaching the conversation terminal state on rows that are
    mid-conversation turns, where terminating would be wrong.

So a comparison between checkpoints has to be stratified by ground-truth turn
type (self-loop vs advancing) and by whether the turn requires a tool at all.
This script reproduces those tables, and adds the paired statistics that decide
whether a delta is real: a bootstrap CI over per-row deltas and an exact sign
test over the discordant rows. Rows are matched across audits by ``row_index``,
which is stable because every audit replays the same sampler with the same seed
over the same pinned held-out set.

Verified: run against the two stored audits it reproduces every figure in
``docs/cat_a_corpus_v2_heldout_regression.md`` sections 1 and 3 exactly.

Usage:
    .venv-train/bin/python scripts/stratify_heldout_audit.py \
        v3=runs/audit/heldout_ckpt1770_v1corpus.json \
        v4=runs/audit/heldout_ckpt1767_v2corpus.json \
        C2=runs/audit/heldout_c2_ckpt1767_v2corpus.json

The first ``label=path`` pair is the reference for the paired statistics.
"""

from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any


def load_rows(path: str) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return {r["row_index"]: r for r in payload["rows"]}


def gt_is_self_loop(row: dict[str, Any]) -> bool:
    seq = row["gt_state_sequence"]
    return len(seq) == 1 and seq[0]["from"] == seq[0]["to"]


def emits_self_loop(row: dict[str, Any]) -> bool:
    return any(t[0] == t[1] for t in row["pred_trans"])


def canonical_call(call: dict[str, Any]) -> tuple[Any, ...]:
    """Order-insensitive identity of a tool call, for set comparison."""
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {"_": args}
    return (
        call.get("name"),
        tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in args.items())),
    )


def stratify(label: str, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    vals = list(rows.values())
    loop = [r for r in vals if gt_is_self_loop(r)]
    adv = [r for r in vals if not gt_is_self_loop(r)]
    tooled = [r for r in vals if r["n_gt_tools"] > 0]
    untooled = [r for r in vals if r["n_gt_tools"] == 0]

    name_hit = full_hit = 0
    for r in tooled:
        gt_names = {c.get("name") for c in r["gt_tool_calls"]}
        pred_names = {c.get("name") for c in r["pred_tools"]}
        if gt_names & pred_names:
            name_hit += 1
        gt_calls = {canonical_call(c) for c in r["gt_tool_calls"]}
        pred_calls = {canonical_call(c) for c in r["pred_tools"]}
        if gt_calls & pred_calls:
            full_hit += 1

    return {
        "label": label,
        "n": len(vals),
        "composite": mean(r["composite"] for r in vals),
        "state_acc": mean(r["state_acc"] for r in vals),
        "tool_f1": mean(r["tool_f1"] for r in vals),
        "task": mean(r["task"] for r in vals),
        "loop_n": len(loop),
        "loop_state_acc": mean(r["state_acc"] for r in loop),
        "loop_emits_loop": mean(emits_self_loop(r) for r in loop),
        "loop_tool_f1": mean(r["tool_f1"] for r in loop),
        "adv_n": len(adv),
        "adv_state_acc": mean(r["state_acc"] for r in adv),
        "adv_spurious_loop": mean(emits_self_loop(r) for r in adv),
        "adv_tool_f1": mean(r["tool_f1"] for r in adv),
        "tooled_n": len(tooled),
        "tooled_tool_f1": mean(r["tool_f1"] for r in tooled),
        "tooled_no_call": mean(r["n_pred_tools"] == 0 for r in tooled),
        "tooled_name_hit": name_hit / len(tooled),
        "tooled_full_hit": full_hit / len(tooled),
        "untooled_tool_f1": mean(r["tool_f1"] for r in untooled),
        "untooled_spurious": mean(r["n_pred_tools"] > 0 for r in untooled),
        "tools_per_row": mean(r["n_pred_tools"] for r in vals),
        "rows_zero_tools": sum(r["n_pred_tools"] == 0 for r in vals),
    }


def paired_stats(
    ref: dict[int, dict[str, Any]],
    new: dict[int, dict[str, Any]],
    key: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    keys = sorted(set(ref) & set(new))
    deltas = [new[k][key] - ref[k][key] for k in keys]
    n = len(deltas)
    rng = random.Random(seed)
    boot = sorted(mean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))

    better = sum(d > 0 for d in deltas)
    worse = sum(d < 0 for d in deltas)
    discordant = better + worse
    if discordant:
        tail = sum(comb(discordant, i) for i in range(min(better, worse) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0

    return {
        "n_paired": n,
        "mean_delta": mean(deltas),
        "ci_lo": boot[int(0.025 * n_boot)],
        "ci_hi": boot[int(0.975 * n_boot)],
        "better": better,
        "worse": worse,
        "tied": n - discordant,
        "sign_p": p_value,
    }


def main() -> int:
    specs = [arg.split("=", 1) for arg in sys.argv[1:]]
    if len(specs) < 1 or any(len(s) != 2 for s in specs):
        print(__doc__)
        return 2

    data = {label: load_rows(path) for label, path in specs}
    blocks = [stratify(label, rows) for label, rows in data.items()]

    def line(title: str, key: str, fmt: str = "{:.4f}") -> None:
        cells = "  ".join(f"{fmt.format(b[key]):>12}" for b in blocks)
        print(f"{title:<34}{cells}")

    header = "  ".join(f"{b['label']:>12}" for b in blocks)
    print(f"\n{'':<34}{header}")
    print("-" * (34 + 14 * len(blocks)))

    print(f"OVERALL (n={blocks[0]['n']})")
    line("  composite", "composite")
    line("  state_acc (w .4)", "state_acc")
    line("  tool_f1 (w .4)", "tool_f1")
    line("  task (w .2)", "task")

    print(f"\nGT SELF-LOOP rows (n={blocks[0]['loop_n']})")
    line("  state_acc", "loop_state_acc")
    line("  emits a self-loop", "loop_emits_loop")
    line("  tool_f1", "loop_tool_f1")

    print(f"\nGT ADVANCES rows (n={blocks[0]['adv_n']})")
    line("  state_acc", "adv_state_acc")
    line("  spurious self-loop", "adv_spurious_loop")
    line("  tool_f1", "adv_tool_f1")

    print("\nTOOL CALLING")
    print(f"  -- rows REQUIRING a tool (n={blocks[0]['tooled_n']})")
    line("    tool_f1 (real ability)", "tooled_tool_f1")
    line("    emits NO tool call", "tooled_no_call")
    line("    right tool name (any)", "tooled_name_hit")
    line("    exact call (name+args)", "tooled_full_hit")
    print(f"  -- rows requiring NO tool (n={blocks[0]['n'] - blocks[0]['tooled_n']})")
    line("    tool_f1 (silence credit)", "untooled_tool_f1")
    line("    spurious tool call", "untooled_spurious")
    print("  -- corpus-wide")
    line("    pred tool calls / row", "tools_per_row")
    line("    rows with zero tool calls", "rows_zero_tools", "{:d}")

    ref_label = specs[0][0]
    ref_rows = data[ref_label]
    print(f"\n\nPAIRED vs {ref_label} (10k bootstrap, exact sign test)")
    for label, rows in list(data.items())[1:]:
        for key in ("composite", "state_acc", "tool_f1"):
            s = paired_stats(ref_rows, rows, key)
            sig = "" if s["ci_lo"] <= 0 <= s["ci_hi"] else "  *"
            print(
                f"  {label} {key:<11} delta={s['mean_delta']:+.4f}  "
                f"CI95=[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
                f"{label}-better={s['better']} worse={s['worse']} tied={s['tied']}  "
                f"p={s['sign_p']:.2g}{sig}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
