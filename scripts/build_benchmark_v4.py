#!/usr/bin/env python3
"""Build the v4 benchmark: v3 minus the multi-tool conversations, plus clean replacements.

Deterministic, no model calls: it reads the replacements that
``generate_benchmark_replacements.py`` accepted (a DVC-tracked directory) and
its manifest, which names the v3 row each replacement stands in for.

    python scripts/build_benchmark_v4.py \\
        --stratum text  data/output/benchmark/task_a_v3       data/output/benchmark/task_a_v4 \\
        --stratum voice data/output/benchmark/task_a_voice_v2 data/output/benchmark/task_a_voice_v3 \\
        --replacements data/interim/task_a_benchmark_v4_replacements

Every v3 file is copied without the replaced rows, row order otherwise kept;
the accepted replacements for each modality go in their own file,
``replacements_v4.jsonl``. The build fails if a replaced row is missing, if a
replacement's ``conversation_id`` collides with an existing one, or if the row
counts do not add up. See CLAUDE.md R28.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPLACEMENTS_FILE = "replacements_v4.jsonl"


def build(
    strata: list[tuple[str, Path, Path]],
    manifest: list[dict[str, Any]],
    replacements: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Write every stratum; return per-modality counts. Raises on any inconsistency."""
    removed = {m["replaces"] for m in manifest if m.get("replaces") and m.get("accepted")}
    unfilled = [m for m in manifest if not m.get("accepted")]
    if unfilled:
        raise ValueError(f"{len(unfilled)} slot(s) were never filled; refusing to build a partial v4")

    seen_removed: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for modality, source, target in strata:
        target.mkdir(parents=True, exist_ok=True)
        kept = 0
        ids: set[str] = set()
        for file in sorted(source.glob("*.jsonl")):
            lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            out: list[str] = []
            for line_no, line in enumerate(lines, start=1):
                key = f"{source}/{file.name}:{line_no}"
                if key in removed:
                    seen_removed.add(key)
                    continue
                out.append(line)
                ids.add(json.loads(line).get("conversation_id"))
            (target / file.name).write_text("".join(l + "\n" for l in out), encoding="utf-8")
            kept += len(out)

        added = replacements.get(modality, [])
        clashes = sorted({s["conversation_id"] for s in added} & ids)
        if clashes:
            raise ValueError(f"{modality}: replacement ids already in use: {clashes}")
        (target / REPLACEMENTS_FILE).write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in added), encoding="utf-8"
        )
        counts[modality] = {"kept": kept, "added": len(added), "total": kept + len(added)}

    missing = sorted(removed - seen_removed)
    if missing:
        raise ValueError(f"{len(missing)} row(s) to replace were not found, e.g. {missing[:3]}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stratum", nargs=3, action="append", required=True, metavar=("MODALITY", "INPUT_DIR", "OUTPUT_DIR"))
    parser.add_argument("--replacements", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.replacements / "manifest.json").read_text(encoding="utf-8"))
    replacements = {
        modality: [json.loads(l) for l in (args.replacements / f"replacements_{modality}.jsonl").read_text().splitlines() if l.strip()]
        for modality in ("text", "voice")
        if (args.replacements / f"replacements_{modality}.jsonl").exists()
    }
    strata = [(m, Path(i), Path(o)) for m, i, o in args.stratum]
    try:
        counts = build(strata, manifest, replacements)
    except ValueError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
