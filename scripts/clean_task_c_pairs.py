#!/usr/bin/env python3
"""Clean Task C playbook->graph JSONL pairs before fine-tuning.

Validation applied per record:
  1. Required fields present.
  2. `messages` is exactly [system, user, assistant].
  3. The assistant content parses as JSON and equals the parsed `graph` field.
  4. `graph` is structurally valid (mirrors eval/graph_extraction_eval predicate).
  5. `verification` has the expected shape.
Across files, duplicate playbooks (identical user-message text) are dropped,
keeping the first occurrence in sorted-filename order.

Usage:
    python scripts/clean_task_c_pairs.py \\
        --input-dir data/output/sft/task_c \\
        --output-dir data/output/sft/task_c_cleaned
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

_REQUIRED_FIELDS = ("pair_id", "graph_id", "register", "language", "verification", "graph", "messages")
_VERIFICATION_KEYS = {"anchor_coverage", "edge_ref_coverage", "back_extraction"}


def _bfs(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


def _structurally_valid(graph: dict) -> str | None:
    """Mirror of eval.graph_extraction_eval.check_structural_validity. None = valid."""
    nodes = graph.get("nodes", [])
    if not nodes:
        return "no nodes"
    ids = {n.get("id", n.get("name", "")) for n in nodes}

    initial = graph.get("initial_state", "")
    if not initial or initial not in ids:
        return "initial state not in nodes"

    terminals = graph.get("terminal_states", [])
    if not terminals:
        return "no terminal states"
    for ts in terminals:
        if ts not in ids:
            return f"terminal {ts} not in nodes"

    adjacency: dict[str, list[str]] = {nid: [] for nid in ids}
    for edge in graph.get("edges", []):
        src, dst = edge.get("from_state", ""), edge.get("to_state", "")
        if src in adjacency and dst in ids:
            adjacency[src].append(dst)
    reachable = _bfs(initial, adjacency)
    for ts in terminals:
        if ts not in reachable:
            return f"terminal {ts} unreachable"

    rev: dict[str, list[str]] = {nid: [] for nid in ids}
    for edge in graph.get("edges", []):
        src, dst = edge.get("from_state", ""), edge.get("to_state", "")
        if dst in rev and src in ids:
            rev[dst].append(src)
    backward: set[str] = set()
    for ts in terminals:
        backward |= _bfs(ts, rev)
    for nid in ids:
        if nid not in reachable and nid not in backward:
            return f"orphan node {nid}"
    return None


def clean_record(record: dict) -> tuple[dict | None, str | None]:
    """Return (record, None) if the record is valid, else (None, drop_reason)."""
    for fld in _REQUIRED_FIELDS:
        if fld not in record:
            return None, f"missing_field:{fld}"

    msgs = record["messages"]
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        return None, "bad_messages_shape"

    try:
        parsed = json.loads(msgs[2]["content"])
    except (json.JSONDecodeError, TypeError):
        return None, "assistant_not_json"
    if parsed != record["graph"]:
        return None, "assistant_graph_mismatch"

    structural = _structurally_valid(record["graph"])
    if structural is not None:
        return None, f"structural:{structural}"

    verification = record.get("verification")
    if not isinstance(verification, dict) or set(verification) != _VERIFICATION_KEYS:
        return None, "bad_verification_shape"

    return record, None


def _load_records(src: Path) -> list[dict]:
    records: list[dict] = []
    with open(src) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"Error: {src}:{lineno}: {exc}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Task C playbook->graph JSONL pairs.")
    parser.add_argument("--input-dir", required=True, metavar="DIR",
                        help="Directory containing raw *.jsonl files.")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Destination directory for cleaned *.jsonl files.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-file progress.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {input_dir}")
    src_files = sorted(input_dir.glob("*.jsonl"))
    if not src_files:
        sys.exit(f"Error: no *.jsonl files found in {input_dir}")

    seen_playbooks: set[str] = set()
    drop_reasons: Counter[str] = Counter()
    grand_total = grand_kept = grand_dup = 0

    for src in src_files:
        kept_lines: list[str] = []
        for record in _load_records(src):
            grand_total += 1
            cleaned, reason = clean_record(record)
            if cleaned is None:
                drop_reasons[reason] += 1
                continue
            playbook = cleaned["messages"][1].get("content", "")
            if playbook in seen_playbooks:
                grand_dup += 1
                drop_reasons["duplicate_playbook"] += 1
                continue
            seen_playbooks.add(playbook)
            kept_lines.append(json.dumps(cleaned, ensure_ascii=False))
            grand_kept += 1

        if not args.quiet:
            print(f"{src.name}: kept {len(kept_lines)}")
        if not args.dry_run:
            dst = output_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w") as fh:
                fh.write("\n".join(kept_lines))
                if kept_lines:
                    fh.write("\n")

    print("=" * 60)
    print(f"Input files    : {len(src_files)}")
    print(f"Total records  : {grand_total}")
    print(f"Kept           : {grand_kept}")
    print(f"Duplicates     : {grand_dup}")
    for reason, count in sorted(drop_reasons.items()):
        print(f"  dropped [{reason}]: {count}")


if __name__ == "__main__":
    main()
