"""Task C rendering verification: deterministic faithfulness checks.

Given a rendered playbook and its gold graph (name-keyed interchange shape), this
module derives per-rendering state IDs from mention order, checks the anchor
contract (every canonical state name present verbatim), tool coverage, the
initial-state-first rule, and edge-reference coverage (see
docs/data_generation_recipes_task_c.md, §Faithfulness Verification).

Segmentation note: segments split on blank lines and headings only, NOT on list
items — a state's heading and its transition bullets belong to one segment, which
is what lets the programmatic `state_script` register pass the edge-reference gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_EDGE_COVERAGE_THRESHOLD = 0.90
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s|\*\*|\d+(?:\.\d+)*[.)]?\s)")


def _anchor_re(name: str) -> re.Pattern[str]:
    """Boundary-safe, case-sensitive matcher for a SCREAMING_SNAKE state name."""
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def find_anchor_occurrences(text: str, state_names: Sequence[str]) -> list[tuple[int, str]]:
    """Return (position, state_name) for every anchor occurrence, sorted by position."""
    occ: list[tuple[int, str]] = []
    for name in state_names:
        for match in _anchor_re(name).finditer(text):
            occ.append((match.start(), name))
    return sorted(occ)


def assign_state_ids(graph_dict: dict[str, Any], playbook: str) -> dict[str, str] | None:
    """Map each state name to S<n> by first-mention order. None if any state is unanchored."""
    names = list(graph_dict["states"])
    first: dict[str, int] = {}
    for pos, name in find_anchor_occurrences(playbook, names):
        first.setdefault(name, pos)
    if len(first) != len(names):
        return None
    ordered = sorted(names, key=lambda n: first[n])
    return {name: f"S{i + 1}" for i, name in enumerate(ordered)}


def graph_to_eval_shape(graph_dict: dict[str, Any], state_ids: dict[str, str]) -> dict[str, Any]:
    """Convert the name-keyed interchange graph to the id-keyed eval schema."""
    nodes = [
        {
            "id": state_ids[sd["name"]],
            "name": sd["name"],
            "tools": list(sd.get("tools", [])),
            "entry_actions": list(sd.get("entry_actions", [])),
        }
        for sd in graph_dict["state_details"]
    ]
    nodes.sort(key=lambda n: int(n["id"][1:]))
    edges = [
        {
            "from_state": state_ids[t["from"]],
            "to_state": state_ids[t["to"]],
            "condition": t.get("condition", ""),
            "priority": t.get("priority", 0),
        }
        for t in graph_dict["transitions"]
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "initial_state": state_ids[graph_dict["initial"]],
        "terminal_states": [state_ids[t] for t in graph_dict["terminal"]],
    }


@dataclass
class EdgeRefResult:
    coverage: float
    failed_edges: list[tuple[str, str]]
    branch_ok: bool


def _segments(text: str) -> list[str]:
    """Split into segments at blank lines and heading lines (not list items)."""
    segs: list[list[str]] = [[]]
    for line in text.splitlines():
        if not line.strip():
            segs.append([])
        elif _HEADING_RE.match(line):
            segs.append([line])
        else:
            segs[-1].append(line)
    return ["\n".join(s) for s in segs if s]


def check_edge_references(playbook: str, graph_dict: dict[str, Any]) -> EdgeRefResult:
    """Check that each edge's endpoints are co-mentioned (same or adjacent anchored segment)."""
    names = list(graph_dict["states"])
    anchored: list[set[str]] = []
    for seg in _segments(playbook):
        present = {n for n in names if _anchor_re(n).search(seg)}
        if present:
            anchored.append(present)

    edges = graph_dict["transitions"]
    failed: list[tuple[str, str]] = []
    branch_ok = True
    for edge in edges:
        src, dst = edge["from"], edge["to"]
        ok = any(
            src in anchored[i] and (dst in anchored[i] or (i + 1 < len(anchored) and dst in anchored[i + 1]))
            for i in range(len(anchored))
        )
        if not ok:
            failed.append((src, dst))
            if edge.get("priority", 0) > 0:
                branch_ok = False

    coverage = 1.0 if not edges else (len(edges) - len(failed)) / len(edges)
    return EdgeRefResult(coverage=coverage, failed_edges=failed, branch_ok=branch_ok)


def check_distractor_purity(
    distractors: Iterable[str],
    state_names: Iterable[str],
    tool_names: Iterable[str],
) -> list[str]:
    """Return distractor texts that leak any state name or tool name."""
    terms = list(state_names) + list(tool_names)
    offending: list[str] = []
    for d in distractors:
        if any(_anchor_re(t).search(d) for t in terms):
            offending.append(d)
    return offending


@dataclass
class VerificationReport:
    accepted: bool
    corrections: list[str]
    anchor_coverage: float
    tool_coverage: float
    edge_ref_coverage: float
    initial_first: bool
    state_ids: dict[str, str] | None
    back_extraction: dict[str, Any] | None = None


def verify_rendering(
    playbook: str,
    graph_dict: dict[str, Any],
    tool_names: Iterable[str],
) -> VerificationReport:
    """Run the deterministic gates cheapest-first, itemizing corrections for repair."""
    names = list(graph_dict["states"])
    tool_list = list(tool_names)
    corrections: list[str] = []

    anchored = {n for n in names if _anchor_re(n).search(playbook)}
    for name in names:
        if name not in anchored:
            corrections.append(f"missing state anchor: {name}")
    anchor_coverage = len(anchored) / len(names) if names else 1.0

    present_tools = [t for t in tool_list if _anchor_re(t).search(playbook)]
    for tool in tool_list:
        if tool not in present_tools:
            corrections.append(f"missing tool mention: {tool}")
    tool_coverage = len(present_tools) / len(tool_list) if tool_list else 1.0

    occ = find_anchor_occurrences(playbook, names)
    initial = graph_dict["initial"]
    initial_anchored = initial in anchored
    initial_first = bool(initial_anchored and occ and occ[0][1] == initial)
    if initial_anchored and not initial_first:
        corrections.append("initial state must be introduced before any other state")

    edge = check_edge_references(playbook, graph_dict)
    if edge.coverage < _EDGE_COVERAGE_THRESHOLD or not edge.branch_ok:
        for src, dst in edge.failed_edges:
            corrections.append(f"edge reference missing: {src} -> {dst}")

    state_ids = assign_state_ids(graph_dict, playbook)
    return VerificationReport(
        accepted=not corrections,
        corrections=corrections,
        anchor_coverage=anchor_coverage,
        tool_coverage=tool_coverage,
        edge_ref_coverage=edge.coverage,
        initial_first=initial_first,
        state_ids=state_ids,
    )
