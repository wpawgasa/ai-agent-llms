"""Task A dataset quality profiler — spec conformance, distributions, and diagnostics.

This complements :mod:`llm_workflow_agents.data.data_validator`. ``data_validator``
answers *"is each sample structurally well-formed?"* (required fields, graph shape,
continuity, reachability, tool-state coherence). This module answers the next two
questions a reviewer asks:

1. **Spec conformance** — does the dataset match the *intended complexity tier*?
   Compares each sample against the canonical :data:`COMPLEXITY_SPECS` (number of
   spine states, tool-count floor, chain depth, loop/back-edge count, branching).

2. **Qualification diagnostics** — distributions a reviewer eyeballs (behavior mix,
   language split, inbound/outbound, self-loop share, arrow-glyph consistency,
   tool-response→later-call propagation) plus a *structural-ceiling* analysis that
   distinguishes a generator undershoot from a hard, domain-imposed impossibility.

It is deliberately read-only and dependency-light (stdlib + the project's own
registries) so it can be run as a CLI by the ``dataset-verifier`` subagent:

    python -m llm_workflow_agents.data.quality_profiler <file.jsonl> [--json]

The CLI prints a human report by default, or a machine-readable JSON blob with
``--json`` (used by the agent to drive its qualitative pass).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY

# Accept either the unicode arrow (most data) or the ASCII fallback.
_STATE_RE = re.compile(
    r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:->|→)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# An identifier-looking token: uppercase prefix + a digit (e.g. ACC987654, CLM-5544,
# RES-889922). Used to detect a value produced by a tool feeding a later tool call.
_ID_RE = re.compile(r"[A-Z]{2,}[-_]?\w*\d[\w-]*")
_ERROR_RE = re.compile(
    r"error|fail|unavailable|invalid|ขัดข้อง"  # TH "ขัดข้อง"
    r"|ไม่สำเร็จ",  # TH "ไม่สำเร็จ"
    re.IGNORECASE,
)

# Canonical (maximum attainable) state count per domain, from the registry.
_DOMAIN_CEILING: dict[str, int] = {n: len(d.states) for n, d in DOMAIN_REGISTRY.items()}


@dataclass
class ProfileReport:
    """Structured output of :func:`profile_task_a`."""

    path: str = ""
    level: str = ""
    total: int = 0
    # Hard structural defects (should be zero). Sample-id-prefixed strings.
    defects: list[str] = field(default_factory=list)
    # Spec conformance: samples violating each tier expectation.
    spec_violations: list[str] = field(default_factory=list)
    # Distributions / diagnostics (informational).
    distributions: dict[str, Any] = field(default_factory=dict)
    # Structural-ceiling notes: when a spec floor is unreachable for a domain.
    ceiling_notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are zero hard structural defects."""
        return not self.defects


def defective_conversation_ids(rep: ProfileReport) -> set[str]:
    """Conversation IDs with at least one hard defect.

    Hard defects recorded by :func:`profile_task_a` are ``"<conversation_id>: ..."``
    strings; file-level defects use a ``"file: ..."`` prefix. This isolates the
    per-sample id parsing so callers (e.g. the target-driven generation loop) can
    filter unqualified samples without re-implementing the prefix contract.
    """
    return {
        d.split(":", 1)[0].strip()
        for d in rep.defects
        if ":" in d and not d.startswith("file:")
    }


def _back_edges(graph: dict[str, Any]) -> list[tuple[str, str]]:
    """Transitions that jump to an earlier-or-equal state index (loops), excluding
    self-loops. State order is the declared ``states`` list order."""
    order = {st: i for i, st in enumerate(graph.get("states", []))}
    out = []
    for t in graph.get("transitions", []):
        fr, to = t.get("from", ""), t.get("to", "")
        if fr != to and order.get(to, 1 << 30) <= order.get(fr, -1):
            out.append((fr, to))
    return out


def _propagation_hops(messages: list[dict[str, Any]]) -> int:
    """Count tool-call argument values that were *produced* by an earlier tool
    response and were not supplied by the user — i.e. real chain propagation."""
    produced: set[str] = set()
    user_supplied: set[str] = set()
    hops = 0
    for m in messages:
        content = m.get("content", "")
        role = m.get("role")
        if role == "tool":
            produced.update(_ID_RE.findall(content))
        elif role == "user":
            user_supplied.update(_ID_RE.findall(content))
        elif role == "assistant":
            for tc in _TOOL_RE.finditer(content):
                try:
                    call = json.loads(tc.group(1))
                except json.JSONDecodeError:
                    continue
                args = json.dumps(call.get("arguments", {}))
                for v in _ID_RE.findall(args):
                    if v in produced and v not in user_supplied:
                        hops += 1
    return hops


def profile_task_a(path: Path) -> ProfileReport:
    """Profile a Task A workflow JSONL against its declared complexity tier."""
    samples = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    rep = ProfileReport(path=str(path), total=len(samples))
    if not samples:
        rep.defects.append("file: no samples")
        return rep

    levels = {s.get("complexity_level") for s in samples}
    rep.level = ", ".join(sorted(x for x in levels if x))

    behav = Counter()
    lang = Counter()
    initiator = Counter()
    source = Counter()
    domains = Counter()
    ntools = Counter()
    nstates: list[int] = []
    arrow_uni = arrow_ascii = 0
    trans_total = self_loops = undeclared = 0
    prop_samples = 0
    max_hops = 0
    recovery_samples = 0
    domain_produced: dict[str, int] = defaultdict(int)  # max states produced per domain

    for s in samples:
        cid = s.get("conversation_id", "?")
        lvl = s.get("complexity_level", "")
        spec = COMPLEXITY_SPECS.get(ComplexityLevel(lvl)) if lvl in ComplexityLevel.__members__ else None
        wg = s.get("workflow_graph", {})
        states = set(wg.get("states", []))
        declared = {(t.get("from"), t.get("to")) for t in wg.get("transitions", [])}
        gt = s.get("ground_truth", {})
        msgs = s.get("messages", [])

        behav[s.get("user_behavior")] += 1
        lang[s.get("language")] += 1
        initiator[s.get("conversation_initiator")] += 1
        source[s.get("generation_source")] += 1
        dom = s.get("domain", "?")
        domains[dom] += 1
        ntools[s.get("num_tools")] += 1
        ns = s.get("num_states", len(states))
        nstates.append(ns)
        domain_produced[dom] = max(domain_produced[dom], ns)

        # --- spec conformance (informational thresholds, not hard defects) ---
        if spec is not None:
            lo, hi = spec.target_path_len
            if not (lo <= ns <= hi):
                rep.spec_violations.append(
                    f"{cid}: num_states={ns} outside {lvl} target_path_len {spec.target_path_len}"
                )
            nt = s.get("num_tools", 0)
            if nt < spec.num_tools:
                rep.spec_violations.append(
                    f"{cid}: num_tools={nt} below {lvl} floor {spec.num_tools}"
                )
            cd = s.get("chain_depth", 0)
            if cd < spec.chain_depth:
                rep.spec_violations.append(
                    f"{cid}: chain_depth={cd} below {lvl} spec {spec.chain_depth}"
                )
            loops = len(_back_edges(wg))
            if spec.num_loops[0] > 0 and loops < spec.num_loops[0]:
                rep.spec_violations.append(
                    f"{cid}: {loops} back-edges below {lvl} num_loops floor {spec.num_loops[0]}"
                )

        # --- hard structural defects (should be zero) ---
        if wg.get("initial") and wg["initial"] not in states:
            rep.defects.append(f"{cid}: initial '{wg['initial']}' not in states")
        for term in wg.get("terminal", []):
            if term not in states:
                rep.defects.append(f"{cid}: terminal '{term}' not in states")
        for t in wg.get("transitions", []):
            if t.get("from") not in states:
                rep.defects.append(f"{cid}: edge from '{t.get('from')}' not in states")
            if t.get("to") not in states:
                rep.defects.append(f"{cid}: edge to '{t.get('to')}' not in states")

        # ground-truth transition legality (self-loops are always legal)
        for tr in gt.get("state_sequence", []):
            trans_total += 1
            fr, to = tr.get("from"), tr.get("to")
            if fr == to:
                self_loops += 1
            elif (fr, to) not in declared:
                undeclared += 1
                rep.defects.append(f"{cid}: GT transition {fr}->{to} not a declared edge")

        # message STATE sequence must equal ground-truth state_sequence
        msg_states: list[tuple[str, str]] = []
        for m in msgs:
            if m.get("role") == "assistant":
                for mt in _STATE_RE.finditer(m.get("content", "")):
                    msg_states.append((mt.group(1), mt.group(2)))
                    seg = m["content"][mt.start() : mt.end()]
                    arrow_uni += "→" in seg
                    arrow_ascii += "->" in seg
        gt_seq = [(t.get("from"), t.get("to")) for t in gt.get("state_sequence", [])]
        if msg_states != gt_seq:
            rep.defects.append(
                f"{cid}: message STATE seq (len {len(msg_states)}) != "
                f"ground_truth.state_sequence (len {len(gt_seq)})"
            )

        # role sequencing must match the declared initiator
        roles = [m.get("role") for m in msgs]
        if roles and roles[0] != "system":
            rep.defects.append(f"{cid}: first role '{roles[0]}' != system")
        init = s.get("conversation_initiator")
        if len(roles) > 1:
            if init == "agent" and roles[1] != "assistant":
                rep.defects.append(f"{cid}: outbound but messages[1]='{roles[1]}'")
            if init == "user" and roles[1] != "user":
                rep.defects.append(f"{cid}: inbound but messages[1]='{roles[1]}'")

        # tool_call JSON validity + schema membership
        schema_names = {
            (ts.get("function") or ts).get("name") for ts in s.get("tool_schemas", [])
        }
        for m in msgs:
            for tc in _TOOL_RE.finditer(m.get("content", "")):
                try:
                    obj = json.loads(tc.group(1))
                except json.JSONDecodeError as e:
                    rep.defects.append(f"{cid}: invalid tool_call JSON: {e}")
                    continue
                name = obj.get("name")
                if name and name not in schema_names:
                    rep.defects.append(f"{cid}: tool_call '{name}' not in tool_schemas")

        # propagation + recovery diagnostics
        hops = _propagation_hops(msgs)
        if hops:
            prop_samples += 1
            max_hops = max(max_hops, hops)
        if any(m.get("role") == "tool" and _ERROR_RE.search(m.get("content", "")) for m in msgs):
            recovery_samples += 1

    # --- structural-ceiling analysis (distinguishes undershoot vs impossibility) ---
    # Done once per (level, domain) against canonical registry ceilings.
    seen_pairs: set[tuple[str, str]] = set()
    for s in samples:
        lvl = s.get("complexity_level", "")
        dom = s.get("domain", "?")
        if (lvl, dom) in seen_pairs:
            continue
        seen_pairs.add((lvl, dom))
        spec = COMPLEXITY_SPECS.get(ComplexityLevel(lvl)) if lvl in ComplexityLevel.__members__ else None
        if spec is None:
            continue
        floor, top = spec.target_path_len
        ceiling = _DOMAIN_CEILING.get(dom)
        if ceiling is None:
            continue
        if ceiling < floor:
            rep.ceiling_notes.append(
                f"{lvl}/{dom}: IMPOSSIBLE — canonical graph has {ceiling} states, "
                f"below the {lvl} floor of {floor}. Exclude this domain from {lvl} or extend the graph."
            )
        elif ceiling < top:
            rep.ceiling_notes.append(
                f"{lvl}/{dom}: capped — canonical max is {ceiling} states; the upper "
                f"{lvl} target ({top}) is unreachable. Produced max so far: {domain_produced.get(dom)}."
            )
        elif domain_produced.get(dom, 0) < floor:
            rep.ceiling_notes.append(
                f"{lvl}/{dom}: UNDERSHOOT — canonical graph allows {ceiling} states "
                f"(>= floor {floor}) but generator produced only {domain_produced.get(dom)}. "
                f"Subgraph selection, not a hard ceiling."
            )

    rep.distributions = {
        "num_states": {"min": min(nstates), "max": max(nstates), "values": sorted(set(nstates))},
        "num_tools": dict(sorted((k, v) for k, v in ntools.items() if k is not None)),
        "language": dict(lang),
        "conversation_initiator": dict(initiator),
        "generation_source": dict(source),
        "user_behavior": dict(behav),
        "domains": dict(domains),
        "arrows": {"unicode": arrow_uni, "ascii": arrow_ascii},
        "self_loops": {
            "count": self_loops,
            "total_transitions": trans_total,
            "pct": round(100 * self_loops / trans_total, 1) if trans_total else 0.0,
        },
        "undeclared_nonloop_transitions": undeclared,
        "tool_chain_propagation": {
            "samples_with_propagation": prop_samples,
            "max_hops_single_convo": max_hops,
        },
        "recovery_samples": recovery_samples,
    }
    return rep


def _format_report(rep: ProfileReport) -> str:
    lines = [
        f"=== Task A Quality Profile: {rep.path} ===",
        f"level(s): {rep.level}    samples: {rep.total}",
        "",
        f"HARD STRUCTURAL DEFECTS: {len(rep.defects)}"
        + ("  ✅ none" if not rep.defects else ""),
    ]
    for d in rep.defects[:40]:
        lines.append(f"  ✗ {d}")
    if len(rep.defects) > 40:
        lines.append(f"  … +{len(rep.defects) - 40} more")

    lines.append("")
    lines.append(f"SPEC-CONFORMANCE VIOLATIONS: {len(rep.spec_violations)}")
    for v in rep.spec_violations[:40]:
        lines.append(f"  • {v}")
    if len(rep.spec_violations) > 40:
        lines.append(f"  … +{len(rep.spec_violations) - 40} more")

    if rep.ceiling_notes:
        lines.append("")
        lines.append("STRUCTURAL-CEILING NOTES:")
        for n in rep.ceiling_notes:
            lines.append(f"  ⚑ {n}")

    lines.append("")
    lines.append("DISTRIBUTIONS:")
    for k, v in rep.distributions.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Task A workflow JSONL file")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    rep = profile_task_a(args.path)
    if args.json:
        print(json.dumps(asdict(rep), ensure_ascii=False, indent=2))
    else:
        print(_format_report(rep))
    # Exit non-zero only on hard structural defects (spec violations are advisory).
    return 1 if rep.defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
