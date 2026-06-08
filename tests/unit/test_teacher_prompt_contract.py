"""The teacher-prompt WORKFLOW CONTRACT must exactly match what the repair loop enforces."""

from __future__ import annotations

import random

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
import llm_workflow_agents.data.generate_workflows as gw

# L1–L3 draw from all 18 domains, so this sweep covers every domain. L4/L5 are
# restricted to specific high-state-count domains (see _select_domain); the
# parity property is structural and level-independent, so L1–L3 is sufficient.
_LEVELS = [ComplexityLevel.L1, ComplexityLevel.L2, ComplexityLevel.L3]


def _repair_loop_sets(workflow):
    """Recompute exactly what generate_workflow_dataset's repair loop accepts."""
    allowed = {s.name: set(s.tools) for s in workflow.states}
    id_to_name = {s.id: s.name for s in workflow.states}
    valid = {
        (id_to_name.get(t.from_state, t.from_state),
         id_to_name.get(t.to_state, t.to_state))
        for t in workflow.transitions
    } | {
        (id_to_name.get(sid, sid), id_to_name.get(sid, sid))
        for sid in {t.from_state for t in workflow.transitions}
    }
    legal_edges = {(a, b) for (a, b) in valid if a != b}
    return legal_edges, allowed


def test_contract_edges_match_repair_loop():
    for level in _LEVELS:
        spec = COMPLEXITY_SPECS[level]
        for key, dom in DOMAIN_REGISTRY.items():
            rng = random.Random(0)
            wf = gw.select_subgraph(dom, spec, rng, "service")
            c_edges, _c_tools = gw._workflow_contract(wf)
            legal, _allowed = _repair_loop_sets(wf)
            assert c_edges == legal, f"{key}/{level}: edge mismatch {c_edges ^ legal}"


def test_contract_tools_match_repair_loop():
    for level in _LEVELS:
        spec = COMPLEXITY_SPECS[level]
        for key, dom in DOMAIN_REGISTRY.items():
            rng = random.Random(0)
            wf = gw.select_subgraph(dom, spec, rng, "service")
            _c_edges, c_tools = gw._workflow_contract(wf)
            _legal, allowed = _repair_loop_sets(wf)
            as_sets = {k: set(v) for k, v in c_tools.items()}
            assert as_sets == allowed, f"{key}/{level}: tool map mismatch"


def test_rendered_prompt_contains_full_contract():
    spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
    dom = DOMAIN_REGISTRY["account_management"]
    rng = random.Random(0)
    wf = gw.select_subgraph(dom, spec, rng, "service")
    tools = [t for t in dom.tools]
    prompt = gw._build_teacher_prompt(
        wf, tools, "cooperative", spec, dom, "en", "service",
    )
    assert "WORKFLOW CONTRACT" in prompt
    edges, tools_by_state = gw._workflow_contract(wf)
    # Every legal edge appears as an "X → Y" line.
    for src, dst in edges:
        assert f"{src} → {dst}" in prompt, f"missing edge line {src} → {dst}"
    # Every state's tool permission appears verbatim.
    for s in wf.states:
        st = tools_by_state[s.name]
        if st:
            assert f"{s.name}: {', '.join(st)}" in prompt, f"missing tools for {s.name}"
        else:
            assert f"{s.name}: (text only" in prompt, f"missing text-only line for {s.name}"
