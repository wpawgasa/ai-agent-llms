"""The generator builds and walks one-tool-per-state graphs correctly.

With ``single_tool_states`` a router must present every option (the model
routes on the customer's request) while each conversation walks one of them.
A domain without routers must build and walk exactly as before.
"""

from __future__ import annotations

import random

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
from llm_workflow_agents.data.generate_workflows import select_subgraph, walk_path
from llm_workflow_agents.data.single_tool_graph import multi_tool_states, split_multi_tool_states


def _router_names(domain) -> set[str]:
    return {e.src for e in domain.edges if e.route}


def _graphs(key: str, level: str, seeds: range):
    domain = split_multi_tool_states(DOMAIN_REGISTRY[key])
    for seed in seeds:
        rng = random.Random(seed)
        graph = select_subgraph(domain, COMPLEXITY_SPECS[level], rng)
        yield domain, graph, rng


def _names(graph) -> dict[str, str]:
    return {s.id: s.name for s in graph.states}


def test_no_state_in_a_built_graph_offers_two_tools() -> None:
    for key in ("sales", "surveys", "banking", "billing_payments", "technical_support", "utilities", "ecommerce"):
        for _, graph, _ in _graphs(key, "L3", range(20)):
            assert all(len(s.tools) <= 1 for s in graph.states), key


def test_an_included_router_lists_every_route_and_every_branch_continues() -> None:
    seen = 0
    for key in ("billing_payments", "technical_support", "ecommerce", "banking"):
        for domain, graph, _ in _graphs(key, "L3", range(30)):
            names = _names(graph)
            for router in _router_names(domain) & set(names.values()):
                seen += 1
                expected = {e.dst for e in domain.edges if e.src == router and e.route}
                routed = {names[t.to_state] for t in graph.transitions if names[t.from_state] == router and t.route}
                assert routed == expected, (key, router)
                for branch in expected:
                    assert any(names[t.from_state] == branch for t in graph.transitions), (key, branch)
    assert seen > 0, "no router was ever included; the test proves nothing"


def test_walks_take_different_routes_and_always_end_at_a_terminal() -> None:
    chosen: set[str] = set()
    for domain, graph, rng in _graphs("billing_payments", "L3", range(60)):
        names = _names(graph)
        path = walk_path(graph, domain, "cooperative", "service", rng)
        assert path and path[-1].to_state in graph.terminal_states
        for step in path:
            if step.route:
                chosen.add(names[step.to_state])
    assert len(chosen) > 1, f"every walk took the same route: {chosen}"


def test_a_domain_without_routers_builds_and_walks_as_before() -> None:
    # The router step draws no randomness, so the same seed gives the same
    # graph and the same walk whether or not the rewrite ran.
    key = next(k for k, d in DOMAIN_REGISTRY.items() if not multi_tool_states(d))
    domain = DOMAIN_REGISTRY[key]
    assert split_multi_tool_states(domain) is domain
    for seed in range(10):
        a, b = random.Random(seed), random.Random(seed)
        ga, gb = select_subgraph(domain, COMPLEXITY_SPECS["L3"], a), select_subgraph(domain, COMPLEXITY_SPECS["L3"], b)
        assert ga.to_dict() == gb.to_dict()
        pa = walk_path(ga, domain, "cooperative", "service", a)
        pb = walk_path(gb, domain, "cooperative", "service", b)
        assert [(t.from_state, t.to_state) for t in pa] == [(t.from_state, t.to_state) for t in pb]
