"""One tool per state: split every multi-tool state of a domain graph.

A state that offers two tools leaves the model guessing which to call first,
and a state instruction like "satisfaction or NPS" cannot say how many calls
finish it (CLAUDE.md R28). Each multi-tool state declares a ``tool_mode``:

- ``sequence``: every tool is called, in order -> one state per tool, chained
  by tool_success.
- ``choice``: the customer's request decides which tool -> a text-only router
  that keeps the state's name, with a ``route`` edge to one state per tool.
"""

from __future__ import annotations

import dataclasses

import pytest

from llm_workflow_agents.data.domain_registry import (
    DOMAIN_REGISTRY,
    DomainSpec,
    Edge,
    StateNode,
    validate_domain,
)
from llm_workflow_agents.data.single_tool_graph import (
    multi_tool_states,
    split_multi_tool_states,
)


def _state(domain: DomainSpec, name: str) -> StateNode:
    return next(s for s in domain.states if s.name == name)


def _edges(domain: DomainSpec, src: str | None = None, dst: str | None = None) -> list[Edge]:
    return [e for e in domain.edges if (src is None or e.src == src) and (dst is None or e.dst == dst)]


class TestEveryRegisteredDomain:

    @pytest.mark.parametrize("key", sorted(DOMAIN_REGISTRY))
    def test_split_leaves_at_most_one_tool_per_state_and_stays_valid(self, key: str) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY[key])
        assert multi_tool_states(split) == []
        validate_domain(split)

    @pytest.mark.parametrize("key", sorted(DOMAIN_REGISTRY))
    def test_split_loses_no_tool_and_no_state_name(self, key: str) -> None:
        original = DOMAIN_REGISTRY[key]
        split = split_multi_tool_states(original)
        assert {t for s in split.states for t in s.tools} == {t for s in original.states for t in s.tools}
        assert {s.name for s in original.states} <= {s.name for s in split.states}

    def test_every_multi_tool_state_declares_its_mode(self) -> None:
        for domain in DOMAIN_REGISTRY.values():
            for state in domain.states:
                if len(state.tools) > 1:
                    assert state.tool_mode in ("sequence", "choice"), f"{domain.name}.{state.name}"


class TestSequence:

    def test_create_proposal_becomes_quote_then_send(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["sales"])
        assert _state(split, "CREATE_PROPOSAL").tools == ("create_quote",)
        assert _state(split, "SEND_PROPOSAL").tools == ("send_proposal",)
        (chain,) = _edges(split, "CREATE_PROPOSAL", "SEND_PROPOSAL")
        assert (chain.trigger, chain.optional) == ("tool_success", False)

    def test_success_exits_leave_from_the_last_state_only(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["sales"])
        assert [e.src for e in _edges(split, dst="CLOSE_DEAL") if e.trigger == "tool_success"] == ["SEND_PROPOSAL"]

    def test_other_exits_stay_available_from_every_step(self) -> None:
        # A follow-up can be scheduled whether or not the proposal was sent yet.
        split = split_multi_tool_states(DOMAIN_REGISTRY["sales"])
        assert {"CREATE_PROPOSAL", "SEND_PROPOSAL"} <= {e.src for e in _edges(split, dst="FOLLOW_UP")}

    def test_sequence_order_overrides_the_listed_order(self) -> None:
        # banking lists (report_fraud, block_card) but blocks the card first.
        split = split_multi_tool_states(DOMAIN_REGISTRY["banking"])
        assert _state(split, "FRAUD_INVESTIGATION").tools == ("block_card",)
        assert _edges(split, "FRAUD_INVESTIGATION", "REPORT_FRAUD")

    def test_each_step_says_which_step_it_is(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["sales"])
        assert "step 2 of 2" in _state(split, "SEND_PROPOSAL").instruction


class TestChoice:

    def test_router_keeps_the_name_and_loses_the_tools(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["billing_payments"])
        router = _state(split, "APPLY_ADJUSTMENT")
        assert router.tools == ()
        assert "[ISSUE_REFUND]" in router.instruction and "[WAIVE_LATE_FEE]" in router.instruction

    def test_router_routes_to_one_state_per_tool(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["billing_payments"])
        routes = [e for e in _edges(split, "APPLY_ADJUSTMENT") if e.route]
        assert {e.dst for e in routes} == {"ISSUE_REFUND", "WAIVE_LATE_FEE"}
        assert _state(split, "ISSUE_REFUND").tools == ("issue_refund",)

    def test_a_router_has_exactly_one_spine_successor(self) -> None:
        # validate_domain requires it; the walk ignores the flag at a router and
        # picks among its routes uniformly.
        for domain in DOMAIN_REGISTRY.values():
            split = split_multi_tool_states(domain)
            for state in split.states:
                if any(e.route for e in _edges(split, state.name)):
                    spine = [e for e in _edges(split, state.name) if not e.optional]
                    assert len(spine) == 1, f"{domain.name}.{state.name}: {spine}"

    def test_each_branch_carries_the_tool_exits(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["billing_payments"])
        assert {e.src for e in _edges(split, dst="CONFIRM_ACTION") if e.trigger == "tool_success"} >= {
            "ISSUE_REFUND", "WAIVE_LATE_FEE",
        }

    def test_non_tool_exits_stay_on_the_router(self) -> None:
        split = split_multi_tool_states(DOMAIN_REGISTRY["healthcare"])
        assert [e.src for e in _edges(split, dst="CONFIRM_DETAILS") if e.trigger == "always"] == ["PROCESS_REQUEST"]

    def test_incoming_edges_still_reach_the_router(self) -> None:
        original = DOMAIN_REGISTRY["billing_payments"]
        split = split_multi_tool_states(original)
        assert {e.src for e in _edges(original, dst="APPLY_ADJUSTMENT")} <= {e.src for e in _edges(split, dst="APPLY_ADJUSTMENT")}


class TestNaming:

    def test_a_tool_split_in_two_states_gets_distinct_state_names(self) -> None:
        # banking offers block_card in PROCESS_REQUEST (choice) and in
        # FRAUD_INVESTIGATION (sequence); the two new states must not collide.
        split = split_multi_tool_states(DOMAIN_REGISTRY["banking"])
        names = [s.name for s in split.states]
        assert len(names) == len(set(names))
        assert sum(1 for s in split.states if s.tools == ("block_card",)) == 2


class TestUnchanged:

    def test_a_domain_without_multi_tool_states_is_returned_as_is(self) -> None:
        clean = next(d for d in DOMAIN_REGISTRY.values() if not multi_tool_states(d))
        assert split_multi_tool_states(clean) is clean


class TestValidation:

    def test_a_multi_tool_state_without_a_mode_is_rejected(self) -> None:
        domain = DOMAIN_REGISTRY["sales"]
        states = tuple(
            dataclasses.replace(s, tool_mode="") if s.name == "CREATE_PROPOSAL" else s for s in domain.states
        )
        with pytest.raises(ValueError, match="tool_mode"):
            validate_domain(dataclasses.replace(domain, states=states))
