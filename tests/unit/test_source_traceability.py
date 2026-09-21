"""Data-quality checks for Task A conversations: can every value be traced?

These checks exist because the Phase 1 benchmark and the SFT corpus score or
train a model on values it could never have produced (CLAUDE.md R28): gold tool
calls whose required arguments appear nowhere in the conversation or the
prompt, and assistant prose that states specific facts no tool returned.
"""

from __future__ import annotations

import json

from llm_workflow_agents.data.source_traceability import (
    collect_identifiers,
    find_identifier_reuse,
    find_mergeable_stay_pairs,
    find_multi_tool_states,
    find_orphan_tool_results,
    find_unsourced_argument_values,
    find_unsourced_facts,
    is_identifier_shaped,
    value_is_sourced,
)


def _call(state: str, name: str, args: dict, *, to: str | None = None, prose: str = "") -> dict:
    marker = f"[STATE: {state} → {to or state}]"
    body = (prose + "\n") if prose else ""
    return {
        "role": "assistant",
        "content": f"{marker}\n{body}<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>",
    }


def _say(frm: str, to: str, text: str) -> dict:
    return {"role": "assistant", "content": f"[STATE: {frm} → {to}] {text}"}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _tool(payload: dict) -> dict:
    return {"role": "tool", "content": json.dumps(payload)}


def _schema(name: str, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in required}, "required": required},
        },
    }


# --------------------------------------------------------------------------- helpers


class TestIsIdentifierShaped:

    def test_prefixed_codes_are_identifiers(self) -> None:
        for value in ("INT-5541", "CUST-882", "POL-998877", "ACC987654", "PLAN_50GB"):
            assert is_identifier_shaped(value), value

    def test_amounts_names_and_dates_are_not(self) -> None:
        for value in ("1000000", "Bangkok", "2024-11-25", "premium", "5", "Chiang Mai"):
            assert not is_identifier_shaped(value), value


class TestValueIsSourced:

    def test_verbatim_match(self) -> None:
        assert value_is_sourced("CUST-882", "my id is CUST-882")

    def test_case_and_separators_are_ignored(self) -> None:
        assert value_is_sourced("INT-5541", "reference int 5541 on file")
        assert value_is_sourced(1000000, "coverage of 1,000,000 baht")

    def test_absent_value_is_not_sourced(self) -> None:
        assert not value_is_sourced("INT-5541", "my id is CUST-882")

    def test_short_numbers_do_not_match_inside_longer_ones(self) -> None:
        # "5" is not sourced by "15,000": a digit run must match a whole number.
        assert not value_is_sourced(5, "the premium is 15,000")
        assert value_is_sourced(5, "I'd give it a 5")


# --------------------------------------------------------------------------- 1. arguments


class TestFindUnsourcedArgumentValues:

    def test_value_given_by_the_user_is_sourced(self) -> None:
        messages = [_user("My customer id is CUST-882"), _call("S", "lookup", {"customer_id": "CUST-882"})]
        assert find_unsourced_argument_values(messages, [_schema("lookup", ["customer_id"])]) == []

    def test_value_from_an_earlier_tool_result_is_sourced(self) -> None:
        messages = [
            _user("find my order"),
            _call("S", "search", {"q": "order"}),
            _tool({"order_id": "ORD-42"}),
            _call("S", "cancel", {"order_id": "ORD-42"}),
        ]
        schemas = [_schema("search", ["q"]), _schema("cancel", ["order_id"])]
        assert find_unsourced_argument_values(messages, schemas) == []

    def test_value_nowhere_in_context_is_flagged_confident(self) -> None:
        # The L1_006 shape: the user gave a customer id; gold uses an interaction id.
        messages = [_user("I'd give it a 5. My ID is CUST-882."), _call("R", "collect_csat", {"interaction_id": "INT-5541", "score": "5"})]
        found = find_unsourced_argument_values(messages, [_schema("collect_csat", ["interaction_id", "score"])])
        assert [(f.argument, f.value, f.confidence) for f in found] == [("interaction_id", "INT-5541", "confident")]

    def test_value_in_the_prompt_or_session_context_is_sourced(self) -> None:
        messages = [_user("rate: 5"), _call("R", "collect_csat", {"interaction_id": "INT-5541", "score": "5"})]
        schemas = [_schema("collect_csat", ["interaction_id", "score"])]
        assert find_unsourced_argument_values(messages, schemas, prompt_text="Session: INT-5541") == []
        assert find_unsourced_argument_values(messages, schemas, session_context={"interaction_id": "INT-5541"}) == []

    def test_non_identifier_value_is_needs_review(self) -> None:
        # The user said the city in Thai; the argument is the English name.
        messages = [_user("บินจากกรุงเทพ"), _call("S", "search_flights", {"origin": "Bangkok"})]
        found = find_unsourced_argument_values(messages, [_schema("search_flights", ["origin"])])
        assert [(f.value, f.confidence) for f in found] == [("Bangkok", "needs_review")]

    def test_an_argument_named_like_an_identifier_is_confident(self) -> None:
        # One-letter prefixes and plan slugs are not identifier-SHAPED, but a
        # customer_id nobody gave is invented all the same.
        messages = [_user("upgrade me"), _call("S", "change_plan", {"customer_id": "C-8821", "new_plan_id": "unlimited_50gb"})]
        found = find_unsourced_argument_values(messages, [_schema("change_plan", ["customer_id", "new_plan_id"])])
        assert {(f.argument, f.confidence) for f in found} == {("customer_id", "confident"), ("new_plan_id", "confident")}

    def test_a_free_text_argument_stays_needs_review(self) -> None:
        messages = [_user("the site crashed at checkout"), _call("S", "log", {"description": "website crashed during checkout"})]
        found = find_unsourced_argument_values(messages, [_schema("log", ["description"])])
        assert [(f.argument, f.confidence) for f in found] == [("description", "needs_review")]

    def test_the_calling_turn_cannot_source_its_own_value(self) -> None:
        messages = [_user("hi"), _call("S", "lookup", {"customer_id": "CUST-1"}, prose="I found CUST-1 for you.")]
        found = find_unsourced_argument_values(messages, [_schema("lookup", ["customer_id"])])
        assert [f.value for f in found] == ["CUST-1"]

    def test_optional_arguments_are_not_checked(self) -> None:
        messages = [_user("5 stars"), _call("R", "collect_csat", {"score": "5", "comments": "INT-9"})]
        assert find_unsourced_argument_values(messages, [_schema("collect_csat", ["score"])]) == []

    def test_booleans_are_ignored(self) -> None:
        messages = [_user("yes"), {"role": "assistant", "content": '[STATE: S → S]\n<tool_call>{"name": "opt_in", "arguments": {"consent": true}}</tool_call>'}]
        assert find_unsourced_argument_values(messages, [_schema("opt_in", ["consent"])]) == []


# --------------------------------------------------------------------------- 2. facts


class TestFindUnsourcedFacts:

    def test_invented_code_in_prose_is_flagged(self) -> None:
        messages = [_user("thanks"), _say("R", "T", "You can use code PREM20 for your discount.")]
        assert [(f.value, f.confidence) for f in find_unsourced_facts(messages)] == [("PREM20", "confident")]

    def test_fact_reported_from_a_tool_result_is_sourced(self) -> None:
        messages = [
            _user("find me a flight"),
            _call("S", "search_flights", {"q": "x"}),
            _tool({"flight": "TG910", "price": "35,000"}),
            _say("S", "C", "I found flight TG910 at 35,000 baht."),
        ]
        assert find_unsourced_facts(messages) == []

    def test_invented_amounts_and_percentages_are_needs_review(self) -> None:
        # Amounts may be legitimate arithmetic on sourced values, so they need a look.
        messages = [_user("ok"), _say("R", "T", "I've applied a 15% discount; your balance is 1,100 baht.")]
        found = {(f.value, f.confidence) for f in find_unsourced_facts(messages)}
        assert found == {("15%", "needs_review"), ("1,100", "needs_review")}

    def test_acronyms_without_digits_are_not_facts(self) -> None:
        messages = [_user("ok"), _say("S", "S", "We comply with PDPA. Flights from BKK to LHR are available.")]
        assert find_unsourced_facts(messages) == []

    def test_markers_and_tool_calls_are_not_prose(self) -> None:
        messages = [_user("my id is CUST-1"), _call("S", "lookup", {"customer_id": "CUST-1"})]
        assert find_unsourced_facts(messages) == []

    def test_user_turns_are_never_checked(self) -> None:
        assert find_unsourced_facts([_user("my order is ORD-77")]) == []

    def test_allowlisted_tokens_are_skipped(self) -> None:
        messages = [_user("ok"), _say("S", "S", "Please note COVID-19 rules; data is AES-256 encrypted.")]
        assert find_unsourced_facts(messages) == []


# --------------------------------------------------------------------------- 3. multi-tool states


class TestFindMultiToolStates:

    def test_state_offering_two_tools_is_flagged(self) -> None:
        graph = {"state_details": [{"name": "R", "tools": ["collect_csat", "collect_nps"]}, {"name": "G", "tools": []}]}
        found = find_multi_tool_states(graph, [])
        assert [(f.state, f.kind, f.tools) for f in found] == [("R", "offers", ("collect_csat", "collect_nps"))]

    def test_state_using_two_tools_is_flagged(self) -> None:
        graph = {"state_details": [{"name": "R", "tools": ["collect_csat", "collect_nps"]}]}
        messages = [_call("R", "collect_csat", {}), _tool({}), _call("R", "collect_nps", {})]
        kinds = {(f.state, f.kind) for f in find_multi_tool_states(graph, messages)}
        assert kinds == {("R", "offers"), ("R", "uses")}

    def test_repeating_the_same_tool_is_not_multi_tool(self) -> None:
        # A retry after a tool error calls the same tool twice in one state.
        graph = {"state_details": [{"name": "R", "tools": ["verify"]}]}
        messages = [_call("R", "verify", {}), _tool({"error": "x"}), _call("R", "verify", {})]
        assert find_multi_tool_states(graph, messages) == []

    def test_dict_shaped_state_details_are_accepted(self) -> None:
        graph = {"state_details": {"R": {"tools": ["a", "b"]}}}
        assert [f.state for f in find_multi_tool_states(graph, [])] == ["R"]


# --------------------------------------------------------------------------- 4. identifier reuse


class TestIdentifierReuse:

    def test_collects_identifiers_from_every_role(self) -> None:
        messages = [_user("id CUST-1"), _call("S", "x", {"a": "ORD-2"}), _tool({"r": "TX-101"})]
        assert collect_identifiers(messages) == {"CUST-1", "ORD-2", "TX-101"}

    def test_prompt_vocabulary_is_not_an_entity_value(self) -> None:
        # State names like AUTHENTICATE_2FA are identifier-shaped but are graph
        # vocabulary shared by every conversation in the domain.
        messages = [_say("AUTHENTICATE_2FA", "AUTHENTICATE_2FA", "code sent"), _call("S", "x", {"a": "ORD-2"})]
        prompt = "### [AUTHENTICATE_2FA]\nplans: PLAN_50GB"
        assert collect_identifiers(messages, vocabulary_text=prompt) == {"ORD-2"}

    def test_values_in_more_than_one_row_are_reported(self) -> None:
        rows = {"a.jsonl:1": {"TX-101", "CUST-1"}, "a.jsonl:2": {"TX-101"}, "b.jsonl:1": {"ORD-9"}}
        assert find_identifier_reuse(rows) == {"TX-101": ["a.jsonl:1", "a.jsonl:2"]}


# --------------------------------------------------------------------------- 5. consecutive turns


class TestFindMergeableStayPairs:

    def test_two_self_loops_in_one_state_are_mergeable(self) -> None:
        messages = [_user("hi"), _say("S", "S", "One moment."), _call("S", "lookup", {})]
        assert find_mergeable_stay_pairs(messages) == [(1, 2)]

    def test_advance_then_stay_is_kept_as_two_turns(self) -> None:
        # This pair is how the stay convention enters a tool state. Never flag it.
        messages = [_user("hi"), _say("W", "S", "Let me check."), _call("S", "lookup", {})]
        assert find_mergeable_stay_pairs(messages) == []

    def test_turns_separated_by_a_tool_result_are_not_consecutive(self) -> None:
        messages = [_call("S", "lookup", {}), _tool({}), _say("S", "S", "Done.")]
        assert find_mergeable_stay_pairs(messages) == []

    def test_self_loops_in_different_states_are_not_mergeable(self) -> None:
        messages = [_user("hi"), _say("A", "A", "x"), _say("B", "B", "y")]
        assert find_mergeable_stay_pairs(messages) == []


class TestFindOrphanToolResults:

    def test_a_result_after_a_call_is_fine(self) -> None:
        messages = [_user("hi"), _call("S", "lookup", {}), _tool({"ok": True}), _tool({"ok": True})]
        assert find_orphan_tool_results(messages) == []

    def test_a_result_after_an_announcement_is_an_orphan(self) -> None:
        # "Let me get you qualified in our system" -- and no call.
        messages = [_user("hi"), _say("S", "S", "Let me get you qualified in our system."), _tool({"status": "qualified"})]
        (problem,) = find_orphan_tool_results(messages)
        assert "message 2" in problem

    def test_a_result_with_nothing_before_it_is_an_orphan(self) -> None:
        assert len(find_orphan_tool_results([_tool({"ok": True})])) == 1
