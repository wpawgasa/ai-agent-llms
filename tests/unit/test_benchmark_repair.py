"""Mechanical repairs for the Task A benchmark (CLAUDE.md R28).

Each repair is planned once into a ledger and replayed deterministically, so a
frozen test set never depends on code that may change later (the R25 pattern).
"""

from __future__ import annotations

import json
import random

import pytest

from llm_workflow_agents.data.benchmark_repair import (
    apply_fact_edits,
    apply_identifier_remap,
    apply_stay_merges,
    plan_identifier_remap,
    plan_session_context,
    plan_stay_merges,
)


def _call(state: str, name: str, args: dict, prose: str = "") -> dict:
    body = f"\n{prose}" if prose else ""
    return {
        "role": "assistant",
        "content": f"[STATE: {state} → {state}]{body}\n<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>",
        "annotations": {"state_transition": {"from": state, "to": state}, "tool_calls": [{"name": name, "arguments": args}]},
    }


def _say(frm: str, to: str, text: str) -> dict:
    return {
        "role": "assistant",
        "content": f"[STATE: {frm} → {to}] {text}",
        "annotations": {"state_transition": {"from": frm, "to": to}},
    }


def _sample(messages: list[dict], **extra: object) -> dict:
    assistant = [m for m in messages if m["role"] == "assistant"]
    sample = {
        "conversation_id": "L1_001",
        "messages": [{"role": "system", "content": "You are an agent."}] + messages,
        "tool_schemas": [
            {"type": "function", "function": {"name": "lookup", "parameters": {"required": ["customer_id"]}}},
            {"type": "function", "function": {"name": "collect_csat", "parameters": {"required": ["interaction_id"]}}},
        ],
        "workflow_graph": {"state_details": []},
        "ground_truth": {
            "state_sequence": [m["annotations"]["state_transition"] for m in assistant],
            "tool_calls": [c for m in assistant for c in m["annotations"].get("tool_calls", [])],
            "tool_chain_dependencies": [m["annotations"]["tool_calls"] for m in assistant if m["annotations"].get("tool_calls")],
            "terminal_state": "TERMINAL",
        },
    }
    sample.update(extra)
    return sample


# --------------------------------------------------------------------------- identifiers


class TestPlanIdentifierRemap:

    def _conversation(self) -> dict:
        return _sample([
            {"role": "user", "content": "My customer id is CUST-882."},
            _call("S", "lookup", {"customer_id": "CUST-882"}),
            {"role": "tool", "content": json.dumps({"status": "ok", "order": "ORD-4415"})},
            _say("S", "TERMINAL", "Your order ORD-4415 is on its way."),
        ])

    def test_simple_identifiers_get_fresh_values_of_the_same_shape(self) -> None:
        decisions = plan_identifier_remap(self._conversation(), "", forbidden=set(), rng=random.Random(0), taken=set())
        remapped = {d.old: d.new for d in decisions if d.reason == "remapped"}
        assert set(remapped) == {"CUST-882", "ORD-4415"}
        assert remapped["CUST-882"].startswith("CUST-") and len(remapped["CUST-882"]) == len("CUST-882")
        assert remapped["ORD-4415"] != "ORD-4415"

    def test_new_values_avoid_forbidden_and_taken_values(self) -> None:
        taken: set[str] = set()
        forbidden = {f"CUST-{n}" for n in range(100, 1000) if n != 555}
        decisions = plan_identifier_remap(self._conversation(), "", forbidden=forbidden, rng=random.Random(1), taken=taken)
        assert {d.new for d in decisions if d.old == "CUST-882"} == {"CUST-555"}
        assert "CUST-555" in taken

    def test_prompt_vocabulary_is_never_remapped(self) -> None:
        decisions = plan_identifier_remap(self._conversation(), "plan ORD-4415", forbidden=set(), rng=random.Random(0), taken=set())
        assert "ORD-4415" not in {d.old for d in decisions}

    def test_general_knowledge_tokens_are_never_remapped(self) -> None:
        # AES-256 is identifier-shaped; remapping it produced "AES-616 encryption".
        sample = _sample([{"role": "user", "content": "is it safe?"}, _say("S", "S", "We use AES-256 encryption.")])
        (decision,) = plan_identifier_remap(sample, "", forbidden=set(), rng=random.Random(0), taken=set())
        assert (decision.old, decision.new, decision.reason) == ("AES-256", None, "general_knowledge")

    def test_complex_identifiers_are_skipped(self) -> None:
        # Changing digits inside PLAN_50GB changes what the plan is.
        sample = _sample([{"role": "user", "content": "switch me to PLAN_50GB"}, _say("S", "S", "Done.")])
        (decision,) = plan_identifier_remap(sample, "", forbidden=set(), rng=random.Random(0), taken=set())
        assert (decision.old, decision.new, decision.reason) == ("PLAN_50GB", None, "complex_shape")

    def test_identifier_inside_a_longer_token_is_skipped(self) -> None:
        # Rewriting INV-5544 but not INV-5544-SETUP named one invoice two ways.
        sample = _sample([{"role": "user", "content": "Refund INV-5544, the INV-5544-SETUP fee."}, _say("S", "S", "ok")])
        decisions = {d.old: d.reason for d in plan_identifier_remap(sample, "", forbidden=set(), rng=random.Random(0), taken=set())}
        assert decisions["INV-5544"] == "embedded_in_longer_token"

    def test_digits_spoken_on_their_own_block_the_remap(self) -> None:
        # The user also says "882" alone; rewriting only CUST-882 would contradict it.
        sample = _sample([{"role": "user", "content": "It's CUST-882, the one ending 882."}, _say("S", "S", "Thanks.")])
        (decision,) = plan_identifier_remap(sample, "", forbidden=set(), rng=random.Random(0), taken=set())
        assert decision.reason == "digits_referenced_elsewhere"


class TestApplyIdentifierRemap:

    def test_every_occurrence_is_rewritten_consistently(self) -> None:
        sample = TestPlanIdentifierRemap()._conversation()
        out = apply_identifier_remap(sample, {"CUST-882": "CUST-317"})
        text = json.dumps(out)
        assert "CUST-882" not in text
        assert out["messages"][1]["content"] == "My customer id is CUST-317."
        assert out["messages"][2]["annotations"]["tool_calls"][0]["arguments"]["customer_id"] == "CUST-317"
        assert out["ground_truth"]["tool_calls"][0]["arguments"]["customer_id"] == "CUST-317"
        assert "CUST-317" in out["messages"][2]["content"]

    def test_the_input_is_not_modified(self) -> None:
        sample = TestPlanIdentifierRemap()._conversation()
        apply_identifier_remap(sample, {"CUST-882": "CUST-317"})
        assert "CUST-882" in json.dumps(sample)

    def test_longer_identifiers_sharing_a_prefix_are_left_alone(self) -> None:
        sample = _sample([{"role": "user", "content": "TX-101 and TX-1010"}, _say("S", "S", "ok")])
        out = apply_identifier_remap(sample, {"TX-101": "TX-202"})
        assert out["messages"][1]["content"] == "TX-202 and TX-1010"

    def test_system_message_and_schemas_are_untouched(self) -> None:
        sample = _sample([{"role": "user", "content": "CUST-882"}, _say("S", "S", "ok")])
        sample["messages"][0]["content"] = "Example: CUST-882"
        out = apply_identifier_remap(sample, {"CUST-882": "CUST-317"})
        assert out["messages"][0]["content"] == "Example: CUST-882"


# --------------------------------------------------------------------------- session context


class TestPlanSessionContext:

    def test_confident_unsourced_argument_goes_into_context(self) -> None:
        sample = _sample([
            {"role": "user", "content": "5 stars"},
            _call("R", "collect_csat", {"interaction_id": "INT-7302"}),
        ])
        assert plan_session_context(sample, prompt_text="") == {"interaction_id": "INT-7302"}

    def test_sourced_and_needs_review_values_stay_out(self) -> None:
        sample = _sample([
            {"role": "user", "content": "my id is CUST-1"},
            _call("S", "lookup", {"customer_id": "CUST-1"}),
        ])
        assert plan_session_context(sample, prompt_text="") == {}

    def test_one_value_used_twice_is_stated_once(self) -> None:
        sample = _sample([
            {"role": "user", "content": "5"},
            _call("R", "collect_csat", {"interaction_id": "INT-7302"}),
            {"role": "tool", "content": "{}"},
            _call("R", "collect_csat", {"interaction_id": "INT-7302"}),
        ])
        assert plan_session_context(sample, prompt_text="") == {"interaction_id": "INT-7302"}

    def test_two_values_for_one_argument_get_distinct_keys(self) -> None:
        sample = _sample([
            {"role": "user", "content": "5"},
            _call("R", "collect_csat", {"interaction_id": "INT-7302"}),
            {"role": "tool", "content": "{}"},
            _call("R", "collect_csat", {"interaction_id": "INT-8841"}),
        ])
        assert plan_session_context(sample, prompt_text="") == {
            "interaction_id": "INT-7302",
            "interaction_id_2": "INT-8841",
        }


# --------------------------------------------------------------------------- merges


class TestStayMerges:

    def _pair(self) -> dict:
        return _sample([
            {"role": "user", "content": "check it, id CUST-1"},
            _say("S", "S", "One moment please."),
            _call("S", "lookup", {"customer_id": "CUST-1"}),
            {"role": "tool", "content": "{}"},
            _say("S", "TERMINAL", "Done."),
        ])

    def test_prose_then_call_in_one_state_is_planned(self) -> None:
        assert plan_stay_merges(self._pair()) == [(2, 3)]

    def test_advance_then_stay_is_never_planned(self) -> None:
        sample = _sample([
            {"role": "user", "content": "id CUST-1"},
            _say("G", "S", "Let me check."),
            _call("S", "lookup", {"customer_id": "CUST-1"}),
        ])
        assert plan_stay_merges(sample) == []

    def test_a_run_ending_in_prose_is_not_planned(self) -> None:
        sample = _sample([{"role": "user", "content": "hi"}, _say("S", "S", "a"), _say("S", "S", "b")])
        assert plan_stay_merges(sample) == []

    def test_barge_in_turns_are_never_merged(self) -> None:
        sample = self._pair()
        sample["messages"][2]["content"] += " <unspoken>"
        assert plan_stay_merges(sample) == []

    def test_three_turn_run_merges_into_the_call_turn(self) -> None:
        sample = _sample([
            {"role": "user", "content": "id CUST-1"},
            _say("S", "S", "Sure."),
            _say("S", "S", "Checking now."),
            _call("S", "lookup", {"customer_id": "CUST-1"}),
        ])
        assert plan_stay_merges(sample) == [(2, 3, 4)]

    def test_apply_keeps_one_turn_and_ground_truth_aligned(self) -> None:
        sample = self._pair()
        out = apply_stay_merges(sample, plan_stay_merges(sample))
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        merged = out["messages"][2]["content"]
        assert merged.startswith("[STATE: S → S]")
        assert merged.index("One moment please.") < merged.index("<tool_call>")
        assert out["messages"][2]["annotations"]["tool_calls"][0]["name"] == "lookup"
        assert out["ground_truth"]["state_sequence"] == [{"from": "S", "to": "S"}, {"from": "S", "to": "TERMINAL"}]
        assert out["ground_truth"]["tool_chain_dependencies"] == sample["ground_truth"]["tool_chain_dependencies"]

    def test_apply_refuses_misaligned_ground_truth(self) -> None:
        sample = self._pair()
        sample["ground_truth"]["state_sequence"].pop()
        with pytest.raises(ValueError, match="state_sequence"):
            apply_stay_merges(sample, plan_stay_merges(sample))


# --------------------------------------------------------------------------- fact edits


class TestApplyFactEdits:

    def _row(self) -> dict:
        return _sample([
            {"role": "user", "content": "anything else?"},
            _say("S", "S", "Use code PREM20. My customer ID is CID-1274."),
            {"role": "user", "content": "ok"},
            _say("S", "TERMINAL", "The format is like RX566609002."),
        ])

    def test_session_context_rewrite_and_accept(self) -> None:
        edits = [
            {"fact": "PREM20", "action": "session_context", "field": "survey_reward_code", "value": "PREM20"},
            {"fact": "CID-1274", "action": "rewrite", "body_index": 1, "old": " My customer ID is CID-1274.", "new": ""},
            {"fact": "RX566609002", "action": "accept_example"},
        ]
        out = apply_fact_edits(self._row(), edits)
        assert out["session_context"] == {"survey_reward_code": "PREM20"}
        assert out["messages"][2]["content"].endswith("Use code PREM20.")
        assert "RX566609002" in out["messages"][4]["content"]

    def test_a_fact_missing_from_the_row_is_an_error(self) -> None:
        # Fact values are post-remap; a changed remap must not apply silently.
        with pytest.raises(ValueError, match="not found"):
            apply_fact_edits(self._row(), [{"fact": "PREM99", "action": "accept_example"}])

    def test_a_session_context_clash_is_an_error(self) -> None:
        row = self._row()
        row["session_context"] = {"survey_reward_code": "OTHER1"}
        with pytest.raises(ValueError, match="already holds"):
            apply_fact_edits(row, [{"fact": "PREM20", "action": "session_context", "field": "survey_reward_code", "value": "PREM20"}])

    def test_a_rewrite_must_match_exactly_once(self) -> None:
        with pytest.raises(ValueError, match="exactly once"):
            apply_fact_edits(self._row(), [{"fact": "PREM20", "action": "rewrite", "body_index": 1, "old": "nowhere", "new": "x"}])
