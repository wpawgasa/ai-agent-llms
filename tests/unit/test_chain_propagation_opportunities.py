"""Chain propagation must measure value carrying, not corpus coincidence.

The old definition scored every consecutive pair of tool calls, counting a pair
as a failure whenever the second call's arguments happened to share no value
with the first call's response -- which is most pairs, because most consecutive
calls are independent. Scoring the benchmark's ground truth against itself gave
0.2843, below what every model scored, so the metric had no headroom and ranked
nothing (CLAUDE.md R26 follow-up).

The definition here scores only real propagation opportunities: an argument of
call N+1 whose value comes from call N's response and appears nowhere earlier in
the conversation, so the only way to produce it is to read the tool result.
"""

from __future__ import annotations

import glob
import json

import pytest

from llm_workflow_agents.eval.tool_chain_propagation import (
    evaluate_chain_propagation,
    find_propagation_opportunities,
)


def _conv(messages: list[dict]) -> dict:
    return {"messages": messages}


def _call(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>",
    }


def _result(payload: dict) -> dict:
    return {"role": "tool", "content": json.dumps(payload)}


class TestFindPropagationOpportunities:

    def test_value_from_the_previous_response_is_an_opportunity(self) -> None:
        conv = _conv([
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            _call("cancel", {"order_id": "ORD-42"}),
            _result({"status": "cancelled"}),
        ])
        opps = find_propagation_opportunities(conv["messages"])
        assert [(o.argument_path, o.expected_value) for o in opps] == [("order_id", "ORD-42")]

    def test_independent_consecutive_calls_are_not_an_opportunity(self) -> None:
        # The old metric counted this pair and scored it zero, for every model.
        conv = _conv([
            _call("collect_csat", {"interaction_id": "INT-5541", "score": 5}),
            _result({"status": "success"}),
            _call("collect_nps", {"customer_id": "CUST-882", "score": 9}),
            _result({"status": "success"}),
        ])
        assert find_propagation_opportunities(conv["messages"]) == []

    def test_value_already_in_earlier_context_is_not_an_opportunity(self) -> None:
        # The user supplied the id, so repeating it proves nothing about the
        # model reading the tool result.
        conv = _conv([
            {"role": "user", "content": "my order is ORD-42"},
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            _call("cancel", {"order_id": "ORD-42"}),
        ])
        assert find_propagation_opportunities(conv["messages"]) == []

    def test_no_tool_calls_yields_nothing(self) -> None:
        assert find_propagation_opportunities([{"role": "user", "content": "hi"}]) == []


class TestEvaluateChainPropagation:

    def _chained(self) -> dict:
        return _conv([
            {"role": "user", "content": "cancel my latest order"},
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            _call("cancel", {"order_id": "ORD-42"}),
            _result({"status": "cancelled"}),
        ])

    def test_ground_truth_against_itself_is_perfect(self) -> None:
        gt = self._chained()
        m = evaluate_chain_propagation([gt], [gt])
        assert m.chain_propagation_accuracy == 1.0
        assert m.total_chains == 1

    def test_carrying_the_wrong_value_scores_zero(self) -> None:
        gt = self._chained()
        pred = _conv([
            {"role": "user", "content": "cancel my latest order"},
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            _call("cancel", {"order_id": "ORD-99"}),
        ])
        m = evaluate_chain_propagation([pred], [gt])
        assert m.chain_propagation_accuracy == 0.0
        assert m.total_chains == 1

    def test_never_making_the_second_call_scores_zero(self) -> None:
        gt = self._chained()
        pred = _conv([
            {"role": "user", "content": "cancel my latest order"},
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            {"role": "assistant", "content": "I have cancelled it."},
        ])
        m = evaluate_chain_propagation([pred], [gt])
        assert m.chain_propagation_accuracy == 0.0
        assert m.missed_calls == 1

    def test_an_extra_unrelated_call_does_not_break_alignment(self) -> None:
        gt = self._chained()
        pred = _conv([
            {"role": "user", "content": "cancel my latest order"},
            _call("lookup", {"customer": "jane"}),
            _result({"order_id": "ORD-42"}),
            _call("check_policy", {"customer": "jane"}),
            _result({"ok": True}),
            _call("cancel", {"order_id": "ORD-42"}),
        ])
        m = evaluate_chain_propagation([pred], [gt])
        assert m.chain_propagation_accuracy == 1.0

    def test_conversations_without_opportunities_report_no_chains(self) -> None:
        flat = _conv([
            _call("collect_csat", {"interaction_id": "INT-1", "score": 5}),
            _result({"status": "success"}),
            _call("collect_nps", {"customer_id": "CUST-1", "score": 9}),
        ])
        m = evaluate_chain_propagation([flat], [flat])
        assert m.total_chains == 0
        # compute_full_workflow_success reads total_chains == 0 as "not a failure"
        assert m.chain_propagation_accuracy == 0.0

    def test_empty(self) -> None:
        assert evaluate_chain_propagation([], []).chain_propagation_accuracy == 0.0


class TestAgainstTheRealBenchmarkCorpus:
    """The ceiling check that condemned the old definition."""

    def _corpus(self) -> list[dict]:
        rows: list[dict] = []
        for path in sorted(glob.glob("data/output/benchmark/task_a_v2/*.jsonl")):
            with open(path) as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        return rows

    def test_benchmark_ground_truth_self_scores_perfectly(self) -> None:
        rows = self._corpus()
        if not rows:
            pytest.skip("benchmark corpus not materialized (dvc pull task_a_benchmark_text_v2)")
        convs = [{"messages": r.get("messages", [])} for r in rows]
        m = evaluate_chain_propagation(convs, convs)
        assert m.total_chains > 0, "corpus has no propagation opportunities at all"
        assert m.chain_propagation_accuracy == 1.0
