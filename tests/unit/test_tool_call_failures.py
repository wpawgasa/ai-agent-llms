"""The failure taxonomy must separate a formatting difference from a wrong value.

`tool_call_f1` scores both as zero, which is right for a headline metric and
useless for deciding what to fix: one is a comparator change, the other needs
training data (CLAUDE.md R30 records the cost of guessing which). These tests
pin the boundary, and in particular that the module never waves through a
genuinely different value.
"""

from __future__ import annotations

from llm_workflow_agents.eval.tool_call_failures import (
    align_calls,
    classify_call,
    value_is_visible,
    values_equivalent,
)

CONTEXT = "Customer ACC-4471 asked to book for May 10, 2024 in Bangkok, budget 2000 baht."


def _call(name: str, **arguments):
    return {"name": name, "arguments": arguments}


class TestEquivalence:

    def test_a_date_written_another_way_is_equivalent(self) -> None:
        assert values_equivalent("2024-05-10", "May 10, 2024")
        assert values_equivalent("10 May 2024", "2024-05-10")
        assert values_equivalent("2024-05-10", "10th May 2024")

    def test_a_different_date_is_not(self) -> None:
        assert not values_equivalent("2024-05-10", "2024-05-11")
        assert not values_equivalent("May 10, 2024", "June 10, 2024")

    def test_an_ambiguous_order_is_left_as_a_real_mismatch(self) -> None:
        # 10/05/2024 could be either; the parser reads it one way for both sides,
        # so the pair stays a mismatch rather than being waved through.
        assert not values_equivalent("2024-05-10", "10/05/2024")

    def test_case_space_and_punctuation_are_equivalent(self) -> None:
        assert values_equivalent("Bangkok", " bangkok ")
        assert values_equivalent("UNLIMITED_MAX", "unlimited max")
        assert values_equivalent("A-1", "a1")

    def test_numbers_and_bools_written_as_text_are_equivalent(self) -> None:
        assert values_equivalent(100, "100")
        assert values_equivalent(100.0, 100)
        assert values_equivalent(True, "yes")
        assert not values_equivalent(100, "200")
        assert not values_equivalent(True, "no")

    def test_a_different_word_is_not_equivalent(self) -> None:
        assert not values_equivalent("Bangkok", "Chiang Mai")
        assert not values_equivalent("ACC-4471", "ACC-4472")

    def test_structures_compare_by_content_not_key_order(self) -> None:
        assert values_equivalent({"a": 1, "b": 2}, {"b": 2, "a": 1})
        assert not values_equivalent({"a": 1}, {"a": 2})


class TestVisibility:

    def test_a_value_present_in_the_context_is_visible(self) -> None:
        assert value_is_visible("ACC-4471", CONTEXT)
        assert value_is_visible("acc 4471", CONTEXT)      # loosely compared

    def test_a_value_absent_from_the_context_is_not(self) -> None:
        assert not value_is_visible("ACC-9999", CONTEXT)

    def test_a_very_short_value_is_never_called_unknowable(self) -> None:
        assert value_is_visible("A", CONTEXT)


class TestClassifyCall:

    def test_a_matching_call_produces_no_failures(self) -> None:
        gold = _call("book", account="ACC-4471", date="2024-05-10")
        assert classify_call(gold, dict(gold), CONTEXT) == []

    def test_no_predicted_call(self) -> None:
        (failure,) = classify_call(_call("book", account="ACC-4471"), None, CONTEXT)
        assert failure.bucket == "no_call"

    def test_wrong_tool_name(self) -> None:
        (failure,) = classify_call(_call("book"), _call("cancel"), CONTEXT)
        assert failure.bucket == "wrong_name"

    def test_missing_argument(self) -> None:
        (failure,) = classify_call(
            _call("book", account="ACC-4471"), _call("book"), CONTEXT
        )
        assert failure.bucket == "missing_argument"

    def test_formatting_difference_is_equivalent_value(self) -> None:
        (failure,) = classify_call(
            _call("book", date="2024-05-10"), _call("book", date="May 10, 2024"), CONTEXT
        )
        assert failure.bucket == "equivalent_value"

    def test_wrong_value_that_was_in_the_context_is_a_copy_failure(self) -> None:
        (failure,) = classify_call(
            _call("book", account="ACC-4471"), _call("book", account="ACC-1234"), CONTEXT
        )
        assert failure.bucket == "value_in_context"

    def test_gold_value_absent_from_the_context_is_unknowable(self) -> None:
        (failure,) = classify_call(
            _call("book", ref="INT-5541"), _call("book", ref="INT-0001"), CONTEXT
        )
        assert failure.bucket == "value_unknowable"

    def test_an_extra_predicted_argument_is_not_a_failure(self) -> None:
        # tool_call_f1's subtree rule ignores extra keys; this must agree with it.
        gold = _call("book", account="ACC-4471")
        assert classify_call(gold, _call("book", account="ACC-4471", note="x"), CONTEXT) == []


class TestAlign:

    def test_calls_pair_by_name_not_by_index(self) -> None:
        gold = [_call("verify"), _call("book")]
        predicted = [_call("book"), _call("verify")]
        pairs = align_calls(gold, predicted)
        assert [(g["name"], p and p["name"]) for g, p in pairs] == [("verify", "verify"), ("book", "book")]

    def test_a_gold_call_with_no_counterpart_pairs_with_none(self) -> None:
        pairs = align_calls([_call("verify"), _call("book")], [_call("verify")])
        assert pairs[1][1] is None

    def test_one_predicted_call_is_consumed_once(self) -> None:
        pairs = align_calls([_call("verify"), _call("verify")], [_call("verify")])
        assert pairs[0][1] is not None and pairs[1][1] is None
