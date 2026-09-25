"""Gold tool-call values must obey the enum their own schema declares.

Found by the failure taxonomy on the v4 benchmark: 7 wrong-value failures were
the benchmark demanding a value its own schema forbids —

    search_products.sort_by      expected 'cheapest_ever'
                                 enum ['relevance','price_low','price_high','rating']
    verify_identity.verification_method  expected 'password'
                                 enum ['otp','pin','security_question']

The model answered `price_low` and `pin`, which the schema allows and the
prompt shows, and was scored wrong. This check is cheap, needs no model, and
would have caught all seven at generation time — the same role
`find_orphan_tool_results` plays in the generator's repair loop.
"""

from __future__ import annotations

from llm_workflow_agents.data.source_traceability import find_enum_violations

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {"type": "string", "enum": ["relevance", "price_low", "price_high"]},
                    "query": {"type": "string"},
                },
            },
        },
    }
]


def _turn(**arguments):
    return {
        "role": "assistant",
        "content": "[STATE: S → S]",
        "annotations": {"tool_calls": [{"name": "search_products", "arguments": arguments}]},
    }


def test_a_gold_value_inside_the_enum_passes() -> None:
    assert find_enum_violations([_turn(sort_by="price_low")], SCHEMAS) == []


def test_a_gold_value_outside_the_enum_is_reported() -> None:
    (violation,) = find_enum_violations([_turn(sort_by="cheapest_ever")], SCHEMAS)
    assert violation.tool == "search_products" and violation.argument == "sort_by"
    assert violation.value == "cheapest_ever"
    assert violation.allowed == ["relevance", "price_low", "price_high"]
    assert "cheapest_ever" in violation.describe()


def test_case_and_spacing_do_not_count_as_violations() -> None:
    # The scorer is strict about these, but they are a different defect
    # (formatting), and reporting them here would bury the real ones.
    assert find_enum_violations([_turn(sort_by="Price_Low")], SCHEMAS) == []
    assert find_enum_violations([_turn(sort_by="price low")], SCHEMAS) == []


def test_an_argument_without_an_enum_is_not_checked() -> None:
    assert find_enum_violations([_turn(query="anything at all")], SCHEMAS) == []


def test_an_unknown_tool_is_not_checked() -> None:
    turn = {
        "role": "assistant",
        "content": "x",
        "annotations": {"tool_calls": [{"name": "other_tool", "arguments": {"sort_by": "nope"}}]},
    }
    assert find_enum_violations([turn], SCHEMAS) == []


def test_calls_written_as_text_are_checked_too() -> None:
    # The corpus writes tool calls as <tool_call> text; annotations may be absent.
    turn = {
        "role": "assistant",
        "content": '[STATE: S → S]\n<tool_call>{"name": "search_products", '
                   '"arguments": {"sort_by": "cheapest_ever"}}</tool_call>',
    }
    (violation,) = find_enum_violations([turn], SCHEMAS)
    assert violation.value == "cheapest_ever"


def test_a_list_valued_argument_checks_each_member() -> None:
    schemas = [{
        "type": "function",
        "function": {
            "name": "search_products",
            "parameters": {"type": "object", "properties": {
                "tags": {"type": "array", "items": {"enum": ["new", "sale"]}}}},
        },
    }]
    assert find_enum_violations([_turn(tags=["new"])], schemas) == []
    (violation,) = find_enum_violations([_turn(tags=["new", "bogus"])], schemas)
    assert violation.value == "bogus"
