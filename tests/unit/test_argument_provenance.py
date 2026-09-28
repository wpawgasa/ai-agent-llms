"""Who owns a tool argument's value, declared in the schema.

Measured on the v4 benchmark (R31): of the failing gold arguments, 10% hold a
value that appears nowhere the model could read (an external system supplies
it) and 18% are free text where a paraphrase is correct. Scoring those against
the model mixes work it owns with values it was never given.

`source` declares the owner per argument:

    system     supplied by the runtime (session context); never the model's
    user       stated by the customer; the model must copy it
    derived    the model decides it from the conversation
    free_text  a paraphrase is correct, so exact match cannot score it

`user` and `derived` are the agent-decidable set. Everything here is inert
until a schema declares a source: an undeclared argument stays decidable, so
no existing score moves.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.argument_provenance import (
    AGENT_DECIDABLE,
    agent_decidable_call,
    argument_source,
    split_arguments,
)

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "book",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "source": "system"},
                    "date": {"type": "string", "source": "user"},
                    "channel": {"type": "string", "enum": ["sms", "email"], "source": "derived"},
                    "notes": {"type": "string", "source": "free_text"},
                    "legacy": {"type": "string"},
                },
            },
        },
    }
]


def _call(**arguments):
    return {"name": "book", "arguments": arguments}


class TestDeclaration:

    def test_a_declared_source_is_read(self) -> None:
        assert argument_source(SCHEMAS, "book", "account_id") == "system"
        assert argument_source(SCHEMAS, "book", "channel") == "derived"

    def test_an_undeclared_argument_has_no_source(self) -> None:
        assert argument_source(SCHEMAS, "book", "legacy") is None

    def test_an_unknown_tool_or_argument_has_no_source(self) -> None:
        assert argument_source(SCHEMAS, "other", "date") is None
        assert argument_source(SCHEMAS, "book", "nope") is None

    def test_an_unrecognised_source_is_an_error_not_a_silent_default(self) -> None:
        bad = [{"function": {"name": "book", "parameters": {"properties": {
            "x": {"source": "magic"}}}}}]
        with pytest.raises(ValueError, match="magic"):
            argument_source(bad, "book", "x")

    def test_user_and_derived_are_the_agent_decidable_set(self) -> None:
        assert AGENT_DECIDABLE == frozenset({"user", "derived"})


class TestSplit:

    def test_arguments_split_by_owner(self) -> None:
        decidable, supplied = split_arguments(
            SCHEMAS, _call(account_id="A-1", date="2024-05-10", channel="sms", notes="blah")
        )
        assert decidable == {"date": "2024-05-10", "channel": "sms"}
        assert supplied == {"account_id": "A-1", "notes": "blah"}

    def test_an_undeclared_argument_stays_decidable(self) -> None:
        decidable, supplied = split_arguments(SCHEMAS, _call(legacy="x"))
        assert decidable == {"legacy": "x"} and supplied == {}

    def test_a_call_with_no_schema_at_all_is_unchanged(self) -> None:
        decidable, supplied = split_arguments([], _call(anything="x"))
        assert decidable == {"anything": "x"} and supplied == {}


class TestAgentDecidableCall:

    def test_the_call_keeps_only_what_the_agent_decides(self) -> None:
        call = agent_decidable_call(SCHEMAS, _call(account_id="A-1", date="2024-05-10"))
        assert call == {"name": "book", "arguments": {"date": "2024-05-10"}}

    def test_a_call_whose_every_argument_is_supplied_keeps_its_name(self) -> None:
        # The model still has to make the CALL; only its values were given.
        call = agent_decidable_call(SCHEMAS, _call(account_id="A-1", notes="blah"))
        assert call == {"name": "book", "arguments": {}}

    def test_undeclared_schemas_leave_every_call_untouched(self) -> None:
        call = _call(a="1", b="2")
        assert agent_decidable_call([], call) == call
