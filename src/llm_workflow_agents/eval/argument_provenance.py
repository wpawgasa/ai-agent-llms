"""Who owns a tool argument's value: declared in the schema, read here.

A tool call mixes values of very different kinds, and scoring them alike makes
the headline number mean less than it looks:

    book(account_id="ACC-4471",     <- the runtime knows this; the model cannot
         date="2024-05-10",         <- the customer said it; the model must copy
         channel="sms",             <- the model decides it from the dialogue
         notes="app is much faster") <- a paraphrase is correct

Measured on the v4 benchmark (CLAUDE.md R31), 10% of failing gold arguments
hold a value that appears nowhere the model could read and 18% are free text
where the model wrote a faithful paraphrase and scored zero. Neither is the
model's work, and neither can be fixed by training it harder.

A parameter spec may therefore declare::

    "account_id": {"type": "string", "source": "system"}

``system``     an external system supplies it (the session-context rule, R28)
``user``       the customer states it; the model must carry it across turns
``derived``    the model decides it from the conversation
``free_text``  a paraphrase is correct, so exact match cannot score it

``user`` and ``derived`` form the **agent-decidable** set — the arguments a
score can fairly attribute to the model.

**This module is inert until a schema declares something.** An undeclared
argument counts as decidable, so every existing metric is unchanged; nothing
here is subtracted from a score by inference. An unrecognised source raises
rather than defaulting, because a typo that silently drops an argument from
scoring would inflate the number in the direction nobody would question.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

SYSTEM = "system"
USER = "user"
DERIVED = "derived"
FREE_TEXT = "free_text"

SOURCES = frozenset({SYSTEM, USER, DERIVED, FREE_TEXT})
#: The arguments a score may fairly attribute to the model.
AGENT_DECIDABLE = frozenset({USER, DERIVED})


def _properties(schemas: Iterable[Mapping[str, Any]] | None, tool: str) -> dict[str, Any]:
    for schema in schemas or []:
        function = schema.get("function", schema)
        if function.get("name") == tool:
            return (function.get("parameters") or {}).get("properties") or {}
    return {}


def argument_source(
    schemas: Iterable[Mapping[str, Any]] | None, tool: str, argument: str
) -> str | None:
    """The declared owner of ``tool.argument``, or None when undeclared.

    Raises ``ValueError`` on a source outside :data:`SOURCES` — a typo must not
    quietly remove an argument from scoring.
    """
    spec = _properties(schemas, tool).get(argument)
    if not isinstance(spec, Mapping):
        return None
    source = spec.get("source")
    if source is None:
        return None
    if source not in SOURCES:
        raise ValueError(
            f"{tool}.{argument} declares source {source!r}; expected one of {sorted(SOURCES)}"
        )
    return source


def split_arguments(
    schemas: Iterable[Mapping[str, Any]] | None, call: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a call's arguments into (agent-decidable, supplied-or-unscoreable)."""
    tool = call.get("name") or ""
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return ({}, {})
    decidable: dict[str, Any] = {}
    supplied: dict[str, Any] = {}
    for argument, value in arguments.items():
        source = argument_source(schemas, tool, argument)
        if source is None or source in AGENT_DECIDABLE:
            decidable[argument] = value
        else:
            supplied[argument] = value
    return decidable, supplied


def agent_decidable_call(
    schemas: Iterable[Mapping[str, Any]] | None, call: Mapping[str, Any]
) -> dict[str, Any]:
    """``call`` with only the arguments a score may attribute to the model.

    The tool NAME is kept whatever its arguments' sources are: choosing to make
    the call, and which tool to call, is the model's work even when every value
    was handed to it.
    """
    decidable, _ = split_arguments(schemas, call)
    out = dict(call)
    out["arguments"] = decidable
    return out
