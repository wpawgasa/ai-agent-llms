"""The proposal rules, pinned. The script applies nothing; these decide what it suggests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from propose_argument_sources import build_proposal, propose_source  # noqa: E402


def _evidence(**kw):
    base = {"tool": "t", "argument": "x", "occurrences": 10, "sourced_share": 0.9,
            "median_length": 8, "has_enum": False, "samples": []}
    return {**base, **kw}


def test_a_prose_field_is_free_text_whatever_else_holds() -> None:
    assert propose_source(_evidence(argument="description", sourced_share=0.0)) == "free_text"
    assert propose_source(_evidence(argument="notes", has_enum=True)) == "free_text"


def test_long_values_are_free_text_even_under_another_name() -> None:
    assert propose_source(_evidence(argument="resolution", median_length=120)) == "free_text"


def test_a_value_rarely_visible_to_the_model_is_system_supplied() -> None:
    assert propose_source(_evidence(sourced_share=0.05)) == "system"


def test_an_enum_the_model_can_see_is_derived() -> None:
    assert propose_source(_evidence(has_enum=True, sourced_share=0.8)) == "derived"


def test_everything_else_is_user_supplied() -> None:
    assert propose_source(_evidence()) == "user"


def test_a_visible_value_is_never_called_system() -> None:
    # The accusation that a value is unknowable must need evidence (R31).
    assert propose_source(_evidence(sourced_share=1.0)) != "system"


def test_the_proposal_carries_its_evidence() -> None:
    stats = {("book", "account_id"): {"occurrences": 3, "sourced": 0, "lengths": [7, 7, 7],
                                      "has_enum": False, "samples": ["ACC-1"]}}
    proposal = build_proposal(stats)
    (row,) = proposal["pairs"]
    assert row["proposed_source"] == "system"
    assert row["sourced_share"] == 0.0 and row["occurrences"] == 3
    assert row["samples"] == ["ACC-1"]
    assert proposal["summary"] == {"system": 1}
