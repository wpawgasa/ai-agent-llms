"""Declared argument owners change what they declare, and nothing else.

Scoring a free-text argument by similarity is a metric change (R27): it must
move free-text and system-supplied arguments only, stay completely inert when
no ledger is in force, and leave each predicted call on the turn the model made
it — otherwise a reported gain is partly an artefact of the tool.
"""

from __future__ import annotations

import json

import pytest

from llm_workflow_agents.eval.argument_scoring import (
    DEFAULT_LEDGER,
    EXACT_MATCH,
    ArgumentScoring,
    canonicalize_call,
    canonicalize_conversation,
    load_ledger,
)

SOURCES = ArgumentScoring(
    sources={
        "close_case.resolution_summary": "free_text",
        "send_sms.policy_number": "system",
        "block_card.reason": "derived",
    }
)


@pytest.fixture
def counters() -> dict[str, int]:
    return {}


def _call(name: str, **arguments: object) -> dict[str, object]:
    return {"name": name, "arguments": dict(arguments)}


# --------------------------------------------------------------- one call


def test_a_faithful_paraphrase_of_a_free_text_argument_is_accepted(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge to the customer")
    got = _call("close_case", resolution_summary="Refunded the customer for the duplicate charge")
    assert canonicalize_call(gold, got, SOURCES, counters) == gold
    assert counters["free_text_accepted"] == 1


def test_a_different_answer_in_a_free_text_argument_is_still_wrong(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge")
    got = _call("close_case", resolution_summary="Customer was transferred to billing")
    assert canonicalize_call(gold, got, SOURCES, counters) == got
    assert counters["free_text_rejected"] == 1


def test_a_paraphrase_that_changes_a_number_is_rejected(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded 8,900 baht to the customer")
    got = _call("close_case", resolution_summary="Refunded 4,250 baht to the customer")
    assert canonicalize_call(gold, got, SOURCES, counters) == got
    assert counters["free_text_rejected"] == 1


def test_an_undeclared_argument_is_left_exactly_as_the_model_wrote_it(counters) -> None:
    gold = _call("transfer_funds", amount=500, account="ACC-1")
    got = _call("transfer_funds", amount=400, account="ACC-1")
    assert canonicalize_call(gold, got, SOURCES, counters) == got
    assert counters == {}


def test_a_derived_argument_is_compared_exactly_despite_reading_like_prose(counters) -> None:
    gold = _call("block_card", reason="suspicious_activity")
    got = _call("block_card", reason="suspicious activity reported by the customer")
    assert canonicalize_call(gold, got, SOURCES, counters) == got
    assert counters == {}


def test_a_system_supplied_argument_is_filled_in_because_the_runtime_would_supply_it(counters) -> None:
    gold = _call("send_sms", policy_number="POL-4471", message="Your premium is due")
    got = _call("send_sms", policy_number="POL-0000", message="Your premium is due")
    fixed = canonicalize_call(gold, got, SOURCES, counters)
    assert fixed["arguments"]["policy_number"] == "POL-4471"
    assert counters["system_supplied"] == 1


def test_a_missing_call_stays_missing(counters) -> None:
    assert canonicalize_call(_call("close_case", resolution_summary="x"), None, SOURCES, counters) is None


def test_a_free_text_argument_the_model_omitted_is_not_filled_in(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge")
    got = _call("close_case")
    assert canonicalize_call(gold, got, SOURCES, counters) == got
    assert counters == {}


# --------------------------------------------------------------- inertness


def test_exact_match_returns_the_prediction_untouched(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge")
    got = _call("close_case", resolution_summary="Refunded the charge that was duplicated")
    assert canonicalize_call(gold, got, EXACT_MATCH, counters) is got
    assert counters == {}


def test_exact_match_leaves_a_whole_conversation_identical() -> None:
    gold = [[_call("close_case", resolution_summary="Refunded the duplicate charge")], []]
    predicted = [[_call("close_case", resolution_summary="Refunded the duplicated charge")], []]
    assert canonicalize_conversation(gold, predicted, EXACT_MATCH) is predicted


def test_a_ledger_that_does_not_exist_scores_by_exact_match(tmp_path) -> None:
    assert load_ledger(tmp_path / "absent.json") is EXACT_MATCH
    assert load_ledger(None) is EXACT_MATCH


def test_a_loaded_ledger_records_what_produced_the_score(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": {"a.b": "free_text"}}), encoding="utf-8")
    scoring = load_ledger(path, threshold=0.5, backend="token_f1")
    recorded = scoring.to_dict()
    assert recorded["rule"] == "declared_sources"
    assert recorded["threshold"] == 0.5
    assert recorded["declared_pairs"] == 1
    assert len(recorded["ledger_sha256"]) == 64


# --------------------------------------------------------------- conversation


def test_each_rewritten_call_stays_on_the_turn_the_model_made_it() -> None:
    gold = [
        [_call("close_case", resolution_summary="Refunded the duplicate charge")],
        [_call("block_card", reason="suspicious_activity")],
    ]
    predicted = [
        [_call("close_case", resolution_summary="Refunded the duplicated charge")],
        [_call("block_card", reason="suspicious_activity")],
    ]
    out = canonicalize_conversation(gold, predicted, SOURCES)
    assert [len(turn) for turn in out] == [1, 1]
    assert out[0][0]["arguments"]["resolution_summary"] == "Refunded the duplicate charge"
    assert out[1][0] == predicted[1][0]


def test_a_call_made_one_turn_late_is_still_paired_with_its_gold_call() -> None:
    gold = [[_call("close_case", resolution_summary="Refunded the duplicate charge")], []]
    predicted = [[], [_call("close_case", resolution_summary="Refunded the duplicated charge")]]
    out = canonicalize_conversation(gold, predicted, SOURCES)
    assert out[0] == []
    assert out[1][0]["arguments"]["resolution_summary"] == "Refunded the duplicate charge"


def test_a_call_with_no_gold_counterpart_is_left_alone() -> None:
    predicted = [[_call("close_case", resolution_summary="Refunded the duplicated charge")]]
    assert canonicalize_conversation([[]], predicted, SOURCES)[0][0] == predicted[0][0]


# --------------------------------------------------------------- the real ledger


@pytest.mark.skipif(not DEFAULT_LEDGER.exists(), reason="sources ledger not materialized")
def test_the_reviewed_ledger_declares_only_known_sources() -> None:
    scoring = load_ledger(DEFAULT_LEDGER)
    assert scoring.enabled
    assert set(scoring.sources.values()) <= {"free_text", "system", "user", "derived"}
    # Every key is tool.argument, so a stray bare argument name cannot silently
    # match nothing.
    assert all(key.count(".") == 1 and all(key.split(".")) for key in scoring.sources)


def test_the_default_ledger_is_found_from_any_working_directory(tmp_path, monkeypatch) -> None:
    """A relative default would silently score the old way from a runner's cwd."""
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_LEDGER.is_absolute()
    if DEFAULT_LEDGER.exists():
        assert load_ledger(DEFAULT_LEDGER).enabled
