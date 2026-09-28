"""Rescoring under declared argument sources touches only what it declares.

The rescore is a metric change (R27): it must move free-text and
system-supplied arguments and NOTHING else, or a reported gain is partly an
artefact of the tool rather than of the question being asked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "rescore_with_argument_sources", SCRIPTS / "rescore_with_argument_sources.py"
)
rescore_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore_module)

rewrite_call = rescore_module.rewrite_call
blended_delta = rescore_module.blended_delta

SOURCES = {
    "close_case.resolution_summary": "free_text",
    "send_sms.policy_number": "system",
    "block_card.reason": "derived",
}


@pytest.fixture
def counters() -> dict[str, int]:
    return {"system_supplied": 0, "free_text_accepted": 0, "free_text_rejected": 0}


def _call(name: str, **arguments: object) -> dict[str, object]:
    return {"name": name, "arguments": dict(arguments)}


def test_a_faithful_paraphrase_of_a_free_text_argument_is_accepted(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge to the customer")
    got = _call("close_case", resolution_summary="Refunded the customer for the duplicate charge")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == gold
    assert counters["free_text_accepted"] == 1


def test_a_different_answer_in_a_free_text_argument_is_still_wrong(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge")
    got = _call("close_case", resolution_summary="Customer was transferred to billing")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == got
    assert counters["free_text_rejected"] == 1


def test_a_paraphrase_that_changes_a_number_is_rejected(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded 8,900 baht to the customer")
    got = _call("close_case", resolution_summary="Refunded 4,250 baht to the customer")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == got
    assert counters["free_text_rejected"] == 1


def test_an_undeclared_argument_is_left_exactly_as_the_model_wrote_it(counters) -> None:
    gold = _call("transfer_funds", amount=500, account="ACC-1")
    got = _call("transfer_funds", amount=400, account="ACC-1")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == got
    assert counters == {"system_supplied": 0, "free_text_accepted": 0, "free_text_rejected": 0}


def test_a_derived_argument_is_compared_exactly_despite_reading_like_prose(counters) -> None:
    gold = _call("block_card", reason="suspicious_activity")
    got = _call("block_card", reason="suspicious activity reported by customer")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == got
    assert counters["free_text_accepted"] == 0


def test_a_system_supplied_argument_is_filled_in_because_the_runtime_would_supply_it(counters) -> None:
    gold = _call("send_sms", policy_number="POL-4471", message="Your premium is due")
    got = _call("send_sms", policy_number="POL-0000", message="Your premium is due")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters)["arguments"]["policy_number"] == "POL-4471"
    assert counters["system_supplied"] == 1


def test_a_missing_call_stays_missing(counters) -> None:
    gold = _call("close_case", resolution_summary="anything")
    assert rewrite_call(gold, None, SOURCES, 0.40, "token_f1", counters) is None


def test_a_free_text_argument_the_model_omitted_is_not_filled_in(counters) -> None:
    gold = _call("close_case", resolution_summary="Refunded the duplicate charge")
    got = _call("close_case")
    assert rewrite_call(gold, got, SOURCES, 0.40, "token_f1", counters) == got
    assert counters["free_text_accepted"] == 0


def test_the_quality_delta_weights_the_two_strata_as_phase_1_does() -> None:
    result = {
        "strata": {
            "text": {"before": {"tool_call_f1": 0.60}, "after": {"tool_call_f1": 0.70}},
            "voice": {"before": {"tool_call_f1": 0.70}, "after": {"tool_call_f1": 0.75}},
        }
    }
    assert blended_delta(result) == pytest.approx(0.4 * (0.7 * 0.10 + 0.3 * 0.05))


def test_a_single_stratum_run_contributes_only_its_own_stratum() -> None:
    result = {"strata": {"text": {"before": {"tool_call_f1": 0.6}, "after": {"tool_call_f1": 0.7}}}}
    assert blended_delta(result) == pytest.approx(0.4 * 0.7 * 0.10)
