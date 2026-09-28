"""Free-text similarity must accept paraphrases and still refuse changed facts.

Calibrated on 105 real pairs from the v4 benchmark: at threshold 0.40 the
default backend accepts 78.1% of genuine paraphrases and 1.0% of other rows'
values. The fact guard exists because of the single worst false accept found —
two SMS reminders 0.92 similar while differing in policy number, date and
amount, which a semantic model would rate HIGHER still.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.free_text_similarity import (
    DEFAULT_THRESHOLD,
    facts_in,
    register_backend,
    similarity,
    token_f1,
    values_match,
)

SMS_A = "กธ.7076205678 ครบชำระเบี้ยฯ 20 มิถุนายน 2569 จำนวน 8,900 บาท"
SMS_B = "กธ.7076201234 ครบชำระเบี้ยฯ 15 มิถุนายน 2569 จำนวน 4,250 บาท"


class TestFactGuard:

    def test_a_changed_identifier_or_amount_scores_zero_however_similar_the_prose(self) -> None:
        assert token_f1(SMS_A, SMS_B) > 0.8      # the prose really is near-identical
        assert similarity(SMS_A, SMS_B) == 0.0   # and it is still wrong
        assert not values_match(SMS_A, SMS_B)

    def test_a_dropped_number_scores_zero(self) -> None:
        assert similarity("Applied $50 credit for the outage", "Applied a credit for the outage") == 0.0

    def test_formatting_of_the_same_number_is_not_a_changed_fact(self) -> None:
        assert facts_in("จำนวน 8,900 บาท", "จำนวน 8900 บาท")
        assert facts_in("due 2024-05-10", "due 20240510")

    def test_prose_with_no_facts_is_compared_normally(self) -> None:
        assert similarity("the website kept crashing", "the website kept crashing at checkout") > 0.5


class TestParaphrase:

    def test_a_faithful_paraphrase_passes_the_threshold(self) -> None:
        assert values_match(
            "Agent was rude and did not solve the issue.",
            "The agent was rude and didn't solve my issue at all",
        )

    def test_a_clearly_different_complaint_does_not(self) -> None:
        assert not values_match(
            "the website kept crashing during checkout",
            "the delivery driver arrived three days late",
        )

    def test_the_known_false_accepts_are_documented_not_assumed_away(self) -> None:
        """Two unrelated complaints sharing a content word can pass.

        'Agent was rude and did not solve the issue' scores 0.421 against
        'Delivery was three days late and the driver was rude'. Removing
        stopwords fixes this pair and measures WORSE overall (70.5% vs 78.1%
        of real paraphrases kept at a matched 1.0% false-accept rate), so the
        threshold knowingly admits about 1% of these.
        """
        assert values_match(
            "Agent was rude and did not solve the issue.",
            "Delivery was three days late and the driver was rude",
        )

    def test_thai_prose_is_compared_by_bigrams_not_whole_strings(self) -> None:
        assert values_match("แอปฯ บนมือถือทำงานเร็วขึ้นกว่าเมื่อก่อนเยอะเลย",
                            "แอปฯ บนมือถือทำงานเร็วขึ้นกว่าเมื่อก่อนเยอะเลยนะ ประทับใจ")

    def test_the_known_gap_is_cross_language_and_is_not_papered_over(self) -> None:
        # The default backend cannot bridge this; a multilingual encoder would.
        # Asserted so the limitation is visible rather than assumed away.
        assert similarity("Rear-end collision", "รถชนท้าย") == 0.0


class TestBackends:

    def test_an_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown similarity backend"):
            similarity("a b c", "a b c", backend="magic")

    def test_a_registered_backend_is_used_but_still_obeys_the_fact_guard(self) -> None:
        register_backend("always_one", lambda a, b: 1.0)
        assert similarity("no facts here", "totally different", backend="always_one") == 1.0
        assert similarity("amount 8,900", "amount 4,250", backend="always_one") == 0.0

    def test_the_default_threshold_is_the_calibrated_one(self) -> None:
        assert DEFAULT_THRESHOLD == 0.40
