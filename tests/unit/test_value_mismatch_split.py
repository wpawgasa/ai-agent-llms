"""Splitting a wrong value by WHY it is wrong.

`value_in_context` (320 of 540 failures on the best v4 run) is the bucket that
decides the next project, and it is not one thing. Reading samples showed at
least three causes with three different fixes:

    expected 'Bangkok'       got 'กรุงเทพ'          the same city, other language
    expected 'three months'  got 'next three months'  the same answer, other words
    expected '2023-10-15'    got '2024-10-15'         a different date

Only the third is a capability gap worth training against. The first is a data
convention nobody stated (Thai conversations carrying English gold arguments);
the second is a phrasing boundary the schema never pinned.
"""

from __future__ import annotations

from llm_workflow_agents.eval.tool_call_failures import refine_value_mismatch

THAI_CONTEXT = "ลูกค้าต้องการบินจากกรุงเทพไปเชียงใหม่ วันที่ 15 ตุลาคม"
EN_CONTEXT = "Customer wants to fly from Bangkok to Chiang Mai on 2023-10-15, budget 2000."


def test_a_value_in_the_other_script_is_cross_script() -> None:
    assert refine_value_mismatch("Bangkok", "กรุงเทพ", THAI_CONTEXT) == "cross_script"
    assert refine_value_mismatch("เชียงใหม่", "Chiang Mai", EN_CONTEXT) == "cross_script"


def test_one_value_containing_the_other_is_phrasing() -> None:
    assert refine_value_mismatch("three months", "next three months", EN_CONTEXT) == "phrasing"
    assert refine_value_mismatch("บริษัท เอ็กซ์ จำกัด", "เอ็กซ์ จำกัด", THAI_CONTEXT) == "phrasing"


def test_shared_words_without_containment_is_partial_overlap() -> None:
    assert refine_value_mismatch(
        "premium fiber plan", "fiber plan upgrade", EN_CONTEXT
    ) == "partial_overlap"


def test_two_different_dates_are_a_different_value() -> None:
    assert refine_value_mismatch("2023-10-15", "2024-10-15", EN_CONTEXT) == "different_value"


def test_two_different_numbers_are_a_different_value() -> None:
    assert refine_value_mismatch(2000, 3000, EN_CONTEXT) == "different_value"
    assert refine_value_mismatch("ACC-4471", "ACC-1234", EN_CONTEXT) == "different_value"


def test_unrelated_words_are_a_different_value() -> None:
    assert refine_value_mismatch("Bangkok", "Chiang Mai", EN_CONTEXT) == "different_value"


def test_a_near_identical_number_does_not_hide_behind_overlap() -> None:
    # "2000 baht" vs "3000 baht" shares a word; the numbers differ, so it is a
    # different value, not phrasing.
    assert refine_value_mismatch("2000 baht", "3000 baht", EN_CONTEXT) == "different_value"


def test_cross_script_wins_only_when_scripts_actually_differ() -> None:
    # Both Thai, different words -> not cross_script.
    assert refine_value_mismatch("กรุงเทพ", "เชียงใหม่", THAI_CONTEXT) == "different_value"
