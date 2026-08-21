"""The enriched system prompt must select voice mode.

Four consumers build prompts through this one function: eval/agent_benchmark.py,
scripts/heldout_composite_audit.py, GRPO rollouts, and SFT training. Before this
change none of them told a model to chunk its speech, so a voice row measured
whether the model guessed an unstated convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
from llm_workflow_agents.data.voice_convention import render_voice_format_rules

CORPUS = Path("data/output/sft/task_a_splits/test.jsonl")


def _row(**overrides):
    # workflow_graph mirrors the real shape (list-of-dict state_details with a
    # "name" key on each entry — see data/output/sft/task_a_splits/test.jsonl
    # row 0), not an invented one. An empty-dict state_details never renders a
    # "Workflow script" section, which starves force_rebuild's marker-strip of
    # anything to find and makes force_rebuild tests meaningless.
    sample = {
        "workflow_graph": {
            "states": ["GREETING", "VERIFY_IDENTITY", "TERMINAL"],
            "state_details": [
                {
                    "name": "GREETING",
                    "tools": [],
                    "entry_actions": [],
                    "instruction": "Greet the customer and ask how you can help.",
                },
                {
                    "name": "VERIFY_IDENTITY",
                    "tools": [],
                    "entry_actions": [],
                    "instruction": "Verify the customer's identity.",
                },
                {
                    "name": "TERMINAL",
                    "tools": [],
                    "entry_actions": [],
                    "instruction": "Thank the customer and close the conversation.",
                },
            ],
            "transitions": [
                {
                    "from": "GREETING",
                    "to": "VERIFY_IDENTITY",
                    "condition": "proceed to identity verification",
                    "priority": 0,
                },
                {
                    "from": "VERIFY_IDENTITY",
                    "to": "TERMINAL",
                    "condition": "proceed to resolution",
                    "priority": 0,
                },
            ],
            "initial": "GREETING",
            "terminal": ["TERMINAL"],
        },
        "tool_schemas": [],
        "messages": [],
        "language": "en",
        "complexity_level": "L1",
    }
    sample.update(overrides)
    return sample


def test_voice_sample_gets_the_block():
    out = build_enriched_system_prompt(_row(modality="voice"), "You are an agent.")
    assert render_voice_format_rules() in out


def test_text_sample_does_not_get_the_block():
    out = build_enriched_system_prompt(_row(modality="text"), "You are an agent.")
    assert render_voice_format_rules() not in out


def test_absent_modality_is_treated_as_text():
    """Every pre-existing corpus row predates the field."""
    out = build_enriched_system_prompt(_row(), "You are an agent.")
    assert render_voice_format_rules() not in out


def test_force_rebuild_regenerates_the_block():
    sample = _row(modality="voice")
    once = build_enriched_system_prompt(sample, "You are an agent.")
    twice = build_enriched_system_prompt(sample, once, force_rebuild=True)
    assert twice.count(render_voice_format_rules()) == 1


def test_idempotent_without_force_rebuild():
    sample = _row(modality="voice")
    once = build_enriched_system_prompt(sample, "You are an agent.")
    assert build_enriched_system_prompt(sample, once) == once


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not materialized")
def test_real_text_rows_render_byte_identically(tmp_path):
    """The property that keeps R17's 0.7595 comparable.

    Baseline is captured from the CURRENT implementation before the voice block
    lands, written to a file, and compared after. Regenerate the baseline only
    when a change to the text prompt is intended.
    """
    baseline = Path("tests/fixtures/text_prompt_baseline.json")
    rows = [json.loads(x) for x in CORPUS.read_text().splitlines()[:20]]
    rendered = [
        build_enriched_system_prompt(r, r["messages"][0]["content"], force_rebuild=True)
        for r in rows
    ]
    assert baseline.exists(), "run the baseline capture step first"
    assert json.loads(baseline.read_text()) == rendered
