"""GRPO loader must carry the source conversation's modality through, and
must not otherwise change what it emits for prompt/ground_truth (Task 10
follow-up, round 1: the held-out audit's voice_format_compliance guardrail
was structurally unable to fire because modality never survived
_load_grpo_jsonl -> _sample_prompts)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.training.grpo import _load_grpo_jsonl


def _voice_conv() -> dict:
    return {
        "modality": "voice",
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "<S>Hi, I need help.</S>"},
            {
                "role": "assistant",
                "content": "<S>Sure, let me look that up.</S>",
                "annotations": {
                    "state_transition": {"from": "GREETING", "to": "LOOKUP"}
                },
            },
        ],
        "ground_truth": {"terminal_state": "", "terminal_reached": False},
    }


def _text_conv_no_modality_field() -> dict:
    """Predates the modality field entirely (all 5,549 pre-existing rows)."""
    return {
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Hi, I need help."},
            {
                "role": "assistant",
                "content": "Sure, let me look that up.",
                "annotations": {
                    "state_transition": {"from": "GREETING", "to": "LOOKUP"}
                },
            },
        ],
        "ground_truth": {"terminal_state": "", "terminal_reached": False},
    }


def _write(tmp_path: Path, convs: list[dict]) -> Path:
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in convs) + "\n")
    return tmp_path


def test_modality_column_present_for_voice_conversation(tmp_path):
    _write(tmp_path, [_voice_conv()])
    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 1
    assert ds[0]["modality"] == "voice"


def test_modality_defaults_to_text_when_field_absent(tmp_path):
    _write(tmp_path, [_text_conv_no_modality_field()])
    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 1
    assert ds[0]["modality"] == "text"


def test_row_count_and_prompt_ground_truth_content_unchanged(tmp_path):
    """Regression guard: adding the modality column must not perturb the
    existing prompt/ground_truth emission the GRPO training path, DPO
    preference-pair mining, and negative mining all depend on."""
    _write(tmp_path, [_text_conv_no_modality_field()])
    ds = _load_grpo_jsonl(tmp_path, split="train")

    assert len(ds) == 1
    row = ds[0]

    assert row["prompt"] == [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Hi, I need help."},
    ]

    gt = json.loads(row["ground_truth"])
    assert gt["state_sequence"] == [{"from": "GREETING", "to": "LOOKUP"}]
    assert gt["tool_calls"] == []
    assert gt["terminal_state"] == ""
    # terminal_reached is False here because the fixture's ground_truth sets
    # terminal_reached=False overall, not because this test is asserting
    # anything about which row is "final".
    assert gt["terminal_reached"] is False
    assert gt["messages"] == [
        {"role": "assistant", "content": "Sure, let me look that up."}
    ]


def test_outbound_opener_regression_still_two_rows(tmp_path):
    """Same fixture as test_grpo_outbound.py's opener regression, plus a
    modality assertion, to prove the two features compose."""
    conv = {
        "modality": "text",
        "messages": [
            {"role": "system", "content": "You are a sales agent."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING -> GREETING]\nHi, calling to offer a promotion.",
                "annotations": {
                    "state_transition": {"from": "GREETING", "to": "GREETING"}
                },
            },
            {"role": "user", "content": "Oh, sure."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING -> QUALIFY_LEAD]\nGreat, let me check your account.",
                "annotations": {
                    "state_transition": {"from": "GREETING", "to": "QUALIFY_LEAD"}
                },
            },
        ],
        "conversation_initiator": "agent",
        "ground_truth": {"terminal_state": "", "terminal_reached": False},
    }
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(conv) + "\n")

    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 2
    assert all(r["modality"] == "text" for r in ds)
