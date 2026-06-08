"""GRPO loader must keep the outbound opener (assistant preceded by system)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.training.grpo import _load_grpo_jsonl


def _write(tmp_path: Path) -> Path:
    conv = {
        "messages": [
            {"role": "system", "content": "You are a sales agent."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → GREETING]\nHi, calling to offer a promotion.",
                "annotations": {"state_transition": {"from": "GREETING", "to": "GREETING"}},
            },
            {"role": "user", "content": "Oh, sure."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → QUALIFY_LEAD]\nGreat, let me check your account.",
                "annotations": {"state_transition": {"from": "GREETING", "to": "QUALIFY_LEAD"}},
            },
        ],
        "conversation_initiator": "agent",
        "ground_truth": {"terminal_state": "", "terminal_reached": False},
    }
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(conv) + "\n")
    return tmp_path


def test_opener_becomes_a_row_with_system_only_prompt(tmp_path):
    _write(tmp_path)
    ds = _load_grpo_jsonl(tmp_path, split="train")
    prompts = [r["prompt"] for r in ds]
    # The opener row's prompt is exactly the system message.
    assert any(len(p) == 1 and p[0]["role"] == "system" for p in prompts), \
        f"opener row missing; prompts={prompts}"
    # Two assistant turns → two rows.
    assert len(ds) == 2
