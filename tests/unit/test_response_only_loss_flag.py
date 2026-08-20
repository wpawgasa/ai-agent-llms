"""A message marked loss:false must never become a training target.

The orchestrator writes the <unspoken> barge-in marker into the model's own
past turn. The model does not write it. Training on that turn would teach the
model to emit the marker. See risk R15 for what a uniform edit teaches.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.sft import render_response_only_sample

@pytest.fixture(scope="module")
def tokenizer():
    # Reuse the repo's existing helper: it skips cleanly on an offline machine
    # instead of erroring. Qwen/Qwen2.5-0.5B-Instruct is in the local HF cache.
    from tests.unit.test_training import _load_tokenizer_or_skip

    return _load_tokenizer_or_skip("Qwen/Qwen2.5-0.5B-Instruct")


def _messages(loss_flag):
    first = {"role": "assistant", "content": "Interrupted reply"}
    if loss_flag is not None:
        first["loss"] = loss_flag
    return [
        {"role": "user", "content": "Hello"},
        first,
        {"role": "user", "content": "Wait"},
        {"role": "assistant", "content": "Sorry to interrupt. Here it is."},
    ]


def test_absent_loss_key_keeps_the_turn_as_a_target(tokenizer):
    out = render_response_only_sample(_messages(None), tokenizer, 512)
    assert any(label != -100 for label in out["labels"])


def test_loss_true_keeps_the_turn_as_a_target(tokenizer):
    absent = render_response_only_sample(_messages(None), tokenizer, 512)
    explicit = render_response_only_sample(_messages(True), tokenizer, 512)
    assert absent["labels"] == explicit["labels"]


def test_loss_false_masks_that_turn_but_not_the_next(tokenizer):
    masked = render_response_only_sample(_messages(False), tokenizer, 512)
    unmasked = render_response_only_sample(_messages(None), tokenizer, 512)
    n_masked = sum(1 for label in masked["labels"] if label != -100)
    n_unmasked = sum(1 for label in unmasked["labels"] if label != -100)
    assert 0 < n_masked < n_unmasked


def test_grpo_loader_skips_a_loss_false_turn(tmp_path):
    """build_preference_pairs.py and mine_model_negatives.py both read their
    rows through _load_grpo_jsonl, so this one guard covers all three."""
    import json

    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    conv = {
        "workflow_graph": {"transitions": []},
        "ground_truth": {"terminal_state": "END", "terminal_reached": True},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "[STATE: A → A]\nkept"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "[STATE: A → A]\nskipped", "loss": False},
        ],
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(conv) + "\n")

    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 1
