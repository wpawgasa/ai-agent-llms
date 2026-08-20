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


class TestCorpusHasLossFalse:
    """`_corpus_has_loss_false` backs the all_tokens safety warning below.

    Its whole job is deciding whether to warn — it must never itself raise,
    even on a corpus so broken that `_load_split` would crash on it later.
    """

    def test_missing_directory_returns_false(self, tmp_path):
        from llm_workflow_agents.training.sft import _corpus_has_loss_false

        assert _corpus_has_loss_false(str(tmp_path / "does_not_exist")) is False

    def test_empty_file_returns_false(self, tmp_path):
        import json

        from llm_workflow_agents.training.sft import _corpus_has_loss_false

        (tmp_path / "train.jsonl").write_text("")
        assert _corpus_has_loss_false(str(tmp_path)) is False

    def test_malformed_line_does_not_raise_and_is_skipped(self, tmp_path):
        import json

        from llm_workflow_agents.training.sft import _corpus_has_loss_false

        (tmp_path / "train.jsonl").write_text("{not valid json\n")
        # Does not raise, and finds no evidence in the unparseable line.
        assert _corpus_has_loss_false(str(tmp_path)) is False

    def test_malformed_line_does_not_hide_a_real_match_in_the_same_file(self, tmp_path):
        import json

        from llm_workflow_agents.training.sft import _corpus_has_loss_false

        conv = {"messages": [{"role": "assistant", "content": "x", "loss": False}]}
        (tmp_path / "train.jsonl").write_text(
            "{not valid json\n" + json.dumps(conv) + "\n"
        )
        assert _corpus_has_loss_false(str(tmp_path)) is True

    def test_malformed_line_logs_a_warning(self, tmp_path):
        import json

        from structlog.testing import capture_logs

        from llm_workflow_agents.training.sft import _corpus_has_loss_false

        (tmp_path / "train.jsonl").write_text("{not valid json\n")
        with capture_logs() as logs:
            _corpus_has_loss_false(str(tmp_path))
        assert any(
            log["event"] == "corpus_has_loss_false_check_skipped_lines"
            for log in logs
        )


def _write_sft_config(tmp_path, loss_mask, has_loss_false):
    """A minimal, valid-enough sft.yaml that reaches the loss_mask warning
    block in train_sft and then exits early on a missing model.config_path —
    before any model is loaded. Fast enough for a unit test."""
    import json

    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    asst_msg = {"role": "assistant", "content": "hi"}
    if has_loss_false:
        asst_msg["loss"] = False
    conv = {"messages": [{"role": "user", "content": "hello"}, asst_msg]}
    (data_dir / "train.jsonl").write_text(json.dumps(conv) + "\n")
    (data_dir / "validation.jsonl").write_text(json.dumps(conv) + "\n")

    cfg_path = tmp_path / "sft.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "stage": "sft",
                "training": {"loss_mask": loss_mask},
                "data": {"source": str(data_dir)},
                # model.config_path deliberately omitted: train_sft returns
                # an error SFTResult right after the loss_mask warning block,
                # so this stays a fast unit test with no model load.
            }
        )
    )
    return cfg_path


class TestAllTokensLossFlagWarning:
    """The sole guard against a silent all_tokens run over voice data
    teaching the model to emit the orchestrator-written <unspoken> marker
    (risk R15's failure shape: a problem invisible until a held-out audit).
    Assert on the emitted log record itself, not just on
    `_corpus_has_loss_false`'s return value — the return value being right
    while the warning never reaches the log is exactly what this must catch.
    """

    def test_fires_under_all_tokens_with_a_loss_false_message(self, tmp_path):
        from structlog.testing import capture_logs

        from llm_workflow_agents.training.sft import train_sft

        cfg_path = _write_sft_config(tmp_path, "all_tokens", has_loss_false=True)
        with capture_logs() as logs:
            train_sft(cfg_path)
        assert any(log["event"] == "all_tokens_ignores_loss_flag" for log in logs)

    def test_does_not_fire_under_response_only_with_the_same_corpus(self, tmp_path):
        from structlog.testing import capture_logs

        from llm_workflow_agents.training.sft import train_sft

        cfg_path = _write_sft_config(tmp_path, "response_only", has_loss_false=True)
        with capture_logs() as logs:
            train_sft(cfg_path)
        assert not any(log["event"] == "all_tokens_ignores_loss_flag" for log in logs)

    def test_does_not_fire_under_all_tokens_with_no_loss_key_anywhere(self, tmp_path):
        from structlog.testing import capture_logs

        from llm_workflow_agents.training.sft import train_sft

        cfg_path = _write_sft_config(tmp_path, "all_tokens", has_loss_false=False)
        with capture_logs() as logs:
            train_sft(cfg_path)
        assert not any(log["event"] == "all_tokens_ignores_loss_flag" for log in logs)
