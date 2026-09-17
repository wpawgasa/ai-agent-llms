"""response_only trains exactly each assistant turn's content and terminator.

Spans come from character offsets over one render, so a token that straddles a
content boundary (Gemma-4 merges `>` and `[` into `>[`) is trained rather than
cut, and the template's turn header is never trained.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.sft import render_response_only_sample

CONVERSATION = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "check my eligibility"},
    {"role": "assistant", "content": '[STATE: CHECK → CHECK]\n<tool_call>{"name": "check", "arguments": {"id": "1100101928374"}}</tool_call>'},
    {"role": "tool", "content": '{"status": "documents_required"}'},
    {"role": "assistant", "content": "[STATE: CHECK → DOCS]\nPlease send your passport."},
    {"role": "user", "content": "ok"},
    {"role": "assistant", "content": "[STATE: DOCS → DOCS]\nThanks.", "loss": False},
]


def _gemma(repo):
    from tests.unit.test_training import _load_tokenizer_or_skip

    return _load_tokenizer_or_skip(repo)


def _trained_text(tok, out):
    return tok.decode(
        [i for i, l in zip(out["input_ids"], out["labels"]) if l != -100], skip_special_tokens=False
    )


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it", "Qwen/Qwen2.5-0.5B-Instruct"])
def test_every_assistant_content_is_trained_whole(repo):
    tok = _gemma(repo)
    trained = _trained_text(tok, render_response_only_sample(CONVERSATION, tok, 8192))
    for m in CONVERSATION:
        if m["role"] == "assistant" and m.get("loss", True):
            assert m["content"] in trained, m["content"][:40]


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it"])
def test_gemma_turn_after_a_tool_result_keeps_its_bracket_and_the_call_its_close(repo):
    tok = _gemma(repo)
    trained = _trained_text(tok, render_response_only_sample(CONVERSATION, tok, 8192))
    assert "[STATE: CHECK → DOCS]" in trained
    assert "</tool_call>" in trained


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it"])
def test_gemma_header_tool_result_and_user_text_are_not_trained(repo):
    tok = _gemma(repo)
    trained = _trained_text(tok, render_response_only_sample(CONVERSATION, tok, 8192))
    assert "<|turn>model" not in trained
    assert "documents_required" not in trained
    assert "check my eligibility" not in trained
    assert "Thanks." not in trained, "a loss:false turn must stay untrained"


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it"])
def test_gemma_trains_the_end_of_turn_marker(repo):
    tok = _gemma(repo)
    trained = _trained_text(tok, render_response_only_sample(CONVERSATION, tok, 8192))
    assert trained.count("<turn|>") >= 2


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it"])
def test_gemma_input_ids_equal_the_template_tokenization(repo):
    from llm_workflow_agents.data.tool_turns import to_text_tool_turns
    from llm_workflow_agents.training._utils import normalize_chat_template_ids

    tok = _gemma(repo)
    out = render_response_only_sample(CONVERSATION, tok, 8192)
    expected = normalize_chat_template_ids(
        tok.apply_chat_template(
            [{k: v for k, v in m.items() if k in ("role", "content")} for m in to_text_tool_turns(CONVERSATION)],
            tokenize=True,
        )
    )
    assert out["input_ids"] == expected


def test_consecutive_assistant_turns_are_each_trained():
    tok = _gemma("google/gemma-4-12B-it")
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[STATE: A → A]\nOne moment."},
        {"role": "assistant", "content": "[STATE: A → B]\nDone."},
    ]
    trained = _trained_text(tok, render_response_only_sample(conv, tok, 8192))
    assert "[STATE: A → A]\nOne moment." in trained and "[STATE: A → B]\nDone." in trained


def test_left_truncation_keeps_the_last_turn_trained():
    tok = _gemma("google/gemma-4-12B-it")
    out = render_response_only_sample(CONVERSATION, tok, 40)
    assert len(out["input_ids"]) == 40 and any(l != -100 for l in out["labels"])
