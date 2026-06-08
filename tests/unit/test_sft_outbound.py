"""SFT response-only masking must keep the outbound opener (assistant) unmasked."""

from __future__ import annotations

from llm_workflow_agents.training.sft import render_response_only_sample


class _StubTok:
    """Deterministic, prefix-extending chat template: 1 token per char."""

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        s = "".join(f"{m['role']}|{m['content']}\n" for m in msgs)
        return [ord(c) % 256 for c in s]


def test_outbound_opener_tokens_are_unmasked():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "[STATE: G -> G] Hi, calling about X."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "[STATE: G -> V] Let me check."},
    ]
    out = render_response_only_sample(messages, _StubTok(), max_seq_length=10_000)
    labels = out["labels"]
    assert len(labels) == len(out["input_ids"])
    # At least some tokens are unmasked (the two assistant turns).
    assert any(l != -100 for l in labels)
    # System tokens (the prefix) are masked.
    assert labels[0] == -100


def test_system_then_assistant_only_keeps_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "opener"},
    ]
    out = render_response_only_sample(messages, _StubTok(), max_seq_length=10_000)
    # The very last token belongs to the assistant opener → unmasked.
    assert out["labels"][-1] != -100
