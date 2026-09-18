"""No render path may hand a raw `tool` message to a chat template again.

The Gemma-4 template silently drops `tool` messages that do not answer
structured `tool_calls`; that is how every Gemma-4 SFT run on the Task A corpus
trained without a single tool result (docs/superpowers/specs/
2026-09-17-gemma4-tool-results-design.md). Each test drives a real render path
with a tokenizer that refuses a `tool` role, and checks the tool result's text
reaches the rendered input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TOOL_TEXT = '{"status": "documents_required"}'

CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "check my eligibility"},
    {
        "role": "assistant",
        "content": '[STATE: CHECK → CHECK]\n<tool_call>{"name": "check", "arguments": {"id": "1100101928374"}}</tool_call>',
        "annotations": {
            "state_transition": {"from": "CHECK", "to": "CHECK"},
            "tool_calls": [{"name": "check", "arguments": {"id": "1100101928374"}}],
        },
    },
    {"role": "tool", "content": TOOL_TEXT},
    {
        "role": "assistant",
        "content": "[STATE: CHECK → DOCS]\nPlease send your passport.",
        "annotations": {"state_transition": {"from": "CHECK", "to": "DOCS"}},
    },
]


class _NoToolRoleTok:
    """Renders like a simple template, but fails loudly on a `tool` role."""

    def _render(self, msgs, add_generation_prompt=False):
        for m in msgs:
            assert m["role"] != "tool", "a raw tool message reached apply_chat_template"
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        return text + ("<assistant>" if add_generation_prompt else "")

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **_):
        text = self._render(msgs, add_generation_prompt)
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        ids = [ord(c) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_sft_response_only_render_includes_and_masks_the_tool_result():
    from llm_workflow_agents.training.sft import render_response_only_sample

    out = render_response_only_sample(CONVERSATION, _NoToolRoleTok(), max_seq_length=10_000)
    rendered = "".join(chr(i) for i in out["input_ids"])
    trained = "".join(chr(i) for i, l in zip(out["input_ids"], out["labels"]) if l != -100)
    assert TOOL_TEXT in rendered
    assert TOOL_TEXT not in trained
    assert "[STATE: CHECK → DOCS]" in trained


def test_grpo_loader_puts_tool_results_in_prompts_and_admits_post_tool_turns(tmp_path: Path):
    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    row = {"messages": CONVERSATION, "ground_truth": {"terminal_state": "DOCS"}}
    (tmp_path / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 2, "the assistant turn after the tool result must be a row"
    post_tool = ds[1]["prompt"]
    assert all(m["role"] != "tool" for m in post_tool)
    assert any(TOOL_TEXT in m["content"] for m in post_tool)


def test_trajectory_segment_render_includes_the_tool_result():
    from llm_workflow_agents.training.trajectory_rollout import _segment_suffix_ids

    ids = _segment_suffix_ids(_NoToolRoleTok(), [{"role": "tool", "content": TOOL_TEXT}])
    assert TOOL_TEXT in "".join(chr(i) for i in ids)


def test_benchmark_text_mode_sends_no_tool_role_and_no_tools(monkeypatch):
    from llm_workflow_agents.eval import agent_benchmark as ab

    seen = []

    def fake_call(endpoint, model, messages, temperature, tools=None, **_):
        seen.append((messages, tools))
        # Call the tool on the first turn: a result is shown only for a call
        # the model made (test_replay_tool_result_gating.py).
        if len(seen) == 1:
            return CONVERSATION[2]["content"], [], 1.0, 1.0
        return "[STATE: CHECK → DOCS]\nok", [], 1.0, 1.0

    monkeypatch.setattr(ab, "_call_vllm", fake_call)
    sample = {"messages": CONVERSATION, "tool_schemas": [{"type": "function", "function": {"name": "check"}}]}
    ab._replay_conversation("http://x", "m", sample, tool_turn_format="text")
    assert seen, "the model was never called"
    for messages, tools in seen:
        assert tools is None
        assert all(m["role"] != "tool" for m in messages)
    assert any(TOOL_TEXT in m["content"] for m in seen[-1][0])


def test_benchmark_native_mode_is_unchanged(monkeypatch):
    from llm_workflow_agents.eval import agent_benchmark as ab

    seen = []

    def fake_call(endpoint, model, messages, temperature, tools=None, **_):
        seen.append(tools)
        return "[STATE: CHECK → DOCS]\nok", [], 1.0, 1.0

    monkeypatch.setattr(ab, "_call_vllm", fake_call)
    sample = {"messages": CONVERSATION, "tool_schemas": [{"type": "function", "function": {"name": "check"}}]}
    ab._replay_conversation("http://x", "m", sample)
    assert seen and all(tools == sample["tool_schemas"] for tools in seen)


def test_prompt_filter_finds_carried_values_in_converted_turns():
    from llm_workflow_agents.data.tool_turns import to_text_tool_turns
    from llm_workflow_agents.training.prompt_filters import propagated_arguments

    prompt = to_text_tool_turns([
        {"role": "tool", "content": '{"order_id": "ORD-9912"}'},
    ])
    gt = [{"name": "cancel", "arguments": {"order_id": "ORD-9912"}}]
    assert propagated_arguments(prompt, gt)[0]["value"] == "ORD-9912"


@pytest.mark.parametrize("repo", ["google/gemma-4-12B-it", "unsloth/gemma-4-E4B-it"])
def test_real_gemma_templates_now_render_every_tool_result(repo):
    from tests.unit.test_training import _load_tokenizer_or_skip
    from llm_workflow_agents.data.tool_turns import to_text_tool_turns

    tok = _load_tokenizer_or_skip(repo)
    raw = [{k: v for k, v in m.items() if k in ("role", "content")} for m in CONVERSATION]
    assert TOOL_TEXT not in tok.apply_chat_template(raw, tokenize=False), "template no longer drops tool messages"
    assert TOOL_TEXT in tok.apply_chat_template(to_text_tool_turns(raw), tokenize=False)
