"""Conversation-level dataset + gold-script index for trajectory GRPO.

The per-turn GRPO setup gives the reward almost no resolution: scored on 206
real C2 completions it takes 11 distinct values and sits at exactly 1.0 on
81.1% of them, so groups tie and GRPO's advantage is zero (the 2026-08-16
diagnostic logged `reward_std 0, frac_reward_zero_std 1` at every step).

`reward_business_logic_trajectory` is the project's designed answer —
aggregating over T turns turns that discrete lattice into a near-continuous
distribution. It needs one row per *conversation* rather than one per assistant
turn, plus a `prompt_key -> GoldScript` index the rollout uses to replay gold
user/tool turns between the model's own turns.

The skip-don't-crash behaviour is the load-bearing part: `build_gold_script`
raises `ValueError` when a conversation's gold `state_sequence` length does not
match its assistant-turn count, and one such row must not take down a load of
thousands.
"""

import json

import pytest

from llm_workflow_agents.training.grpo import _load_grpo_trajectory_dataset
from llm_workflow_agents.training.trajectory_rollout import prompt_key


def _conversation(cid, n_turns=2, bad_gt=False):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"u0-{cid}"},
    ]
    for t in range(n_turns):
        messages.append({"role": "assistant", "content": f"[STATE: A → B] a{t}"})
        if t < n_turns - 1:
            messages.append({"role": "user", "content": f"u{t + 1}"})
    seq = [{"from": "A", "to": "B"} for _ in range(n_turns - (1 if bad_gt else 0))]
    return {
        "conversation_id": cid,
        "messages": messages,
        "workflow_graph": {"transitions": [{"from": "A", "to": "B"}]},
        "ground_truth": {
            "state_sequence": seq,
            "tool_calls": [{"name": "t", "arguments": {}}],
            "terminal_state": "B",
            "terminal_reached": True,
        },
    }


def _write(tmp_path, rows, split="train"):
    p = tmp_path / f"{split}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_one_row_per_conversation(tmp_path):
    d = _write(tmp_path, [_conversation("c1"), _conversation("c2")])
    ds, index = _load_grpo_trajectory_dataset(d, split="train")
    assert len(ds) == 2
    assert len(index) == 2


def test_prompt_is_the_pre_assistant_prefix(tmp_path):
    d = _write(tmp_path, [_conversation("c1")])
    ds, _ = _load_grpo_trajectory_dataset(d, split="train")
    prompt = ds[0]["prompt"]
    assert [m["role"] for m in prompt] == ["system", "user"]


def test_index_is_keyed_by_prompt_key_of_the_row_prompt(tmp_path):
    """The rollout looks scripts up by prompt_key — the two must agree exactly."""
    d = _write(tmp_path, [_conversation("c1"), _conversation("c2")])
    ds, index = _load_grpo_trajectory_dataset(d, split="train")
    for row in ds:
        assert prompt_key(row["prompt"]) in index


def test_ground_truth_carries_the_trajectory_reward_keys(tmp_path):
    d = _write(tmp_path, [_conversation("c1")])
    ds, _ = _load_grpo_trajectory_dataset(d, split="train")
    gt = json.loads(ds[0]["ground_truth"])
    for key in (
        "state_sequence",
        "tool_calls",
        "terminal_state",
        "terminal_reached",
        "valid_transitions",
    ):
        assert key in gt, key


def test_valid_transitions_come_from_the_workflow_graph(tmp_path):
    d = _write(tmp_path, [_conversation("c1")])
    ds, _ = _load_grpo_trajectory_dataset(d, split="train")
    gt = json.loads(ds[0]["ground_truth"])
    assert gt["valid_transitions"] == [["A", "B"]]


def test_malformed_conversation_is_skipped_not_fatal(tmp_path):
    """A gold_transitions/assistant-turn mismatch must not kill the whole load."""
    d = _write(tmp_path, [_conversation("ok"), _conversation("bad", bad_gt=True)])
    ds, index = _load_grpo_trajectory_dataset(d, split="train")
    assert len(ds) == 1
    assert len(index) == 1


def test_all_malformed_raises_rather_than_returning_empty(tmp_path):
    """An empty training set must fail loudly, not start a no-op run."""
    d = _write(tmp_path, [_conversation("bad", bad_gt=True)])
    with pytest.raises(ValueError):
        _load_grpo_trajectory_dataset(d, split="train")


def test_prompt_key_collisions_are_deduped_not_silently_mismapped(tmp_path):
    """Two conversations sharing a prompt must not both stay in the dataset.

    The rollout resolves a script by `prompt_key(prompt)`, so a dict keyed that
    way keeps only the LAST colliding conversation. Left alone, every earlier
    colliding row replays a different conversation's gold segments and is scored
    against its transitions and tool calls — silent, per-row corruption of the
    reward. Measured on the real corpus: 2,558 conversations collapse to 2,420
    keys, so 138 rows (5.4%) were affected, one key colliding 8 ways.

    Rows and index must therefore agree exactly.
    """
    a = _conversation("dup-a")
    b = _conversation("dup-b")
    b["messages"][0]["content"] = a["messages"][0]["content"]
    b["messages"][1]["content"] = a["messages"][1]["content"]  # same prompt prefix

    ds, index = _load_grpo_trajectory_dataset(_write(tmp_path, [a, b]), split="train")
    assert len(ds) == len(index), "every row must own exactly one script"
    assert len(ds) == 1


def test_dedup_keeps_a_row_whose_script_is_its_own(tmp_path):
    """The surviving row's script must be the one built from that same row."""
    a = _conversation("dup-a", n_turns=2)
    b = _conversation("dup-b", n_turns=3)
    b["messages"][0]["content"] = a["messages"][0]["content"]
    b["messages"][1]["content"] = a["messages"][1]["content"]

    ds, index = _load_grpo_trajectory_dataset(_write(tmp_path, [a, b]), split="train")
    row = ds[0]
    script = index[prompt_key(row["prompt"])]
    gt = json.loads(row["ground_truth"])
    assert len(script.gold_transitions) == len(gt["state_sequence"])
    assert len(script.segments) == len(script.gold_transitions)


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_grpo_trajectory_dataset(tmp_path, split="nope")
