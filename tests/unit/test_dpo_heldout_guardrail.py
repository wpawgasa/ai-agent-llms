"""dpo.py's R5 held-out guardrail must draw only from the fingerprint slice
scripts/mine_model_negatives.py reserves (Task 1/3) — never from rows that
could also be mined as DPO training negatives, or the guardrail is no longer
independent of what the policy was just trained to prefer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint
from llm_workflow_agents.training.dpo import _build_heldout_callback


def _row(cid: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"hello from {cid}"}],
        "ground_truth": json.dumps({"tool_calls": [], "state_sequence": []}),
    }


class _FakeDataset(list):
    pass


def test_heldout_rows_are_restricted_to_the_reserved_fingerprints(tmp_path):
    all_rows = [_row(f"c{i}") for i in range(10)]
    fake_ds = _FakeDataset(all_rows)

    # Reserve only c0, c1, c2's prefix fingerprints.
    reserved_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in all_rows[:3]
    }

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=reserved_fps,
        ),
    ):
        callback = _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 50},
            max_new_tokens=64,
        )

    selected_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in callback.held_out_rows
    }
    assert selected_fps == reserved_fps
    assert len(callback.held_out_rows) == 3


def test_respects_eval_held_out_num_prompts_cap(tmp_path):
    all_rows = [_row(f"c{i}") for i in range(10)]
    fake_ds = _FakeDataset(all_rows)
    reserved_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in all_rows
    }  # all 10 reserved

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=reserved_fps,
        ),
    ):
        callback = _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 4},
            max_new_tokens=64,
        )

    assert len(callback.held_out_rows) == 4


def test_reserved_fraction_config_key_is_passed_through(tmp_path):
    all_rows = [_row("c0")]
    fake_ds = _FakeDataset(all_rows)

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=set(),
        ) as mock_reserve,
    ):
        _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={
                "heldout_data_source": "data/output/grpo/task_a",
                "guardrail_reserved_fraction": 0.35,
                "guardrail_reserved_seed": 7,
            },
            monitoring_cfg={"eval_held_out_num_prompts": 50},
            max_new_tokens=64,
        )

    mock_reserve.assert_called_once_with(
        Path("data/output/grpo/task_a"),
        split="validation",
        reserved_fraction=0.35,
        seed=7,
    )


def test_evaluate_releases_cuda_cache_so_the_next_step_can_allocate():
    """The guardrail generates mid-training, on the same GPU as the optimizer.

    `model.generate()` allocates a KV cache per prompt. Without an explicit
    release those blocks stay in PyTorch's caching allocator, so the training
    step that follows the eval starts with less memory than the one before it.
    Measured 2026-08-18 (checkpoints/dpo_cat_a_smoke7/): three DPO steps ran,
    the guardrail fired at step 3, and step 4 then died asking for 52 MiB with
    31.9 MiB free and 525 MiB "reserved but unallocated". The real config
    generates 50 prompts per eval, not the 2 that produced that failure.
    """
    rows = [_row("c0")]
    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=_FakeDataset(rows),
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value={user_turn_fingerprint({"messages": rows[0]["prompt"]})},
        ),
    ):
        callback = _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 1},
            max_new_tokens=8,
        )

    with (
        patch("torch.cuda.empty_cache") as empty_cache,
        patch(
            "llm_workflow_agents.training.dpo.heldout_composite_score",
            return_value=0.5,
        ),
    ):
        callback._evaluate()

    assert empty_cache.called, (
        "guardrail must release its generation KV cache before training resumes"
    )


def test_evaluate_restores_use_cache_after_generate_flips_it():
    """model.generate() turns use_cache back on; the callback must undo that.

    HF Trainer sets ``model.config.use_cache = False`` at train start under
    gradient checkpointing. ``model.generate()`` flips it back to ``True`` to
    speed up decoding and, on a bare model, leaves it there. If the callback
    does not restore it, every training step after the first guardrail eval
    runs with a KV cache gradient checkpointing had disabled — the leading
    hypothesis for R19 / docs/dpo_memory_ceiling_investigation.md §5.
    """
    rows = [_row("c0")]

    class _FakeConfig:
        def __init__(self) -> None:
            self.use_cache = False  # what HF Trainer sets under grad checkpointing

    model = MagicMock()
    model.config = _FakeConfig()

    def _generate(**kwargs):
        model.config.use_cache = True  # what model.generate() does in real HF
        return MagicMock()

    model.generate.side_effect = _generate

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=_FakeDataset(rows),
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value={user_turn_fingerprint({"messages": rows[0]["prompt"]})},
        ),
    ):
        callback = _build_heldout_callback(
            model=model,
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 1},
            max_new_tokens=8,
        )

    with (
        patch("torch.cuda.empty_cache"),
        patch(
            "llm_workflow_agents.training.dpo.heldout_composite_score",
            return_value=0.5,
        ),
    ):
        callback._evaluate()

    assert model.config.use_cache is False, (
        "guardrail left use_cache=True after generate() — the next training "
        "step would allocate a KV cache it didn't before the eval"
    )
