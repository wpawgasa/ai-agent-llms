"""Trainer kwargs handed to TRL's DPOConfig by `training/dpo.py`.

R16 all over again, one objective further down the pipeline. TRL's
`DataCollatorForPreference` concatenates `prompt_ids + chosen_ids` (and the
rejected counterpart) and slices the result to `DPOConfig.max_length`, whose
default is **1024**, with `truncation_mode='keep_start'`.

Cat A prompts are median ~4,400 tokens and never shorter than ~2,400, while
`chosen` and `rejected` differ ONLY in the trailing assistant turn. So at the
default the first 1024 tokens are pure system prompt, the two sequences
truncate to byte-identical token ids, and `completion_mask` is all zeros:
no token is scored, the implicit margin is exactly 0, the loss sits at
log 2, and the gradient is exactly 0 for every step of the run.

That is the DPO analogue of the tied-group failure that made GRPO unlearnable
(R18) — and, like R16, it fails silently rather than crashing. The configured
ceiling must reach `DPOConfig`.

Requires a real `trl` import, so — like `test_dpo_config_kwargs.py` — this
only collects under an environment with `trl` installed (`.venv-train`), not
CI's lightweight `dev` extra.
"""

import dataclasses

import pytest

pytest.importorskip("trl")
from trl import DPOConfig  # noqa: E402

from llm_workflow_agents.training.dpo import (
    _dpo_trainer_kwargs,
    _filter_dpo_config_kwargs,
)


def _built(dpo_cfg: dict, output_dir: str) -> DPOConfig:
    """Build a real DPOConfig the same way train_dpo does."""
    kwargs = _dpo_trainer_kwargs(dpo_cfg, "dpo", output_dir=output_dir)
    kept, _dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    return DPOConfig(**kept)


def test_configured_max_seq_length_reaches_dpo_config(tmp_path):
    cfg = _built({"max_seq_length": 8192}, str(tmp_path))
    assert cfg.max_length == 8192, (
        f"collator would truncate chosen/rejected to {cfg.max_length} tokens, "
        "not the configured 8192"
    )


def test_does_not_silently_fall_back_to_the_trl_default(tmp_path):
    cfg = _built({"max_seq_length": 4096}, str(tmp_path))
    assert cfg.max_length == 4096
    assert cfg.max_length != 1024, "fell back to the TRL default"
    assert cfg.truncation_mode == "keep_start"


def test_max_length_survives_the_trl_version_filter(tmp_path):
    """A ceiling dropped by _filter_dpo_config_kwargs is a ceiling not applied."""
    kwargs = _dpo_trainer_kwargs({"max_seq_length": 8192}, "dpo", output_dir=str(tmp_path))
    kept, dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    assert "max_length" in kept
    assert "max_length" not in dropped


def test_omitted_max_seq_length_defaults_to_8192_not_1024(tmp_path):
    cfg = _built({}, str(tmp_path))
    assert cfg.max_length == 8192


def test_eval_batch_size_follows_the_train_batch_size(tmp_path):
    """TRL defaults per_device_eval_batch_size to 8.

    DPO's eval forward scores chosen AND rejected, so 8 becomes 16 sequences of
    up to 8192 tokens against Gemma-4's 262,144-token vocab — an instant OOM on
    an 80GB card. It only survives at the TRL default because the collator is
    truncating everything to 1024, so raising max_length without pinning this
    trades a silent no-op run for a crash at the first eval.
    """
    cfg = _built({"max_seq_length": 8192, "per_device_train_batch_size": 1}, str(tmp_path))
    assert cfg.per_device_eval_batch_size == 1


def test_explicit_eval_batch_size_is_respected(tmp_path):
    cfg = _built(
        {"per_device_train_batch_size": 1, "per_device_eval_batch_size": 2},
        str(tmp_path),
    )
    assert cfg.per_device_eval_batch_size == 2


def test_beta_is_set_for_dpo_but_not_for_orpo(tmp_path):
    dpo = _dpo_trainer_kwargs({"beta": 0.1}, "dpo", output_dir=str(tmp_path))
    orpo = _dpo_trainer_kwargs({"beta": 0.1}, "orpo", output_dir=str(tmp_path))
    assert dpo["beta"] == 0.1
    assert "beta" not in orpo


def test_max_length_is_a_real_dpoconfig_field(tmp_path):
    """Guards the assertion above against a TRL rename making it vacuous."""
    assert "max_length" in {f.name for f in dataclasses.fields(DPOConfig)}


def test_precompute_ref_log_probs_reaches_dpo_config(tmp_path):
    """The reference forward is what OOMs a 26B Gemma-4 DPO step.

    Unsloth's compiled trainer materializes four full [2, S, 262144] fp32
    logits tensors per step — policy and reference, each `.contiguous()`-copied
    while the original is live. That is ~33 GiB at this corpus's median prompt
    length. `precompute_ref_log_probs` makes the trainer read cached reference
    logps from the batch instead, removing the reference forward and its two
    tensors entirely.
    """
    cfg = _built({"precompute_ref_log_probs": True}, str(tmp_path))
    assert cfg.precompute_ref_log_probs is True


def test_precompute_ref_log_probs_defaults_to_off(tmp_path):
    assert _built({}, str(tmp_path)).precompute_ref_log_probs is False


def test_precompute_ref_batch_size_is_forwarded(tmp_path):
    cfg = _built(
        {"precompute_ref_log_probs": True, "precompute_ref_batch_size": 4},
        str(tmp_path),
    )
    assert cfg.precompute_ref_batch_size == 4


def test_use_liger_kernel_reaches_dpo_config(tmp_path):
    """The fused Liger DPO loss is the only lever that removes the [2,S,262144]
    fp32 logits tensor from BOTH forward and backward.

    precompute_ref_log_probs deletes the reference half; the policy half still
    OOMs a 26B Gemma-4 at 6144 tokens in `autograd.backward`. Unsloth's compiled
    trainer branches to `_compute_loss_liger` on this flag, which computes the
    loss without ever materializing logits.
    """
    cfg = _built({"use_liger_kernel": True}, str(tmp_path))
    assert cfg.use_liger_kernel is True


def test_use_liger_kernel_defaults_to_off(tmp_path):
    assert _built({}, str(tmp_path)).use_liger_kernel is False


def test_shipped_cat_a_config_does_not_enable_liger_and_precompute_together():
    """TRL rejects the combination at trainer __init__ — after the 26B load.

    `Liger DPO loss does not support precomputing reference log probabilities.`
    Liger derives the reference inside its fused kernel without materializing
    logits, so precompute is redundant there, not additive. Enabling both is
    always a mistake, and it costs a model load to find out.
    """
    import yaml

    with open("configs/training/dpo_cat_a.yaml") as fh:
        dpo = yaml.safe_load(fh)["dpo"]
    assert not (
        dpo.get("use_liger_kernel") and dpo.get("precompute_ref_log_probs")
    ), "use_liger_kernel and precompute_ref_log_probs are mutually exclusive"
