"""Checkpoint output-dir resolution for DPO/ORPO runs.

Ports the R13 regression cover from `test_sft_output_dir.py` /
`test_grpo_output_dir.py` to the preference-learning path, so it inherits the
same guarantee from day one instead of rediscovering the bug: an explicit
`output_dir` is the only way to give a run its own checkpoint directory, and a
`run_phase2_*.sh`-style timestamp suffix on the config filename must never leak
into the checkpoint path.
"""

from pathlib import Path

from llm_workflow_agents.training.dpo import _resolve_output_dir

MODEL = "gemma-4-26B-A4B-it"


def test_run_stamped_config_resolves_to_stable_dir():
    """A run stamp is provenance for the config, not part of the output path."""
    assert _resolve_output_dir(
        {}, Path(".runs/dpo_cat_a/dpo_cat_a_20260817T101500Z.yaml"), MODEL
    ) == Path("checkpoints/dpo_cat_a/gemma-4-26B-A4B-it")


def test_unstamped_config_keeps_its_stem():
    assert _resolve_output_dir(
        {}, Path("configs/training/dpo_cat_a.yaml"), MODEL
    ) == Path("checkpoints/dpo_cat_a/gemma-4-26B-A4B-it")


def test_two_runs_of_same_config_share_one_dir():
    """Distinct run stamps must not fan out into distinct checkpoint dirs."""
    a = _resolve_output_dir({}, Path("dpo_cat_a_20260817T101500Z.yaml"), MODEL)
    b = _resolve_output_dir({}, Path("dpo_cat_a_20260818T093000Z.yaml"), MODEL)
    assert a == b


def test_explicit_output_dir_wins():
    """The supported way to give a DPO/ORPO run its own directory."""
    assert _resolve_output_dir(
        {"output_dir": "dpo_cat_a_orpo"},
        Path("dpo_cat_a_20260817T101500Z.yaml"),
        MODEL,
    ) == Path("checkpoints/dpo_cat_a_orpo/gemma-4-26B-A4B-it")


def test_explicit_output_dir_survives_a_stamped_name():
    """An explicit name is used verbatim — never stamp-stripped."""
    assert _resolve_output_dir(
        {"output_dir": "cell_20260817T101500Z"}, Path("dpo_cat_a.yaml"), MODEL
    ) == Path("checkpoints/cell_20260817T101500Z/gemma-4-26B-A4B-it")


def test_only_a_trailing_stamp_is_stripped():
    """A stamp-shaped substring mid-name is not a run stamp."""
    assert _resolve_output_dir(
        {}, Path("dpo_20260817T101500Z_ablation.yaml"), MODEL
    ) == Path("checkpoints/dpo_20260817T101500Z_ablation/gemma-4-26B-A4B-it")


def test_model_basename_is_flattened():
    """A full HF repo id must not nest the checkpoint under an org directory."""
    assert _resolve_output_dir(
        {}, Path("dpo_cat_a.yaml"), "google/gemma-4-26B-A4B-it"
    ) == Path("checkpoints/dpo_cat_a/gemma-4-26B-A4B-it")
