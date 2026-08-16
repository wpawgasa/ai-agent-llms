"""Checkpoint output-dir resolution for GRPO runs.

Ports the R13 regression cover from `test_sft_output_dir.py` to the GRPO path,
which had the identical defect: `grpo.py` derived the checkpoint directory from
`Path(config_path).stem`, so the moment `run_phase2_grpo.sh` gains a run-stamped
patched config (as `run_phase2_sft.sh` already has), every GRPO run would write
its weights to a per-run path the DVC stage does not track.

The explicit `output_dir` key is also how a run is given its own directory —
without it a second GRPO run silently overwrites the first.
"""

from pathlib import Path

from llm_workflow_agents.training.grpo import _resolve_output_dir

MODEL = "gemma-4-26B-A4B-it"


def test_run_stamped_config_resolves_to_stable_dir():
    """A run stamp is provenance for the config, not part of the output path."""
    assert _resolve_output_dir(
        {}, Path(".runs/grpo_cat_a/grpo_cat_a_20260816T101500Z.yaml"), MODEL
    ) == Path("checkpoints/grpo_cat_a/gemma-4-26B-A4B-it")


def test_unstamped_config_keeps_its_stem():
    assert _resolve_output_dir(
        {}, Path("configs/training/grpo_cat_a.yaml"), MODEL
    ) == Path("checkpoints/grpo_cat_a/gemma-4-26B-A4B-it")


def test_two_runs_of_same_config_share_one_dir():
    """Distinct run stamps must not fan out into distinct checkpoint dirs."""
    a = _resolve_output_dir({}, Path("grpo_cat_a_20260816T101500Z.yaml"), MODEL)
    b = _resolve_output_dir({}, Path("grpo_cat_a_20260817T093000Z.yaml"), MODEL)
    assert a == b


def test_explicit_output_dir_wins():
    """The supported way to give a GRPO run its own directory."""
    assert _resolve_output_dir(
        {"output_dir": "grpo_cat_a_c2"},
        Path("grpo_cat_a_20260816T101500Z.yaml"),
        MODEL,
    ) == Path("checkpoints/grpo_cat_a_c2/gemma-4-26B-A4B-it")


def test_explicit_output_dir_survives_a_stamped_name():
    """An explicit name is used verbatim — never stamp-stripped."""
    assert _resolve_output_dir(
        {"output_dir": "cell_20260816T101500Z"}, Path("grpo_cat_a.yaml"), MODEL
    ) == Path("checkpoints/cell_20260816T101500Z/gemma-4-26B-A4B-it")


def test_only_a_trailing_stamp_is_stripped():
    """A stamp-shaped substring mid-name is not a run stamp."""
    assert _resolve_output_dir(
        {}, Path("grpo_20260816T101500Z_ablation.yaml"), MODEL
    ) == Path("checkpoints/grpo_20260816T101500Z_ablation/gemma-4-26B-A4B-it")


def test_model_basename_is_flattened():
    """A full HF repo id must not nest the checkpoint under an org directory."""
    assert _resolve_output_dir(
        {}, Path("grpo_cat_a.yaml"), "google/gemma-4-26B-A4B-it"
    ) == Path("checkpoints/grpo_cat_a/gemma-4-26B-A4B-it")
