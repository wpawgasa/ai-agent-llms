"""Checkpoint output-dir resolution for SFT runs.

Regression cover for the 2026-07-22 fault: R13 made the patched SFT config
run-specific (`sft_cat_a_<RUN_TS>.yaml`), and `sft.py` derived the checkpoint
directory from that filename stem — so a full 1,770-step run wrote its weights
to `checkpoints/sft_cat_a_20260722T051246Z/...` while the DVC stage's declared
output `checkpoints/sft_cat_a/...` captured only `train.log`.
"""

from pathlib import Path

from llm_workflow_agents.training.sft import _resolve_output_dir

MODEL = "google/gemma-4-26B-A4B-it"


def test_run_stamped_config_resolves_to_stable_dir():
    """The R13 timestamp must not leak into the checkpoint path."""
    assert _resolve_output_dir(
        {}, Path(".runs/sft_cat_a/sft_cat_a_20260722T051246Z.yaml"), MODEL
    ) == Path("checkpoints/sft_cat_a/gemma-4-26B-A4B-it")


def test_unstamped_config_keeps_its_stem():
    assert _resolve_output_dir(
        {}, Path("configs/training/sft_cat_a.yaml"), MODEL
    ) == Path("checkpoints/sft_cat_a/gemma-4-26B-A4B-it")


def test_two_runs_of_same_config_share_one_dir():
    """Distinct run stamps must not fan out into distinct checkpoint dirs."""
    a = _resolve_output_dir({}, Path("sft_cat_a_20260722T051246Z.yaml"), MODEL)
    b = _resolve_output_dir({}, Path("sft_cat_a_20260723T094500Z.yaml"), MODEL)
    assert a == b


def test_explicit_output_dir_wins():
    """The supported way to give a factorial cell its own directory."""
    assert _resolve_output_dir(
        {"output_dir": "sft_cat_a_c1"},
        Path("sft_cat_a_20260722T051246Z.yaml"),
        MODEL,
    ) == Path("checkpoints/sft_cat_a_c1/gemma-4-26B-A4B-it")


def test_explicit_output_dir_survives_a_stamped_name():
    """An explicit name is used verbatim — never stamp-stripped."""
    assert _resolve_output_dir(
        {"output_dir": "cell_20260722T051246Z"}, Path("sft_cat_a.yaml"), MODEL
    ) == Path("checkpoints/cell_20260722T051246Z/gemma-4-26B-A4B-it")


def test_only_a_trailing_stamp_is_stripped():
    """A stamp-shaped substring mid-name is not a run stamp."""
    assert _resolve_output_dir(
        {}, Path("sft_20260722T051246Z_ablation.yaml"), MODEL
    ) == Path("checkpoints/sft_20260722T051246Z_ablation/gemma-4-26B-A4B-it")
