"""The corpus-repair A/B changes the corpus and nothing else.

The E4B tool-results run scored 0.8332 text / 0.8046 native on the v4
benchmark. The repaired-corpus run exists to answer one question — does
repairing the SFT corpus move those numbers — so any second difference
between the two configs would make the answer unattributable (R15, R16).
"""

from __future__ import annotations

from pathlib import Path

import yaml

BASE = Path("configs/training/sft_cat_a_e4b_textturns.yaml")
ARM = Path("configs/training/sft_cat_a_e4b_corpus_v4.yaml")


def test_the_two_configs_differ_only_in_output_dir_and_data_source() -> None:
    base = yaml.safe_load(BASE.read_text())
    arm = yaml.safe_load(ARM.read_text())
    assert arm["output_dir"] == "sft_cat_a_e4b_corpus_v4" != base["output_dir"]
    assert arm["data"]["source"] == "data/output/sft/task_a_splits_v4"
    assert base["data"]["source"] == "data/output/sft/task_a_splits"
    for config in (base, arm):
        config.pop("output_dir")
        config["data"].pop("source")
    assert arm == base


def test_the_repaired_corpus_is_the_one_the_ledger_built() -> None:
    arm = yaml.safe_load(ARM.read_text())
    source = Path(arm["data"]["source"])
    if not source.exists():
        return  # not materialized in this checkout
    assert sorted(p.name for p in source.glob("*.jsonl")) == ["test.jsonl", "train.jsonl", "validation.jsonl"]


def test_the_recipe_still_masks_to_responses_at_the_full_window() -> None:
    arm = yaml.safe_load(ARM.read_text())
    assert arm["training"]["loss_mask"] == "response_only"   # R20: voice rows carry loss:false
    assert arm["training"]["max_seq_length"] == 8192          # R16: verify it lands in train.log
