"""The Cat A DPO checkpoint stage must stay frozen, and stay honest.

Two reasons, and neither is theoretical.

**Cost.** The stage's `cmd` is a 500-step DPO run: five 100-step chunks, each
preceded by a 5 000-row reference-logprob precompute, plus a held-out audit
between chunks. That is a multi-hour GPU job. `frozen: true` is what stops an
accidental `dvc repro` from starting one.

**Provenance.** The weights on disk did not come from one clean invocation. The
first run was interrupted at chunk 3, the second resumed from `checkpoint-200`,
and `save_total_limit: 3` then pruned `checkpoint-100` and `checkpoint-200`. A
bare `cmd` line implies a one-shot run that did not happen, so the `desc` has to
say so. R21 is the precedent for a stage whose `cmd` stopped describing its
artifact and nothing noticed for months.

The stage replaced a bare `.dvc` pointer, which recorded the artifact but
declared no dependencies — so it could never go stale when the pairs were
rebuilt or C2 moved. R12 (`task_a_splits` drift) and R13 (the misleading frozen
config) are what silent staleness costs here.

See CLAUDE.md R22.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DVC_YAML = Path(__file__).resolve().parents[2] / "dvc.yaml"
STAGE = "task_a_dpo_gemma4_26b_a4b"


@pytest.fixture(scope="module")
def stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert STAGE in stages, f"{STAGE} stage is missing from dvc.yaml"
    return stages[STAGE]


def test_stage_is_frozen(stage: dict) -> None:
    assert stage.get("frozen") is True, (
        f"{STAGE} must stay frozen. Its cmd is a multi-hour GPU run, and the "
        "weights came from two invocations that a single cmd cannot reproduce. "
        "See CLAUDE.md R22."
    )


def test_description_warns_the_cmd_is_not_reproducible(stage: dict) -> None:
    """A reader must not take the cmd as a recipe for these weights."""
    desc = stage.get("desc", "")
    assert "PROVENANCE, NOT REPRODUCTION" in desc
    assert "TWO invocations" in desc
    assert "checkpoint-200" in desc, "desc must name where the resume began"
    assert "save_total_limit" in desc, "desc must explain the missing checkpoints"


def test_description_states_the_result(stage: dict) -> None:
    """The stage records a null result; a reader should not have to run it to learn that."""
    desc = stage.get("desc", "")
    assert "0.7566" in desc, "desc must state the DPO score"
    assert "0.7595" in desc, "desc must state the C2 baseline it is measured against"


def test_stage_declares_the_inputs_that_would_invalidate_it(stage: dict) -> None:
    """The whole point of a stage over a bare .dvc pointer is these edges."""
    deps = set(stage.get("deps", []))
    for required in (
        "src/llm_workflow_agents/training/dpo.py",
        "configs/training/dpo_cat_a.yaml",
        "data/output/preference/task_a/cap6144",
        "checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it",
    ):
        assert required in deps, (
            f"{required} must be a declared dep — without it the checkpoints "
            "never go stale when their input changes"
        )


def test_output_path_is_the_tracked_checkpoint_dir(stage: dict) -> None:
    assert stage.get("outs") == ["checkpoints/dpo_cat_a/gemma-4-26B-A4B-it"]


def test_the_superseded_bare_pointer_is_gone() -> None:
    """Two records of one artifact would drift. The stage is the record."""
    pointer = DVC_YAML.parent / "checkpoints/dpo_cat_a/gemma-4-26B-A4B-it.dvc"
    assert not pointer.exists(), (
        "the bare .dvc pointer was replaced by the dvc.yaml stage; keeping both "
        "gives two sources of truth for the same 690 MB artifact"
    )
