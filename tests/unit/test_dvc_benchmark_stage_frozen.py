"""Both Phase 1 benchmark stages must stay frozen.

`data/output/benchmark/task_a` holds 258 conversations, 250 of them
teacher-generated across two models, plus one hand-added file. It is the data
behind the current Cat A ranking. The stage's `cmd` runs the *placeholder*
generator, which would replace all 258 rows with 1 000 placeholder ones.

That mismatch sat dormant for months because nothing had touched the stage's
declared dependencies. The voice branches changed `generate_workflows.py`,
which IS a declared dependency, so `dvc repro` would now fire it rather than
skip it. `frozen: true` is what stops that.

`data/output/benchmark/task_a_voice` holds 250 conversations, the voice half
of the Cat A composite blend. It was left unfrozen while it had never been
generated — freezing it then would have prevented the run that creates it.
That run has since happened and the data is registered with DVC (2026-09-11),
so the same hazard applies: its declared deps (`generate_workflows.py`,
`voice_convention.py`, the tool-schema template) already show as modified,
and an unfrozen `dvc repro` would silently regenerate and swap the data
CLAUDE.md R20/R24/R25 measure against.

This test exists because unfreezing is a one-word edit whose cost is the
corpus, and nothing else would notice. See CLAUDE.md R21.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DVC_YAML = Path(__file__).resolve().parents[2] / "dvc.yaml"
STAGE = "task_a_benchmark"
VOICE_STAGE = "task_a_benchmark_voice"


@pytest.fixture(scope="module")
def stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert STAGE in stages, f"{STAGE} stage is missing from dvc.yaml"
    return stages[STAGE]


@pytest.fixture(scope="module")
def voice_stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert VOICE_STAGE in stages, f"{VOICE_STAGE} stage is missing from dvc.yaml"
    return stages[VOICE_STAGE]


def test_stage_is_frozen(stage: dict) -> None:
    assert stage.get("frozen") is True, (
        f"{STAGE} must stay frozen. Its cmd runs the placeholder generator and "
        "would overwrite 258 teacher-generated conversations with 1 000 "
        "placeholder ones. See CLAUDE.md R21."
    )


def test_description_does_not_claim_the_artifact_is_placeholder_data(stage: dict) -> None:
    """The old desc said '1 000 placeholder conversations'. The files are not that."""
    desc = stage.get("desc", "")
    assert "1 000 placeholder conversations for Phase 1" not in desc
    assert "No teacher model or API key required" not in desc


def test_description_states_what_the_files_actually_are(stage: dict) -> None:
    desc = stage.get("desc", "")
    assert "258" in desc, "desc must state the real conversation count"
    assert "FROZEN" in desc, "desc must say the stage is frozen"
    for model in ("gemini-3-flash-preview", "gemini-3-5-flash"):
        assert model in desc, f"desc must name the teacher model {model}"


def test_output_path_is_unchanged(stage: dict) -> None:
    """Freezing must not have moved the artifact the ranking depends on."""
    assert stage.get("outs") == ["data/output/benchmark/task_a"]


def test_voice_stratum_is_a_separate_additive_stage(voice_stage: dict) -> None:
    """The voice stratum must never be folded into the frozen text stage."""
    assert voice_stage.get("outs") == ["data/output/benchmark/task_a_voice"]


def test_voice_stage_is_frozen(voice_stage: dict) -> None:
    assert voice_stage.get("frozen") is True, (
        f"{VOICE_STAGE} must stay frozen. It has been generated and registered "
        "with DVC; its cmd and declared deps (generate_workflows.py, "
        "voice_convention.py, the tool-schema template) can drift and an "
        "unfrozen `dvc repro` would silently regenerate and swap the data "
        "CLAUDE.md R20/R24/R25 measure against. See CLAUDE.md R21."
    )


def test_voice_description_states_the_stage_is_frozen(voice_stage: dict) -> None:
    desc = voice_stage.get("desc", "")
    assert "FROZEN" in desc, "desc must say the stage is frozen"
    assert "250" in desc, "desc must state the real conversation count"
