"""The v4 benchmark stage must stay frozen and must never touch its inputs.

`task_a_v4` and `task_a_voice_v3` are v3 without the 19 conversations that use
two tools in one state, plus 20 clean replacements generated one tool per
state (CLAUDE.md R28). The replacements come from a DVC-tracked directory; an
unfrozen `dvc repro` after a change to the build code would rebuild the test
sets that scores are measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DVC_YAML = Path(__file__).resolve().parents[2] / "dvc.yaml"
STAGE = "task_a_benchmark_v4"
INPUTS = ["data/output/benchmark/task_a_v3", "data/output/benchmark/task_a_voice_v2"]
OUTPUTS = ["data/output/benchmark/task_a_v4", "data/output/benchmark/task_a_voice_v3"]
REPLACEMENTS = "data/interim/task_a_benchmark_v4_replacements"


@pytest.fixture(scope="module")
def stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert STAGE in stages, f"{STAGE} stage is missing from dvc.yaml"
    return stages[STAGE]


def test_stage_is_frozen(stage: dict) -> None:
    assert stage.get("frozen") is True, f"{STAGE} must stay frozen. See CLAUDE.md R28."


def test_writes_only_its_own_directories(stage: dict) -> None:
    assert stage["outs"] == OUTPUTS
    assert not set(INPUTS) & set(stage["outs"])


def test_reads_v3_and_the_replacements(stage: dict) -> None:
    for dep in INPUTS + [REPLACEMENTS]:
        assert dep in stage["deps"]


def test_never_deletes_an_input(stage: dict) -> None:
    cmd = " ".join(stage["cmd"].split())
    for path in INPUTS:
        assert f"rm -rf {path} " not in cmd + " "
