"""The repaired benchmark stage must stay frozen and must never touch its inputs.

`task_a_v3` and `task_a_voice_v2` are the v2 text and voice strata with their
untraceable values repaired (CLAUDE.md R28), replayed from two DVC-tracked
ledgers. The cmd is deterministic, but its deps include the repair code, so an
unfrozen `dvc repro` after any change to that code would rebuild the test sets
that scores are measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DVC_YAML = Path(__file__).resolve().parents[2] / "dvc.yaml"
STAGE = "task_a_benchmark_repair_v3"
INPUTS = ["data/output/benchmark/task_a_v2", "data/output/benchmark/task_a_voice"]
OUTPUTS = ["data/output/benchmark/task_a_v3", "data/output/benchmark/task_a_voice_v2"]
LEDGER = "data/interim/task_a_benchmark_repair_ledger"


@pytest.fixture(scope="module")
def stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert STAGE in stages, f"{STAGE} stage is missing from dvc.yaml"
    return stages[STAGE]


def test_stage_is_frozen(stage: dict) -> None:
    assert stage.get("frozen") is True, (
        f"{STAGE} must stay frozen: a change to the repair code would otherwise "
        "rebuild the repaired benchmark. See CLAUDE.md R28."
    )


def test_writes_only_its_own_directories(stage: dict) -> None:
    assert stage["outs"] == OUTPUTS
    assert not set(INPUTS) & set(stage["outs"])


def test_reads_both_strata_and_the_ledger(stage: dict) -> None:
    for dep in INPUTS + [LEDGER]:
        assert dep in stage["deps"]


def test_replays_both_ledgers(stage: dict) -> None:
    cmd = " ".join(stage["cmd"].split())
    assert f"--ledger {LEDGER}/ledger.json" in cmd
    assert f"--facts-ledger {LEDGER}/facts.json" in cmd


def test_never_deletes_an_input(stage: dict) -> None:
    cmd = " ".join(stage["cmd"].split())
    for path in INPUTS:
        assert f"rm -rf {path} " not in cmd + " "
