"""The v2 text benchmark stage must stay frozen and must never touch v1.

`data/output/benchmark/task_a_v2` is the Phase 1 text stratum brought onto the
tool-call stay rule (CLAUDE.md R25): the same 258 conversations as the frozen
v1 stratum, with 25 relabelled and 143 authored inserts replayed from a
DVC-tracked ledger. Its cmd is deterministic, but its deps include the repair
code, so an unfrozen `dvc repro` after any change to that code would rebuild
the test set that scores are measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DVC_YAML = Path(__file__).resolve().parents[2] / "dvc.yaml"
STAGE = "task_a_benchmark_text_v2"
V1_OUT = "data/output/benchmark/task_a"


@pytest.fixture(scope="module")
def stage() -> dict:
    stages = yaml.safe_load(DVC_YAML.read_text())["stages"]
    assert STAGE in stages, f"{STAGE} stage is missing from dvc.yaml"
    return stages[STAGE]


def test_stage_is_frozen(stage: dict) -> None:
    assert stage.get("frozen") is True, (
        f"{STAGE} must stay frozen: a change to the repair code would otherwise "
        "rebuild the v2 text benchmark. See CLAUDE.md R25."
    )


def test_writes_only_its_own_directory(stage: dict) -> None:
    assert stage["outs"] == ["data/output/benchmark/task_a_v2"]
    assert V1_OUT not in stage["outs"]


def test_reads_v1_and_the_ledger(stage: dict) -> None:
    assert V1_OUT in stage["deps"]
    assert "data/interim/task_a_benchmark_remediation_ledger" in stage["deps"]


def test_never_writes_into_v1(stage: dict) -> None:
    cmd = " ".join(stage["cmd"].split())
    assert f"--output-dir {V1_OUT} " not in cmd + " "
    assert f"rm -rf {V1_OUT} " not in cmd + " "
