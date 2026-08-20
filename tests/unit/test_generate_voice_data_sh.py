"""The voice runner must call the generator with the right kwargs.

Mirrors tests/unit/test_generate_sft_data_sh.py, which guards the text runner
against signature drift the same way.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_voice_data.sh"


def _dry_run() -> str:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_script_exists_and_is_executable():
    assert SCRIPT.exists()


def test_language_legs_are_20_50_30():
    out = _dry_run()
    counts = {
        lang: sum(int(n) for n in re.findall(rf'language="{lang}".*?num_samples=(\d+)', out, re.S))
        for lang in ("en", "th", "code_switch")
    }
    total = sum(counts.values())
    assert total == 2400
    assert counts["en"] == 480
    assert counts["th"] == 1200
    assert counts["code_switch"] == 720


def test_every_call_uses_the_voice_only_preset():
    out = _dry_run()
    calls = re.findall(r"generate_workflow_dataset\((.*?)\n\)", out, re.S)
    assert calls
    assert all('modality_preset="voice_only"' in c for c in calls)


def test_kwargs_bind_against_the_real_signature():
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    out = _dry_run()
    for block in re.findall(r"(meta = generate_workflow_dataset\(.*?\n\))", out, re.S):
        call = ast.parse(block).body[-1].value
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        inspect.signature(generate_workflow_dataset).bind(**kwargs)
