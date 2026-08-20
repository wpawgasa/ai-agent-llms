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


def _dry_run(*args: str) -> str:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
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
        # output_dir is a Path(...) call, not a literal -- ast.literal_eval
        # can't evaluate it. Mirrors test_generate_sft_data_sh.py's
        # _kwargs_of/test_emitted_kwargs_match_the_real_signature, which
        # excludes output_dir the same way and re-adds a stand-in before
        # binding.
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call.keywords
            if kw.arg != "output_dir"
        }
        kwargs["output_dir"] = "."
        inspect.signature(generate_workflow_dataset).bind(**kwargs)


# --- --smoke-test / --teacher-model precedence ------------------------------


def test_smoke_test_defaults_to_offline_placeholder_teacher():
    out = _dry_run("--smoke-test")
    assert "teacher_model=None" in out
    assert "teacher_model='gemini" not in out


@pytest.mark.parametrize(
    "args",
    [
        ("--smoke-test", "--teacher-model", "gemini-3.5-flash"),
        ("--teacher-model", "gemini-3.5-flash", "--smoke-test"),
    ],
    ids=["smoke-then-teacher", "teacher-then-smoke"],
)
def test_explicit_teacher_model_overrides_smoke_test_default(args):
    out = _dry_run(*args)
    assert "teacher_model='gemini-3.5-flash'" in out
    assert "teacher_model=None" not in out
