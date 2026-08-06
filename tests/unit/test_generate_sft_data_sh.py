"""Dry-run tests for ``scripts/generate_sft_data.sh``.

``--dry-run`` prints the exact ``python3 -c`` blocks it would execute without
calling a teacher, so the whole argument -> kwarg path is testable offline. The
generation itself is not exercised (it needs API keys and hours).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "scripts" / "generate_sft_data.sh"
PY_LOOP = REPO / "scripts" / "generate_sft_until_target.py"

_MARKER = "[DRY RUN] python3 -c "


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SH), "--dry-run", *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def _blocks(stdout: str) -> list[str]:
    return stdout.split(_MARKER)[1:]


def test_dry_run_emits_one_block_per_level_and_language():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert len(_blocks(r.stdout)) == 15  # 5 levels x 3 language legs


def test_emitted_python_is_syntactically_valid():
    """A malformed kwarg (e.g. an empty ``retry_budget=``) must not reach a run."""
    r = _run("--samples-per-leg", "2")
    assert r.returncode == 0, r.stderr
    for block in _blocks(r.stdout):
        code = block.split("print(f'  ->")[0]
        ast.parse(code)


def _kwargs_of(block: str) -> dict[str, object]:
    code = block.split("print(f'  ->")[0]
    call = ast.parse(code).body[-1].value  # meta = generate_workflow_dataset(...)
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg != "output_dir"}


def test_defaults_emit_no_override():
    r = _run("--samples-per-leg", "2")
    kwargs = _kwargs_of(_blocks(r.stdout)[0])
    assert kwargs["retry_budget"] is None
    assert kwargs["retry_exhaustion"] is None
    assert kwargs["require_tool_stay"] is True


def test_flags_reach_the_emitted_kwargs():
    r = _run("--samples-per-leg", "2", "--retry-budget", "3",
             "--retry-exhaustion", "handoff_in_state", "--no-require-tool-stay")
    assert r.returncode == 0, r.stderr
    for block in _blocks(r.stdout):
        kwargs = _kwargs_of(block)
        assert kwargs["retry_budget"] == 3
        assert kwargs["retry_exhaustion"] == "handoff_in_state"
        assert kwargs["require_tool_stay"] is False


def test_emitted_kwargs_match_the_real_signature():
    """Guards against the kwargs drifting away from generate_workflow_dataset."""
    import inspect

    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    r = _run("--samples-per-leg", "2", "--retry-budget", "2")
    kwargs = _kwargs_of(_blocks(r.stdout)[0])
    kwargs["output_dir"] = "."
    inspect.signature(generate_workflow_dataset).bind(**kwargs)


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5"])
def test_invalid_retry_budget_is_rejected(bad):
    r = _run("--retry-budget", bad)
    assert r.returncode != 0
    assert "Invalid --retry-budget" in r.stderr


def test_invalid_retry_exhaustion_is_rejected():
    r = _run("--retry-exhaustion", "bogus")
    assert r.returncode != 0
    assert "Unknown --retry-exhaustion" in r.stderr


# --- CURRICULUM single-sourcing ----------------------------------------------


def test_curriculum_totals_come_from_the_python_script():
    """The shell must report the Python script's numbers, not its own copy."""
    ns: dict = {}
    for node in ast.parse(PY_LOOP.read_text()).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "CURRICULUM":
            ns["CURRICULUM"] = ast.literal_eval(node.value)
            break
    assert "CURRICULUM" in ns, "CURRICULUM disappeared from generate_sft_until_target.py"

    r = _run()
    assert r.returncode == 0, r.stderr
    for level, per_leg in ns["CURRICULUM"].items():
        assert f"  --- {level} ({per_leg * 3} samples) ---" in r.stdout
    total = sum(v * 3 for v in ns["CURRICULUM"].values())
    assert f"(~{total} total)" in r.stdout


def test_shell_has_no_second_copy_of_the_curriculum_numbers():
    """A literal `[L1]=1000` block would silently re-introduce the drift."""
    text = SH.read_text()
    assert "declare -A CURRICULUM=(" not in text
