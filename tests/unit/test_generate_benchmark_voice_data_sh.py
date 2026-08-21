"""The voice benchmark runner must match the text stratum on everything but modality."""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_benchmark_voice_data.sh"


def _dry_run() -> str:
    return subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                          capture_output=True, text=True, check=True).stdout


def _blocks(out: str) -> list[str]:
    return re.findall(r"(meta = generate_workflow_dataset\(.*?\n\))", out, re.S)


def test_script_exists():
    assert SCRIPT.exists()


def test_five_levels_of_fifty():
    blocks = _blocks(_dry_run())
    assert len(blocks) == 5
    counts = [int(re.search(r"num_samples=(\d+)", b).group(1)) for b in blocks]
    assert counts == [50, 50, 50, 50, 50]
    assert sum(counts) == 250


def test_every_call_is_voice_only():
    assert all('modality_preset="voice_only"' in b for b in _blocks(_dry_run()))


def test_no_language_argument_so_it_mixes_like_the_text_stratum():
    """The text stratum passes no language and draws en/th evenly. Matching it
    keeps modality the only difference between the strata."""
    assert all("language=" not in b for b in _blocks(_dry_run()))


def test_one_teacher_model_for_every_level():
    models = {re.search(r"teacher_model='([^']+)'", b).group(1) for b in _blocks(_dry_run())}
    assert len(models) == 1


def test_kwargs_bind_against_the_real_signature():
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    for block in _blocks(_dry_run()):
        call = ast.parse(block).body[-1].value
        kwargs = {kw.arg: ast.literal_eval(kw.value)
                  for kw in call.keywords if kw.arg != "output_dir"}
        kwargs["output_dir"] = "."
        inspect.signature(generate_workflow_dataset).bind(**kwargs)
