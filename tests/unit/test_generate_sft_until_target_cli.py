"""CLI + qualification-filter tests for ``scripts/generate_sft_until_target.py``.

Deliberately not an end-to-end generation test: a real run needs teacher API
keys and minutes of wall clock. What is covered here is everything that can go
wrong *without* the teacher — flag parsing, the "auto" -> None translation, and
the profiler-report filtering that ``--no-require-tool-stay`` and the new
``stay_dropped`` counter are built on.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "generate_sft_until_target.py"


def _load_script_module():
    """Import the script by path (``scripts/`` is not a package)."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("_gen_sft_until_target", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(REPO / "scripts"))


gs = _load_script_module()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


# --- argparse ----------------------------------------------------------------


def test_dry_run_accepts_retry_flags(tmp_path):
    result = _run(
        "--dry-run",
        "--levels", "L1",
        "--languages", "en",
        "--retry-budget", "2",
        "--retry-exhaustion", "error_path",
        "--no-require-tool-stay",
        "--output-dir", str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "Retry budget:  2" in result.stdout
    assert "Exhaustion: error_path" in result.stdout
    assert "Tool-stay gate: OFF" in result.stdout


def test_dry_run_defaults_report_per_level_and_gate_on(tmp_path):
    result = _run("--dry-run", "--levels", "L5", "--languages", "en",
                  "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Retry budget:  per-level" in result.stdout
    assert "Exhaustion: auto" in result.stdout
    assert "Tool-stay gate: on" in result.stdout


def test_require_tool_stay_can_be_re_enabled_after_the_negative_flag(tmp_path):
    """The flag pair shares a dest, so last-one-wins must hold."""
    result = _run("--dry-run", "--levels", "L1", "--languages", "en",
                  "--no-require-tool-stay", "--require-tool-stay",
                  "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Tool-stay gate: on" in result.stdout


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_retry_budget_below_one_is_rejected(tmp_path, bad):
    """Budget counts the first attempt, so <1 has no coherent meaning."""
    result = _run("--dry-run", "--levels", "L1", "--languages", "en",
                  "--retry-budget", bad, "--output-dir", str(tmp_path))
    assert result.returncode != 0
    assert "--retry-budget must be >= 1" in (result.stdout + result.stderr)


def test_unknown_retry_exhaustion_is_rejected(tmp_path):
    result = _run("--dry-run", "--levels", "L1", "--languages", "en",
                  "--retry-exhaustion", "bogus", "--output-dir", str(tmp_path))
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize(
    "cli_value,expected",
    [
        ("auto", None),
        (None, None),
        ("error_path", "error_path"),
        ("handoff_in_state", "handoff_in_state"),
        # "none" is a real policy ("do nothing special"), NOT the no-override
        # sentinel — it must survive translation as the string it is.
        ("none", "none"),
    ],
)
def test_auto_exhaustion_translates_to_none_for_the_library(cli_value, expected):
    assert gs._resolve_retry_exhaustion(cli_value) == expected


# --- the defect-string contract ----------------------------------------------


def test_stay_defect_marker_matches_the_real_emitter():
    """``_STAY_DEFECT_MARKER`` must appear in what find_tool_stay_violations emits.

    This is the load-bearing assumption behind both --no-require-tool-stay and
    stay_dropped. If the violation message is ever reworded, the marker silently
    stops matching: the gate becomes unconditional and stay_dropped becomes a
    permanent zero, with no error anywhere. Hence a direct assertion against the
    real function rather than a copy of its text.
    """
    from llm_workflow_agents.data.state_convention import find_tool_stay_violations

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": '[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>',
        },
    ]
    violations = find_tool_stay_violations(messages)
    assert violations, "fixture no longer produces a stay violation"
    assert all(gs._STAY_DEFECT_MARKER in v for v in violations)


# --- qualification filtering --------------------------------------------------


def _report(*defects: str):
    from llm_workflow_agents.data.quality_profiler import ProfileReport

    rep = ProfileReport()
    rep.defects.extend(defects)
    return rep


_STAY = (
    "assistant turn 2 issues a <tool_call> but annotates an advancing transition "
    "[A -> B]; a tool-execution turn must annotate [A -> A] and advance on a later turn"
)


def test_stay_only_conversations_survive_when_the_gate_is_off():
    rep = _report(f"C1: {_STAY}", "C2: terminal 'X' not in states")

    on = gs._disqualified_ids(rep, require_tool_stay=True)
    off = gs._disqualified_ids(rep, require_tool_stay=False)

    assert on == {"C1", "C2"}
    assert off == {"C2"}, "C1's only defect was the stay violation"


def test_gate_off_still_drops_a_stay_violator_with_another_defect():
    rep = _report(f"C1: {_STAY}", "C1: terminal 'X' not in states")
    assert gs._disqualified_ids(rep, require_tool_stay=False) == {"C1"}


def test_file_level_defects_are_not_mistaken_for_conversation_ids():
    rep = _report("file: no samples", f"C1: {_STAY}")
    assert gs._disqualified_ids(rep, require_tool_stay=True) == {"C1"}
    assert gs._disqualified_ids(rep, require_tool_stay=False) == set()


def test_stay_violating_ids_ignores_other_defects():
    rep = _report(f"C1: {_STAY}", "C2: terminal 'X' not in states")
    assert gs._stay_violating_ids(rep) == {"C1"}


def test_filtering_does_not_mutate_the_report():
    rep = _report(f"C1: {_STAY}", "C2: terminal 'X' not in states")
    before = list(rep.defects)
    gs._disqualified_ids(rep, require_tool_stay=False)
    gs._stay_violating_ids(rep)
    assert rep.defects == before


# --- generate_leg observability (offline: placeholder teacher) ----------------


def test_generate_leg_reports_stay_counters_offline(tmp_path):
    """A placeholder-teacher leg runs without API keys and must emit the keys."""
    out = tmp_path / "leg.jsonl"
    stats = gs.generate_leg(
        level="L1",
        language="en",
        target=2,
        teacher_model=None,
        out_file=out,
        base_seed=11,
        seed_offset=0,
        batch_size=2,
        max_iterations=3,
        behavior_preset="default",
        intent_category="default",
        initiation="default",
        keep_intermediates=False,
    )
    for key in ("stay_dropped", "stay_violating", "retry_budget",
                "retry_exhaustion", "require_tool_stay"):
        assert key in stats, f"missing stats key {key}"
    assert stats["require_tool_stay"] is True
    # The placeholder generator emits conforming self-loops, so a clean leg.
    assert stats["stay_violating"] == 0
    assert stats["stay_dropped"] == 0
    assert stats["kept"] == 2
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(rows) == 2


def test_generate_leg_threads_retry_budget_into_the_corpus(tmp_path):
    """--retry-budget must reach the baked workflow script, not just the stats."""
    out = tmp_path / "leg.jsonl"
    stats = gs.generate_leg(
        level="L1",
        language="en",
        target=2,
        teacher_model=None,
        out_file=out,
        base_seed=11,
        seed_offset=0,
        batch_size=2,
        max_iterations=3,
        behavior_preset="default",
        intent_category="default",
        initiation="default",
        keep_intermediates=False,
        retry_budget=3,
    )
    assert stats["retry_budget"] == 3
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    scripts = " ".join(r.get("workflow_script", "") for r in rows)
    assert "3 attempts total" in scripts
    assert "do NOT retry" not in scripts
