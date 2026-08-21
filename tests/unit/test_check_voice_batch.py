"""The voice batch gate must fail a degraded batch and pass a clean one.

Spec risk 2: a teacher model that fails on every sample yields a batch of
format-perfect placeholder rows. Zero format violations, fifteen success
lines. This gate is the only thing that can tell the two apart, so its own
failure modes are worth a test file.

The four checks under test:
  1. placeholder share (with the offline-run exemption)
  2. format re-check against the artifact on disk
  3. modality labelling
  4. barge-in realisation, which WARNS and must never fail
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_voice_batch.py"

_spec = importlib.util.spec_from_file_location("check_voice_batch", SCRIPT)
check_voice_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_voice_batch)


# --- fixtures ---------------------------------------------------------------

_VOICE_TURNS = [
    {"role": "system", "content": "You are a customer service agent."},
    {"role": "user", "content": "I want to check my balance."},
    {
        "role": "assistant",
        "content": (
            "[STATE: GREETING → VERIFY]\n"
            "<S>Of course.</S><S>Let me check that for you.</S>\n"
            '<tool_call>{"name": "get_balance", "arguments": {"id": "A1"}}</tool_call>'
        ),
    },
    {"role": "tool", "content": '{"balance": 100}'},
    {
        "role": "assistant",
        "content": "[STATE: VERIFY → DONE]\n<S>Your balance is one hundred.</S>\n[END_CONVERSATION]",
    },
]


def _write_leg(
    directory: Path,
    name: str,
    *,
    teacher_model: str | None,
    sources: dict[str, int],
    num_samples: int | None = None,
    barge_in_requested: int = 0,
    barge_in_realized: int = 0,
    modality: str = "voice",
    messages: list[dict] | None = None,
    row_generation_sources: list[str] | None = None,
    row_barge_ins: list[bool] | None = None,
) -> None:
    """Write one leg: a .jsonl of conversations plus its .stats.json sidecar.

    Every row also carries ``generation_source`` and ``barge_in`` — real
    output always does (see ``ConversationSample``), and
    ``count_teacher_realized_barge_ins`` reads exactly those two fields off
    disk. By default every row is attributed to ``"teacher"`` and the first
    ``barge_in_realized`` of them carry ``barge_in: True``, which reproduces
    the pre-Task-3 world where "realised" and "realised by the teacher" were
    the same number. Pass ``row_generation_sources`` / ``row_barge_ins``
    (one entry per row) to build a batch that mixes sources.
    """
    directory.mkdir(parents=True, exist_ok=True)
    n = num_samples if num_samples is not None else sum(sources.values())
    rows = []
    for i in range(n):
        source = (
            row_generation_sources[i] if row_generation_sources is not None else "teacher"
        )
        realized = (
            row_barge_ins[i] if row_barge_ins is not None else i < barge_in_realized
        )
        rows.append(
            {
                "conversation_id": f"{name}_{i:03d}",
                "modality": modality,
                "messages": messages if messages is not None else _VOICE_TURNS,
                "generation_source": source,
                "barge_in": realized,
            }
        )
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (directory / f"{name}.stats.json").write_text(
        json.dumps(
            {
                "complexity_level": "L1",
                "num_samples": n,
                "teacher_model": teacher_model,
                "seed": 4242,
                "output_file": f"{name}.jsonl",
                "stats": {
                    "generation_source_counts": sources,
                    "repair_fallbacks": 0,
                    "repair_retries": 0,
                    "teacher_call_failures": 0,
                    "barge_in_requested": barge_in_requested,
                    "barge_in_realized": barge_in_realized,
                    "modality_distribution": {"text": 0, "voice": n},
                },
            }
        )
    )


def _run(directory: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input-dir", str(directory), *args],
        capture_output=True,
        text=True,
        env=env,
    )


# --- the script itself ------------------------------------------------------


def test_script_is_executable():
    """The DVC stage and the runner both invoke it; the mode bit must be committed."""
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


# --- check 1: placeholder share --------------------------------------------


def test_all_teacher_batch_passes(tmp_path):
    _write_leg(tmp_path, "leg1", teacher_model="gemini-3.5-flash", sources={"teacher": 10})
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_fully_degraded_teacher_batch_fails(tmp_path):
    """The exact shape of spec risk 2: a teacher was named, nothing came from it."""
    _write_leg(
        tmp_path, "leg1", teacher_model="gemini-3.5-flash", sources={"placeholder_fallback": 10}
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FAILED" in result.stderr
    assert "100.0%" in result.stderr
    # The failure must not be attributable to format: a degraded batch is
    # format-clean by construction, which is the whole reason the gate exists.
    assert "every row on disk passes find_voice_violations" in result.stdout


def test_share_just_over_threshold_fails(tmp_path):
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 88, "placeholder_fallback": 12},
    )
    result = _run(tmp_path, "--max-placeholder-share", "0.10")
    assert result.returncode == 1


def test_share_at_threshold_passes(tmp_path):
    """The threshold is inclusive: 10% of a batch is not "more than 10%"."""
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 90, "placeholder_fallback": 10},
    )
    result = _run(tmp_path, "--max-placeholder-share", "0.10")
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_run_is_exempt(tmp_path):
    """A batch generated with no teacher model asked for placeholders."""
    _write_leg(tmp_path, "leg1", teacher_model=None, sources={"placeholder": 10})
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "does not apply" in result.stdout


def test_mixed_teacher_and_offline_legs_are_not_exempt(tmp_path):
    """One offline leg must not buy an exemption for the teacher legs."""
    _write_leg(tmp_path, "leg1", teacher_model=None, sources={"placeholder": 10})
    _write_leg(
        tmp_path, "leg2", teacher_model="gemini-3.5-flash", sources={"placeholder_fallback": 10}
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FAILED" in result.stderr


def test_share_is_aggregated_across_legs(tmp_path):
    """Per-leg shares mean nothing; the batch is what gets merged."""
    _write_leg(tmp_path, "leg1", teacher_model="gemini-3.5-flash", sources={"teacher": 100})
    _write_leg(
        tmp_path, "leg2", teacher_model="gemini-3.5-flash", sources={"placeholder_fallback": 100}
    )
    result = _run(tmp_path)
    assert result.returncode == 1  # 50% overall


# --- check 2: format on disk -----------------------------------------------


def test_format_violation_on_disk_fails(tmp_path):
    """The check that matters runs against the artifact, not an in-memory object."""
    broken = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        # Spoken text outside a chunk: rule 3.
        {"role": "assistant", "content": "[STATE: A → B]\nHello there, no chunk here."},
    ]
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 1},
        messages=broken,
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "do not satisfy the voice format on disk" in result.stderr


def test_skip_format_check_only_aggregates(tmp_path):
    broken = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "[STATE: A → B]\nunchunked speech"},
    ]
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 1},
        messages=broken,
    )
    assert _run(tmp_path, "--skip-format-check").returncode == 0


# --- check 3: modality labelling -------------------------------------------


def test_text_labelled_row_in_a_voice_batch_fails(tmp_path):
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 1},
        modality="text",
    )
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "modality is 'text'" in result.stderr


# --- check 4: barge-in realisation warns, never fails -----------------------


def test_missing_barge_ins_warn_but_do_not_fail(tmp_path):
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 100},
        barge_in_requested=25,
        barge_in_realized=0,
    )
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[warn]" in result.stderr
    assert "PASSED" in result.stdout


def test_realised_barge_ins_do_not_warn(tmp_path):
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 100},
        barge_in_requested=25,
        barge_in_realized=24,
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    assert "[warn]" not in result.stderr


def test_placeholder_fallback_realised_barge_ins_do_not_count_toward_teacher(tmp_path):
    """Task 3 let the placeholder realise a barge-in too (generation_source
    "placeholder"/"placeholder_fallback"). The warning exists to catch the
    TEACHER dropping interruptions, so a placeholder_fallback row that
    realised one must not be credited to the teacher's delivery — even
    though the sidecar's aggregate barge_in_realized (which Task 3 did not
    change) counts it.
    """
    n = 100
    row_generation_sources = ["teacher"] * 30 + ["placeholder_fallback"] * 70
    # The teacher itself realised NONE of what it was asked for; every
    # realised barge-in in this batch came from the placeholder fallback.
    row_barge_ins = [False] * 30 + [True] * 70
    _write_leg(
        tmp_path,
        "leg1",
        teacher_model="gemini-3.5-flash",
        sources={"teacher": 30, "placeholder_fallback": 70},
        num_samples=n,
        barge_in_requested=25,
        barge_in_realized=70,
        row_generation_sources=row_generation_sources,
        row_barge_ins=row_barge_ins,
    )
    # A 70% placeholder_fallback share would also trip the unrelated
    # placeholder-share gate; disable it here so only the barge-in check
    # under test can fail this run.
    result = _run(tmp_path, "--max-placeholder-share", "1.0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[warn]" in result.stderr
    assert "realised 0" in result.stderr
    assert "70 more were realised by non-teacher rows" in result.stderr


# --- refusing to guess ------------------------------------------------------


def test_missing_directory_fails(tmp_path):
    result = _run(tmp_path / "nope")
    assert result.returncode == 2


def test_no_sidecars_fails(tmp_path):
    """Without sidecars the gate cannot tell teacher from placeholder at all."""
    (tmp_path / "leg1.jsonl").write_text(json.dumps({"modality": "voice", "messages": []}) + "\n")
    result = _run(tmp_path)
    assert result.returncode == 2
    assert "no *.stats.json sidecars" in result.stderr


def test_empty_directory_of_sidecars_only_fails(tmp_path):
    _write_leg(tmp_path, "leg1", teacher_model="gemini-3.5-flash", sources={"teacher": 1})
    (tmp_path / "leg1.jsonl").unlink()
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "no conversations found" in result.stderr


# --- summarise() unit level -------------------------------------------------


def test_summarise_counts_both_placeholder_source_names():
    sidecars = [
        {
            "num_samples": 4,
            "teacher_model": "gemini-3.5-flash",
            "stats": {"generation_source_counts": {"placeholder": 1, "placeholder_fallback": 1, "teacher": 2}},
        }
    ]
    summary = check_voice_batch.summarise(sidecars)
    assert summary["placeholder"] == 2
    assert summary["scored"] == 4
    assert summary["placeholder_share"] == pytest.approx(0.5)
    assert summary["offline_run"] is False
