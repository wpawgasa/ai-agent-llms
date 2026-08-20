"""Tests for scripts/heldout_composite_audit.py's modality plumbing and the
voice-guardrail silent-failure warning (Task 10 follow-up, round 1).

No GPU is available in this environment, so these test the seams that don't
require loading a checkpoint or generating completions: the source-corpus
voice detector, the warning decision function, and (via preflight_entropy_diag)
that a conversation's modality actually survives sampling into the shape the
audit reads as `conv`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from heldout_composite_audit import _corpus_has_voice, _voice_dropped_warning  # noqa: E402


def _write_jsonl(tmp_path: Path, name: str, convs: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(c) for c in convs) + "\n")
    return p


class TestCorpusHasVoice:
    def test_true_when_a_voice_conversation_is_present(self, tmp_path):
        _write_jsonl(tmp_path, "test.jsonl", [
            {"modality": "text", "messages": []},
            {"modality": "voice", "messages": []},
        ])
        assert _corpus_has_voice(tmp_path, "test") is True

    def test_false_for_all_text_corpus(self, tmp_path):
        _write_jsonl(tmp_path, "test.jsonl", [
            {"modality": "text", "messages": []},
            {"messages": []},  # pre-existing row, no field at all
        ])
        assert _corpus_has_voice(tmp_path, "test") is False

    def test_false_when_split_file_is_missing(self, tmp_path):
        assert _corpus_has_voice(tmp_path, "nope") is False


class TestVoiceDroppedWarning:
    def test_fires_when_corpus_has_voice_but_zero_voice_rows_reached_the_audit(
        self, tmp_path
    ):
        _write_jsonl(tmp_path, "test.jsonl", [{"modality": "voice", "messages": []}])
        msg = _voice_dropped_warning(tmp_path, "test", voice_rows=[])
        assert msg is not None
        assert "voice_format_compliance" in msg
        assert "did NOT fire" in msg

    def test_does_not_fire_on_an_all_text_corpus(self, tmp_path):
        _write_jsonl(tmp_path, "test.jsonl", [{"modality": "text", "messages": []}])
        assert _voice_dropped_warning(tmp_path, "test", voice_rows=[]) is None

    def test_does_not_fire_when_voice_rows_did_reach_the_audit(self, tmp_path):
        _write_jsonl(tmp_path, "test.jsonl", [{"modality": "voice", "messages": []}])
        assert (
            _voice_dropped_warning(
                tmp_path, "test", voice_rows=[{"modality": "voice"}]
            )
            is None
        )


class TestModalityReachesSampledPrompts:
    """The plumbing fix: _load_grpo_jsonl -> _sample_prompts -> conv.get("modality")."""

    def test_voice_conversation_modality_survives_sampling(self, tmp_path):
        conv = {
            "modality": "voice",
            "messages": [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "<S>Hi, I need help.</S>"},
                {
                    "role": "assistant",
                    "content": "<S>Sure, let me look that up.</S>",
                    "annotations": {
                        "state_transition": {"from": "GREETING", "to": "LOOKUP"}
                    },
                },
            ],
            "ground_truth": {"terminal_state": "", "terminal_reached": False},
        }
        _write_jsonl(tmp_path, "test.jsonl", [conv])

        from preflight_entropy_diag import _sample_prompts

        prompts = _sample_prompts(tmp_path, "test", n_prompts=1, seed=42)
        assert len(prompts) == 1
        assert prompts[0]["modality"] == "voice"

    def test_missing_modality_field_defaults_to_text_after_sampling(self, tmp_path):
        conv = {
            "messages": [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "Hi, I need help."},
                {
                    "role": "assistant",
                    "content": "Sure, let me look that up.",
                    "annotations": {
                        "state_transition": {"from": "GREETING", "to": "LOOKUP"}
                    },
                },
            ],
            "ground_truth": {"terminal_state": "", "terminal_reached": False},
        }
        _write_jsonl(tmp_path, "test.jsonl", [conv])

        from preflight_entropy_diag import _sample_prompts

        prompts = _sample_prompts(tmp_path, "test", n_prompts=1, seed=42)
        assert len(prompts) == 1
        assert prompts[0]["modality"] == "text"


class TestSummariseByModality:
    """Spec section 5 forbids blending; the audit must report each modality.

    A blended mean_composite looks exactly like the 0.7595-comparable number
    and moves the pre-registered 0.75 bar without a decision.
    """

    @staticmethod
    def _row(modality, composite, state_acc=0.0, tool_f1=0.0, task=0.0):
        row = {
            "composite": composite,
            "state_acc": state_acc,
            "tool_f1": tool_f1,
            "task": task,
        }
        if modality is not None:
            row["modality"] = modality
        return row

    def test_reports_each_modality_separately(self):
        from heldout_composite_audit import summarise_by_modality

        rows = [
            self._row("text", 0.8),
            self._row("text", 0.6),
            self._row("voice", 0.2),
        ]
        out = summarise_by_modality(rows)
        assert set(out) == {"text", "voice"}
        assert out["text"]["n_rows"] == 2
        assert out["text"]["mean_composite"] == 0.7
        assert out["voice"]["n_rows"] == 1
        assert out["voice"]["mean_composite"] == 0.2

    def test_blended_mean_hides_a_collapsed_modality(self):
        """The number this separation exists to prevent."""
        from heldout_composite_audit import summarise_by_modality

        rows = [self._row("text", 0.90)] * 9 + [self._row("voice", 0.10)]
        blended = sum(r["composite"] for r in rows) / len(rows)
        out = summarise_by_modality(rows)
        assert blended == pytest.approx(0.82)  # would read as a pass
        assert out["voice"]["mean_composite"] == 0.10  # the real story

    def test_missing_modality_counts_as_text(self):
        from heldout_composite_audit import summarise_by_modality

        out = summarise_by_modality([self._row(None, 0.5)])
        assert set(out) == {"text"}
        assert out["text"]["n_rows"] == 1

    def test_single_modality_sample_is_not_mixed(self):
        from heldout_composite_audit import summarise_by_modality

        out = summarise_by_modality([self._row("text", 0.5), self._row("text", 0.7)])
        assert len(out) == 1  # main() sets mixed_modality from this length

    def test_every_component_is_reported_per_modality(self):
        from heldout_composite_audit import summarise_by_modality

        out = summarise_by_modality(
            [self._row("voice", 0.5, state_acc=0.4, tool_f1=0.6, task=0.8)]
        )
        assert out["voice"]["mean_state_acc"] == 0.4
        assert out["voice"]["mean_tool_f1"] == 0.6
        assert out["voice"]["mean_task"] == 0.8


class TestModalityFlag:
    def test_audit_exposes_a_modality_flag(self):
        """Mirrors build_heldout_clean_set.py so one command audits one modality."""
        import subprocess

        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "heldout_composite_audit.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--modality" in out.stdout
        for choice in ("text", "voice", "all"):
            assert choice in out.stdout
