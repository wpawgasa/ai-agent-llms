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
