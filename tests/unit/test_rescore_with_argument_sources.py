"""The from-log rescore weights the two strata the way Phase 1 does.

The rewriting itself is `eval/argument_scoring.py`, covered by
`test_argument_scoring.py`; what is specific to the script is turning a change
in tool F1 into a change in the blended quality score.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "rescore_with_argument_sources", SCRIPTS / "rescore_with_argument_sources.py"
)
rescore_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore_module)

blended_delta = rescore_module.blended_delta


def test_the_quality_delta_weights_the_two_strata_as_phase_1_does() -> None:
    result = {
        "strata": {
            "text": {"before": {"tool_call_f1": 0.60}, "after": {"tool_call_f1": 0.70}},
            "voice": {"before": {"tool_call_f1": 0.70}, "after": {"tool_call_f1": 0.75}},
        }
    }
    assert blended_delta(result) == pytest.approx(0.4 * (0.7 * 0.10 + 0.3 * 0.05))


def test_a_single_stratum_run_contributes_only_its_own_stratum() -> None:
    result = {"strata": {"text": {"before": {"tool_call_f1": 0.6}, "after": {"tool_call_f1": 0.7}}}}
    assert blended_delta(result) == pytest.approx(0.4 * 0.7 * 0.10)


def test_a_run_that_did_not_move_reports_no_change() -> None:
    result = {
        "strata": {
            "text": {"before": {"tool_call_f1": 0.6}, "after": {"tool_call_f1": 0.6}},
            "voice": {"before": {"tool_call_f1": 0.7}, "after": {"tool_call_f1": 0.7}},
        }
    }
    assert blended_delta(result) == pytest.approx(0.0)
