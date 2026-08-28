"""The configured `dpo.method` must be checked against the installed TRL early.

TRL 1.0.0 ships `DPOConfig`/`DPOTrainer` but NO `ORPOConfig`/`ORPOTrainer`.
`train_dpo` imported them lazily at the point of trainer construction — which
is *after* `FastLanguageModel.from_pretrained` has pulled a 26B checkpoint onto
the GPU and `_load_dpo_dataset` has read ~650 MB of preference pairs. A config
typo or an unavailable algorithm therefore cost a full model load before
surfacing as a bare `ImportError`.

Resolving the pair up front turns that into an immediate, actionable error.
"""

import pytest

from llm_workflow_agents.training.dpo import _resolve_trl_classes


def _trl_has_orpo() -> bool:
    pytest.importorskip("trl")
    import trl

    return hasattr(trl, "ORPOConfig") and hasattr(trl, "ORPOTrainer")


def test_dpo_resolves_to_the_trl_dpo_pair():
    pytest.importorskip("trl")
    from trl import DPOConfig, DPOTrainer

    assert _resolve_trl_classes("dpo") == (DPOConfig, DPOTrainer)


def test_orpo_either_resolves_or_fails_with_an_actionable_message():
    if _trl_has_orpo():
        from trl import ORPOConfig, ORPOTrainer

        assert _resolve_trl_classes("orpo") == (ORPOConfig, ORPOTrainer)
        return

    with pytest.raises(RuntimeError) as exc:
        _resolve_trl_classes("orpo")
    msg = str(exc.value)
    assert "orpo" in msg.lower(), "error must name the method that failed"
    assert "trl" in msg.lower(), "error must name the package that lacks it"
    assert "dpo" in msg.lower(), "error must point at the working alternative"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        _resolve_trl_classes("kto")


def test_filter_shares_the_same_resolution_path():
    """One import site, so an unavailable method fails identically everywhere."""
    from llm_workflow_agents.training.dpo import _filter_dpo_config_kwargs

    if _trl_has_orpo():
        pytest.skip("TRL provides ORPO; nothing to fail")
    with pytest.raises(RuntimeError):
        _filter_dpo_config_kwargs({"learning_rate": 5e-6}, "orpo")


# --------------------------------------------------------------------------- #
# Row-processing path guard (see training/dpo.py)
# --------------------------------------------------------------------------- #


class _LegacyDPOTrainer:
    """Stand-in for TRL 0.23.1's DPOTrainer.

    It picks the row-processing path from the MODEL's type rather than from
    the ``processing_class`` it was handed, then dereferences ``.tokenizer``
    unconditionally.
    """

    def __init__(self, model, processing_class):
        self.is_vision_model = (
            model.config.model_type in {"gemma4"}
        )
        self.tokenizer = processing_class.tokenizer


class _ModernDPOTrainer:
    """Stand-in for TRL 1.0.0's DPOTrainer: branches on the processing class."""

    def __init__(self, model, processing_class):
        if isinstance(processing_class, ProcessorMixin):  # noqa: F821
            self.tokenizer = processing_class.tokenizer
            self._is_vlm = True
        else:
            self.tokenizer = processing_class
            self._is_vlm = False


def test_legacy_trainer_that_assumes_a_processor_is_rejected():
    """A TRL whose DPOTrainer assumes a processor must fail before the 26B load."""
    from llm_workflow_agents.training.dpo import (
        _assert_dpo_row_processing_support,
    )

    with pytest.raises(RuntimeError) as exc:
        _assert_dpo_row_processing_support(_LegacyDPOTrainer)
    msg = str(exc.value)
    # Must name the remedy, not just the symptom.
    assert "install_train.sh" in msg
    assert "trl" in msg.lower()


def test_modern_trainer_that_branches_on_processing_class_is_accepted():
    from llm_workflow_agents.training.dpo import (
        _assert_dpo_row_processing_support,
    )

    _assert_dpo_row_processing_support(_ModernDPOTrainer)  # must not raise
