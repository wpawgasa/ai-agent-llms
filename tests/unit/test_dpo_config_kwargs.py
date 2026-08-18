"""DPOConfig/ORPOConfig kwarg compatibility across TRL versions.

Same defect class as `test_grpo_config_kwargs.py` (R16): TRL renames or drops
config fields between releases, and a config written for one version can raise
`TypeError` at `DPOConfig(**kwargs)`/`ORPOConfig(**kwargs)` — after the model
and both dataset splits have already loaded. `_filter_dpo_config_kwargs` drops
unsupported keys and reports them, so a dropped key is always a loud warning,
never a silent no-op.

Requires a real `trl` import (`DPOConfig`), so — like
`test_grpo_config_kwargs.py` — this only collects under an environment with a
working torch/transformers/trl stack (`.venv-train`), not a lightweight dev
venv without CUDA-linked torchvision.

`ORPOConfig` is imported defensively rather than at module scope: TRL 1.0.0
dropped it, and a hard top-level import made THIS FILE fail at collection,
which aborted the entire `pytest tests/unit` run — one unavailable algorithm
hiding all 1,500+ unrelated tests. The ORPO cases skip instead.
"""

import dataclasses

import pytest
from trl import DPOConfig

from llm_workflow_agents.training.dpo import _filter_dpo_config_kwargs

try:  # TRL 1.0.0 ships no ORPO; see training/dpo.py::_resolve_trl_classes
    from trl import ORPOConfig
except ImportError:  # pragma: no cover - depends on installed TRL
    ORPOConfig = None

requires_orpo = pytest.mark.skipif(
    ORPOConfig is None, reason="installed TRL provides no ORPOConfig"
)

DPO_SUPPORTED = {f.name for f in dataclasses.fields(DPOConfig)}
ORPO_SUPPORTED = (
    {f.name for f in dataclasses.fields(ORPOConfig)} if ORPOConfig else set()
)


def test_supported_dpo_keys_pass_through_unchanged():
    kwargs = {"learning_rate": 5e-6, "beta": 0.1}
    kept, dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    assert kept == kwargs
    assert dropped == []


def test_unsupported_key_is_dropped_and_reported():
    kwargs = {"learning_rate": 5e-6, "definitely_not_a_trl_field": 123}
    kept, dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    assert kept == {"learning_rate": 5e-6}
    assert dropped == ["definitely_not_a_trl_field"]


def test_dropped_keys_are_sorted_for_stable_logging():
    kwargs = {"zzz_nope": 1, "aaa_nope": 2, "learning_rate": 5e-6}
    _, dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    assert dropped == ["aaa_nope", "zzz_nope"]


def test_result_actually_constructs_a_dpoconfig(tmp_path):
    kwargs = {"output_dir": str(tmp_path), "learning_rate": 5e-6, "beta": 0.1}
    kept, dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    cfg = DPOConfig(**kept)  # must not raise
    assert cfg.learning_rate == 5e-6
    assert dropped == []


@requires_orpo
def test_result_actually_constructs_an_orpoconfig(tmp_path):
    kwargs = {"output_dir": str(tmp_path), "learning_rate": 5e-6}
    kept, dropped = _filter_dpo_config_kwargs(kwargs, "orpo")
    cfg = ORPOConfig(**kept)  # must not raise
    assert cfg.learning_rate == 5e-6
    assert dropped == []


@requires_orpo
def test_dpo_only_field_is_dropped_under_orpo():
    """`beta` is a DPO field; ORPOConfig may not accept it under this key."""
    kwargs = {"beta": 0.1, "learning_rate": 5e-6}
    _, dropped = _filter_dpo_config_kwargs(kwargs, "orpo")
    if "beta" not in ORPO_SUPPORTED:
        assert "beta" in dropped


def test_empty_kwargs_are_handled():
    assert _filter_dpo_config_kwargs({}, "dpo") == ({}, [])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        _filter_dpo_config_kwargs({}, "kto")
