"""Config loading and method resolution for the DPO/ORPO entry point.

`training/dpo.py` serves both algorithms from one config shape (`stage: dpo`,
`dpo.method: dpo | orpo`) rather than a second training/orpo.py, mirroring how
`sft.py` serves all three task categories from one entry point. These tests
cover the validation logic only — no torch/trl/unsloth import required.
"""


import pytest
import yaml

from llm_workflow_agents.training.dpo import (
    _load_dpo_config,
    _resolve_method,
)


def test_load_dpo_config_accepts_stage_dpo(tmp_path):
    cfg_path = tmp_path / "dpo_cat_a.yaml"
    cfg_path.write_text(yaml.safe_dump({"stage": "dpo", "dpo": {"method": "dpo"}}))
    config = _load_dpo_config(cfg_path)
    assert config["stage"] == "dpo"


def test_load_dpo_config_rejects_wrong_stage(tmp_path):
    cfg_path = tmp_path / "sft_cat_a.yaml"
    cfg_path.write_text(yaml.safe_dump({"stage": "sft"}))
    with pytest.raises(ValueError, match="stage='dpo'"):
        _load_dpo_config(cfg_path)


def test_load_dpo_config_handles_empty_file(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")
    with pytest.raises(ValueError):
        _load_dpo_config(cfg_path)


def test_resolve_method_defaults_to_dpo():
    assert _resolve_method({}) == "dpo"


def test_resolve_method_accepts_orpo():
    assert _resolve_method({"method": "orpo"}) == "orpo"


def test_resolve_method_is_case_insensitive():
    assert _resolve_method({"method": "ORPO"}) == "orpo"


def test_resolve_method_rejects_unknown_method():
    with pytest.raises(ValueError, match=r"dpo\.method"):
        _resolve_method({"method": "kto"})
