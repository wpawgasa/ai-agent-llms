"""The configured warmup ratio must reach every trainer config.

transformers removed ``TrainingArguments.warmup_ratio`` between 5.6.2 and
5.17.0, and ``warmup_steps`` became a float where a value below 1 is a ratio of
total steps. ``SFTConfig(warmup_ratio=...)`` then raised ``TypeError`` half an
hour into the E4B SFT run, after the corpus had been rendered. GRPO and DPO
fared worse: their kwarg filters dropped the field and trained with no warmup,
logging only a warning.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llm_workflow_agents.training._utils import warmup_kwargs


@dataclasses.dataclass
class _OldArgs:
    warmup_ratio: float = 0.0
    warmup_steps: int = 0


@dataclasses.dataclass
class _NewArgs:
    warmup_steps: float = 0


def test_uses_warmup_ratio_where_the_field_exists():
    assert warmup_kwargs(0.05, _OldArgs) == {"warmup_ratio": 0.05}


def test_falls_back_to_float_warmup_steps_where_warmup_ratio_was_removed():
    assert warmup_kwargs(0.05, _NewArgs) == {"warmup_steps": 0.05}


def test_non_dataclass_config_is_read_from_its_signature():
    # train_specialist's tests replace transformers with a MagicMock, so
    # TrainingArguments arrives as a plain callable rather than a dataclass.
    def old_style(output_dir, warmup_ratio=0.0):
        return None

    def new_style(output_dir, warmup_steps=0.0):
        return None

    assert warmup_kwargs(0.05, old_style) == {"warmup_ratio": 0.05}
    assert warmup_kwargs(0.05, new_style) == {"warmup_steps": 0.05}


def test_zero_warmup_is_allowed():
    assert warmup_kwargs(0.0, _NewArgs) == {"warmup_steps": 0.0}


@pytest.mark.parametrize("bad", [1.0, 5, -0.1])
def test_rejects_values_the_new_api_would_read_as_step_counts(bad):
    # warmup_steps >= 1 means an absolute step count under transformers 5.17,
    # so passing a "ratio" of 1 or more would silently change its meaning.
    with pytest.raises(ValueError):
        warmup_kwargs(bad, _NewArgs)


@pytest.mark.parametrize("config_name", ["SFTConfig", "GRPOConfig", "DPOConfig"])
def test_installed_trl_config_warms_up_five_percent_of_steps(
    config_name: str, tmp_path: Path
):
    trl = pytest.importorskip("trl")
    config_cls = getattr(trl, config_name)

    cfg = config_cls(output_dir=str(tmp_path), **warmup_kwargs(0.05, config_cls))

    assert cfg.get_warmup_steps(1000) == 50


def test_installed_training_arguments_warm_up_five_percent_of_steps(tmp_path: Path):
    transformers = pytest.importorskip("transformers")

    cfg = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        **warmup_kwargs(0.05, transformers.TrainingArguments),
    )

    assert cfg.get_warmup_steps(1000) == 50
