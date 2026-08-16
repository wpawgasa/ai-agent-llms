"""GRPOConfig kwarg compatibility across TRL versions.

TRL 1.0.0 removed `max_prompt_length` from `GRPOConfig`, so passing the key
straight through raised `TypeError: GRPOConfig.__init__() got an unexpected
keyword argument 'max_prompt_length'` and killed the run at construction —
after the model and both dataset splits had already loaded.

The filter below drops unsupported keys so a config written for one TRL version
still launches on another. The load-bearing part is that it is *loud*: R16 was
caused by a length parameter being dropped silently (`max_seq_length` became
`max_length` in TRL 0.23+, the guarded branch turned into a no-op, and every Cat
A run trained on a 1024-token window for months without a single warning). A
dropped length knob must always name itself in the log.
"""

import dataclasses

from trl import GRPOConfig

from llm_workflow_agents.training.grpo import _filter_grpo_config_kwargs

SUPPORTED = {f.name for f in dataclasses.fields(GRPOConfig)}


def test_supported_keys_pass_through_unchanged():
    kwargs = {"learning_rate": 1e-6, "beta": 0.05, "max_steps": 50}
    kept, dropped = _filter_grpo_config_kwargs(kwargs)
    assert kept == kwargs
    assert dropped == []


def test_unsupported_key_is_dropped_and_reported():
    kwargs = {"learning_rate": 1e-6, "definitely_not_a_trl_field": 123}
    kept, dropped = _filter_grpo_config_kwargs(kwargs)
    assert kept == {"learning_rate": 1e-6}
    assert dropped == ["definitely_not_a_trl_field"]


def test_dropped_keys_are_sorted_for_stable_logging():
    kwargs = {"zzz_nope": 1, "aaa_nope": 2, "learning_rate": 1e-6}
    _, dropped = _filter_grpo_config_kwargs(kwargs)
    assert dropped == ["aaa_nope", "zzz_nope"]


def test_result_actually_constructs_a_grpoconfig(tmp_path):
    """The whole point: the filtered dict must be accepted by the installed TRL."""
    kwargs = {
        "output_dir": str(tmp_path),
        "learning_rate": 1e-6,
        "max_prompt_length": 7680,  # removed in TRL 1.0.0
    }
    kept, dropped = _filter_grpo_config_kwargs(kwargs)
    cfg = GRPOConfig(**kept)  # must not raise
    assert cfg.learning_rate == 1e-6
    if "max_prompt_length" not in SUPPORTED:
        assert dropped == ["max_prompt_length"]


def test_empty_kwargs_are_handled():
    assert _filter_grpo_config_kwargs({}) == ({}, [])
