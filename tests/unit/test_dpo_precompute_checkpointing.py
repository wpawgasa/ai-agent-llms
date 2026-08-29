"""Gradient checkpointing must be paused while the reference logprobs precompute.

WHY (CLAUDE.md R19, docs/dpo_memory_ceiling_investigation.md section 11):
Unsloth loads this model with offloaded gradient checkpointing — the banner
reads `Unsloth: Will smartly offload gradients to save VRAM!`. That mechanism
copies each layer's hidden states to host memory, expecting a backward pass to
consume and free them.

`precompute_ref_log_probs` runs forward-only under `no_grad`. There is no
backward pass, so nothing ever drains the offloaded activations. They
accumulate at one hidden-state tensor per row until the kernel OOM-kills the
process.

Measured directly: CPU bfloat16 tensors whose element counts divide exactly by
`2 * hidden_size`, e.g. 34,276,352 = 2 x 6,086 tokens x 2,816 — the chosen and
rejected sequences of one row. Growth was 65 MB per row, linear, projecting to
~320 GB over the 5,000-row split.

The fix flips only the per-module `gradient_checkpointing` flag and leaves
`_gradient_checkpointing_func` alone. Removing the function instead — which is
what `from_pretrained(use_gradient_checkpointing=False)` does — leaves the flag
set and the model raises
`AttributeError: 'Gemma4TextDecoderLayer' object has no attribute
'_gradient_checkpointing_func'` on the first forward. These tests pin that
asymmetry, because the obvious fix is the broken one.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.dpo import _gradient_checkpointing_paused


class _Layer:
    """Stands in for a decoder layer carrying Unsloth's offload hook."""

    def __init__(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled
        self._gradient_checkpointing_func = lambda *a, **k: None


class _Model:
    def __init__(self, flags: list[bool]) -> None:
        self._layers = [_Layer(f) for f in flags]

    def modules(self):
        return iter(self._layers)


def test_flag_is_off_inside_the_context():
    model = _Model([True, True, True])
    with _gradient_checkpointing_paused(model):
        assert [m.gradient_checkpointing for m in model.modules()] == [False] * 3


def test_flag_is_restored_on_exit():
    model = _Model([True, True, True])
    with _gradient_checkpointing_paused(model):
        pass
    assert [m.gradient_checkpointing for m in model.modules()] == [True] * 3


def test_the_offload_function_is_left_intact():
    """Removing the function while the flag stays set is the failure mode that
    `from_pretrained(use_gradient_checkpointing=False)` produces. Never do it."""
    model = _Model([True])
    layer = next(model.modules())
    func = layer._gradient_checkpointing_func
    with _gradient_checkpointing_paused(model):
        assert layer._gradient_checkpointing_func is func
    assert layer._gradient_checkpointing_func is func


def test_a_module_that_was_already_off_stays_off():
    """Restore what was there, not a blanket True — a model loaded without
    checkpointing must not come out of the context with it switched on."""
    model = _Model([True, False, True])
    with _gradient_checkpointing_paused(model):
        pass
    assert [m.gradient_checkpointing for m in model.modules()] == [True, False, True]


def test_flag_is_restored_when_the_body_raises():
    """The precompute runs inside this context. If it fails, training must not
    silently continue without checkpointing and OOM on the GPU instead."""
    model = _Model([True, True])
    with pytest.raises(RuntimeError):
        with _gradient_checkpointing_paused(model):
            raise RuntimeError("precompute blew up")
    assert [m.gradient_checkpointing for m in model.modules()] == [True, True]


def test_a_model_with_no_checkpointing_is_a_no_op():
    model = _Model([False, False])
    with _gradient_checkpointing_paused(model):
        assert [m.gradient_checkpointing for m in model.modules()] == [False, False]
    assert [m.gradient_checkpointing for m in model.modules()] == [False, False]


# --------------------------------------------------------------------------- #
# The switch that actually reaches Unsloth's buffer pool
# --------------------------------------------------------------------------- #
#
# Flipping the per-module `gradient_checkpointing` flag does NOT stop the
# offload: Unsloth installs it below that flag by replacing
# `torch.utils.checkpoint.CheckpointFunction`, so neither the module flag nor
# TRL's own `disable_gradient_checkpointing` guard reaches it. Measured — with
# the flag paused, the leak was unchanged at 65.3 MB/row, and the global
# CPU_BUFFERS pool grew from 200 buffers / 4.36 GB to 735 / 33.49 GB, matching
# the retained-tensor total to the decimal at every sample.
#
# `unsloth_zoo.gradient_checkpointing` exports the pair that does reach it.


class _FakeUnslothGC:
    """Stands in for unsloth_zoo.gradient_checkpointing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def unpatch_unsloth_smart_gradient_checkpointing(self) -> None:
        self.calls.append("unpatch")

    def patch_unsloth_smart_gradient_checkpointing(self, dtype=None) -> None:
        self.calls.append("patch")


@pytest.fixture
def fake_unsloth_gc(monkeypatch):
    import sys

    fake = _FakeUnslothGC()
    monkeypatch.setitem(
        sys.modules, "unsloth_zoo.gradient_checkpointing", fake
    )
    return fake


def test_unsloth_offload_is_unpatched_for_the_duration(fake_unsloth_gc):
    model = _Model([True])
    with _gradient_checkpointing_paused(model):
        assert fake_unsloth_gc.calls == ["unpatch"], (
            "the offload must be unpatched on entry; flipping the module flag "
            "alone leaves CPU_BUFFERS growing"
        )


def test_unsloth_offload_is_restored_on_exit(fake_unsloth_gc):
    model = _Model([True])
    with _gradient_checkpointing_paused(model):
        pass
    assert fake_unsloth_gc.calls == ["unpatch", "patch"]


def test_unsloth_offload_is_restored_when_the_body_raises(fake_unsloth_gc):
    model = _Model([True])
    with pytest.raises(RuntimeError):
        with _gradient_checkpointing_paused(model):
            raise RuntimeError("precompute blew up")
    assert fake_unsloth_gc.calls == ["unpatch", "patch"], (
        "training after a failed precompute must get its offload back"
    )


def test_missing_unsloth_zoo_is_tolerated(monkeypatch):
    """The context manager must not break a non-Unsloth environment."""
    import builtins

    real_import = builtins.__import__

    def _no_unsloth(name, *a, **k):
        if name.startswith("unsloth_zoo"):
            raise ImportError("no unsloth_zoo here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_unsloth)
    model = _Model([True])
    with _gradient_checkpointing_paused(model):
        assert next(model.modules()).gradient_checkpointing is False
    assert next(model.modules()).gradient_checkpointing is True
