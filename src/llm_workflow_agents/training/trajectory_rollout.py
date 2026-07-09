"""Multi-turn trajectory rollout for GRPO (Cat A workflow conversations).

Replaces GRPO's per-turn single-completion rollout with a whole-conversation
*replay* rollout: the model free-runs its assistant turns while the gold
conversation's user turns and tool results are replayed as a fixed script, and
the rollout truncates when the model's emitted state transition leaves the gold
path. A trajectory-level reward then yields one scalar per rollout — restoring
the within-group reward variance that per-turn scoring structurally collapses
(see ``docs/superpowers/specs/2026-07-09-multiturn-trajectory-reward-design.md``).

This module holds the pure, unit-testable trajectory logic (gold-script
construction, turn alignment) plus the in-process ``model.generate`` rollout and
its TRL ``rollout_func`` adapter. It imports ``trl``/``torch`` lazily inside the
functions that need them so the module imports on a box without the training
stack.
"""

from __future__ import annotations

import hashlib
import json


def assert_trajectory_rollout_support() -> None:
    """Fail fast if the installed TRL lacks the rollout_func/env_mask pathway.

    The trajectory rollout depends on TRL 1.0.0's ``GRPOTrainer`` accepting a
    custom ``rollout_func`` and honoring an ``env_mask`` extra field (which masks
    injected gold user/tool tokens out of the policy-gradient loss). TRL 0.24.0
    and earlier lack both. Unsloth also monkey-patches ``GRPOTrainer`` at import,
    so we inspect the *installed, possibly-patched* ``_generate`` source rather
    than trusting the version string alone.

    Raises:
        RuntimeError: if ``GRPOTrainer._generate`` does not reference both
            ``rollout_func`` and ``env_mask``.
    """
    import inspect

    import trl
    from trl.trainer.grpo_trainer import GRPOTrainer

    src = inspect.getsource(GRPOTrainer._generate)
    if "rollout_func" not in src or "env_mask" not in src:
        raise RuntimeError(
            f"TRL {trl.__version__}: GRPOTrainer._generate lacks the "
            "rollout_func/env_mask pathway required by trajectory GRPO "
            "(present in trl==1.0.0). Check the training env's trl pin and "
            "whether Unsloth's RL patch replaced _generate."
        )
