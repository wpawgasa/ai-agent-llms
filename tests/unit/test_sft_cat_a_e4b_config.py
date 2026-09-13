"""The E4B SFT arm may differ from the C2 corpus-v3 arm in output_dir only.

docs/training_plan_voice_and_e4b.md designs the two runs as a size comparison
on one corpus, so any other difference between the two configs would confound
model size with the recipe. R15 and R16 are both comparisons that had to be
withdrawn for that reason. This test makes the parity executable instead of a
comment someone has to remember.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
E4B_SFT = PROJECT_ROOT / "configs/training/sft_cat_a_e4b.yaml"
C2_V3_SFT = PROJECT_ROOT / "configs/training/sft_cat_a_c2_corpus_v3.yaml"
E4B_MODEL = PROJECT_ROOT / "configs/models_exp_a/gemma4_e4b.yaml"


def _load(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def test_e4b_sft_differs_from_c2_corpus_v3_only_in_output_dir():
    e4b = _flatten(_load(E4B_SFT))
    c2 = _flatten(_load(C2_V3_SFT))

    differing = sorted(
        key for key in e4b.keys() | c2.keys() if e4b.get(key) != c2.get(key)
    )

    assert differing == ["output_dir"]


def test_e4b_sft_writes_to_its_own_checkpoint_directory():
    # R13: without an explicit output_dir the run-stamped patched config moves
    # the checkpoint path, and reusing a C2 directory would overwrite weights.
    assert _load(E4B_SFT)["output_dir"] == "sft_cat_a_e4b"
    assert _load(C2_V3_SFT)["output_dir"] != "sft_cat_a_e4b"


def test_e4b_sft_uses_response_only_at_8192():
    # response_only is mandatory on a corpus with voice rows (R20), and 8192 is
    # the window R16 found silently collapsed to 1024 before c5906e7.
    training = _load(E4B_SFT)["training"]
    assert training["loss_mask"] == "response_only"
    assert training["max_seq_length"] == 8192


def test_neither_arm_freezes_gate_proj_via_freeze_router():
    # freeze_router: true calls _freeze_modules(model, ["mlp.gate"]), a
    # substring match that freezes every mlp.gate_proj LoRA adapter and never
    # Gemma-4's router. The C2 v2 checkpoint trained with it has gate_proj
    # lora_B exactly zero in 30/30 layers.
    assert _load(E4B_SFT)["lora"]["freeze_router"] is False
    assert _load(C2_V3_SFT)["lora"]["freeze_router"] is False


def test_e4b_model_config_fills_the_cat_a_role():
    model_cfg = _load(E4B_MODEL)
    assert model_cfg["model"]["name"] == "google/gemma-4-E4B-it"
    assert model_cfg["category"] == "A"
    assert "task_a" in model_cfg["benchmark_tasks"]


def test_e4b_model_config_does_not_force_the_trl_framework():
    # The 26B pins framework: trl to dodge Unsloth's MoE expert-LoRA crash.
    # E4B is dense, and copying that pin would change the training path
    # between the two arms without anyone deciding to.
    assert "framework" not in _load(E4B_MODEL).get("training", {})
