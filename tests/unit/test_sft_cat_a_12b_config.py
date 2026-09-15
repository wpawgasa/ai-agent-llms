"""The 12B SFT arm may differ from the E4B and C2 corpus-v3 arms in output_dir only.

Same reasoning as tests/unit/test_sft_cat_a_e4b_config.py: the arms are a
comparison of model size on one corpus and one recipe, so any other difference
would confound size with the recipe (R15, R16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_workflow_agents.training.sft import _resolve_lora_targets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_12B = PROJECT_ROOT / "configs/training/sft_cat_a_12b.yaml"
SFT_E4B = PROJECT_ROOT / "configs/training/sft_cat_a_e4b.yaml"
SFT_C2_V3 = PROJECT_ROOT / "configs/training/sft_cat_a_c2_corpus_v3.yaml"
MODEL_12B = PROJECT_ROOT / "configs/models_exp_a/gemma4_12b.yaml"


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


def _differing(a: Path, b: Path) -> list[str]:
    fa, fb = _flatten(_load(a)), _flatten(_load(b))
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))


def test_12b_differs_from_e4b_only_in_output_dir():
    assert _differing(SFT_12B, SFT_E4B) == ["output_dir"]


def test_12b_differs_from_c2_corpus_v3_only_in_output_dir():
    assert _differing(SFT_12B, SFT_C2_V3) == ["output_dir"]


def test_12b_writes_to_its_own_checkpoint_directory():
    assert _load(SFT_12B)["output_dir"] == "sft_cat_a_12b"


def test_12b_model_resolves_lora_targets():
    # sft.py returns an error before training when this list is empty.
    name = _load(MODEL_12B)["model"]["name"]
    assert _resolve_lora_targets(_load(SFT_12B), model_name=name) == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
