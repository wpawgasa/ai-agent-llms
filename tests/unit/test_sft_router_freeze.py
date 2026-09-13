"""sft.py must take freeze_router's patterns from the LoRA registry.

It hardcoded ["mlp.gate"], which names a real router only on Qwen MoE and GLM.
On Gemma-4 the router is layers.N.router and on NemotronH it is
layers.N.mixer.gate, so the flag froze nothing it was meant to and, through a
substring match, froze mlp.gate_proj LoRA adapters instead.
"""

from __future__ import annotations

from llm_workflow_agents.training.sft import _router_freeze_patterns


def test_gemma4_moe_resolves_to_its_router():
    assert _router_freeze_patterns({}, "google/gemma-4-26B-A4B-it") == ("router",)


def test_qwen_moe_resolves_to_mlp_gate():
    assert _router_freeze_patterns({}, "Qwen/Qwen3.5-35B-A3B") == ("mlp.gate",)


def test_nemotron_resolves_to_mixer_gate():
    assert _router_freeze_patterns({}, "nvidia/Nemotron-3-Nano-30B") == ("mixer.gate",)


def test_dense_model_resolves_to_nothing():
    assert _router_freeze_patterns({}, "google/gemma-4-E4B-it") == ()
