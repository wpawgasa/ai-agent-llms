"""Shared test fixtures for all tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_workflow_agents.config.schema import (
    ComplexityLevel,
    ComplexitySpec,
    InferenceConfig,
    ModelConfig,
    ModelFamily,
    ServingConfig,
)


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


gpu_available = pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU")


# --- Paths ---


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def templates_dir(project_root: Path) -> Path:
    return project_root / "data" / "templates"


# --- Model Config Fixtures ---


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Minimal model config for testing."""
    return ModelConfig(
        name="test/model-3b",
        family=ModelFamily.QWEN,
        params_total=3_000_000_000,
        precision="bfloat16",
        serving=ServingConfig(
            engine="vllm",
            tool_call_parser="hermes",
            chat_template="chatml",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        ),
        inference=InferenceConfig(
            temperature_deterministic=0.0,
            temperature_stochastic=0.7,
            stochastic_trials=5,
            max_tokens=2048,
        ),
    )


@pytest.fixture
def l1_complexity_spec() -> ComplexitySpec:
    return ComplexitySpec(
        level=ComplexityLevel.L1,
        target_path_len=(3, 4),
        num_branches=(0, 0),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=1,
        num_states=(3, 4),
        branching_factor=(1, 2),
        chain_depth=0,
        nesting_depth=0,
        domain="faq_lookup",
    )


@pytest.fixture
def l3_complexity_spec() -> ComplexitySpec:
    return ComplexitySpec(
        level=ComplexityLevel.L3,
        target_path_len=(8, 12),
        num_branches=(2, 3),
        num_loops=(0, 1),
        include_recovery=True,
        num_tools=4,
        num_states=(8, 12),
        branching_factor=(2, 4),
        chain_depth=2,
        nesting_depth=2,
        domain="booking_payment",
    )


# --- Temporary Output Fixtures ---


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output = tmp_path / "output"
    output.mkdir()
    return output
