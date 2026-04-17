"""Tests for configuration schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_workflow_agents.config.schema import (
    COMPLEXITY_SPECS,
    USER_BEHAVIOR_DISTRIBUTION,
    ComplexityLevel,
    InferenceConfig,
    ModelConfig,
    ModelFamily,
    QuantizationMethodConfig,
    ServingConfig,
)
from llm_workflow_agents.config.loader import load_model_config, load_all_model_configs


class TestConfigSchema:
    """Tests for Pydantic config models."""

    def test_model_config_defaults(self) -> None:
        cfg = ModelConfig(
            name="test/model",
            family=ModelFamily.QWEN,
            params_total=3_000_000_000,
        )
        assert cfg.precision == "bfloat16"
        assert cfg.context_length == 131072
        assert cfg.serving.engine == "vllm"
        assert cfg.inference.temperature_deterministic == 0.0

    def test_model_config_custom_serving(self) -> None:
        cfg = ModelConfig(
            name="test/model",
            family=ModelFamily.GEMMA,
            params_total=27_000_000_000,
            serving=ServingConfig(
                tool_call_parser="pythonic",
                chat_template="gemma",
                gpu_memory_utilization=0.85,
            ),
        )
        assert cfg.serving.tool_call_parser == "pythonic"
        assert cfg.serving.gpu_memory_utilization == 0.85

    def test_inference_config_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.stochastic_trials == 5
        assert cfg.max_tokens == 2048

    def test_quantization_config_defaults(self) -> None:
        cfg = QuantizationMethodConfig(name="fp8")
        assert len(cfg.quality_tasks) == 5
        assert len(cfg.performance_metrics) == 8
        assert cfg.benchmark_runs == 5

    def test_model_family_enum(self) -> None:
        assert ModelFamily.QWEN == "qwen"
        assert ModelFamily.GEMMA == "gemma"
        assert ModelFamily.MISTRAL == "mistral"

    def test_complexity_level_enum(self) -> None:
        assert ComplexityLevel.L1 == "L1"
        assert ComplexityLevel.L5 == "L5"


class TestComplexitySpecs:
    """Tests for complexity level specifications."""

    def test_all_levels_defined(self) -> None:
        for level in ComplexityLevel:
            assert level in COMPLEXITY_SPECS

    def test_l1_simplest(self) -> None:
        spec = COMPLEXITY_SPECS[ComplexityLevel.L1]
        assert spec.num_tools == 1
        assert spec.chain_depth == 0
        assert spec.domain == "faq_lookup"

    def test_l5_most_complex(self) -> None:
        spec = COMPLEXITY_SPECS[ComplexityLevel.L5]
        assert spec.num_tools == 7
        assert spec.chain_depth == 4
        assert spec.domain == "multi_dept_workflow"

    def test_complexity_increases(self) -> None:
        levels = [ComplexityLevel.L1, ComplexityLevel.L2, ComplexityLevel.L3,
                  ComplexityLevel.L4, ComplexityLevel.L5]
        for i in range(len(levels) - 1):
            curr = COMPLEXITY_SPECS[levels[i]]
            next_ = COMPLEXITY_SPECS[levels[i + 1]]
            assert curr.num_tools <= next_.num_tools
            assert curr.chain_depth <= next_.chain_depth


class TestUserBehaviorDistribution:
    """Tests for user behavior distribution."""

    def test_sums_to_one(self) -> None:
        total = sum(USER_BEHAVIOR_DISTRIBUTION.values())
        assert abs(total - 1.0) < 1e-9

    def test_cooperative_majority(self) -> None:
        assert USER_BEHAVIOR_DISTRIBUTION["cooperative"] == 0.60


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_load_model_config(self, configs_dir: Path) -> None:
        config_path = configs_dir / "models" / "cat_a" / "gemma3_27b.yaml"
        assert config_path.exists(), f"Expected config at {config_path}"
        cfg = load_model_config(config_path)
        assert cfg.family == ModelFamily.GEMMA
        assert cfg.params_total == 27_000_000_000
        assert cfg.category == "A"
        assert "task_a" in cfg.benchmark_tasks

    def test_load_all_model_configs(self, configs_dir: Path) -> None:
        model_dir = configs_dir / "models" / "cat_a"
        assert model_dir.exists(), f"Expected model dir at {model_dir}"
        configs = load_all_model_configs(model_dir)
        assert len(configs) == 12
        assert "gemma3_27b" in configs

    def test_load_cat_bc_configs(self, configs_dir: Path) -> None:
        model_dir = configs_dir / "models" / "cat_bc"
        assert model_dir.exists(), f"Expected model dir at {model_dir}"
        configs = load_all_model_configs(model_dir)
        assert len(configs) == 7
