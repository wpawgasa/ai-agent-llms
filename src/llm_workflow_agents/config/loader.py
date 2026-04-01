"""YAML configuration loader with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_workflow_agents.config.schema import (
    ExperimentConfig,
    ModelConfig,
    QuantizationMethodConfig,
    ServingDeploymentConfig,
    TrainingModelConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: Path) -> ModelConfig:
    """Load a model configuration from YAML."""
    raw = _load_yaml(path)
    model_data = raw.get("model", raw)
    # Merge top-level category and benchmark_tasks into model data
    for key in ("category", "benchmark_tasks", "serving", "inference"):
        if key in raw and key not in model_data:
            model_data[key] = raw[key]
    return ModelConfig(**model_data)


def load_training_model_config(path: Path) -> TrainingModelConfig:
    """Load a training model configuration (model + LoRA + training) from YAML."""
    raw = _load_yaml(path)
    return TrainingModelConfig(**raw)


def load_quantization_config(path: Path) -> QuantizationMethodConfig:
    """Load a quantization method configuration from YAML."""
    raw = _load_yaml(path)
    method_data = raw.get("method", raw)
    return QuantizationMethodConfig(**method_data)


def load_serving_config(path: Path) -> ServingDeploymentConfig:
    """Load a serving deployment configuration from YAML."""
    raw = _load_yaml(path)
    return ServingDeploymentConfig(**raw)


def load_config(path: Path) -> ExperimentConfig:
    """Load a top-level experiment configuration from YAML."""
    raw = _load_yaml(path)
    return ExperimentConfig(**raw)


def load_all_model_configs(config_dir: Path) -> dict[str, ModelConfig]:
    """Load all model configs from a directory."""
    configs: dict[str, ModelConfig] = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        cfg = load_model_config(yaml_file)
        configs[yaml_file.stem] = cfg
    return configs
