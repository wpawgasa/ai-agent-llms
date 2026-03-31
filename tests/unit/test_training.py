"""Tests for training module — LoRA targets, config builders, and merge utility.

These tests run without GPU/CUDA by testing the configuration and
registry logic only. Actual training requires GPU and is covered by
integration tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_workflow_agents.config.schema import (
    LoRAConfig,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
    TrainingModelConfig,
)
from llm_workflow_agents.training.lora_targets import (
    LORA_TARGET_MODULES,
    LoRATargetSpec,
    detect_model_key,
    get_lora_target_spec,
    get_trainable_param_summary,
)
from llm_workflow_agents.training.train_specialist import (
    TrainingResult,
    _build_peft_config,
    _build_training_arguments,
    _freeze_modules,
)
from llm_workflow_agents.training.train_graph_extractor import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
)
from llm_workflow_agents.training.merge_adapter import merge_and_export


# --- Fixtures ---


@pytest.fixture
def qwen_training_config() -> TrainingModelConfig:
    return TrainingModelConfig(
        model=ModelConfig(
            name="Qwen/Qwen2.5-3B-Instruct",
            family=ModelFamily.QWEN,
            params_total=3_000_000_000,
        ),
        lora=LoRAConfig(
            rank=64,
            alpha=128,
            dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
        training=TrainingConfig(
            learning_rate=2e-4,
            effective_batch_size=32,
            gradient_accumulation_steps=8,
            num_epochs=3,
            max_seq_length=4096,
        ),
    )


@pytest.fixture
def glm_training_config() -> TrainingModelConfig:
    return TrainingModelConfig(
        model=ModelConfig(
            name="THUDM/glm-4-9b-chat",
            family=ModelFamily.GLM,
            params_total=4_700_000_000,
        ),
        lora=LoRAConfig(rank=64, alpha=64),
        training=TrainingConfig(
            learning_rate=5e-5,
            effective_batch_size=8,
            gradient_accumulation_steps=8,
        ),
    )


@pytest.fixture
def gemma_training_config() -> TrainingModelConfig:
    return TrainingModelConfig(
        model=ModelConfig(
            name="google/gemma-2b",
            family=ModelFamily.GEMMA,
            params_total=2_000_000_000,
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        training=TrainingConfig(learning_rate=2e-4),
    )


# --- LoRA Target Registry Tests ---


class TestLoRATargetRegistry:
    """Tests for the LoRA target module registry."""

    def test_all_models_registered(self) -> None:
        expected = {"qwen25_3b", "qwen35_4b", "glm47_flash", "gemma_2b", "gemma3_4b"}
        assert set(LORA_TARGET_MODULES.keys()) == expected

    def test_qwen25_targets(self) -> None:
        spec = LORA_TARGET_MODULES["qwen25_3b"]
        assert "q_proj" in spec.target_modules
        assert "gate_proj" in spec.target_modules
        assert len(spec.warnings) == 0

    def test_qwen35_has_deltanet_warning(self) -> None:
        spec = LORA_TARGET_MODULES["qwen35_4b"]
        assert "in_proj_qkv" in spec.target_modules
        assert any("DeltaNet" in w for w in spec.warnings)

    def test_glm_has_freeze_modules(self) -> None:
        spec = LORA_TARGET_MODULES["glm47_flash"]
        assert "mlp.gate" in spec.modules_to_freeze
        assert any("60GB" in w for w in spec.warnings)

    def test_glm_targets_include_mla(self) -> None:
        spec = LORA_TARGET_MODULES["glm47_flash"]
        assert "q_a_proj" in spec.target_modules
        assert "kv_a_proj_with_mqa" in spec.target_modules

    def test_gemma_targets(self) -> None:
        for key in ("gemma_2b", "gemma3_4b"):
            spec = LORA_TARGET_MODULES[key]
            assert "q_proj" in spec.target_modules
            assert len(spec.target_modules) == 7


class TestDetectModelKey:
    """Tests for model name pattern detection."""

    def test_detect_qwen25(self) -> None:
        assert detect_model_key("Qwen/Qwen2.5-3B-Instruct") == "qwen25_3b"

    def test_detect_qwen35(self) -> None:
        assert detect_model_key("Qwen/Qwen3.5-4B") == "qwen35_4b"

    def test_detect_glm(self) -> None:
        assert detect_model_key("THUDM/glm-4-9b-chat") == "glm47_flash"

    def test_detect_gemma_2b(self) -> None:
        assert detect_model_key("google/gemma-2b") == "gemma_2b"

    def test_detect_gemma3_4b(self) -> None:
        assert detect_model_key("google/gemma-3-4b-it") == "gemma3_4b"

    def test_unknown_model(self) -> None:
        assert detect_model_key("unknown/model-7b") is None


class TestGetLoRATargetSpec:
    """Tests for the priority-based target spec resolver."""

    def test_explicit_targets_highest_priority(self) -> None:
        spec = get_lora_target_spec(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            explicit_targets=["custom_proj"],
        )
        assert spec.target_modules == ["custom_proj"]

    def test_model_specific_registry(self) -> None:
        spec = get_lora_target_spec(model_name="Qwen/Qwen2.5-3B-Instruct")
        assert "q_proj" in spec.target_modules
        assert len(spec.target_modules) == 7

    def test_family_fallback(self) -> None:
        spec = get_lora_target_spec(
            model_name="unknown/qwen-model",
            model_family=ModelFamily.QWEN,
        )
        assert "q_proj" in spec.target_modules

    def test_no_match_returns_empty(self) -> None:
        spec = get_lora_target_spec(model_name="totally/unknown-model")
        assert spec.target_modules == []

    def test_glm_returns_warnings(self) -> None:
        spec = get_lora_target_spec(model_name="THUDM/glm-4-9b-chat")
        assert len(spec.warnings) > 0
        assert "mlp.gate" in spec.modules_to_freeze


class TestGetTrainableParamSummary:
    """Tests for parameter summary computation."""

    def test_summary_with_mock_model(self) -> None:
        import torch

        model = MagicMock()
        # Create mock parameters
        trainable_param = torch.nn.Parameter(torch.randn(10, 10))
        frozen_param = torch.nn.Parameter(torch.randn(20, 20), requires_grad=False)
        model.parameters.return_value = [trainable_param, frozen_param]

        summary = get_trainable_param_summary(model)
        assert summary["trainable_params"] == 100
        assert summary["total_params"] == 500
        assert 19.0 < summary["trainable_pct"] < 21.0

    def test_summary_all_frozen(self) -> None:
        import torch

        model = MagicMock()
        frozen = torch.nn.Parameter(torch.randn(10), requires_grad=False)
        model.parameters.return_value = [frozen]

        summary = get_trainable_param_summary(model)
        assert summary["trainable_params"] == 0
        assert summary["trainable_pct"] == 0.0


# --- PEFT Config Builder Tests ---


class TestBuildPeftConfig:
    """Tests for PEFT config dict builder."""

    def test_basic_config(self, qwen_training_config: TrainingModelConfig) -> None:
        kwargs = _build_peft_config(qwen_training_config)
        assert kwargs["r"] == 64
        assert kwargs["lora_alpha"] == 128
        assert kwargs["lora_dropout"] == 0.05
        assert kwargs["task_type"] == "CAUSAL_LM"
        assert kwargs["bias"] == "none"
        assert "target_modules" in kwargs
        assert "q_proj" in kwargs["target_modules"]

    def test_config_with_no_explicit_targets(self) -> None:
        config = TrainingModelConfig(
            model=ModelConfig(
                name="Qwen/Qwen2.5-3B-Instruct",
                family=ModelFamily.QWEN,
                params_total=3_000_000_000,
            ),
            lora=LoRAConfig(rank=32, alpha=64, target_modules=[]),
        )
        kwargs = _build_peft_config(config)
        # Should fall back to registry
        assert "target_modules" in kwargs
        assert "q_proj" in kwargs["target_modules"]

    def test_config_preserves_modules_to_save(self) -> None:
        config = TrainingModelConfig(
            model=ModelConfig(
                name="test/model",
                family=ModelFamily.QWEN,
                params_total=1_000_000_000,
            ),
            lora=LoRAConfig(
                rank=16,
                alpha=32,
                target_modules=["q_proj"],
                modules_to_save=["embed_tokens"],
            ),
        )
        kwargs = _build_peft_config(config)
        assert kwargs["modules_to_save"] == ["embed_tokens"]


# --- Training Arguments Builder Tests ---


class TestBuildTrainingArguments:
    """Tests for HF TrainingArguments dict builder."""

    def test_basic_args(self, qwen_training_config: TrainingModelConfig, tmp_path: Path) -> None:
        kwargs = _build_training_arguments(qwen_training_config, tmp_path)
        assert kwargs["output_dir"] == str(tmp_path)
        assert kwargs["learning_rate"] == 2e-4
        assert kwargs["num_train_epochs"] == 3
        assert kwargs["per_device_train_batch_size"] == 4  # 32 / 8
        assert kwargs["gradient_accumulation_steps"] == 8

    def test_bf16_precision(self, qwen_training_config: TrainingModelConfig, tmp_path: Path) -> None:
        kwargs = _build_training_arguments(qwen_training_config, tmp_path)
        assert kwargs["bf16"] is True

    def test_fp16_precision(self, tmp_path: Path) -> None:
        config = TrainingModelConfig(
            model=ModelConfig(
                name="test/model",
                family=ModelFamily.QWEN,
                params_total=1_000_000_000,
            ),
            training=TrainingConfig(mixed_precision="fp16"),
        )
        kwargs = _build_training_arguments(config, tmp_path)
        assert kwargs.get("fp16") is True
        assert "bf16" not in kwargs

    def test_gradient_checkpointing_kwargs(
        self, qwen_training_config: TrainingModelConfig, tmp_path: Path
    ) -> None:
        kwargs = _build_training_arguments(qwen_training_config, tmp_path)
        assert kwargs["gradient_checkpointing"] is True
        assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    def test_eval_strategy_matches_save(
        self, qwen_training_config: TrainingModelConfig, tmp_path: Path
    ) -> None:
        kwargs = _build_training_arguments(qwen_training_config, tmp_path)
        assert kwargs["eval_strategy"] == "steps"
        assert kwargs["eval_steps"] == 500
        assert kwargs["save_steps"] == 500


# --- Freeze Modules Tests ---


class TestFreezeModules:
    """Tests for selective module freezing."""

    def test_freeze_matching_params(self) -> None:
        import torch

        # Build a nested module hierarchy so named_parameters yields dotted names
        model = torch.nn.Module()
        mlp = torch.nn.Module()
        gate = torch.nn.Module()
        gate.register_parameter("weight", torch.nn.Parameter(torch.randn(5)))
        up = torch.nn.Module()
        up.register_parameter("weight", torch.nn.Parameter(torch.randn(5)))
        mlp.add_module("gate", gate)
        mlp.add_module("up", up)
        model.add_module("mlp", mlp)
        attn = torch.nn.Module()
        attn.register_parameter("q_proj", torch.nn.Parameter(torch.randn(5)))
        model.add_module("attn", attn)

        frozen = _freeze_modules(model, ["mlp.gate"])
        assert frozen == 1
        assert not dict(model.named_parameters())["mlp.gate.weight"].requires_grad
        assert dict(model.named_parameters())["mlp.up.weight"].requires_grad
        assert dict(model.named_parameters())["attn.q_proj"].requires_grad

    def test_freeze_no_matches(self) -> None:
        import torch

        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.randn(5)))
        model.add_module("layer", layer)

        frozen = _freeze_modules(model, ["nonexistent"])
        assert frozen == 0
        assert dict(model.named_parameters())["layer.weight"].requires_grad


# --- Training Result Tests ---


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_default_values(self) -> None:
        result = TrainingResult()
        assert result.checkpoint_path is None
        assert result.best_eval_loss is None
        assert result.total_steps == 0
        assert result.error is None

    def test_with_values(self, tmp_path: Path) -> None:
        result = TrainingResult(
            checkpoint_path=tmp_path / "best",
            best_eval_loss=0.42,
            total_steps=1500,
            metrics={"train_loss": 0.5},
            param_summary={"trainable_params": 1000},
        )
        assert result.best_eval_loss == 0.42
        assert result.total_steps == 1500


# --- Graph Extraction Tests ---


class TestGraphExtractor:
    """Tests for graph extraction module constants and structure."""

    def test_system_prompt_defined(self) -> None:
        assert "workflow graph" in GRAPH_EXTRACTION_SYSTEM_PROMPT.lower()
        assert "JSON" in GRAPH_EXTRACTION_SYSTEM_PROMPT
        assert "nodes" in GRAPH_EXTRACTION_SYSTEM_PROMPT
        assert "edges" in GRAPH_EXTRACTION_SYSTEM_PROMPT

    def test_system_prompt_instructs_json_only(self) -> None:
        assert "ONLY valid JSON" in GRAPH_EXTRACTION_SYSTEM_PROMPT


# --- Merge Adapter Tests ---


class TestMergeAdapter:
    """Tests for merge_and_export error handling."""

    def test_adapter_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            merge_and_export(
                base_model="test/model",
                adapter_path=tmp_path / "nonexistent",
                output_path=tmp_path / "output",
            )


# --- Module Import Tests ---


class TestModuleImports:
    """Verify that training module imports work without GPU."""

    def test_import_training_package(self) -> None:
        from llm_workflow_agents import training
        assert hasattr(training, "LORA_TARGET_MODULES")
        assert hasattr(training, "TrainingResult")
        assert hasattr(training, "train")
        assert hasattr(training, "train_graph_extractor")
        assert hasattr(training, "merge_and_export")

    def test_import_lora_targets(self) -> None:
        from llm_workflow_agents.training.lora_targets import LORA_TARGET_MODULES
        assert len(LORA_TARGET_MODULES) == 5

    def test_import_training_result(self) -> None:
        from llm_workflow_agents.training.train_specialist import TrainingResult
        assert TrainingResult is not None
