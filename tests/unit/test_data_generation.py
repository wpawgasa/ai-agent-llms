"""Tests for data generation modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.config.schema import (
    COMPLEXITY_SPECS,
    TOOL_ERROR_RATE,
    USER_BEHAVIOR_DISTRIBUTION,
    ComplexityLevel,
)
from llm_workflow_agents.data.generate_workflows import (
    ConversationSample,
    DatasetMetadata,
    WorkflowGraph,
    WorkflowState,
    WorkflowTransition,
    generate_workflow_dataset,
    _select_domain,
)
from llm_workflow_agents.data.domain_registry import (
    ALL_DOMAIN_NAMES,
    DOMAIN_REGISTRY,
    DomainSpec,
    classify_intent,
)
from llm_workflow_agents.data.generate_workflows import INTENT_CATEGORY_PRESETS
from llm_workflow_agents.data.generate_tool_call_data import (
    DatasetSplits,
    generate_tool_call_dataset,
)
from llm_workflow_agents.data.generate_graph_pairs import (
    GraphNode,
    GraphEdge,
    WorkflowGraphOutput,
    generate_graph_pairs,
)


class TestWorkflowGeneration:
    """Tests for Experiment A data generation."""

    def test_generate_l1_dataset(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=10,
            output_dir=tmp_output_dir,
            seed=42,
        )
        assert isinstance(result, DatasetMetadata)
        assert result.num_samples == 10
        assert len(result.output_files) == 1
        assert result.output_files[0].exists()

    def test_generate_l3_dataset(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L3",
            num_samples=5,
            output_dir=tmp_output_dir,
            seed=42,
        )
        assert result.num_samples == 5

    def test_output_jsonl_valid(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=5,
            output_dir=tmp_output_dir,
            seed=42,
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                assert "conversation_id" in sample
                assert "complexity_level" in sample
                assert "messages" in sample
                assert "workflow_graph" in sample
                assert sample["complexity_level"] == "L1"

    def test_workflow_graph_structure(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L2",
            num_samples=3,
            output_dir=tmp_output_dir,
            seed=42,
        )
        with open(result.output_files[0]) as f:
            sample = json.loads(f.readline())
            graph = sample["workflow_graph"]
            assert "states" in graph
            assert "transitions" in graph
            assert "initial" in graph
            assert "terminal" in graph

    def test_behavior_distribution_approximate(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=100,
            output_dir=tmp_output_dir,
            seed=42,
        )
        behaviors: dict[str, int] = result.stats["behavior_distribution"]
        total = sum(behaviors.values())
        # Cooperative should be roughly 60% (±15% for small sample)
        assert behaviors["cooperative"] / total > 0.40

    def test_seed_determinism(self, tmp_output_dir: Path) -> None:
        dir1 = tmp_output_dir / "run1"
        dir2 = tmp_output_dir / "run2"
        r1 = generate_workflow_dataset("L1", num_samples=5, output_dir=dir1, seed=42)
        r2 = generate_workflow_dataset("L1", num_samples=5, output_dir=dir2, seed=42)

        with open(r1.output_files[0]) as f1, open(r2.output_files[0]) as f2:
            assert f1.read() == f2.read()

    def test_messages_start_with_system(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=3,
            output_dir=tmp_output_dir,
            seed=42,
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                messages = sample["messages"]
                assert messages[0]["role"] == "system"


class TestToolCallDataGeneration:
    """Tests for Experiment B data generation."""

    def test_generate_dataset(self, tmp_output_dir: Path) -> None:
        result = generate_tool_call_dataset(
            external_sources=[],
            custom_synthetic_size=100,
            output_dir=tmp_output_dir,
            seed=42,
        )
        assert isinstance(result, DatasetSplits)
        assert result.train_path.exists()
        assert result.val_path.exists()
        assert result.test_path.exists()

    def test_split_ratios(self, tmp_output_dir: Path) -> None:
        result = generate_tool_call_dataset(
            external_sources=[],
            custom_synthetic_size=100,
            output_dir=tmp_output_dir,
            seed=42,
        )
        total = result.train_size + result.val_size + result.test_size
        assert total > 0
        # Train should be majority
        assert result.train_size > result.val_size
        assert result.train_size > result.test_size

    def test_negative_examples_included(self, tmp_output_dir: Path) -> None:
        result = generate_tool_call_dataset(
            external_sources=[],
            custom_synthetic_size=100,
            negative_ratio=0.15,
            output_dir=tmp_output_dir,
            seed=42,
        )
        # Check that negative samples exist
        has_negative = False
        with open(result.train_path) as f:
            for line in f:
                sample = json.loads(line)
                if sample.get("source") == "synthetic_negative":
                    has_negative = True
                    break
        assert has_negative


class TestGraphPairGeneration:
    """Tests for Experiment C data generation."""

    def test_generate_pairs_placeholder(self, tmp_output_dir: Path) -> None:
        result = generate_graph_pairs(
            workflow_prompts_dir=tmp_output_dir / "nonexistent",
            gold_annotations=20,
            teacher_generated=30,
            augmentation_target=100,
            output_dir=tmp_output_dir,
            seed=42,
        )
        assert isinstance(result, DatasetSplits)
        assert result.train_path.exists()

    def test_graph_output_structure(self) -> None:
        graph = WorkflowGraphOutput(
            nodes=[
                GraphNode(id="S1", name="Start"),
                GraphNode(id="S2", name="End"),
            ],
            edges=[
                GraphEdge(from_state="S1", to_state="S2", condition="done"),
            ],
            initial_state="S1",
            terminal_states=["S2"],
        )
        d = graph.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["initial_state"] == "S1"
        assert d["terminal_states"] == ["S2"]


class TestDomainRegistry:
    """Tests for expanded domain registry."""

    def test_registry_has_18_domains(self) -> None:
        assert len(DOMAIN_REGISTRY) == 18
        assert len(ALL_DOMAIN_NAMES) == 18

    def test_all_domains_have_tools(self) -> None:
        for name, spec in DOMAIN_REGISTRY.items():
            assert len(spec.tools) >= 4, f"{name} has fewer than 4 tools"

    def test_all_domains_have_state_templates(self) -> None:
        for name, spec in DOMAIN_REGISTRY.items():
            assert len(spec.state_templates) >= 7, f"{name} has fewer than 7 state templates"
            assert spec.state_templates[-1] in ("TERMINAL", "RESOLVE", "POST_INCIDENT_REVIEW"), (
                f"{name} last state template is {spec.state_templates[-1]}"
            )

    def test_all_domains_have_intents(self) -> None:
        for name, spec in DOMAIN_REGISTRY.items():
            assert len(spec.intents) >= 4, f"{name} has fewer than 4 intents"

    def test_domain_categories(self) -> None:
        categories = {spec.category for spec in DOMAIN_REGISTRY.values()}
        assert "core_business" in categories
        assert "industry" in categories
        assert "operational" in categories

    def test_tool_schemas_valid_format(self) -> None:
        for name, spec in DOMAIN_REGISTRY.items():
            for tool in spec.tools:
                assert tool["type"] == "function", f"{name}: tool missing type=function"
                func = tool["function"]
                assert "name" in func, f"{name}: tool missing function.name"
                assert "parameters" in func, f"{name}: tool missing function.parameters"

    def test_select_domain_explicit(self) -> None:
        import random
        rng = random.Random(42)
        key, spec = _select_domain(rng, "banking")
        assert key == "banking"
        assert spec.name == "Banking & Financial Services"

    def test_select_domain_legacy_mapping(self) -> None:
        import random
        rng = random.Random(42)
        key, spec = _select_domain(rng, "faq_lookup")
        assert key == "product_info"

    def test_select_domain_random(self) -> None:
        import random
        rng = random.Random(42)
        key, spec = _select_domain(rng, None)
        assert key in ALL_DOMAIN_NAMES

    def test_generate_with_specific_domain(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L2",
            num_samples=5,
            output_dir=tmp_output_dir,
            seed=42,
            domain="healthcare",
        )
        assert result.stats["num_domains"] == 1
        assert "healthcare" in result.stats["domain_distribution"]

    def test_generate_with_random_domains(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L2",
            num_samples=50,
            output_dir=tmp_output_dir,
            seed=42,
            domain=None,
        )
        # With 50 samples across 18 domains, should see multiple domains
        assert result.stats["num_domains"] > 1


class TestIntentCategoryPreset:
    """Tests for intent-category biasing (promo/upsell preset)."""

    def test_classify_intent_known(self) -> None:
        assert classify_intent("upsell_offer") == "upsell_promo"
        assert classify_intent("promotion_inquiry") == "upsell_promo"
        assert classify_intent("policy_renewal") == "upsell_promo"

    def test_classify_intent_fallback(self) -> None:
        assert classify_intent("unknown_intent_xyz") == "service"
        assert classify_intent("complaint_registration") == "service"

    def test_presets_defined(self) -> None:
        for name in ("default", "service_only", "upsell_heavy"):
            assert name in INTENT_CATEGORY_PRESETS
            dist = INTENT_CATEGORY_PRESETS[name]
            assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_unknown_preset_raises(self, tmp_output_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown intent_category_preset"):
            generate_workflow_dataset(
                complexity_level="L1",
                num_samples=2,
                output_dir=tmp_output_dir,
                seed=42,
                intent_category_preset="nonexistent",
            )

    def test_service_only_preset_zero_upsell(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=50,
            output_dir=tmp_output_dir,
            seed=42,
            intent_category_preset="service_only",
        )
        dist = result.stats["intent_category_distribution"]
        assert dist.get("upsell_promo", 0) == 0

    def test_default_preset_biases_upsell(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L2",
            num_samples=200,
            output_dir=tmp_output_dir,
            seed=42,
            domain="sales",
            intent_category_preset="default",
        )
        dist = result.stats["intent_category_distribution"]
        total = sum(dist.values())
        upsell_share = dist.get("upsell_promo", 0) / total
        assert 0.20 <= upsell_share <= 0.45

    def test_upsell_fallback_in_pure_service_domain(self, tmp_output_dir: Path) -> None:
        # government has no upsell intents; selector must fall back cleanly
        result = generate_workflow_dataset(
            complexity_level="L1",
            num_samples=10,
            output_dir=tmp_output_dir,
            seed=42,
            domain="government",
            intent_category_preset="upsell_heavy",
        )
        assert result.num_samples == 10
        assert result.stats["num_domains"] == 1


class TestWorkflowGraphModel:
    """Tests for WorkflowGraph dataclass."""

    def test_to_dict(self) -> None:
        graph = WorkflowGraph(
            states=[
                WorkflowState(id="S1", name="A", tools=["tool1"]),
                WorkflowState(id="S2", name="B"),
            ],
            transitions=[
                WorkflowTransition(from_state="S1", to_state="S2", condition="proceed"),
            ],
            initial_state="S1",
            terminal_states=["S2"],
        )
        d = graph.to_dict()
        assert d["initial"] == "A"
        assert d["terminal"] == ["B"]
        assert len(d["states"]) == 2
        assert len(d["transitions"]) == 1
