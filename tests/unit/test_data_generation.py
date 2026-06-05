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
import llm_workflow_agents.data.generate_workflows as gw
from llm_workflow_agents.data.data_validator import validate_dataset
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

    def test_state_tools_are_rational(self, tmp_output_dir: Path) -> None:
        """Every tool placed in a state must come from that domain's curated
        state→tools map and must be present in the sample's tool_schemas."""
        result = generate_workflow_dataset(
            complexity_level="L4",
            num_samples=8,
            output_dir=tmp_output_dir,
            seed=7,
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                spec = DOMAIN_REGISTRY[sample["domain"]]
                schema_names = {
                    ts["function"]["name"] for ts in sample["tool_schemas"]
                }
                _new_schema_map = (
                    {s.name: set(s.tools) for s in spec.states} if spec.states else {}
                )
                for sd in sample["workflow_graph"]["state_details"]:
                    curated = (
                        _new_schema_map.get(sd["name"], set())
                        if spec.states
                        else set(spec.state_tools.get(sd["name"], ()))
                    )
                    for tool in sd["tools"]:
                        assert tool in curated, (
                            f"{sample['domain']}: '{tool}' placed in '{sd['name']}' "
                            f"is not in the curated map"
                        )
                        assert tool in schema_names, (
                            f"{sample['domain']}: '{tool}' not in tool_schemas"
                        )

    def test_every_working_state_has_instruction(self, tmp_output_dir: Path) -> None:
        """Every non-terminal state carries an instruction in both the graph
        and the rendered workflow_script."""
        result = generate_workflow_dataset(
            complexity_level="L3",
            num_samples=8,
            output_dir=tmp_output_dir,
            seed=11,
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                terminals = set(sample["workflow_graph"]["terminal"])
                for sd in sample["workflow_graph"]["state_details"]:
                    if sd["name"] in terminals:
                        continue
                    assert sd.get("instruction", "").strip(), (
                        f"{sample['domain']}: state '{sd['name']}' has no instruction"
                    )
                # Instruction line is rendered with a language-specific marker.
                script = sample["workflow_script"]
                assert "Instruction:" in script or "คำแนะนำ:" in script

    def test_l5_placeholder_always_reaches_terminal(self, tmp_output_dir: Path):
        result = generate_workflow_dataset(
            complexity_level="L5",
            num_samples=10,
            domain="banking",
            output_dir=tmp_output_dir,
            seed=42,
        )
        samples = []
        with open(result.output_files[0]) as f:
            for line in f:
                samples.append(json.loads(line))
        empty_terminals = [s for s in samples if not s["ground_truth"]["terminal_state"]]
        assert not empty_terminals, f"{len(empty_terminals)} L5 samples have empty terminal_state"


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
            if spec.states:
                # New-schema domain: check StateNode list
                assert len(spec.states) >= 7, f"{name} has fewer than 7 states"
                assert spec.terminals, f"{name} has no terminals"
            else:
                # Legacy-schema domain: check state_templates
                assert len(spec.state_templates) >= 7, f"{name} has fewer than 7 state templates"
                assert spec.state_templates[-1] in (
                    "TERMINAL", "RESOLVE", "POST_INCIDENT_REVIEW",
                ), f"{name} last state template is {spec.state_templates[-1]}"

    def test_all_domains_have_curated_state_maps(self) -> None:
        """Every state template must have a curated instruction + tools entry,
        and every referenced tool must exist in the domain's tool list."""
        for name, spec in DOMAIN_REGISTRY.items():
            tool_names = {t["function"]["name"] for t in spec.tools}
            if spec.states:
                # New-schema domain: validated by validate_domain at import time
                for sn in spec.states:
                    assert sn.instruction.strip(), f"{name}: state '{sn.name}' has no instruction"
                    for tool in sn.tools:
                        assert tool in tool_names, (
                            f"{name}: state '{sn.name}' references unknown tool '{tool}'"
                        )
                continue
            states = set(spec.state_templates)
            for state in spec.state_templates:
                assert state in spec.state_instructions, f"{name}: {state} has no instruction"
                assert spec.state_instructions[state].strip(), f"{name}: {state} empty instruction"
                assert state in spec.state_tools, f"{name}: {state} missing state_tools entry"
            for state, tools in spec.state_tools.items():
                assert state in states, f"{name}: state_tools key {state} is not a state"
                for tool in tools:
                    assert tool in tool_names, f"{name}: {state} references unknown tool {tool}"
            for state in spec.state_instructions:
                assert state in states, f"{name}: instruction key {state} is not a state"

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


class TestPostGenerationRepair:
    """The inline repair loop guarantees teacher-generated samples respect the
    curated tool placement (regenerate, then placeholder fallback)."""

    @staticmethod
    def _incoherent_conversation(workflow, *args, **kwargs):
        """A teacher conversation that calls an off-schema tool → always violates."""
        s0 = workflow.states[0].name
        s1 = workflow.states[1].name if len(workflow.states) > 1 else s0
        return [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": f"[STATE: {s0} → {s1}]\n"
                           "<tool_call>{\"name\": \"ghost_tool\", \"arguments\": {}}</tool_call>",
            },
            {"role": "tool", "content": "{\"status\": \"ok\"}"},
        ]

    def test_irreparable_teacher_falls_back_to_placeholder(
        self, tmp_output_dir: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(gw, "_generate_teacher_conversation", self._incoherent_conversation)
        result = generate_workflow_dataset(
            complexity_level="L3",
            num_samples=4,
            teacher_model="fake-teacher",
            output_dir=tmp_output_dir,
            seed=5,
        )
        # Every requested sample is still emitted...
        assert result.num_samples == 4
        # ...all fell back to the placeholder generator...
        assert result.stats["repair_fallbacks"] == 4
        assert result.stats["repair_retries"] == 4 * 2  # max_repair_retries default
        # ...and the output is fully coherent under the validator.
        val = validate_dataset(result.output_files[0], "workflow")
        assert val.valid, val.errors

    def test_retry_succeeds_without_fallback(
        self, tmp_output_dir: Path, monkeypatch
    ) -> None:
        calls = {"n": 0}

        def flaky(workflow, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return TestPostGenerationRepair._incoherent_conversation(workflow)
            # Coherent: an assistant turn with no tool calls has zero violations.
            s0 = workflow.states[0].name
            s1 = workflow.states[1].name if len(workflow.states) > 1 else s0
            return [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": f"[STATE: {s0} → {s1}]\nLet me help."},
            ]

        monkeypatch.setattr(gw, "_generate_teacher_conversation", flaky)
        result = generate_workflow_dataset(
            complexity_level="L2",
            num_samples=1,
            teacher_model="fake-teacher",
            output_dir=tmp_output_dir,
            seed=5,
        )
        assert result.stats["repair_retries"] >= 1
        assert result.stats["repair_fallbacks"] == 0
        val = validate_dataset(result.output_files[0], "workflow")
        assert val.valid, val.errors

    def test_placeholder_path_needs_no_repair(self, tmp_output_dir: Path) -> None:
        result = generate_workflow_dataset(
            complexity_level="L3",
            num_samples=5,
            output_dir=tmp_output_dir,
            seed=5,
        )
        assert result.num_samples == 5
        assert result.stats["repair_retries"] == 0
        assert result.stats["repair_fallbacks"] == 0


class TestDomainSchema:
    """Tests for the new StateNode/Edge/DomainSpec schema and validate_domain."""

    def _make_minimal_valid_domain(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge,
        )
        return DomainSpec(
            name="Test",
            category="test",
            tools=(),
            intents=("help",),
            entity_slots=(),
            states=(
                StateNode("START", "greet", kind="initial"),
                StateNode("WORK", "do work", tools=("some_tool",)),
                StateNode("END", "close", kind="terminal"),
            ),
            edges=(
                Edge("START", "WORK", "proceed", "always"),
                Edge("WORK", "END", "done", "tool_success"),
            ),
            initial="START",
            terminals=("END",),
        )

    def test_validate_domain_passes_minimal(self):
        from llm_workflow_agents.data.domain_registry import validate_domain
        d = self._make_minimal_valid_domain()
        validate_domain(d)  # should not raise

    def test_validate_domain_rejects_unknown_edge_src(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(Edge("MISSING", "B", "x", "always"),),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="unknown state"):
            validate_domain(d)

    def test_validate_domain_rejects_self_loop(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(
                Edge("A", "A", "loop", "always"),
                Edge("A", "B", "done", "always"),
            ),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="self-loop"):
            validate_domain(d)

    def test_validate_domain_rejects_missing_spine_successor(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        # WORK has only an optional edge — no spine successor
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b"),
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(
                Edge("A", "B", "go", "always"),
                Edge("B", "C", "branch", "intent_match", optional=True, priority=1),
            ),
            initial="A", terminals=("C",),
        )
        with pytest.raises(ValueError, match="spine successor"):
            validate_domain(d)

    def test_validate_domain_rejects_tool_trigger_on_toolless_state(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b"),  # no tools
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(
                Edge("A", "B", "go", "always"),
                Edge("B", "C", "success", "tool_success"),  # tool_success but B has no tools
            ),
            initial="A", terminals=("C",),
        )
        with pytest.raises(ValueError, match="tool_success.*no tools"):
            validate_domain(d)

    def test_validate_domain_rejects_terminal_unreachable(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        # C is declared terminal but not reachable from initial
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(Edge("A", "B", "done", "always"),),
            initial="A", terminals=("B", "C"),
        )
        with pytest.raises(ValueError, match="unreachable"):
            validate_domain(d)

    def test_validate_domain_rejects_invalid_trigger(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain,
        )
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(Edge("A", "B", "x", "fire_photon_torpedoes"),),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="trigger"):
            validate_domain(d)

    def test_all_registry_domains_pass_validate(self):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY, validate_domain
        errors = []
        for key, domain in DOMAIN_REGISTRY.items():
            try:
                validate_domain(domain)
            except (ValueError, Exception) as e:
                errors.append(f"{key}: {e}")
        assert not errors, "\n".join(errors)


class TestSelectSubgraph:
    def test_l1_subgraph_has_3_to_4_states(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        rng = random.Random(42)
        spec = COMPLEXITY_SPECS[ComplexityLevel.L1]
        domain = DOMAIN_REGISTRY["account_management"]
        graph = select_subgraph(domain, spec, rng)
        assert 3 <= len(graph.states) <= 4

    def test_subgraph_no_duplicate_state_names(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        for level in [ComplexityLevel.L1, ComplexityLevel.L2, ComplexityLevel.L3]:
            spec = COMPLEXITY_SPECS[level]
            for key, domain in DOMAIN_REGISTRY.items():
                rng = random.Random(0)
                graph = select_subgraph(domain, spec, rng)
                names = [s.name for s in graph.states]
                assert len(names) == len(set(names)), f"duplicate names in {key} {level}: {names}"

    def test_subgraph_terminal_always_reachable(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L3]
        for key, domain in DOMAIN_REGISTRY.items():
            rng = random.Random(7)
            graph = select_subgraph(domain, spec, rng)
            terminal_names = set(graph.terminal_states)
            terminal_state_names = {
                s.name for s in graph.states
                if s.id in terminal_names or s.name in terminal_names
            }
            assert terminal_state_names, f"no terminal in subgraph for {key}"

    def test_subgraph_transitions_carry_label_and_trigger(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        domain = DOMAIN_REGISTRY["billing_payments"]
        rng = random.Random(1)
        graph = select_subgraph(domain, spec, rng)
        for t in graph.transitions:
            assert t.label, f"transition {t.from_state}->{t.to_state} missing label"
            assert t.trigger, f"transition {t.from_state}->{t.to_state} missing trigger"


class TestWalkPath:
    def _make_graph_and_domain(self, level="L1"):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        rng = random.Random(99)
        spec = COMPLEXITY_SPECS[ComplexityLevel(level)]
        domain = DOMAIN_REGISTRY["account_management"]
        graph = select_subgraph(domain, spec, rng)
        return graph, domain

    def test_walk_reaches_terminal(self):
        from llm_workflow_agents.data.generate_workflows import walk_path
        import random
        graph, domain = self._make_graph_and_domain("L2")
        rng = random.Random(1)
        path = walk_path(graph, domain, "cooperative", "service", rng)
        terminal_ids = set(graph.terminal_states)
        terminal_names = {s.name for s in graph.states if s.id in terminal_ids}
        assert path[-1].to_state in terminal_names or path[-1].to_state in terminal_ids, \
            f"walk did not reach terminal, last state: {path[-1].to_state}"

    def test_walk_all_transitions_are_valid_edges(self):
        from llm_workflow_agents.data.generate_workflows import walk_path
        import random
        graph, domain = self._make_graph_and_domain("L3")
        rng = random.Random(5)
        path = walk_path(graph, domain, "adversarial_probing", "service", rng)
        valid = {(t.from_state, t.to_state) for t in graph.transitions}
        for step in path:
            assert (step.from_state, step.to_state) in valid or \
                   (step.from_state, step.to_state) in {
                       (graph.states[i].name, graph.states[j].name)
                       for i in range(len(graph.states))
                       for j in range(len(graph.states))
                       for t in graph.transitions
                       if t.from_state == graph.states[i].id and t.to_state == graph.states[j].id
                   }, f"walk step {step.from_state}->{step.to_state} not a valid edge"

    def test_upsell_walk_traverses_upsell_arc(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph, walk_path
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        domain = DOMAIN_REGISTRY["account_management"]
        found_upsell = False
        for seed in range(50):
            rng = random.Random(seed)
            graph = select_subgraph(domain, spec, rng, intent_category="upsell_promo")
            upsell_transitions = [t for t in graph.transitions if t.intent_category == "upsell_promo"]
            if not upsell_transitions:
                continue
            rng2 = random.Random(seed)
            path = walk_path(graph, domain, "cooperative", "upsell_promo", rng2)
            upsell_edge_pairs = {(t.from_state, t.to_state) for t in upsell_transitions}
            for step in path:
                if (step.from_state, step.to_state) in upsell_edge_pairs:
                    found_upsell = True
                    break
        assert found_upsell, "No upsell path found across 50 seeds"
