"""Experiment C: Generate (prompt, graph) pairs for graph extraction training.

Constructs 5,000 pairs from workflow prompts with gold annotations,
teacher-generated pairs, and paraphrase augmentation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.data.generate_tool_call_data import DatasetSplits

logger = structlog.get_logger(__name__)


@dataclass
class GraphNode:
    """A node in the workflow graph."""

    id: str
    name: str
    tools: list[str] = field(default_factory=list)
    entry_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tools": self.tools,
            "entry_actions": self.entry_actions,
        }


@dataclass
class GraphEdge:
    """An edge in the workflow graph."""

    from_state: str
    to_state: str
    condition: str
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "condition": self.condition,
            "priority": self.priority,
        }


@dataclass
class WorkflowGraphOutput:
    """Structured workflow graph output for training."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    initial_state: str
    terminal_states: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "initial_state": self.initial_state,
            "terminal_states": self.terminal_states,
        }


def _load_workflow_prompts(prompts_dir: Path) -> list[dict[str, Any]]:
    """Load workflow prompts from Exp A output."""
    prompts: list[dict[str, Any]] = []

    for jsonl_file in sorted(prompts_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    prompts.append(sample)

    if not prompts:
        logger.warning("no_prompts_found", dir=str(prompts_dir))

    return prompts


def _extract_graph_from_workflow(sample: dict[str, Any]) -> WorkflowGraphOutput:
    """Extract a WorkflowGraphOutput from an Exp A conversation sample."""
    wf = sample.get("workflow_graph", {})

    nodes = []
    for state_detail in wf.get("state_details", []):
        nodes.append(
            GraphNode(
                id=state_detail["id"],
                name=state_detail["name"],
                tools=state_detail.get("tools", []),
                entry_actions=state_detail.get("entry_actions", []),
            )
        )

    # Fallback: create nodes from state IDs if no details
    if not nodes:
        for state_id in wf.get("states", []):
            nodes.append(GraphNode(id=state_id, name=state_id))

    edges = []
    for trans in wf.get("transitions", []):
        edges.append(
            GraphEdge(
                from_state=trans["from"],
                to_state=trans["to"],
                condition=trans.get("condition", ""),
                priority=trans.get("priority", 0),
            )
        )

    return WorkflowGraphOutput(
        nodes=nodes,
        edges=edges,
        initial_state=wf.get("initial", nodes[0].id if nodes else "S1"),
        terminal_states=wf.get("terminal", [nodes[-1].id] if nodes else ["S1"]),
    )


def _create_graph_pair(
    prompt_text: str,
    graph: WorkflowGraphOutput,
    pair_id: str,
    source: str,
) -> dict[str, Any]:
    """Create a (prompt, graph) training pair."""
    return {
        "id": pair_id,
        "source": source,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract the workflow graph from the following prompt. "
                    "Output a JSON object with nodes, edges, initial_state, "
                    "and terminal_states."
                ),
            },
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": json.dumps(graph.to_dict())},
        ],
        "graph": graph.to_dict(),
    }


def _augment_with_paraphrases(
    pairs: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Augment pairs via paraphrasing to reach target size.

    In production, uses teacher model for paraphrasing.
    For now, creates variants with minor prompt modifications.
    """
    augmented = list(pairs)

    while len(augmented) < target_size:
        base = rng.choice(pairs)
        user_msg = base["messages"][1]["content"]

        # Simple augmentation: add prefix/suffix variations
        prefixes = [
            "Please process: ",
            "Handle the following workflow: ",
            "Given this workflow description: ",
            "Extract the graph for: ",
            "",
        ]
        prefix = rng.choice(prefixes)
        new_id = f"aug_{len(augmented):05d}"

        new_pair = _create_graph_pair(
            prompt_text=prefix + user_msg,
            graph=WorkflowGraphOutput(
                nodes=[GraphNode(**n) for n in base["graph"]["nodes"]],
                edges=[GraphEdge(**e) for e in base["graph"]["edges"]],
                initial_state=base["graph"]["initial_state"],
                terminal_states=base["graph"]["terminal_states"],
            ),
            pair_id=new_id,
            source="augmented",
        )
        augmented.append(new_pair)

    return augmented[:target_size]


def generate_graph_pairs(
    workflow_prompts_dir: Path = Path("data/output/exp_a"),
    gold_annotations: int = 200,
    teacher_generated: int = 800,
    augmentation_target: int = 5000,
    output_dir: Path = Path("data/output/exp_c"),
    seed: int = 42,
) -> DatasetSplits:
    """Generate (prompt, graph) pairs for graph extraction training.

    Args:
        workflow_prompts_dir: Directory with Exp A workflow JSONL files.
        gold_annotations: Number of gold-standard annotated pairs.
        teacher_generated: Number of teacher-model generated pairs.
        augmentation_target: Total target size after augmentation.
        output_dir: Output directory.
        seed: Random seed.

    Returns:
        DatasetSplits with 4000 train / 500 val / 500 test.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "generating_graph_pairs",
        prompts_dir=str(workflow_prompts_dir),
        gold=gold_annotations,
        teacher=teacher_generated,
        target=augmentation_target,
    )

    # Load workflow prompts from Exp A
    prompts = _load_workflow_prompts(workflow_prompts_dir)

    pairs: list[dict[str, Any]] = []

    if prompts:
        # Use existing workflow data
        for i, sample in enumerate(prompts[:gold_annotations]):
            graph = _extract_graph_from_workflow(sample)
            system_msg = next(
                (m["content"] for m in sample.get("messages", []) if m["role"] == "system"),
                f"Workflow for {sample.get('domain', 'unknown')}",
            )
            pair = _create_graph_pair(system_msg, graph, f"gold_{i:04d}", "gold")
            pairs.append(pair)

        for i, sample in enumerate(
            prompts[gold_annotations : gold_annotations + teacher_generated]
        ):
            graph = _extract_graph_from_workflow(sample)
            system_msg = next(
                (m["content"] for m in sample.get("messages", []) if m["role"] == "system"),
                f"Workflow for {sample.get('domain', 'unknown')}",
            )
            pair = _create_graph_pair(system_msg, graph, f"teacher_{i:04d}", "teacher")
            pairs.append(pair)
    else:
        # Generate placeholder pairs if no prompts available
        logger.warning("no_workflow_prompts_generating_placeholders")
        for i in range(gold_annotations + teacher_generated):
            nodes = [
                GraphNode(id="S1", name="START"),
                GraphNode(id="S2", name="PROCESS"),
                GraphNode(id="S3", name="END"),
            ]
            edges = [
                GraphEdge(from_state="S1", to_state="S2", condition="input_received"),
                GraphEdge(from_state="S2", to_state="S3", condition="processing_complete"),
            ]
            graph = WorkflowGraphOutput(
                nodes=nodes, edges=edges, initial_state="S1", terminal_states=["S3"]
            )
            source = "gold" if i < gold_annotations else "teacher"
            pair = _create_graph_pair(
                f"Placeholder workflow prompt {i}", graph, f"{source}_{i:04d}", source
            )
            pairs.append(pair)

    # Augment to target size
    all_pairs = _augment_with_paraphrases(pairs, augmentation_target, rng)
    rng.shuffle(all_pairs)

    # Split: 4000 train / 500 val / 500 test
    train_data = all_pairs[:4000]
    val_data = all_pairs[4000:4500]
    test_data = all_pairs[4500:5000]

    # Write JSONL
    paths: dict[str, Path] = {}
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for pair in split_data:
                f.write(json.dumps(pair) + "\n")
        paths[split_name] = path

    logger.info(
        "graph_pairs_complete",
        train=len(train_data),
        val=len(val_data),
        test=len(test_data),
    )

    return DatasetSplits(
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        train_size=len(train_data),
        val_size=len(val_data),
        test_size=len(test_data),
    )
