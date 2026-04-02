"""Experiment C: Generate (prompt, graph) pairs for graph extraction training.

Constructs 5,000 pairs from workflow prompts with gold annotations,
teacher-generated pairs, and paraphrase augmentation.

Teacher model generation:
  Set ``teacher_model`` in ``generate_graph_pairs`` to use a live API for
  both the teacher-generated pairs and paraphrase augmentation instead of
  the placeholder implementations.
  Supported prefixes: ``gemini-*``, ``gpt-*``, ``claude-*``.
  Falls back to placeholder on API error.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model
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


_GRAPH_PAIR_SYSTEM_PROMPT = """\
You are a dataset generation expert creating graph-extraction training data.
Given a workflow description prompt and its corresponding graph, produce a
paraphrased version of the prompt that preserves all workflow semantics.

OUTPUT FORMAT — return a JSON object:
{"paraphrased_prompt": "<rewritten prompt text>"}

RULES:
- Keep all state names, tool names, and transition conditions intact.
- Change sentence structure, vocabulary, and ordering only.
- Output ONLY the JSON object — no markdown fences.
"""

_TEACHER_GRAPH_SYSTEM_PROMPT = """\
You are a dataset generation expert creating graph-extraction training data.
Given a workflow description, extract and output the workflow graph as JSON.

OUTPUT FORMAT — return a JSON object:
{
  "prompt": "<clear natural-language description of the workflow>",
  "graph": {
    "nodes": [{"id": "S1", "name": "...", "tools": [...], "entry_actions": [...]}],
    "edges": [{"from_state": "S1", "to_state": "S2", "condition": "...", "priority": 0}],
    "initial_state": "S1",
    "terminal_states": ["SN"]
  }
}

RULES:
- Nodes must include at least one initial and one terminal state.
- Every node (except terminal) must have at least one outgoing edge.
- Output ONLY the JSON object — no markdown fences.
"""


def _augment_with_paraphrases(
    pairs: list[dict[str, Any]],
    target_size: int,
    rng: random.Random,
    teacher_model: str | None = None,
) -> list[dict[str, Any]]:
    """Augment pairs via paraphrasing to reach target size.

    Uses the teacher model for paraphrasing when ``teacher_model`` is set,
    otherwise creates variants with minor prefix modifications.
    """
    augmented = list(pairs)
    _prefixes = [
        "Please process: ",
        "Handle the following workflow: ",
        "Given this workflow description: ",
        "Extract the graph for: ",
        "",
    ]

    while len(augmented) < target_size:
        base = rng.choice(pairs)
        user_msg = base["messages"][1]["content"]
        new_id = f"aug_{len(augmented):05d}"
        graph = WorkflowGraphOutput(
            nodes=[GraphNode(**n) for n in base["graph"]["nodes"]],
            edges=[GraphEdge(**e) for e in base["graph"]["edges"]],
            initial_state=base["graph"]["initial_state"],
            terminal_states=base["graph"]["terminal_states"],
        )

        if teacher_model:
            paraphrased_prompt = _paraphrase_prompt(user_msg, teacher_model, user_msg)
        else:
            paraphrased_prompt = rng.choice(_prefixes) + user_msg

        augmented.append(
            _create_graph_pair(paraphrased_prompt, graph, new_id, "augmented")
        )

    return augmented[:target_size]


def _paraphrase_prompt(original: str, teacher_model: str, fallback: str) -> str:
    """Return a paraphrased version of ``original`` via the teacher model."""
    try:
        raw = call_teacher_model(
            teacher_model,
            _GRAPH_PAIR_SYSTEM_PROMPT,
            f"Paraphrase this workflow prompt:\n\n{original}",
        )
        data = json.loads(raw)
        return data.get("paraphrased_prompt", fallback)
    except Exception as exc:
        logger.warning("paraphrase_fallback", error=str(exc))
        return fallback


def _generate_teacher_graph_pair(
    pair_id: str,
    teacher_model: str,
    domain_hint: str = "customer service",
) -> dict[str, Any] | None:
    """Ask the teacher model to generate a novel (prompt, graph) pair.

    Returns ``None`` on failure so the caller can fall back to placeholder data.
    """
    user_prompt = (
        f"Generate a {domain_hint} workflow with 3–7 states and 2–5 tools. "
        "Produce a natural-language description and the corresponding graph."
    )
    try:
        raw = call_teacher_model(teacher_model, _TEACHER_GRAPH_SYSTEM_PROMPT, user_prompt)
        data = json.loads(raw)
        prompt_text = data.get("prompt", "")
        g = data.get("graph", {})
        if not prompt_text or not g:
            return None
        nodes = [GraphNode(**n) for n in g.get("nodes", [])]
        edges = [GraphEdge(**e) for e in g.get("edges", [])]
        graph = WorkflowGraphOutput(
            nodes=nodes,
            edges=edges,
            initial_state=g.get("initial_state", nodes[0].id if nodes else "S1"),
            terminal_states=g.get("terminal_states", [nodes[-1].id] if nodes else ["S1"]),
        )
        return _create_graph_pair(prompt_text, graph, pair_id, "teacher")
    except Exception as exc:
        logger.warning("teacher_graph_pair_failed", pair_id=pair_id, error=str(exc))
        return None


def generate_graph_pairs(
    workflow_prompts_dir: Path = Path("data/output/exp_a"),
    gold_annotations: int = 200,
    teacher_generated: int = 800,
    augmentation_target: int = 5000,
    teacher_model: str | None = None,
    output_dir: Path = Path("data/output/exp_c"),
    seed: int = 42,
) -> DatasetSplits:
    """Generate (prompt, graph) pairs for graph extraction training.

    Args:
        workflow_prompts_dir: Directory with Exp A workflow JSONL files.
        gold_annotations: Number of gold-standard annotated pairs.
        teacher_generated: Number of teacher-model generated pairs.
        augmentation_target: Total target size after augmentation.
        teacher_model: Teacher model name for live API generation of
            teacher pairs and paraphrase augmentation.
            Supported prefixes: ``gemini-*``, ``gpt-*``, ``claude-*``.
            If ``None``, uses placeholder generation throughout.
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
        teacher_model=teacher_model or "placeholder",
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
            pair_id = f"teacher_{i:04d}"
            if teacher_model:
                pair = _generate_teacher_graph_pair(
                    pair_id,
                    teacher_model,
                    domain_hint=sample.get("domain", "customer service"),
                )
            else:
                pair = None
            if pair is None:
                graph = _extract_graph_from_workflow(sample)
                system_msg = next(
                    (m["content"] for m in sample.get("messages", []) if m["role"] == "system"),
                    f"Workflow for {sample.get('domain', 'unknown')}",
                )
                pair = _create_graph_pair(system_msg, graph, pair_id, "teacher")
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
    all_pairs = _augment_with_paraphrases(pairs, augmentation_target, rng, teacher_model)
    rng.shuffle(all_pairs)

    # Split: 4000 train / 500 val / 500 test
    train_data = all_pairs[:4000]
    val_data = all_pairs[4000:4500]
    test_data = all_pairs[4500:5000]

    # Write JSONL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        path = output_dir / f"{split_name}_{timestamp}.jsonl"
        with open(path, "w") as f:
            for pair in split_data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
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
