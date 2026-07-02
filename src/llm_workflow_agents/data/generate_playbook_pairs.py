"""Task C: generate (natural-language playbook -> WorkflowGraph JSON) training pairs.

See docs/data_generation_recipes_task_c.md for the full recipe. This module is
built in three layers:
  1. Pure planning (this file, part 1): graph pool assembly, deterministic split
     pre-assignment, per-graph rendering plans.
  2. Per-rendering pipeline (part 2): render -> verify -> repair -> row assembly.
  3. Dataset orchestration (part 3): generate_playbook_dataset, stats, merge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import structlog

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
from llm_workflow_agents.data._graph_invention import DOMAIN_BRIEFS, invent_novel_graphs
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
from llm_workflow_agents.data.generate_workflows import _select_domain, select_subgraph

logger = structlog.get_logger(__name__)

# Task C shifts mass toward mid-size graphs vs Task A (tiny L1 graphs saturate).
TASK_C_LEVEL_WEIGHTS: dict[str, float] = {"L1": 0.10, "L2": 0.25, "L3": 0.35, "L4": 0.20, "L5": 0.10}

# The six playbook registers (final home is _playbook_render.Register; string values match).
REGISTERS: tuple[str, ...] = (
    "sop_document",
    "prose_narrative",
    "bullet_quick_reference",
    "wiki_training_notes",
    "state_script",
    "manager_transcript",
)
_ANCHOR_REGISTERS = ("sop_document", "prose_narrative")

_HELDOUT_DOMAINS = ("utilities", "surveys")
_HELDOUT_REGISTER = "manager_transcript"


@dataclass
class GraphPoolEntry:
    """One gold graph in the generation pool (name-keyed interchange shape)."""

    graph_id: str
    source: str  # "registry" | "invented"
    domain: str  # registry key or invented brief slug
    complexity_level: str  # "L1".."L5" for registry, "NA" for invented
    graph: dict[str, Any]
    tool_schemas: list[dict[str, Any]]


@dataclass
class RenderingPlan:
    """A single planned (playbook, graph) rendering."""

    pair_id: str
    graph_id: str
    register: str
    language: str
    distractor_count: int
    paraphrase_density: str
    condition_explicitness: str


def _tool_stub(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name.replace("_", " "),
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def build_graph_pool(
    num_graphs: int,
    invented_ratio: float,
    invention_teacher: str,
    rng: random.Random,
) -> list[GraphPoolEntry]:
    """Assemble a pool of `num_graphs` gold graphs (registry subgraphs + invented)."""
    n_invented_target = round(num_graphs * invented_ratio)
    invented = (
        invent_novel_graphs(DOMAIN_BRIEFS, count=n_invented_target, teacher_model=invention_teacher, rng=rng)
        if n_invented_target
        else []
    )
    n_registry = num_graphs - len(invented)

    # Deterministic interleave of source labels.
    labels = ["invented"] * len(invented) + ["registry"] * n_registry
    rng.shuffle(labels)

    pool: list[GraphPoolEntry] = []
    invented_iter = iter(invented)
    for idx, label in enumerate(labels, start=1):
        graph_id = f"G{idx:04d}"
        if label == "invented":
            inv = next(invented_iter)
            used = {t for sd in inv.graph["state_details"] for t in sd["tools"]}
            pool.append(
                GraphPoolEntry(
                    graph_id=graph_id,
                    source="invented",
                    domain=inv.brief_slug,
                    complexity_level="NA",
                    graph=inv.graph,
                    tool_schemas=[_tool_stub(name) for name in sorted(used)],
                )
            )
        else:
            level = _weighted_choice(rng, TASK_C_LEVEL_WEIGHTS)
            spec = COMPLEXITY_SPECS[ComplexityLevel(level)]
            domain_key, domain_spec = _select_domain(rng, None, spec)
            workflow = select_subgraph(domain_spec, spec, rng)
            graph = workflow.to_dict()
            used = {t for sd in graph["state_details"] for t in sd["tools"]}
            tool_schemas = [t for t in domain_spec.tools if t["function"]["name"] in used]
            pool.append(
                GraphPoolEntry(
                    graph_id=graph_id,
                    source="registry",
                    domain=domain_key,
                    complexity_level=level,
                    graph=graph,
                    tool_schemas=tool_schemas,
                )
            )
    return pool


def assign_splits(
    pool: list[GraphPoolEntry],
    heldout_domains: tuple[str, ...] = _HELDOUT_DOMAINS,
    heldout_brief_fraction: float = 0.20,
    ratios: tuple[float, float, float] = (0.85, 0.10, 0.05),
    seed: int = 142,
) -> dict[str, str]:
    """Assign each graph_id to train/validation/test at the GRAPH level (never row level).

    Held-out registry domains and a fraction of invented briefs go entirely to test.
    """
    rng = random.Random(seed)
    by_id = {e.graph_id: e for e in pool}
    ids = sorted(by_id)
    n = len(ids)

    invented_briefs = sorted({e.domain for e in pool if e.source == "invented"})
    n_held = round(len(invented_briefs) * heldout_brief_fraction) if invented_briefs else 0
    held_briefs = set(rng.sample(invented_briefs, n_held)) if n_held else set()
    heldout = set(heldout_domains) | held_briefs

    test = {gid for gid in ids if by_id[gid].domain in heldout}
    remaining = [gid for gid in ids if gid not in test]
    rng.shuffle(remaining)

    while len(test) < round(n * ratios[2]) and remaining:
        test.add(remaining.pop())
    n_val = round(n * ratios[1])
    val = set(remaining[:n_val])

    return {
        gid: "test" if gid in test else "validation" if gid in val else "train"
        for gid in ids
    }


def plan_renderings(
    pool: list[GraphPoolEntry],
    splits: dict[str, str],
    language_mix: dict[str, float],
    rng: random.Random,
) -> list[RenderingPlan]:
    """Assign 4-6 registers, a language, and difficulty knobs to each graph."""
    plans: list[RenderingPlan] = []
    for entry in pool:
        gid = entry.graph_id
        is_train = splits[gid] == "train"
        candidates = [r for r in REGISTERS if not (is_train and r == _HELDOUT_REGISTER)]

        k = rng.randint(4, min(6, len(candidates)))
        chosen = ["state_script", rng.choice(_ANCHOR_REGISTERS)]
        pool_rest = [r for r in candidates if r not in chosen]
        rng.shuffle(pool_rest)
        while len(chosen) < k and pool_rest:
            chosen.append(pool_rest.pop())
        # Preserve a stable, non-alphabetical order without bias.
        rng.shuffle(chosen)

        for i, register in enumerate(chosen, start=1):
            distractor_count = 0 if rng.random() >= 0.30 else rng.randint(1, 3)
            paraphrase_density = rng.choices(
                ["low", "medium", "high"], weights=[0.40, 0.40, 0.20], k=1
            )[0]
            condition_explicitness = rng.choices(
                ["explicit", "narrative_order", "listing_order"], weights=[0.40, 0.35, 0.25], k=1
            )[0]
            language = _weighted_choice(rng, language_mix)
            plans.append(
                RenderingPlan(
                    pair_id=f"{gid}_r{i}",
                    graph_id=gid,
                    register=register,
                    language=language,
                    distractor_count=distractor_count,
                    paraphrase_density=paraphrase_density,
                    condition_explicitness=condition_explicitness,
                )
            )
    return plans
