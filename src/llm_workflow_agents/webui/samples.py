"""Task A benchmark sample loading and prompt building for the chatbot UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
from llm_workflow_agents.eval.graph_extraction_eval import WorkflowGraph, graph_to_mermaid

_LEVELS = {"L1", "L2", "L3", "L4", "L5"}


def _level_files(data_dir: Path, level: str) -> list[Path]:
    return sorted(data_dir.glob(f"{level.lower()}_*.jsonl"))


def _iter_samples(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def list_samples(data_dir: Path, level: str) -> list[dict[str, Any]]:
    """Return lightweight summaries of all samples at a complexity level."""
    if level not in _LEVELS or not data_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for s in _iter_samples(_level_files(data_dir, level)):
        first_user = next(
            (m.get("content", "") for m in s.get("messages", []) if m.get("role") == "user"),
            "",
        )
        out.append(
            {
                "conversation_id": s.get("conversation_id", ""),
                "domain": s.get("domain", ""),
                "num_states": s.get("num_states", 0),
                "num_tools": s.get("num_tools", 0),
                "preview": first_user[:120],
            }
        )
    return out


def get_sample(data_dir: Path, conversation_id: str) -> dict[str, Any] | None:
    """Find a full sample by conversation_id across all level files."""
    if not data_dir.exists():
        return None
    for s in _iter_samples(sorted(data_dir.glob("*.jsonl"))):
        if s.get("conversation_id") == conversation_id:
            return s
    return None


def build_sample_prompt(sample: dict[str, Any]) -> dict[str, Any]:
    """Build the benchmark's enriched system prompt + tools + seed user message.

    Reuses ``build_enriched_system_prompt`` so the prompt is identical to what
    ``eval.agent_benchmark`` sends for this sample.
    """
    system_msg = next(
        (m for m in sample.get("messages", []) if m.get("role") == "system"),
        None,
    )
    original = system_msg.get("content", "") if system_msg else ""
    enriched = build_enriched_system_prompt(sample, original)
    messages = sample.get("messages", [])
    seed_user = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    # Outbound (support-initiated) conversations open with an assistant turn
    # right after `system` (conversation_initiator == "agent"). Surface that
    # opener so the UI can render it as the first bubble instead of waiting on
    # the user. Inbound conversations leave this empty.
    first_non_system = next((m for m in messages if m.get("role") != "system"), None)
    seed_assistant = (
        first_non_system.get("content", "")
        if first_non_system and first_non_system.get("role") == "assistant"
        else ""
    )
    return {
        "system_prompt": enriched,
        "tools": sample.get("tool_schemas", []),
        "seed_user": seed_user,
        "seed_assistant": seed_assistant,
        "conversation_initiator": sample.get("conversation_initiator", "user"),
        "mermaid": build_workflow_mermaid(sample),
    }


def build_workflow_mermaid(sample: dict[str, Any]) -> str:
    """Render the sample's workflow_graph state machine as Mermaid markup.

    Reuses ``eval.graph_extraction_eval.graph_to_mermaid`` for the base
    node/edge markup, then appends classDef/class lines to highlight the
    initial and terminal states. Returns ``""`` when the sample has no states.
    """
    g = sample.get("workflow_graph") or {}
    states = g.get("states") or []
    if not states:
        return ""
    wg = WorkflowGraph(
        nodes=[{"id": s, "name": s} for s in states],
        edges=[
            {
                "from_state": t.get("from", ""),
                "to_state": t.get("to", ""),
                "condition": t.get("condition", ""),
            }
            for t in (g.get("transitions") or [])
        ],
    )
    lines = [graph_to_mermaid(wg)]
    lines.append("    classDef initial fill:#1f6feb,stroke:#79c0ff,color:#fff;")
    lines.append("    classDef terminal fill:#238636,stroke:#56d364,color:#fff;")
    initial = g.get("initial")
    if initial in states:
        lines.append(f"    class {initial} initial")
    terms = [t for t in (g.get("terminal") or []) if t in states]
    if terms:
        lines.append(f"    class {','.join(terms)} terminal")
    return "\n".join(lines)
