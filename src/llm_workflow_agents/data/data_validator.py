"""Schema validation and quality checks for generated datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    valid: bool = True
    total_samples: int = 0
    valid_samples: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# Required fields per dataset type
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "workflow": [
        "conversation_id",
        "complexity_level",
        "domain",
        "workflow_graph",
        "messages",
    ],
    "tool_call": [
        "id",
        "messages",
    ],
    "graph_pair": [
        "id",
        "messages",
        "graph",
    ],
}


def _validate_workflow_sample(sample: dict[str, Any], idx: int) -> list[str]:
    """Validate a single workflow conversation sample."""
    errors: list[str] = []

    # Check required fields
    for field_name in _REQUIRED_FIELDS["workflow"]:
        if field_name not in sample:
            errors.append(f"Sample {idx}: missing required field '{field_name}'")

    # Validate workflow graph structure
    graph = sample.get("workflow_graph", {})
    if not graph.get("states"):
        errors.append(f"Sample {idx}: workflow_graph has no states")
    if not graph.get("initial"):
        errors.append(f"Sample {idx}: workflow_graph has no initial state")
    if not graph.get("terminal"):
        errors.append(f"Sample {idx}: workflow_graph has no terminal states")

    # Validate messages
    messages = sample.get("messages", [])
    if not messages:
        errors.append(f"Sample {idx}: no messages")
    elif messages[0].get("role") != "system":
        errors.append(f"Sample {idx}: first message should be system role")

    # Second-message role depends on who initiated the conversation.
    initiator = sample.get("conversation_initiator", "user")
    if len(messages) > 1:
        second_role = messages[1].get("role")
        if initiator == "agent":
            if second_role != "assistant":
                errors.append(
                    f"Sample {idx}: outbound (agent-initiated) conversation must have "
                    f"an assistant second message, got '{second_role}'"
                )
        elif second_role != "user":
            errors.append(
                f"Sample {idx}: inbound conversation must have a user second message, "
                f"got '{second_role}'"
            )

    # Validate state transitions are valid
    valid_states = set(graph.get("states", []))
    for msg in messages:
        annotations = msg.get("annotations", {})
        transition = annotations.get("state_transition", {})
        if transition:
            from_state = transition.get("from", "")
            to_state = transition.get("to", "")
            # State names (not IDs) are used in annotations, so we check format only
            if not from_state or not to_state:
                errors.append(f"Sample {idx}: empty state in transition")

    # Validate complexity level
    level = sample.get("complexity_level", "")
    if level not in ("L1", "L2", "L3", "L4", "L5"):
        errors.append(f"Sample {idx}: invalid complexity_level '{level}'")

    # Rationality checks (only when per-state detail is present; older/minimal
    # samples that carry just a flat ``states`` list are skipped).
    errors.extend(_check_workflow_rationality(sample, graph, messages, idx))

    return errors


def _check_workflow_rationality(
    sample: dict[str, Any],
    graph: dict[str, Any],
    messages: list[dict[str, Any]],
    idx: int,
) -> list[str]:
    """Enforce that workflow data is semantically coherent.

    Three checks (see ``.claude/rules/02-data-generation.md``):

    1. Tool-state coherence — every tool listed in a state exists in the
       sample's ``tool_schemas``, and every tool *called* in the ground-truth
       conversation for a state is listed in that state's tools.
    2. Instruction completeness — every non-terminal state has an instruction.
    3. Terminal reachability — at least one terminal is reachable from the
       initial state by following the transitions.
    """
    errors: list[str] = []

    state_details = graph.get("state_details") or []
    terminals = set(graph.get("terminal") or [])
    initial = graph.get("initial", "")

    # 1a. Tool-state coherence: listed tools must exist in tool_schemas.
    if state_details:
        schema_names = set()
        for ts in sample.get("tool_schemas") or []:
            fn = ts.get("function") or ts
            if fn.get("name"):
                schema_names.add(fn["name"])
        if schema_names:
            for sd in state_details:
                sname = sd.get("name", "")
                for tool in sd.get("tools") or []:
                    if tool not in schema_names:
                        errors.append(
                            f"Sample {idx}: state '{sname}' lists tool '{tool}' "
                            f"not present in tool_schemas"
                        )

    # 1b. Tool-state coherence: GT-called tools must be listed in their state.
    if state_details and messages:
        from llm_workflow_agents.data._workflow_script import (
            find_tool_placement_violations,
        )

        listed_by_state = {
            sd.get("name", ""): set(sd.get("tools") or []) for sd in state_details
        }
        for violation in find_tool_placement_violations(listed_by_state, messages):
            errors.append(f"Sample {idx}: ground-truth {violation}")

    # 2. Instruction completeness: every non-terminal state needs an instruction.
    for sd in state_details:
        sname = sd.get("name", "")
        if sname in terminals:
            continue
        if not (sd.get("instruction") or "").strip():
            errors.append(f"Sample {idx}: state '{sname}' has no instruction")

    # 3. Terminal reachability via BFS over transitions (state names).
    transitions = graph.get("transitions") or []
    if initial and terminals and transitions:
        adjacency: dict[str, list[str]] = {}
        for tr in transitions:
            adjacency.setdefault(tr.get("from", ""), []).append(tr.get("to", ""))
        seen = {initial}
        stack = [initial]
        while stack:
            cur = stack.pop()
            for nxt in adjacency.get(cur, []):
                if nxt and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if not (seen & terminals):
            errors.append(
                f"Sample {idx}: no terminal state reachable from initial '{initial}'"
            )

    return errors


def _validate_tool_call_sample(sample: dict[str, Any], idx: int) -> list[str]:
    """Validate a single tool-call training sample."""
    errors: list[str] = []

    for field_name in _REQUIRED_FIELDS["tool_call"]:
        if field_name not in sample:
            errors.append(f"Sample {idx}: missing required field '{field_name}'")

    messages = sample.get("messages", [])
    if not messages:
        errors.append(f"Sample {idx}: no messages")

    return errors


def _validate_graph_pair_sample(sample: dict[str, Any], idx: int) -> list[str]:
    """Validate a single graph pair sample."""
    errors: list[str] = []

    for field_name in _REQUIRED_FIELDS["graph_pair"]:
        if field_name not in sample:
            errors.append(f"Sample {idx}: missing required field '{field_name}'")

    # Validate graph structure
    graph = sample.get("graph", {})
    if not graph.get("nodes"):
        errors.append(f"Sample {idx}: graph has no nodes")
    if not graph.get("initial_state"):
        errors.append(f"Sample {idx}: graph has no initial_state")
    if not graph.get("terminal_states"):
        errors.append(f"Sample {idx}: graph has no terminal_states")

    # Validate structural validity
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {n["id"] for n in nodes}

    initial = graph.get("initial_state", "")
    if initial and initial not in node_ids:
        errors.append(f"Sample {idx}: initial_state '{initial}' not in nodes")

    for terminal in graph.get("terminal_states", []):
        if terminal not in node_ids:
            errors.append(f"Sample {idx}: terminal_state '{terminal}' not in nodes")

    for edge in edges:
        if edge.get("from_state") not in node_ids:
            errors.append(f"Sample {idx}: edge from_state '{edge.get('from_state')}' not in nodes")
        if edge.get("to_state") not in node_ids:
            errors.append(f"Sample {idx}: edge to_state '{edge.get('to_state')}' not in nodes")

    return errors


_VALIDATORS = {
    "workflow": _validate_workflow_sample,
    "tool_call": _validate_tool_call_sample,
    "graph_pair": _validate_graph_pair_sample,
}


def validate_dataset(
    path: Path,
    dataset_type: str = "workflow",
) -> ValidationResult:
    """Validate a dataset JSONL file.

    Args:
        path: Path to JSONL file.
        dataset_type: One of 'workflow', 'tool_call', 'graph_pair'.

    Returns:
        ValidationResult with validity status and details.
    """
    if dataset_type not in _VALIDATORS:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown dataset_type: {dataset_type}"],
        )

    validator = _VALIDATORS[dataset_type]
    result = ValidationResult()

    logger.info("validating_dataset", path=str(path), type=dataset_type)

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            result.total_samples += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                result.errors.append(f"Line {line_num}: invalid JSON: {e}")
                continue

            errors = validator(sample, line_num)
            if errors:
                result.errors.extend(errors)
            else:
                result.valid_samples += 1

    result.valid = len(result.errors) == 0
    result.stats = {
        "total": result.total_samples,
        "valid": result.valid_samples,
        "error_count": len(result.errors),
    }

    logger.info("validation_complete", **result.stats)

    return result
