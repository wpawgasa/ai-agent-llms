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
