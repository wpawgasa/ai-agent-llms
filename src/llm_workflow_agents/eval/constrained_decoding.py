"""Constrained decoding integration for structured JSON output.

Provides helpers to configure Outlines or XGrammar for constraining
model output to valid WorkflowGraph JSON during Experiment C inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# JSON schema for WorkflowGraph (matches data/templates/graph_output_schema.json)
WORKFLOW_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "entry_actions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_state": {"type": "string"},
                    "to_state": {"type": "string"},
                    "condition": {"type": "string"},
                    "priority": {"type": "integer"},
                },
                "required": ["from_state", "to_state", "condition"],
            },
        },
        "initial_state": {"type": "string"},
        "terminal_states": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["nodes", "edges", "initial_state", "terminal_states"],
}


def load_graph_schema(schema_path: Path | None = None) -> dict[str, Any]:
    """Load the WorkflowGraph JSON schema.

    Args:
        schema_path: Path to custom schema file. If None, uses built-in schema.

    Returns:
        JSON schema dict.
    """
    if schema_path is not None and schema_path.exists():
        with open(schema_path) as f:
            return json.load(f)
    return WORKFLOW_GRAPH_SCHEMA


def build_outlines_generator(
    model_name: str,
    schema: dict[str, Any] | None = None,
) -> Any:
    """Build an Outlines JSON-constrained generator.

    Defers outlines import. Returns a generator that produces
    valid JSON matching the WorkflowGraph schema.

    Args:
        model_name: HuggingFace model name.
        schema: JSON schema to constrain output. Defaults to WORKFLOW_GRAPH_SCHEMA.

    Returns:
        Outlines generator object.
    """
    import outlines
    from outlines import models

    if schema is None:
        schema = WORKFLOW_GRAPH_SCHEMA

    model = models.transformers(model_name)
    generator = outlines.generate.json(model, schema)

    logger.info("outlines_generator_built", model=model_name)
    return generator


def build_xgrammar_constraint(
    schema: dict[str, Any] | None = None,
) -> Any:
    """Build an XGrammar JSON constraint for vLLM guided decoding.

    XGrammar integrates with vLLM's guided decoding interface via
    the `guided_json` parameter in the API request.

    Args:
        schema: JSON schema to constrain output. Defaults to WORKFLOW_GRAPH_SCHEMA.

    Returns:
        Schema dict suitable for vLLM's guided_json parameter.
    """
    if schema is None:
        schema = WORKFLOW_GRAPH_SCHEMA

    logger.info("xgrammar_constraint_built")
    return schema


def generate_constrained_graph(
    prompt: str,
    model_name: str,
    base_url: str = "http://localhost:8000/v1",
    schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generate a workflow graph using vLLM with JSON-constrained decoding.

    Uses vLLM's guided_json parameter for constrained output.

    Args:
        prompt: Input workflow description prompt.
        model_name: Model name served by vLLM.
        base_url: vLLM server URL.
        schema: JSON schema constraint. Defaults to WORKFLOW_GRAPH_SCHEMA.

    Returns:
        Parsed WorkflowGraph dict, or None on failure.
    """
    import openai

    if schema is None:
        schema = WORKFLOW_GRAPH_SCHEMA

    client = openai.OpenAI(base_url=base_url, api_key="unused")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the workflow graph from the following prompt. "
                    "Output ONLY valid JSON matching the schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        extra_body={"guided_json": schema},
        temperature=0.0,
        max_tokens=2048,
    )

    content = response.choices[0].message.content if response.choices else None
    if content is None:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("constrained_decode_invalid_json", content=content[:200])
        return None
