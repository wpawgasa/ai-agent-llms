"""Experiment B: Generate tool-call fine-tuning datasets.

Merges external datasets (xlam-60k, ToolBench) with custom synthetic data
into a unified fine-tuning JSONL.

Teacher model generation:
  Set ``teacher_model`` in ``generate_tool_call_dataset`` to a model name to
  call a live API for synthetic sample generation instead of the placeholder
  generator.  Supported prefixes: ``gemini-*``, ``gpt-*``, ``claude-*``.
  Falls back to placeholder on API error.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model

logger = structlog.get_logger(__name__)


@dataclass
class DatasetSplits:
    """Paths and statistics for train/val/test splits."""

    train_path: Path
    val_path: Path
    test_path: Path
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    token_stats: dict[str, Any] = field(default_factory=dict)


# Negative example categories and their proportions (within the 15% negative ratio)
NEGATIVE_CATEGORIES: dict[str, float] = {
    "wrong_tool_selected": 0.05 / 0.15,
    "hallucinated_tool": 0.04 / 0.15,
    "invalid_state_transition": 0.03 / 0.15,
    "error_recovery": 0.03 / 0.15,
}


def _load_external_dataset(source: str) -> list[dict[str, Any]]:
    """Load an external dataset from HuggingFace.

    In production, uses `datasets.load_dataset()`. Returns placeholder data
    for offline development.
    """
    logger.info("loading_external_dataset", source=source)

    try:
        from datasets import load_dataset

        if "xlam" in source.lower():
            ds = load_dataset(source, split="train")
            return [dict(row) for row in ds]
        elif "toolbench" in source.lower():
            ds = load_dataset(source, split="train")
            return [dict(row) for row in ds]
    except Exception:
        logger.warning("external_dataset_unavailable", source=source)

    # Return empty list — caller handles gracefully
    return []


_SYNTHETIC_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_products",
        "description": "Search product catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "max_price": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

_TOOL_CALL_SYSTEM_PROMPT = """\
You are a dataset generation expert creating tool-call training data for LLM fine-tuning.
Given a set of tool schemas, generate a realistic single-turn conversation where a user
requests an action and the assistant responds with the correct tool call.

OUTPUT FORMAT — return a JSON object:
{
  "messages": [
    {"role": "system", "content": "You have access to the following tools: ..."},
    {"role": "user", "content": "<realistic user request>"},
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>",
      "tool_calls": [{"name": "...", "arguments": {...}}]
    }
  ],
  "tools": [<tool schemas>]
}

RULES:
- The user request must be natural and realistic, not a template.
- Arguments must be plausible values, not placeholders.
- Select the single most appropriate tool.
- Output ONLY the JSON object — no markdown fences.
"""


def _generate_placeholder_sample(
    sample_id: str,
    tool_schemas: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one structurally valid placeholder sample."""
    tool = rng.choice(tool_schemas)
    return {
        "id": sample_id,
        "source": "custom_synthetic",
        "messages": [
            {
                "role": "system",
                "content": f"You have access to: {json.dumps([t['name'] for t in tool_schemas])}",
            },
            {"role": "user", "content": f"Placeholder request for {tool['name']}"},
            {
                "role": "assistant",
                "content": f'<tool_call>{json.dumps({"name": tool["name"], "arguments": {}})}</tool_call>',
                "tool_calls": [{"name": tool["name"], "arguments": {}}],
            },
        ],
        "tools": tool_schemas,
    }


def _generate_teacher_sample(
    sample_id: str,
    tool_schemas: list[dict[str, Any]],
    teacher_model: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one sample via the teacher model API, with placeholder fallback."""
    user_prompt = (
        f"Generate a tool-call training sample using these tool schemas:\n"
        f"{json.dumps(tool_schemas, indent=2)}\n\n"
        f"Pick one tool and write a realistic user request that requires it."
    )
    try:
        raw = call_teacher_model(teacher_model, _TOOL_CALL_SYSTEM_PROMPT, user_prompt)
        data = json.loads(raw)
        messages = data.get("messages", [])
        tools = data.get("tools", tool_schemas)
        if not messages:
            raise ValueError("Empty messages from teacher model")
        return {
            "id": sample_id,
            "source": "custom_synthetic",
            "messages": messages,
            "tools": tools,
        }
    except Exception as exc:
        logger.warning("teacher_sample_fallback", sample_id=sample_id, error=str(exc))
        return _generate_placeholder_sample(sample_id, tool_schemas, rng)


def _generate_synthetic_samples(
    num_samples: int,
    teacher_model: str | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate synthetic tool-call training samples.

    Uses the teacher model API when ``teacher_model`` is set, otherwise
    generates structurally valid placeholder samples.
    """
    tool_schemas = _SYNTHETIC_TOOL_SCHEMAS
    samples: list[dict[str, Any]] = []

    for i in range(num_samples):
        sample_id = f"synthetic_{i:05d}"
        if teacher_model:
            sample = _generate_teacher_sample(sample_id, tool_schemas, teacher_model, rng)
        else:
            sample = _generate_placeholder_sample(sample_id, tool_schemas, rng)
        samples.append(sample)

    return samples


def _generate_negative_examples(
    positive_samples: list[dict[str, Any]],
    negative_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate negative examples from positive samples."""
    num_negative = int(len(positive_samples) * negative_ratio / (1 - negative_ratio))
    negatives: list[dict[str, Any]] = []

    for i in range(num_negative):
        # Select negative category
        category = rng.choices(
            list(NEGATIVE_CATEGORIES.keys()),
            weights=list(NEGATIVE_CATEGORIES.values()),
            k=1,
        )[0]

        base = rng.choice(positive_samples) if positive_samples else {}
        negative = {
            "id": f"negative_{i:05d}",
            "source": "synthetic_negative",
            "negative_category": category,
            "messages": [
                {"role": "system", "content": "You have access to tools."},
                {"role": "user", "content": f"Negative example ({category})"},
                {
                    "role": "assistant",
                    "content": f"<tool_call>{json.dumps({'name': 'wrong_tool', 'arguments': {}})}</tool_call>",
                    "is_negative": True,
                },
            ],
            "tools": base.get("tools", []),
        }
        negatives.append(negative)

    return negatives


def _split_dataset(
    samples: list[dict[str, Any]],
    splits: dict[str, float],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Split samples into train/val/test."""
    rng.shuffle(samples)
    n = len(samples)

    train_end = int(n * splits.get("train", 0.85))
    val_end = train_end + int(n * splits.get("val", 0.10))

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def generate_tool_call_dataset(
    external_sources: list[str] | None = None,
    custom_synthetic_size: int = 15000,
    teacher_model: str | None = None,
    negative_ratio: float = 0.15,
    output_dir: Path = Path("data/output/exp_b"),
    seed: int = 42,
) -> DatasetSplits:
    """Generate unified tool-call fine-tuning dataset.

    Args:
        external_sources: HuggingFace dataset IDs to merge.
        custom_synthetic_size: Number of synthetic samples to generate.
        teacher_model: Teacher model name for live API generation.
            Supported prefixes: ``gemini-*``, ``gpt-*``, ``claude-*``.
            If ``None``, uses the local placeholder generator.
        negative_ratio: Fraction of negative examples (0.15 = 15%).
        output_dir: Output directory for JSONL files.
        seed: Random seed.

    Returns:
        DatasetSplits with train/val/test paths and statistics.
    """
    if external_sources is None:
        external_sources = ["Salesforce/xlam-function-calling-60k", "ToolBench"]

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "generating_tool_call_dataset",
        external_sources=external_sources,
        custom_synthetic_size=custom_synthetic_size,
        teacher_model=teacher_model or "placeholder",
        negative_ratio=negative_ratio,
    )

    # Load external datasets
    all_samples: list[dict[str, Any]] = []
    for source in external_sources:
        external = _load_external_dataset(source)
        all_samples.extend(external)
        logger.info("loaded_external", source=source, count=len(external))

    # Generate synthetic samples
    synthetic = _generate_synthetic_samples(custom_synthetic_size, teacher_model, rng)
    all_samples.extend(synthetic)
    logger.info("generated_synthetic", count=len(synthetic))

    # Generate negative examples
    negatives = _generate_negative_examples(all_samples, negative_ratio, rng)
    all_samples.extend(negatives)
    logger.info("generated_negatives", count=len(negatives))

    # Split
    splits = _split_dataset(all_samples, {"train": 0.85, "val": 0.10, "test": 0.05}, rng)

    # Write JSONL files
    paths: dict[str, Path] = {}
    sizes: dict[str, int] = {}
    for split_name, split_data in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for sample in split_data:
                f.write(json.dumps(sample) + "\n")
        paths[split_name] = path
        sizes[split_name] = len(split_data)

    logger.info("dataset_complete", **sizes)

    return DatasetSplits(
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        train_size=sizes["train"],
        val_size=sizes["val"],
        test_size=sizes["test"],
        token_stats={"total_samples": sum(sizes.values())},
    )
