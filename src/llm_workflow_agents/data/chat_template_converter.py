"""Convert unified JSONL to model-specific chat template formats.

Supports: Qwen (ChatML), Gemma, Mistral v3, Nemotron, GLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

ModelFamilyName = Literal["qwen", "gemma", "mistral", "nemotron", "glm"]


@dataclass
class ConversionStats:
    """Statistics from a format conversion."""

    input_samples: int = 0
    output_samples: int = 0
    skipped: int = 0
    tool_calls_converted: int = 0
    errors: list[str] = field(default_factory=list)


def _convert_to_qwen(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert to Qwen ChatML format with <think> support and Hermes tool calls."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "tool":
            converted.append({"role": "tool", "content": content})
        elif role == "assistant" and "<tool_call>" in content:
            # Wrap tool calls in Hermes format
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": role, "content": content})

    return converted


def _convert_to_gemma(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert to Gemma chat template format."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Gemma uses "model" instead of "assistant"
        if role == "assistant":
            converted.append({"role": "model", "content": content})
        elif role == "system":
            # Gemma prepends system to first user message
            converted.append({"role": "system", "content": content})
        elif role == "tool":
            converted.append({"role": "tool", "content": content})
        else:
            converted.append({"role": role, "content": content})

    return converted


def _convert_to_mistral(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert to Mistral Instruct v3 format with tool_calls JSON."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant" and "annotations" in msg:
            annotations = msg["annotations"]
            tool_calls = annotations.get("tool_calls", [])
            if tool_calls:
                mistral_tool_calls = [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for i, tc in enumerate(tool_calls)
                ]
                converted.append(
                    {"role": "assistant", "content": content, "tool_calls": mistral_tool_calls}
                )
            else:
                converted.append({"role": role, "content": content})
        elif role == "tool":
            converted.append({"role": "tool", "content": content})
        else:
            converted.append({"role": role, "content": content})

    return converted


def _convert_to_nemotron(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert to Nemotron chat format."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        converted.append({"role": msg["role"], "content": msg["content"]})
    return converted


def _convert_to_glm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert to GLM ChatML format."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "tool":
            converted.append({"role": "observation", "content": content})
        else:
            converted.append({"role": role, "content": content})

    return converted


# Converter registry
_CONVERTERS: dict[ModelFamilyName, Any] = {
    "qwen": _convert_to_qwen,
    "gemma": _convert_to_gemma,
    "mistral": _convert_to_mistral,
    "nemotron": _convert_to_nemotron,
    "glm": _convert_to_glm,
}


def convert_to_model_format(
    input_jsonl: Path,
    model_family: ModelFamilyName,
    output_path: Path,
) -> ConversionStats:
    """Convert unified JSONL to model-specific chat template format.

    Args:
        input_jsonl: Path to unified JSONL file.
        model_family: Target model family format.
        output_path: Output path for converted JSONL.

    Returns:
        ConversionStats with counts and any errors.
    """
    converter = _CONVERTERS[model_family]
    stats = ConversionStats()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "converting_chat_template",
        input=str(input_jsonl),
        model_family=model_family,
        output=str(output_path),
    )

    with open(input_jsonl) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            stats.input_samples += 1

            try:
                sample = json.loads(line)
                messages = sample.get("messages", [])

                converted_messages = converter(messages)

                # Count tool calls
                for msg in messages:
                    if msg.get("annotations", {}).get("tool_calls"):
                        stats.tool_calls_converted += len(msg["annotations"]["tool_calls"])

                output_sample = {**sample, "messages": converted_messages}
                fout.write(json.dumps(output_sample) + "\n")
                stats.output_samples += 1

            except (json.JSONDecodeError, KeyError) as e:
                stats.skipped += 1
                stats.errors.append(f"Line {line_num}: {e}")
                logger.warning("conversion_error", line=line_num, error=str(e))

    logger.info(
        "conversion_complete",
        input_samples=stats.input_samples,
        output_samples=stats.output_samples,
        skipped=stats.skipped,
    )

    return stats
