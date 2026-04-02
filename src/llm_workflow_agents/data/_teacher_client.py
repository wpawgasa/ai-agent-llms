"""Shared teacher model API client for data generation.

Routes calls to the appropriate provider based on model name prefix:
  ``gemini-*``  → Google GenAI  (env: GEMINI_API_KEY)
  ``gpt-*``     → OpenAI        (env: OPENAI_API_KEY)
  ``claude-*``  → Anthropic     (env: ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def call_teacher_model(teacher_model: str, system_prompt: str, user_prompt: str) -> str:
    """Call a teacher model and return the raw text response.

    Args:
        teacher_model: Model name, e.g. ``"gemini-2.0-flash"``, ``"gpt-4o"``,
            ``"claude-sonnet-4-6"``.
        system_prompt: System / instruction prompt.
        user_prompt: User turn content.

    Returns:
        Raw text response from the model.

    Raises:
        ValueError: If the model prefix is not recognised.
    """
    if teacher_model.startswith("gemini"):
        return _call_gemini(teacher_model, system_prompt, user_prompt)
    if teacher_model.startswith("gpt"):
        return _call_openai(teacher_model, system_prompt, user_prompt)
    if teacher_model.startswith("claude"):
        return _call_anthropic(teacher_model, system_prompt, user_prompt)
    raise ValueError(
        f"Unsupported teacher model {teacher_model!r}. "
        "Expected prefix: gemini-*, gpt-*, or claude-*."
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_gemini(model: str, system_prompt: str, user_prompt: str) -> str:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
    )
    return response.text


def _call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # type: ignore[import-untyped]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_anthropic(model: str, system_prompt: str, user_prompt: str) -> str:
    import anthropic  # type: ignore[import-untyped]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return raw
