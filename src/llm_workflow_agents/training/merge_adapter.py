"""LoRA adapter merge and export utility.

Loads a base model + LoRA adapter, merges weights via
model.merge_and_unload(), and saves the resulting model.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def merge_and_export(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    torch_dtype: str = "bfloat16",
    quantize_merged: str | None = None,
) -> Path:
    """Load base model + LoRA adapter, merge, and save.

    Args:
        base_model: HuggingFace model ID for the base model.
        adapter_path: Path to the LoRA adapter directory.
        output_path: Path to save the merged model.
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_repo_id: Hub repo ID of the form ``"org/repo-name"``. Required
            when ``push_to_hub=True``.
        torch_dtype: Torch dtype string ("bfloat16" or "float16").
        quantize_merged: Optional post-merge quantization ("fp8" or None).

    Returns:
        Path to the saved merged model.

    Raises:
        FileNotFoundError: If adapter_path does not exist.
        ValueError: If push_to_hub is True but hub_repo_id is not provided.
        RuntimeError: If merge fails.
    """
    if push_to_hub and not hub_repo_id:
        raise ValueError("hub_repo_id is required when push_to_hub=True (expected 'org/repo-name')")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    logger.info(
        "merging_adapter",
        base_model=base_model,
        adapter=str(adapter_path),
        output=str(output_path),
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    # Optionally quantize the merged model
    if quantize_merged == "fp8":
        logger.info("quantizing_merged_model", method="fp8")
        model = model.to(torch.float8_e4m3fn)

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("merge_complete", output=str(output_path))

    # Optionally push to hub
    if push_to_hub:
        logger.info("pushing_to_hub", repo_id=hub_repo_id)
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

    return output_path
