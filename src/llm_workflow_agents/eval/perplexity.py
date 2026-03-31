"""Perplexity evaluation on WikiText-2 and C4 datasets.

Measures language modeling quality degradation from KV cache
quantization by computing per-token perplexity on standard benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation on one dataset."""

    dataset: str
    perplexity: float
    avg_neg_log_likelihood: float
    num_tokens: int
    num_sequences: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "perplexity": self.perplexity,
            "avg_neg_log_likelihood": self.avg_neg_log_likelihood,
            "num_tokens": self.num_tokens,
            "num_sequences": self.num_sequences,
        }


def _load_dataset_texts(dataset_name: str, max_samples: int | None = None) -> list[str]:
    """Load evaluation texts from a dataset.

    Defers HuggingFace datasets import.
    """
    from datasets import load_dataset

    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t.strip()]
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for sample in ds:
            texts.append(sample["text"])
            if max_samples and len(texts) >= max_samples:
                break
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples:
        texts = texts[:max_samples]

    return texts


def compute_perplexity_from_losses(losses: list[float]) -> float:
    """Compute perplexity from a list of per-token negative log-likelihoods."""
    if not losses:
        return float("inf")
    avg_nll = sum(losses) / len(losses)
    return math.exp(avg_nll)


def evaluate_perplexity_vllm(
    model_name: str,
    dataset_name: str,
    kv_cache_dtype: str = "auto",
    max_samples: int = 500,
    max_model_len: int = 2048,
    base_url: str = "http://localhost:8000/v1",
) -> PerplexityResult:
    """Evaluate perplexity using a running vLLM server.

    Sends sequences to the vLLM OpenAI-compatible API and computes
    perplexity from log-probabilities in the response.

    Args:
        model_name: Model identifier for the API.
        dataset_name: One of "wikitext2", "c4".
        kv_cache_dtype: KV cache dtype (for logging only, set at server launch).
        max_samples: Maximum number of text samples.
        max_model_len: Maximum sequence length for truncation.
        base_url: vLLM server URL.

    Returns:
        PerplexityResult with computed perplexity.
    """
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="unused")

    logger.info(
        "evaluating_perplexity",
        model=model_name,
        dataset=dataset_name,
        kv_cache_dtype=kv_cache_dtype,
    )

    texts = _load_dataset_texts(dataset_name, max_samples)

    all_losses: list[float] = []
    total_tokens = 0
    num_processed = 0

    for text in texts:
        # Truncate to max_model_len tokens approximately (4 chars per token)
        truncated = text[: max_model_len * 4]
        if not truncated.strip():
            continue
        num_processed += 1

        response = client.completions.create(
            model=model_name,
            prompt=truncated,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )

        if response.choices and response.choices[0].logprobs:
            token_logprobs = response.choices[0].logprobs.token_logprobs
            # Filter None values (first token has no logprob)
            valid = [lp for lp in token_logprobs if lp is not None]
            all_losses.extend([-lp for lp in valid])
            total_tokens += len(valid)

    ppl = compute_perplexity_from_losses(all_losses)
    avg_nll = sum(all_losses) / len(all_losses) if all_losses else 0.0

    result = PerplexityResult(
        dataset=dataset_name,
        perplexity=ppl,
        avg_neg_log_likelihood=avg_nll,
        num_tokens=total_tokens,
        num_sequences=num_processed,
    )

    logger.info("perplexity_complete", **result.to_dict())
    return result


def evaluate_perplexity(
    model_path: str,
    datasets: list[Literal["wikitext2", "c4"]] | None = None,
    kv_cache_dtype: str = "auto",
    max_samples: int = 500,
    base_url: str = "http://localhost:8000/v1",
    max_model_len: int = 2048,
) -> dict[str, float]:
    """Evaluate perplexity on multiple datasets.

    Convenience wrapper that returns {dataset_name: perplexity_value}.

    Args:
        model_path: Model path or name.
        datasets: List of datasets to evaluate on.
        kv_cache_dtype: KV cache dtype for logging.
        max_samples: Max samples per dataset.
        base_url: vLLM server URL.
        max_model_len: Maximum sequence length for truncation.

    Returns:
        Dict mapping dataset name to perplexity value.
    """
    if datasets is None:
        datasets = ["wikitext2", "c4"]

    results: dict[str, float] = {}
    for ds_name in datasets:
        try:
            result = evaluate_perplexity_vllm(
                model_name=model_path,
                dataset_name=ds_name,
                kv_cache_dtype=kv_cache_dtype,
                max_samples=max_samples,
                base_url=base_url,
                max_model_len=max_model_len,
            )
            results[ds_name] = result.perplexity
        except Exception as exc:
            logger.error("perplexity_eval_failed", dataset=ds_name, error=str(exc))
            results[ds_name] = float("inf")

    return results
