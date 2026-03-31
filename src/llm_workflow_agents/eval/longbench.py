"""LongBench 15-task evaluation for KV cache quantization quality.

Evaluates long-context understanding across 15 diverse tasks spanning
single-doc QA, multi-doc QA, summarization, few-shot learning, and
code completion. Measures quality degradation from quantization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# LongBench task categories and their names
LONGBENCH_TASKS: dict[str, list[str]] = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "code": ["lcc", "repobench-p"],
    "synthetic": ["passage_retrieval_en"],
}

ALL_TASK_NAMES: list[str] = [
    task for tasks in LONGBENCH_TASKS.values() for task in tasks
]


@dataclass
class LongBenchTaskResult:
    """Result for a single LongBench task."""

    task_name: str
    category: str
    score: float  # 0.0-100.0
    num_samples: int
    avg_input_length: int = 0
    avg_output_length: int = 0


@dataclass
class LongBenchResult:
    """Aggregated LongBench evaluation result."""

    task_results: list[LongBenchTaskResult] = field(default_factory=list)
    overall_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "tasks": {r.task_name: r.score for r in self.task_results},
        }


def _get_task_category(task_name: str) -> str:
    """Look up category for a task name."""
    for category, tasks in LONGBENCH_TASKS.items():
        if task_name in tasks:
            return category
    return "unknown"


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L (longest common subsequence) F-measure."""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()

    if not pred_words or not ref_words:
        return 0.0

    # LCS length via DP
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def score_task(
    task_name: str,
    predictions: list[str],
    references: list[str],
) -> float:
    """Score predictions against references for a given task.

    Uses F1 for QA tasks, ROUGE-L for summarization, exact match
    for classification, and a heuristic for code tasks.

    Args:
        task_name: LongBench task name.
        predictions: Model predictions.
        references: Ground-truth references.

    Returns:
        Score from 0.0 to 100.0.
    """
    category = _get_task_category(task_name)

    if not predictions or not references:
        return 0.0

    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        if category in ("single_doc_qa", "multi_doc_qa", "few_shot", "synthetic"):
            scores.append(_compute_f1(pred, ref))
        elif category == "summarization":
            scores.append(_compute_rouge_l(pred, ref))
        elif category == "code":
            # Edit similarity for code
            scores.append(_compute_f1(pred, ref))
        else:
            scores.append(1.0 if pred.strip() == ref.strip() else 0.0)

    return (sum(scores) / len(scores)) * 100.0 if scores else 0.0


def evaluate_longbench(
    model_path: str,
    tasks: list[str] | None = None,
    kv_cache_dtype: str = "auto",
    max_samples_per_task: int = 100,
    base_url: str = "http://localhost:8000/v1",
) -> LongBenchResult:
    """Evaluate a model on LongBench tasks via a running vLLM server.

    Args:
        model_path: Model name for the API.
        tasks: Task names to evaluate (default: all 15).
        kv_cache_dtype: KV cache dtype (for logging).
        max_samples_per_task: Max samples per task.
        base_url: vLLM server URL.

    Returns:
        LongBenchResult with per-task and aggregated scores.
    """
    if tasks is None:
        tasks = ALL_TASK_NAMES

    logger.info(
        "evaluating_longbench",
        model=model_path,
        num_tasks=len(tasks),
        kv_cache_dtype=kv_cache_dtype,
    )

    result = LongBenchResult()

    for task_name in tasks:
        try:
            task_result = _evaluate_single_task(
                model_path=model_path,
                task_name=task_name,
                max_samples=max_samples_per_task,
                base_url=base_url,
            )
            result.task_results.append(task_result)
        except Exception as exc:
            logger.error("longbench_task_failed", task=task_name, error=str(exc))
            result.task_results.append(
                LongBenchTaskResult(
                    task_name=task_name,
                    category=_get_task_category(task_name),
                    score=0.0,
                    num_samples=0,
                )
            )

    # Compute aggregated scores
    if result.task_results:
        result.overall_score = (
            sum(r.score for r in result.task_results) / len(result.task_results)
        )

        # Per-category averages
        cat_scores: dict[str, list[float]] = {}
        for r in result.task_results:
            cat_scores.setdefault(r.category, []).append(r.score)
        result.category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in cat_scores.items()
        }

    logger.info("longbench_complete", **result.to_dict())
    return result


def _evaluate_single_task(
    model_path: str,
    task_name: str,
    max_samples: int,
    base_url: str,
) -> LongBenchTaskResult:
    """Evaluate a single LongBench task.

    Loads the task dataset, generates predictions via the API, and scores.
    """
    import openai
    from datasets import load_dataset

    category = _get_task_category(task_name)

    ds = load_dataset("THUDM/LongBench", task_name, split="test")
    samples = list(ds)[:max_samples]

    client = openai.OpenAI(base_url=base_url, api_key="unused")

    predictions: list[str] = []
    references: list[str] = []
    total_input_len = 0

    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("input", "")
        reference = sample.get("answers", [""])[0] if isinstance(sample.get("answers"), list) else sample.get("answers", "")

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        total_input_len += len(prompt.split())

        response = client.completions.create(
            model=model_path,
            prompt=prompt,
            max_tokens=256,
            temperature=0.0,
        )

        pred = response.choices[0].text.strip() if response.choices else ""
        predictions.append(pred)
        references.append(str(reference))

    task_score = score_task(task_name, predictions, references)

    return LongBenchTaskResult(
        task_name=task_name,
        category=category,
        score=task_score,
        num_samples=len(samples),
        avg_input_length=total_input_len // max(len(samples), 1),
    )
