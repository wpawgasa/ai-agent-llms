"""Quantization quality/performance heatmaps.

Rows = models (pre-trained + fine-tuned), columns = methods.
Metrics: PPL delta, tool-call F1 drop, VRAM savings, concurrency multiplier.
Uses seaborn diverging colormap (green = better, red = worse).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def plot_quant_matrix(
    matrix_data: dict[str, Any],
    metric: str = "wikitext2_ppl",
    output_dir: Path = Path("analysis/figures"),
    log_to_wandb: bool = False,
) -> Path:
    """Generate quantization method heatmap for a single metric.

    Args:
        matrix_data: QuantBenchmarkMatrix.to_dict() output with structure:
            {"models": [...], "methods": [...], "results": {"model::method": {...}}}
        metric: Metric to visualize (from quality_mean).
        output_dir: Directory to save figure.
        log_to_wandb: Whether to log chart to W&B.

    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)

    models = matrix_data.get("models", [])
    methods = matrix_data.get("methods", [])
    results = matrix_data.get("results", {})

    # Build heatmap array
    data = np.zeros((len(models), len(methods)))
    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            key = f"{model}::{method}"
            cell = results.get(key, {})
            quality_mean = cell.get("quality_mean", {})
            data[i, j] = quality_mean.get(metric, 0.0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.8)))

    # Use diverging colormap: green = better, red = worse for PPL (lower = better)
    is_lower_better = "ppl" in metric.lower()
    cmap = "RdYlGn_r" if is_lower_better else "RdYlGn"

    short_models = [m.split("/")[-1] for m in models]
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=methods,
        yticklabels=short_models,
        cmap=cmap,
        ax=ax,
    )

    ax.set_title(f"Quantization Matrix — {metric}")
    ax.set_xlabel("Quantization Method")
    ax.set_ylabel("Model")
    fig.tight_layout()

    path = output_dir / f"quant_matrix_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info("quant_matrix_plot_saved", metric=metric, path=str(path))

    if log_to_wandb:
        try:
            import wandb

            wandb.log({f"quant/{metric}_matrix": wandb.Image(str(path))})
        except ImportError:
            pass

    return path


def plot_all_quant_matrices(
    matrix_data: dict[str, Any],
    output_dir: Path = Path("analysis/figures"),
    log_to_wandb: bool = False,
) -> list[Path]:
    """Generate heatmaps for all standard quantization metrics.

    Args:
        matrix_data: QuantBenchmarkMatrix.to_dict() output.
        output_dir: Directory to save figures.
        log_to_wandb: Whether to log charts to W&B.

    Returns:
        List of saved figure paths.
    """
    metrics = [
        "wikitext2_ppl",
        "c4_ppl",
        "longbench_score",
        "needle_accuracy",
        "tool_call_f1",
    ]
    paths = []
    for metric in metrics:
        path = plot_quant_matrix(matrix_data, metric, output_dir, log_to_wandb)
        paths.append(path)
    return paths
