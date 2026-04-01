"""SFT-only vs SFT+RL improvement curves (RQ1 and RQ2).

For each of the 3 winners: pre-trained → SFT → SFT+RL progression
across key metrics with error bars from multiple eval runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def plot_sft_vs_rl(
    progression_data: list[dict[str, Any]],
    output_dir: Path = Path("analysis/figures"),
    log_to_wandb: bool = False,
) -> list[Path]:
    """Generate SFT vs RL progression charts.

    Args:
        progression_data: List of dicts per model with keys:
            - model_name: str
            - category: str ("A", "B", "C")
            - stages: list of {"stage": str, "metrics": dict[str, float],
                                "metrics_std": dict[str, float] (optional)}
              where stages are "pretrained", "sft", "sft_rl"
        output_dir: Directory to save figures.
        log_to_wandb: Whether to log charts to W&B.

    Returns:
        List of saved figure paths.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    metrics_to_plot = [
        "state_transition_accuracy",
        "tool_call_f1",
        "task_completion_rate",
    ]

    for model_data in progression_data:
        model_name = model_data["model_name"]
        category = model_data["category"]
        stages = model_data.get("stages", [])

        if not stages:
            continue

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        stage_names = [s["stage"] for s in stages]
        x = np.arange(len(stage_names))

        for ax, metric in zip(axes, metrics_to_plot):
            values = [s["metrics"].get(metric, 0.0) for s in stages]
            errors = [s.get("metrics_std", {}).get(metric, 0.0) for s in stages]

            bars = ax.bar(x, values, yerr=errors, capsize=5, color=["#95a5a6", "#3498db", "#2ecc71"])
            ax.set_xticks(x)
            ax.set_xticklabels(stage_names, rotation=15)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(0, 1.05)
            ax.set_title(metric.replace("_", " ").title())

        short_name = model_name.split("/")[-1]
        fig.suptitle(f"Cat {category} Winner: {short_name}", fontweight="bold")
        fig.tight_layout()

        path = output_dir / f"sft_vs_rl_cat_{category.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        logger.info("sft_vs_rl_plot_saved", model=model_name, path=str(path))

        if log_to_wandb:
            try:
                import wandb

                wandb.log({f"training/sft_vs_rl_cat_{category}": wandb.Image(str(path))})
            except ImportError:
                pass

    return saved_paths
