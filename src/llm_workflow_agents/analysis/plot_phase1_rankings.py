"""Phase 1 composite score ranking bar charts per category.

Generates bar charts showing composite scores for all candidates
in each category, highlighting the winner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def plot_phase1_rankings(
    rankings: dict[str, list[dict[str, Any]]],
    output_dir: Path = Path("analysis/figures"),
    log_to_wandb: bool = False,
) -> list[Path]:
    """Generate Phase 1 ranking bar charts per category.

    Args:
        rankings: Dict mapping category → list of CompositeScore dicts
            (sorted by weighted_composite descending).
        output_dir: Directory to save figures.
        log_to_wandb: Whether to log charts to W&B.

    Returns:
        List of saved figure paths.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for category, scores in rankings.items():
        if not scores:
            continue

        models = [s["model_name"] for s in scores]
        composites = [s["weighted_composite"] for s in scores]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(models))]
        bars = ax.barh(range(len(models)), composites, color=colors)

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split("/")[-1] for m in models])
        ax.set_xlabel("Composite Score")
        ax.set_title(f"Phase 1 Rankings — Category {category}")
        ax.invert_yaxis()

        # Annotate winner
        if composites:
            ax.annotate(
                "WINNER",
                xy=(composites[0], 0),
                xytext=(composites[0] + 0.02, 0),
                fontweight="bold",
                color="#27ae60",
                va="center",
            )

        fig.tight_layout()
        path = output_dir / f"phase1_rankings_cat_{category.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        logger.info("phase1_ranking_plot_saved", category=category, path=str(path))

        if log_to_wandb:
            try:
                import wandb

                wandb.log({f"phase1/rankings_cat_{category}": wandb.Image(str(path))})
            except ImportError:
                pass

    return saved_paths
