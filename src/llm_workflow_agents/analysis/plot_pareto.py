"""2D Pareto frontier projection plots.

Generates 3 scatter plots from Pareto-optimal (model, quantization) configs:
  1. Quality × Memory
  2. Quality × Latency
  3. Memory × Latency

Highlights Pareto-optimal points with labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_PROJECTION_PAIRS = [
    ("task_completion", "peak_vram_gb", "Quality vs Memory"),
    ("task_completion", "p95_latency_ms", "Quality vs Latency"),
    ("peak_vram_gb", "p95_latency_ms", "Memory vs Latency"),
]


def plot_pareto_projections(
    all_configs: list[dict[str, Any]],
    pareto_configs: list[dict[str, Any]],
    output_dir: Path = Path("analysis/figures"),
    log_to_wandb: bool = False,
) -> list[Path]:
    """Generate 3 × 2D Pareto frontier projection scatter plots.

    Args:
        all_configs: All evaluated (model, quantization) configs with keys:
            model, method, task_completion, peak_vram_gb, p95_latency_ms.
        pareto_configs: Subset of configs on the Pareto frontier.
        output_dir: Directory to save figures.
        log_to_wandb: Whether to log charts to W&B.

    Returns:
        List of 3 saved figure paths.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    pareto_labels = {
        (c.get("model", ""), c.get("method", "")) for c in pareto_configs
    }

    for x_key, y_key, title in _PROJECTION_PAIRS:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot all points
        non_pareto_x = []
        non_pareto_y = []
        pareto_x = []
        pareto_y = []
        pareto_names: list[str] = []

        for cfg in all_configs:
            xv = cfg.get(x_key, 0.0)
            yv = cfg.get(y_key, 0.0)
            key = (cfg.get("model", ""), cfg.get("method", ""))
            if key in pareto_labels:
                pareto_x.append(xv)
                pareto_y.append(yv)
                short = f"{cfg.get('model', '').split('/')[-1]}\n{cfg.get('method', '')}"
                pareto_names.append(short)
            else:
                non_pareto_x.append(xv)
                non_pareto_y.append(yv)

        ax.scatter(non_pareto_x, non_pareto_y, c="#bdc3c7", s=40, alpha=0.6, label="Non-Pareto")
        ax.scatter(pareto_x, pareto_y, c="#e74c3c", s=100, zorder=5, label="Pareto-optimal")

        # Label Pareto points
        for x, y, name in zip(pareto_x, pareto_y, pareto_names):
            ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=7,
                ha="left",
            )

        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(y_key.replace("_", " ").title())
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        filename = f"pareto_{x_key}_vs_{y_key}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        logger.info("pareto_projection_saved", projection=title, path=str(path))

        if log_to_wandb:
            try:
                import wandb

                wandb.log({f"pareto/{x_key}_vs_{y_key}": wandb.Image(str(path))})
            except ImportError:
                pass

    return saved_paths
