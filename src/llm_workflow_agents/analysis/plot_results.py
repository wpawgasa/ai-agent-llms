"""Visualization functions for experiment results.

All matplotlib/seaborn imports are deferred to avoid importing heavy
visualization dependencies in environments where they are unavailable.

Chart coverage:
  Exp A — accuracy vs. complexity (L1-L5)
  Exp B — specialist vs. baseline tool-call F1
  Exp C — graph extraction metrics (Node F1, Edge F1, GED)
  Exp D — quality degradation vs. compression ratio
  E2E   — Pareto frontier (task completion vs. P95 latency)
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def plot_accuracy_vs_complexity(
    results: list[dict[str, Any]],
    output_path: str,
    log_to_wandb: bool = False,
) -> None:
    """Plot Exp A: weighted workflow score vs. complexity level (L1-L5) per model.

    Args:
        results: List of dicts with keys: ``model`` (str), ``complexity`` (int 1-5),
            ``weighted_score`` (float).
        output_path: File path to save the figure (e.g. "results/exp_a/accuracy.png").
        log_to_wandb: If True, log the figure to W&B (requires wandb to be initialized).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="complexity", y="weighted_score", hue="model", marker="o")
    plt.xlabel("Complexity Level")
    plt.ylabel("Weighted Workflow Score")
    plt.title("Exp A: Workflow Score vs. Complexity Level")
    plt.xticks([1, 2, 3, 4, 5], ["L1", "L2", "L3", "L4", "L5"])
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_wandb:
        import wandb
        wandb.log({"exp_a/accuracy_vs_complexity": wandb.Image(output_path)})

    logger.info("plot_accuracy_vs_complexity_saved", output_path=output_path)


def plot_specialist_vs_baseline(
    results: list[dict[str, Any]],
    output_path: str,
    log_to_wandb: bool = False,
) -> None:
    """Plot Exp B: specialist vs. baseline tool-call F1 per model.

    Args:
        results: List of dicts with keys: ``model`` (str), ``condition``
            ("specialist" or "baseline"), ``tool_call_f1`` (float).
        output_path: File path to save the figure.
        log_to_wandb: If True, log the figure to W&B.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="tool_call_f1", hue="condition")
    plt.xlabel("Model")
    plt.ylabel("Tool-Call F1")
    plt.title("Exp B: Specialist vs. Baseline Tool-Call F1")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_wandb:
        import wandb
        wandb.log({"exp_b/specialist_vs_baseline": wandb.Image(output_path)})

    logger.info("plot_specialist_vs_baseline_saved", output_path=output_path)


def plot_graph_metrics(
    results: list[dict[str, Any]],
    output_path: str,
    log_to_wandb: bool = False,
) -> None:
    """Plot Exp C: graph extraction metrics (Node F1, Edge F1, GED) per model.

    Args:
        results: List of dicts with keys: ``model`` (str), ``metric``
            ("node_f1", "edge_f1", or "graph_edit_distance"), ``value`` (float).
        output_path: File path to save the figure.
        log_to_wandb: If True, log the figure to W&B.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="value", hue="metric")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Exp C: Graph Extraction Metrics")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_wandb:
        import wandb
        wandb.log({"exp_c/graph_metrics": wandb.Image(output_path)})

    logger.info("plot_graph_metrics_saved", output_path=output_path)


def plot_quality_degradation(
    results: list[dict[str, Any]],
    output_path: str,
    log_to_wandb: bool = False,
) -> None:
    """Plot Exp D: perplexity delta vs. quantization method per model.

    Args:
        results: List of dicts with keys: ``model`` (str), ``quant_method`` (str),
            ``compression_ratio`` (float), ``perplexity_delta`` (float).
        output_path: File path to save the figure.
        log_to_wandb: If True, log the figure to W&B.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="compression_ratio",
        y="perplexity_delta",
        hue="model",
        style="quant_method",
        s=100,
    )
    plt.xlabel("Compression Ratio")
    plt.ylabel("Perplexity Delta (vs. BF16 baseline)")
    plt.title("Exp D: Quality Degradation vs. Compression Ratio")
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_wandb:
        import wandb
        wandb.log({"exp_d/quality_degradation": wandb.Image(output_path)})

    logger.info("plot_quality_degradation_saved", output_path=output_path)


def plot_pareto_frontier(
    results: list[dict[str, Any]],
    output_path: str,
    pareto_names: list[str] | None = None,
    log_to_wandb: bool = False,
) -> None:
    """Plot E2E: Pareto frontier of task completion rate vs. P95 latency.

    Args:
        results: List of dicts with keys: ``config_name`` (str),
            ``task_completion_rate`` (float), ``p95_latency_ms`` (float),
            ``peak_vram_gb`` (float).
        output_path: File path to save the figure.
        pareto_names: Set of config names that are Pareto-optimal. If provided,
            Pareto-optimal points are highlighted.
        log_to_wandb: If True, log the figure to W&B.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd

    df = pd.DataFrame(results)
    pareto_set = set(pareto_names or [])
    df["pareto"] = df["config_name"].apply(lambda n: "Pareto" if n in pareto_set else "Other")

    palette = {"Pareto": "#e74c3c", "Other": "#95a5a6"}

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="p95_latency_ms",
        y="task_completion_rate",
        hue="pareto",
        size="peak_vram_gb",
        sizes=(40, 300),
        palette=palette,
        alpha=0.8,
    )

    # Annotate Pareto-optimal points
    pareto_df = df[df["pareto"] == "Pareto"]
    for _, row in pareto_df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["p95_latency_ms"], row["task_completion_rate"]),
            fontsize=7,
            ha="left",
            va="bottom",
        )

    plt.xlabel("P95 Latency (ms)")
    plt.ylabel("Task Completion Rate")
    plt.title("E2E Pareto Frontier: Quality vs. Latency")
    plt.legend(title="Config Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_wandb:
        import wandb
        wandb.log({"e2e/pareto_frontier": wandb.Image(output_path)})

    logger.info("plot_pareto_frontier_saved", output_path=output_path)
