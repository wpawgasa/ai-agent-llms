"""Result analysis, Pareto frontier computation, and visualization.

Note: ``plot_results`` is intentionally NOT imported here because it carries
heavy optional dependencies (matplotlib, seaborn, pandas) that are deferred
to first use.  Import it directly when needed::

    from llm_workflow_agents.analysis import plot_results
    plot_results.plot_pareto_frontier(results, "out.png", pareto_names)
"""

from llm_workflow_agents.analysis.pareto import (
    ParetoPoint,
    find_pareto_frontier,
)

__all__ = [
    "ParetoPoint",
    "find_pareto_frontier",
    # plot_results: import directly — deferred matplotlib/seaborn/pandas deps
]
