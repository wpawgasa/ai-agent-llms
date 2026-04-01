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
    # Plot modules: import directly — deferred matplotlib/seaborn deps
    # from llm_workflow_agents.analysis import plot_phase1_rankings
    # from llm_workflow_agents.analysis import plot_sft_vs_rl
    # from llm_workflow_agents.analysis import plot_quant_matrix
    # from llm_workflow_agents.analysis import plot_pareto
    # from llm_workflow_agents.analysis import plot_results
]
