"""Phase 1 orchestrator — run all pre-trained model evaluations.

Execution matrix:
  Task A: 6 Cat A models × L1–L5 (200 conv/level) = 6,000 evaluations
  Task B: 5 Cat B–C models × L2–L3 (400 conv)    = 2,000 evaluations
  Task C: 5 Cat B–C models × 500 pairs × {0-shot, 3-shot} = 5,000 evaluations

Pipeline: launch vLLM sequentially (one model at a time on H100)
→ run tasks → collect latency → aggregate rankings
→ compute composite scores → return winners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from llm_workflow_agents.benchmark.model_selector import (
    CompositeScore,
    compute_composite_scores,
    select_winners,
)
from llm_workflow_agents.benchmark.results_aggregator import (
    Phase1Results,
    aggregate_results,
    save_results,
)
from llm_workflow_agents.benchmark.task_runner import TaskResult, run_task

logger = structlog.get_logger(__name__)


class Phase1Orchestrator:
    """Orchestrate all Phase 1 evaluations.

    Loads the benchmark matrix config and runs each (model, task) pair
    sequentially, then computes composite scores and selects winners.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/benchmark/phase1_matrix.yaml"),
    ) -> None:
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load Phase 1 matrix config."""
        import yaml

        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def _load_model_configs(self, model_paths: list[str]) -> list[dict[str, Any]]:
        """Load model YAML configs from paths."""
        import yaml

        configs = []
        for path in model_paths:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("model", cfg)
            model["_config_path"] = path
            configs.append(model)
        return configs

    def run_all(
        self,
        vllm_endpoint: str = "http://localhost:8000/v1",
        output_path: Path | None = None,
    ) -> Phase1Results:
        """Run all Phase 1 evaluations.

        Args:
            vllm_endpoint: vLLM server URL.
            output_path: Optional path to save results JSON.

        Returns:
            Phase1Results with rankings and winners.
        """
        all_results: list[TaskResult] = []
        scores_by_category: dict[str, list[CompositeScore]] = {}

        # Task A: Cat A models
        task_a_cfg = self.config.get("task_a", {})
        task_a_models = task_a_cfg.get("models", [])
        if task_a_models:
            logger.info("phase1_task_a_start", num_models=len(task_a_models))
            task_a_results = []
            for model_path in task_a_models:
                result = self.run_single(model_path, "task_a", vllm_endpoint)
                task_a_results.append(result)
                all_results.append(result)
            scores_by_category["A"] = compute_composite_scores(
                task_a_results, category="A"
            )

        # Task B: Cat B-C models
        task_b_cfg = self.config.get("task_b", {})
        task_b_models = task_b_cfg.get("models", [])
        if task_b_models:
            logger.info("phase1_task_b_start", num_models=len(task_b_models))
            task_b_results = []
            for model_path in task_b_models:
                result = self.run_single(model_path, "task_b", vllm_endpoint)
                task_b_results.append(result)
                all_results.append(result)
            scores_by_category["B"] = compute_composite_scores(
                task_b_results, category="B"
            )

        # Task C: Cat B-C models
        task_c_cfg = self.config.get("task_c", {})
        task_c_models = task_c_cfg.get("models", [])
        if task_c_models:
            logger.info("phase1_task_c_start", num_models=len(task_c_models))
            task_c_results = []
            for model_path in task_c_models:
                result = self.run_single(model_path, "task_c", vllm_endpoint)
                task_c_results.append(result)
                all_results.append(result)
            scores_by_category["C"] = compute_composite_scores(
                task_c_results, category="C"
            )

        winners = select_winners(scores_by_category)
        results = aggregate_results(all_results, scores_by_category, winners)

        if output_path:
            save_results(results, output_path)

        logger.info("phase1_complete", winners=winners)
        return results

    def run_single(
        self,
        model_config_path: str,
        task: str,
        vllm_endpoint: str = "http://localhost:8000/v1",
    ) -> TaskResult:
        """Run evaluation for a single model on a single task.

        Args:
            model_config_path: Path to model YAML config.
            task: Task identifier ("task_a", "task_b", "task_c").
            vllm_endpoint: vLLM server URL.

        Returns:
            TaskResult with quality and latency metrics.
        """
        import yaml

        with open(model_config_path) as f:
            cfg = yaml.safe_load(f) or {}
        model_name = cfg.get("model", cfg).get("name", model_config_path)

        logger.info("phase1_run_single", model=model_name, task=task)
        return run_task(
            model_name=model_name,
            task=task,
            vllm_endpoint=vllm_endpoint,
        )
