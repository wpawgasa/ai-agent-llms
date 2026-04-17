# Analysis Module

## Overview
`analysis/` handles visualization and reporting across all four phases.

## Files
- `plot_phase1_rankings.py` — Phase 1 composite score ranking bar charts per category
- `plot_sft_vs_rl.py` — SFT-only vs SFT+RL improvement curves (answers RQ1 and RQ2)
- `plot_quant_matrix.py` — Quantization quality/performance heatmaps (quality degradation vs compression ratio)
- `plot_pareto.py` — 2D Pareto frontier projections (quality×memory, quality×latency, memory×latency)

## plot_phase1_rankings.py
- Bar charts per category (A, B, C) showing composite scores for all candidates
- Highlights the winner per category
- W&B integration for chart logging

## plot_sft_vs_rl.py
- For each of the 3 winners: pre-trained → SFT → SFT+RL progression
- Metrics: state_transition_accuracy, tool_call_f1, task_completion_rate
- Error bars from multiple eval runs

## plot_quant_matrix.py
- Heatmap: rows = models (pre-trained + fine-tuned), columns = methods
- Metrics: PPL delta, tool-call F1 drop, VRAM savings, concurrency multiplier
- Seaborn diverging colormap (PPL delta: green = better, red = worse)

## plot_pareto.py
- Input: `integration/pareto.py` output (list of Pareto-optimal configs)
- 3 × 2D scatter plots: quality×memory, quality×latency, memory×latency
- Highlight Pareto frontier configs with labels

## Checklist
- [x] Implement plot_phase1_rankings.py
- [x] Implement plot_sft_vs_rl.py with pre-trained/SFT/SFT+RL progression
- [x] Implement plot_quant_matrix.py with heatmap
- [x] Implement plot_pareto.py with 2D projections
- [x] Add W&B integration for all charts
