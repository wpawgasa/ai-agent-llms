# Testing Strategy

## Unit Tests

| Test File | Module | Key Assertions |
|-----------|--------|----------------|
| test_data_generation.py | data/ | Schema validity, behavior distribution matches spec, tool error rate ~20%, split ratios |
| test_chat_templates.py | data/ | Round-trip conversion fidelity across all 6 model formats |
| test_reward_functions.py | training/rewards/ | Known-answer reward scores, edge cases (invalid JSON → 0.0), component weights sum to 1.0 |
| test_eval_metrics.py | eval/ | Known-answer tests for F1, GED, pass^5 computation |
| test_composite_score.py | eval/composite_score.py | Normalization correctness, weight application, ranking stability |
| test_triton_kernels.py | quantization/ | Numerical correctness: encode→decode approx identity (within quantization error) |

## Integration Tests

| Test | Description |
|------|-------------|
| Phase 1 smoke | 2 models × 1 task × 10 samples → verify composite score computation and winner selection |
| SFT smoke | 50 steps on 100 samples → checkpoint saves → merge → inference |
| GRPO smoke | 10 steps with mock reward → verify reward logging + policy update |
| Reward hacking detector | Synthetic: reward ↑ + held-out ↓ → verify alert fires |
| Quant round-trip | BF16 → TurboQuant encode → decode → compare PPL delta within tolerance |
| E2E pipeline | `run_phase1.sh` on 1 model × 1 task × 10 samples → verify full output |

## Reproducibility
- All experiments use seed-deterministic configuration
- Phase 1 benchmarks: temperature=0.0
- Consistency metrics: temperature=0.7 with 5 trials (pass^5)
- Quantization benchmarks: 3–5 repetitions, 500+ prompts, report mean ± std

## Checklist
- [ ] Implement test_data_generation.py
- [ ] Implement test_chat_templates.py (6 formats)
- [ ] Implement test_reward_functions.py (all 3 reward functions + edge cases)
- [ ] Implement test_eval_metrics.py
- [ ] Implement test_composite_score.py
- [ ] Implement test_triton_kernels.py
- [ ] Set up integration test suite
- [ ] Verify seed determinism across runs
