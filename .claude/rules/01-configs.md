# Configuration Schema Module

## Overview
All YAML configuration files live in `configs/`. Five subdirectories organize configs by purpose.

## Directory Structure
```
configs/
  models/
    cat_a/              # Category A model configs (12 × 15–35B)
    cat_bc/             # Category B–C model configs (5 × 2–5B)
  training/             # SFT + GRPO RL hyperparameter configs
  quantization/         # KV cache quantization method configs
  benchmark/            # Phase 1 benchmark matrix and selection weights
  serving/              # Serving/deployment configs
```

## Model Config Schema
Each model YAML has three sections:
- `model`: name, family, architecture, params_total, params_active, context_length, precision, vram_estimate_gb
- `serving`: engine (vllm), tool_call_parser, chat_template, gpu_memory_utilization, max_model_len
- `category`: "A" or "B"/"C"
- `benchmark_tasks`: which Phase 1 tasks to run

### Category A Models (`configs/models/cat_a/`)
| Config | Model | Params | Tool Parser | BF16 VRAM |
|--------|-------|--------|-------------|-----------|
| gemma3_27b.yaml | google/gemma-3-27b-it | 27B | `gemma` | ~54 GB |
| qwen3_32b.yaml | Qwen/Qwen3-32B | 32B | `hermes` | ~64 GB |
| qwen35_35b_a3b.yaml | Qwen/Qwen3.5-35B-A3B | 35B (3B active) | `qwen3_coder` | ~70 GB |
| mistral_small_24b.yaml | mistralai/Mistral-Small-3.1-24B | 24B | `mistral` | ~48 GB |
| nemotron_30b.yaml | nvidia/Nemotron-3-Nano-30B | 30B (3.6B active) | `nemotron` | ~60 GB |
| glm47_flash.yaml | zai-org/GLM-4.7-Flash | 30B (3.6B active) | `glm4` | ~60 GB |
| qwen36_35b_a3b.yaml | Qwen/Qwen3.6-35B-A3B | 35B (3B active) | `qwen3_coder` | ~70 GB |
| qwen36_35b_a3b_fp8.yaml | Qwen/Qwen3.6-35B-A3B-FP8 | 35B (3B active) | `qwen3_coder` | ~35 GB |

### Category B–C Models (`configs/models/cat_bc/`)
| Config | Model | Params | VRAM |
|--------|-------|--------|------|
| qwen25_3b.yaml | Qwen/Qwen2.5-3B-Instruct | 3B | ~6 GB |
| qwen35_4b.yaml | Qwen/Qwen3.5-4B | 4B (~3B active) | ~8 GB |
| glm47_flash_small.yaml | zai-org/GLM-4.7-Flash | 30B (3.6B active) | ~60 GB |
| gemma_2b.yaml | google/gemma-2b | 2.5B | ~5 GB |
| gemma3_4b.yaml | google/gemma-3-4b-it | 4B | ~8 GB |

## Training Config Schema (`configs/training/`)
Six files: `sft_cat_{a,b,c}.yaml` and `grpo_cat_{a,b,c}.yaml`.

### SFT Config Key Fields
- `stage: sft`, `framework: unsloth`
- `lora`: rank, alpha, dropout, target_modules: "auto", freeze_router: true
- `training`: precision (qlora_4bit or bf16), learning_rate, lr_scheduler, warmup_ratio, effective_batch_size, num_epochs, max_seq_length, gradient_checkpointing, packing
- `data`: source (data/output/task_{a,b,c}), format, splits (0.85/0.10/0.05)
- `logging`: wandb_project, save_steps, eval_steps, metric_for_best_model

### GRPO Config Key Fields
- `stage: grpo`, `framework: unsloth`
- `grpo`: algorithm: GRPO, normalization: DAPO, num_generations: 4, beta: 0.04, epsilon: 0.2, learning_rate: 5e-6, training_steps: 1000, generation_backend: vllm, fp8_rl: true
- `reward`: function name (reward_business_logic / reward_subagent / reward_graph_extraction)
- `monitoring`: eval_held_out_every: 50, reward_hacking_detector: true, kl_divergence_log: true

## Benchmark Config (`configs/benchmark/`)
- `phase1_matrix.yaml` — which models × which tasks to run
- `selection_weights.yaml` — composite score weights per category:
  - Cat A: quality 0.40, latency_p95 0.25, throughput 0.20, memory 0.15
  - Cat B: quality 0.35, latency_p95 0.30, throughput 0.20, memory 0.15
  - Cat C: quality 0.40, latency_p95 0.20, throughput 0.20, memory 0.20

## Quantization Config Schema (`configs/quantization/`)
| Config | Method | Key Feature |
|--------|--------|-------------|
| fp8.yaml | FP8 E4M3 | Native vLLM support |
| kivi.yaml | KIVI 2/4-bit | Asymmetric, no calibration |
| kvquant.yaml | KVQuant | NUQ codebooks, calibration required |
| awq_fp8.yaml | AWQ-INT4 + FP8 KV | Weight quantization + KV FP8 |
| turboquant.yaml | TurboQuant | Beta codebook + rotation + QJL residual, Triton kernels |
| rotorquant.yaml | RotorQuant | Cl(3,0) rotor rotation, Triton kernels |

## Serving Configs (`configs/serving/`)
- `single_model.yaml` — single model serving
- `multi_agent.yaml` — multi-model deployment (orchestrator + specialists)
- `benchmark_e2e.yaml` — full E2E benchmark sweep config

## Checklist
- [x] Create `configs/models/cat_a/` with 6 model YAMLs
- [x] Create `configs/models/cat_bc/` with 5 model YAMLs
- [x] Create `configs/training/` with 6 training YAMLs (sft + grpo × 3 categories)
- [x] Create `configs/benchmark/` with phase1_matrix.yaml and selection_weights.yaml
- [x] Create `configs/quantization/` with 6 method YAMLs (including awq_fp8.yaml)
- [x] Create `configs/serving/` with 3 serving YAMLs
- [x] Validate all configs against schema
