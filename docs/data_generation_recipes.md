# Data Generation Recipes

This document describes the four data generation scripts for Task A (multi-turn workflow conversations). Each script calls `generate_workflow_dataset` from `src/llm_workflow_agents/data/generate_workflows.py`.

## Shared Concepts

### Complexity Levels

| Level | States | Branching | Tools | Chain depth | Domain |
|-------|--------|-----------|-------|-------------|--------|
| L1 | 3–4 | 1–2 | 1 | 0 | faq_lookup |
| L2 | 5–7 | 2–3 | 2 | 1 | order_status_cancel |
| L3 | 8–12 | 3–5 | 4 | 2 | booking_payment |
| L4 | 13–20 | 5–8 | 6 | 3 | it_troubleshoot |
| L5 | 21–30 | 8+ | 7 | 4 | multi_dept_workflow |

### Behavior Presets

| Preset | cooperative | adversarial_probing | digressing | invalid_tool_inputs |
|--------|-------------|---------------------|------------|---------------------|
| `default` | 60% | 15% | 10% | 15% |
| `adversarial` | 45% | 25% | 15% | 15% |
| `balanced` | 25% | 25% | 25% | 25% |

### Language Options

| Value | Effect |
|-------|--------|
| `en` | All turns in English |
| `th` | All turns in Thai (annotations stay in ASCII) |
| `code_switch` | Thai-English code-switching within each conversation — Thai sentence structure with embedded English terms, mirroring real call-centre interactions |
| *(omitted)* | Each sample randomly assigned `en` or `th` (50/50) |

### Seed Allocation

Seeds are assigned per-split to guarantee no sample overlap across datasets:

| Dataset | Seed |
|---------|------|
| SFT | 42 |
| Benchmark | 100 |
| GRPO | 200 |
| Validation | 300 |
| Test | 400 |

### Teacher Models

| Model | API key env var | Used in |
|-------|----------------|---------|
| `gpt-5.4-nano-2026-03-17` | `OPENAI_API_KEY` | SFT (en, code_switch), GRPO (mixed, code_switch), eval |
| `gemini-3-flash-preview` | `GEMINI_API_KEY` | SFT (th), GRPO (mixed) |
| *(none — placeholder)* | — | Benchmark |

---

## Recipes

### Benchmark (`generate_benchmark_data.sh`)

Generates 1 000 structurally valid placeholder conversations for Phase 1 model ranking. No teacher model or API key required.

```
Levels:       L1–L5
Samples/level: 200
Total:        1 000
Language:     mixed (50/50 en/th per sample)
Teacher:      placeholder (local generator)
Behavior:     default
Seed:         100
Output:       data/output/benchmark/task_a/
```

```bash
./scripts/generate_benchmark_data.sh

# Custom output directory
./scripts/generate_benchmark_data.sh --output-dir /mnt/data/output

# Dry run (print commands only)
./scripts/generate_benchmark_data.sh --dry-run
```

---

### SFT (`generate_sft_data.sh`)

Generates ~12 504 curriculum-weighted conversations for supervised fine-tuning. Each complexity level is split evenly across three teacher/language legs to expose the model to diverse linguistic registers from the start.

```
Levels:       L1–L5
Behavior:     adversarial
Seed:         42
Output:       data/output/sft/task_a/
```

**Per-level totals and leg sizes:**

| Level | Total | en (gpt-5.4-nano) | th (gemini-3-flash) | code_switch (gpt-5.4-nano) |
|-------|-------|-------------------|---------------------|---------------------------|
| L1 | 3 000 | 1 000 | 1 000 | 1 000 |
| L2 | 3 000 | 1 000 | 1 000 | 1 000 |
| L3 | 2 502 | 834 | 834 | 834 |
| L4 | 2 001 | 667 | 667 | 667 |
| L5 | 2 001 | 667 | 667 | 667 |
| **Total** | **12 504** | | | |

**Rationale for the adversarial preset:** SFT needs to show the model how to handle user attempts to bypass workflow steps, provide invalid inputs, and go off-topic. Starting from an adversarial distribution produces a more robust SFT baseline before GRPO refines it further.

```bash
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh

# Dry run
./scripts/generate_sft_data.sh --dry-run
```

---

### GRPO (`generate_grpo_data.sh`)

Generates 2 250 prompts for GRPO reward-based fine-tuning. Only L3–L5 are included because the harder levels are where reward shaping delivers the largest gains over SFT alone.

```
Levels:       L3, L4, L5
Samples/level: 750  (3 × 250)
Total:        2 250
Behavior:     balanced
Seed:         200
Output:       data/output/grpo/task_a/
```

**Per-level legs:**

| Leg | Teacher | Language | Samples |
|-----|---------|----------|---------|
| 1 | gpt-5.4-nano-2026-03-17 | mixed (en/th) | 250 |
| 2 | gemini-3-flash-preview | mixed (en/th) | 250 |
| 3 | gpt-5.4-nano-2026-03-17 | code_switch | 250 |

**Rationale for the balanced preset:** GRPO optimises reward across all user behaviour types. A 25/25/25/25 split ensures the policy is not over-fitted to cooperative users and is challenged on adversarial, digressing, and invalid-input turns equally. Code-switching in leg 3 extends this challenge to language-register variation.

```bash
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_grpo_data.sh

# Dry run
./scripts/generate_grpo_data.sh --dry-run
```

---

### Eval (`generate_eval_data.sh`)

Generates held-out validation and test splits. Seeds 300 and 400 guarantee no overlap with any training data.

```
Levels:       L1–L5
Samples/level: 100
Total/split:  500
Language:     mixed (50/50 en/th per sample)
Teacher:      gpt-5.4-nano-2026-03-17
Behavior:     default
Output:       data/output/val/task_a/   (seed=300)
              data/output/test/task_a/  (seed=400)
```

> **Important:** the test split must not be used during training or for hyperparameter selection. Reserve it for final evaluation only.

```bash
# Both splits
OPENAI_API_KEY=sk-... ./scripts/generate_eval_data.sh

# Validation only
OPENAI_API_KEY=sk-... ./scripts/generate_eval_data.sh --split val

# Test only
OPENAI_API_KEY=sk-... ./scripts/generate_eval_data.sh --split test

# Dry run
./scripts/generate_eval_data.sh --dry-run
```

---

## Full Generation Order

Run the scripts in this order. Each depends on having a clean `data/output/` directory and the relevant API keys set.

```bash
# 1. Benchmark (no API key needed)
./scripts/generate_benchmark_data.sh

# 2. SFT training data
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh

# 3. GRPO training prompts
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_grpo_data.sh

# 4. Validation + test splits
OPENAI_API_KEY=sk-... ./scripts/generate_eval_data.sh
```

**Total samples across all splits:** ~15 754 (benchmark 1 000 + SFT 12 504 + GRPO 2 250 + val 500 + test 500).

---

## Output File Naming

Each script writes one JSONL file per (level, language, teacher, preset) combination:

```
{level}_conversations_{lang}_{model}{_preset}_{timestamp}.jsonl
```

Examples:
```
l1_conversations_en_gpt-5-4-nano-2026-03-17_adversarial_20260402_143201.jsonl
l3_conversations_th_gemini-3-flash-preview_adversarial_20260402_143415.jsonl
l4_conversations_code_switch_gpt-5-4-nano-2026-03-17_adversarial_20260402_143602.jsonl
```
