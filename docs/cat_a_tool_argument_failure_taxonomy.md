# What actually stops Cat A reaching 0.9

**2026-09-25.** The composite is `0.4 × max(state_turn, state_seq) + 0.4 × tool_F1
+ 0.2 × completion`, blended 0.7 text / 0.3 voice. On the best checkpoint
(`model/sft-gemma4-e4b-toolresults-on-task-a-v3`, v4 benchmark, text format) the
state term is 0.9613 and completion 0.9057, so **tool F1 is the only lever with
room in it** — and tool F1 is gated by arguments, not by tool selection.

This document is the measured breakdown of the 540 failed tool calls, and the
record of three hypotheses about them that measurement destroyed.

## 1. Only one lever exists

| lever | today | needed for 0.90, others held |
|---|---|---|
| state term | 0.9613 | 1.1019 — impossible |
| completion | 0.9057 | 1.1869 — impossible |
| **tool F1** | **0.6952** | **0.8359 — reachable** |

Perfect state *and* perfect completion, with tool F1 unchanged, caps the
composite at **0.8781**. Tool F1 at 1.0 with everything else unchanged gives
0.9657. Name accuracy is 0.8868 against argument exact match 0.6309, so the
work is in argument values.

## 2. The 540 failures

1,880 gold tool calls, 1,340 matched (71.3%). Reproduce with
`scripts/triage_tool_call_failures.py`; report in
`runs/audit/tool_failures_e4b_toolresults_text_v4.json`.

| bucket | count | % of failures | % of all calls |
|---|---|---|---|
| `value_in_context` — value was visible, model wrote something else | 320 | 59.3% | 17.0% |
| `missing_argument` — gold argument omitted | 123 | 22.8% | 6.5% |
| `value_unknowable` — value nowhere in what the model saw | 123 | 22.8% | 6.5% |
| `no_call` — no call at all (upper bound, see below) | 73 | 13.5% | 3.9% |
| `equivalent_value` — formatting only | 14 | 2.6% | 0.7% |

`no_call` counts turns the harness never asked the model for (an outbound
opener, a segment after a withheld tool result), so it overstates the model's
own omissions.

Splitting the two value buckets by *why* the value differs:

| reason | `value_in_context` | `value_unknowable` |
|---|---|---|
| different_value | 204 | 57 |
| phrasing (one contains the other) | 60 | 7 |
| partial_overlap (shares words) | 33 | 42 |
| cross_script (language) | 23 | 17 |

## 3. Three hypotheses, all refuted by measurement

**"It's the comparator."** `_deep_equals` is strict, so `"2024-05-10"` against
`"May 10, 2024"` scores like a wrong date, and R28 had found its needs-review
bucket was mostly correct normalizations. Measured: `equivalent_value` is **14
of 540**, 2.6%. Fixing the comparator buys 0.7% of calls.

**"It's language."** After reading five printed samples — `'Bangkok'` against
`'กรุงเทพ'`, `'Chiang Mai'` against `'เชียงใหม่'` — this looked like the whole
story, and it was reported as such. Counted, `cross_script` is **23 of 320**,
7.2%. Generalising from the first examples on screen was the error.

**"The benchmark demands values its own schema forbids."** The enum check found
8 such rows in the v4 benchmark, 8 in v2, 34 in the SFT corpus. Reading them in
context: every one is followed by the tool rejecting it —

```
user       "can you recommend something based on 'super_awesome_stuff'?"
assistant  recommend_products(based_on="super_awesome_stuff")
tool       {"error": "Invalid value for 'based_on'. Must be one of: history, trending, similar."}
```

That is the corpus's `invalid_tool_inputs` behaviour, 15% of conversations by
spec. All **42** were deliberate; `find_enum_violations` now skips a call the
next tool message rejects, and reports **0** across every corpus. Had the "fix"
been applied, it would have deleted error-recovery arcs from a frozen benchmark
to chase 8 calls out of 1,880.

## 4. What the enum path really holds

Of 443 wrong-value failures, only 47 involve an enum at all:

| verdict | count |
|---|---|
| no enum declared (untyped strings) | 331 |
| no enum, long free text (descriptions) | 65 |
| **wrong listed option** | **40** |
| gold outside its own enum | 7 — retracted, see §3 |

The 40 are the one class with an unambiguous fix. Checking whether the two turns
before the call name the gold option: gold_cued 17, neither_cued 14, both_cued
7, pred_cued 2. The literal-token test undercounts, because several
`neither_cued` rows are cued in Thai prose without the English enum token
(`"…ส่งทีมตอบโต้เหตุฉุกเฉินระดับวิกฤต"` → gold `critical`). Clear instruction
failures include:

```
"…'history', 'trending' or 'similar'. Which?"  user: "Let's go with trending"  → model: history
"…via SMS, email, or push?"                    user: "SMS."                    → model: email
user: "Show me the actual numbers from last month."                            → model: current
```

Genuinely underdetermined, where the gold is not defensible: `based_on='similar'`
where nothing picks it over `history`; `feedback_type='experience'` for app-speed
praise that reads equally as `product` or `service`.

**Size it honestly: 40 calls is 2.1% of 1,880 — about +0.008 tool F1 and +0.003
composite.** Worth fixing because it is clean, not because it is big.

## 5. Where the mass is, and what 0.9 would take

89% of wrong-value failures are **untyped string arguments**: `timeline`
('three months' vs 'next three months'), `origin` ('Bangkok' vs 'กรุงเทพ'), and
free-text `description` fields where the model writes a faithful paraphrase of
the customer's complaint and scores zero. Nothing in the schema says what
counts as correct for any of them.

Sorting the 540 failures by who owns them:

| class | calls | share of all calls |
|---|---|---|
| convention / comparator (formatting, phrasing, language, paraphrase) | 196 | 10.4% |
| model error (different value, missing argument, no call) | 344 | 18.3% |

Projecting, on the crude assumption that tool F1 tracks the call-match rate:

| scenario | call match | composite |
|---|---|---|
| today | 0.713 | ~0.851 |
| every convention issue fixed | 0.817 | **~0.893** |
| plus half the model errors fixed | 0.909 | ~0.929 |

**Cleaning alone does not reach 0.9.** It needs both a decision about what these
arguments mean and a real improvement in argument selection.

## 6. The open decision

Free-text arguments (`description`, `notes`, `waiver_reason`) are
paraphrase-by-nature and cannot be exact-matched by anyone. Three ways out, in
cost order:

1. **Score them by similarity, or exclude them from the gold argument set.** A
   metric decision: it changes every recorded score and needs the R27 treatment
   — rescore every model and say so.
2. **Constrain them at generation time**, so arguments that admit one right
   answer declare an enum and the prompt shows it.
3. **Train on what remains.** The 40 wrong-listed-option failures plus the
   genuinely wrong values.

Nobody should start (3) expecting 0.9 from it. The decision in (1) is not the
model's to make, which is why it is written down here rather than acted on.

## 7. Option (1), measured: similarity scoring buys +0.014, not 0.9

Section 6 left the free-text decision open. It has now been *measured* rather
than taken, on every v4 run there is.

`scripts/propose_argument_sources.py` proposed an owner for each of the 248
(tool, argument) pairs in the v4 benchmark from how the corpus uses them; the
22 `free_text` proposals were reviewed by hand and **4 were wrong in kind** —
`block_card.reason` and `redirect_call.reason` hold enum-like tokens
(`suspicious_activity`), `modify_order.modifications` is a dict rendered as a
string, `initiate_return.reason` is one word the customer said. The reviewed
declaration is `data/interim/task_a_argument_sources/sources.json`: **18
free_text, 2 derived, 2 user, 1 system**, with a `corrections` block recording
each change and why. Only pairs that differ from the default are listed, so
nothing leaves scoring by omission.

`scripts/rescore_with_argument_sources.py` then recomputes the tool metric from
the stored run logs with those owners applied — `free_text` by
`free_text_similarity.values_match` (fact-guarded, τ = 0.40), `system` supplied
by the runtime, everything else exact. It uses the production evaluator, so the
BEFORE column reproduces each run's recorded `tool_metrics_conversation` rather
than approximating it.

| run (v4, blended quality) | before | after | Δ |
|---|---|---|---|
| sft_cat_a_e4b_textturns_ckpt3168 · text | 0.8332 | **0.8469** | +0.0137 |
| sft_cat_a_e4b_corpus_v4_ckpt3156 · text | 0.8308 | 0.8436 | +0.0128 |
| sft_cat_a_e4b_corpus_v4_ckpt3156 · native | 0.8189 | 0.8322 | +0.0132 |
| sft_cat_a_e4b_ckpt3168 · text | 0.8160 | 0.8283 | +0.0123 |
| sft_cat_a_12b_textturns_ckpt3168 · text | 0.8084 | 0.8222 | +0.0138 |
| sft_cat_a_e4b_textturns_ckpt3168 · native | 0.8046 | 0.8183 | +0.0137 |
| sft_cat_a_12b_textturns_ckpt3168 · native | 0.7864 | 0.8017 | +0.0152 |
| sft_cat_a_e4b_ckpt3168 · native | 0.7763 | 0.7879 | +0.0116 |
| sft_cat_a_12b_ckpt3168 · native | 0.7749 | 0.7873 | +0.0124 |
| sft_cat_a_12b_ckpt3168 · text | 0.7712 | 0.7850 | +0.0138 |
| gemma-4-12B-it untrained · text | 0.7553 | 0.7646 | +0.0094 |
| gemma-4-12B-it untrained · native | 0.7517 | 0.7604 | +0.0087 |
| gemma-4-E4B-it untrained · text | 0.6877 | 0.7035 | +0.0157 |
| gemma-4-E4B-it untrained · native | 0.6867 | 0.7005 | +0.0138 |

Three things to read off it.

**The ranking does not move.** Every run gains between +0.0087 and +0.0157 —
a level shift, not a re-ordering. Checked explicitly within each tool-turn
format: the ordering before and after is identical. So this changes what the
number *means* without changing which model wins, which is the cheapest
possible form of a metric change.

**It does not get to 0.9.** The best run goes 0.8332 → 0.8469, still **0.053
short**. Section 5 projected ~0.893 from fixing *every* convention issue; free
text is only part of that, and this is the measured part. 863 free-text
comparisons were accepted across the 14 runs and **508 were still rejected**.

**The rejections are not all real errors, so +0.0137 is a floor under this
backend.** On the best run, 31 free-text arguments were rejected: 9 by the fact
guard (a number or identifier changed — correctly rejected), 22 by low lexical
similarity, and several of those 22 are genuine paraphrases the deterministic
backend cannot see:

```
0.25  escalate_to_supervisor.reason
      gold 'SLA breach and high severity delivery damage'
      got  'SLA breached by 3 days and package was damaged'
0.17  waive_late_fee.waiver_reason
      gold 'Good payment history customer claim timely payment'
      got  'Customer paid on time previously'
```

Others are correctly rejected and worth seeing, because they are why the guard
exists at all — `log_complaint_trend.description` gold "your system is very
slow" against the model's "the satisfaction score is very low" is a different
complaint, not a paraphrase. A pinned multilingual encoder registered through
`free_text_similarity.register_backend` would lift the first kind without the
second; it would also make the score depend on a model, which is the trade the
module docstring records.

**What is still open, and is not the model's to decide.** This rescore is
ledger-driven and reads the benchmark without changing it. Making it the *live*
metric means declaring `source` inside the tool schemas, which are inside the
frozen `task_a_benchmark_v4` stage — so it is an edit to frozen data, and every
number recorded before it becomes incomparable with every number after. The
v4 set has now had the R27 treatment (all 14 runs rescored above), so the
prerequisite is met; the decision to spend it is not.

## 8. Reproduce

```bash
python scripts/triage_tool_call_failures.py \
    results/exp_a/sft_cat_a_e4b_textturns_ckpt3168_gatedtext_v4_auto.log \
    --data data/output/benchmark/task_a_v4 \
    --data data/output/benchmark/task_a_voice_v3 \
    --out runs/audit/tool_failures_e4b_toolresults_text_v4.json
```

The section 7 rescore, over every v4 run at once:

```bash
python scripts/propose_argument_sources.py \
    --data data/output/benchmark/task_a_v4 \
    --data data/output/benchmark/task_a_voice_v3 \
    --out data/interim/task_a_argument_sources/proposal.json   # then REVIEW it

python scripts/rescore_with_argument_sources.py results/exp_a/*_v4_auto.log \
    --data data/output/benchmark/task_a_v4 \
    --data data/output/benchmark/task_a_voice_v3 \
    --sources data/interim/task_a_argument_sources/sources.json \
    --out runs/audit/rescore_argument_sources_v4.json
```

No GPU for any of it: the run logs keep every `model_response` event. The enum
check runs on data alone and is part of `scripts/triage_task_a_quality.py`.
