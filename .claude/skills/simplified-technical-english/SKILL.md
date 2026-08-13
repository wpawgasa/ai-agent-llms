---
name: simplified-technical-english
description: Use when writing or editing technical documentation, procedures, runbooks, warnings, error messages, release notes, or API docs — especially for readers who are non-native English speakers, or when instructions are long, dense, ambiguous, or get misread. Also use when asked for ASD-STE-100, STE, controlled language, or a plain-English rewrite.
---

# Simplified Technical English

## Overview

A writing contract adapted from ASD-STE-100, the controlled-English standard used for
aerospace and defence maintenance manuals. It exists because a reader who misreads a
procedure breaks something expensive.

**Core principle: one instruction, one sentence, one meaning.**

The output is judged by shape, not by taste. A sentence either fits the contract or it
does not, and you can measure that.

## When to Use

- Procedures, runbooks, migration steps, incident playbooks
- Warnings, cautions, error messages
- API docs and reference material read under time pressure
- Any document whose readers include non-native English speakers
- A reader has already misread the document once

**Do not use for:**

- Prose that must persuade or entertain
- Commit messages — use conventional-commit style
- Narrative design docs, where nuance beats speed

## The Contract

Write each sentence to satisfy all nine:

| # | Rule | Limit |
|---|---|---|
| 1 | One instruction per sentence | 1 |
| 2 | Procedure sentence length | ≤ 20 words |
| 3 | Descriptive sentence length | ≤ 25 words |
| 4 | Paragraph length, one topic each | ≤ 6 sentences |
| 5 | Noun cluster length | ≤ 3 nouns |
| 6 | Start an instruction with its verb | imperative |
| 7 | Use the same word for the same thing | 1 word = 1 meaning |
| 8 | Keep articles: *the*, *a*, *an* | always |
| 9 | Use simple tenses: past, present, future | no perfect/progressive |

In a warning, put the condition first. Put the command second.
**"If the cache is cold, fetch the blobs before you check out."**

## Rewrite Patterns

Each pattern shows a real sentence and its replacement.

**Split on the conjunction.** One clause becomes one sentence.

> The bug is silent in the configuration the project has always used, and loud in the one
> it had never successfully run — because `response_only` was independently broken until
> `bac1d98`. *(41 words)*

> The bug is silent in the usual configuration. It is loud in cell C2. Only C2 uses
> `response_only`. Commit `bac1d98` fixed `response_only`. *(8 / 5 / 4 / 5 words)*

**Name the actor.** Passive hides who acts.

> The collator is truncated to 1024 tokens. → TRL truncates every batch to 1024 tokens.

**Repeat the noun.** A pronoun two sentences from its noun is a guess.

> It overwrites it. → The C2 run overwrites the C0 checkpoint.

**Unstack the nouns.** Break clusters longer than three.

> Cat A SFT corpus role corruption fix → the fix for corrupted roles in the Cat A corpus

**Lift steps into a list.** A sentence with three commas is a list in disguise.

> Run the cleaner, then split the corpus, then filter the prompts, and push the result.

> 1. Run the cleaner.
> 2. Split the corpus.
> 3. Filter the prompts.
> 4. Push the result.

## One Word, One Meaning

Choose one term per concept and keep it for the whole document.

| Keep | Drop |
|---|---|
| start | launch, kick off, fire off, spin up |
| delete | remove, drop, purge, clear |
| fix | resolve, address, handle, sort out |
| check | verify, validate, confirm, ensure |
| fail | break, blow up, die, choke |

Write the list at the top of a long document. Follow it.

## Verify

Do not judge by feel. Measure:

```bash
python check_ste.py path/to/doc.md
```

It reports sentence count, mean and maximum length, the share over 20 words, and the
passive-voice share. It prints every sentence that breaks the contract, so you can fix
them one at a time.

**Gate:** no sentence over 25 words, and under 10% of sentences over 20 words.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Em dashes joining two full clauses | Make them two sentences. |
| A parenthesis carrying a second instruction | Give it its own sentence or a footnote. |
| "This" with no noun after it | Name the thing: "this failure", "this file". |
| Hedges: *should*, *might*, *typically* | State the condition: "If X, then Y." |
| A 30-word bullet | A bullet obeys the same 20-word limit. |
| Rewriting a quotation to fit | Quote exactly. The contract covers your words. |

## Real-World Impact

Three technical documents were measured in one repository, before any rewrite.
Between 33% and 46% of their sentences ran over 20 words. The longest reached 47.
The split and actor patterns moved a section under the gate. No fact was lost.
