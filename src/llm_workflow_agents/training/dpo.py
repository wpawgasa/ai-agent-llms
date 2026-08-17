"""Unsloth DPO/ORPO entry point for Cat A preference fine-tuning.

R18 (`docs/grpo_reward_resolution_investigation.md`) established that GRPO
cannot learn on this task: the per-turn reward takes 11 distinct values across
206 real completions and is exactly 1.0 on 81.1% of them, so a GRPO group ties
and the advantage is identically zero. Three independent fixes (tool-bearing
prompt mix, trajectory aggregation, higher sampling temperature) each moved the
needle and none produced a usable learning signal.

A preference objective sidesteps the problem entirely: every pair carries a
guaranteed margin (`chosen` is the gold turn, `rejected` is a documented
failure C2 actually makes), so no reward variance is required. It is also
cheaper per step than GRPO, since there is no generation in the training loop —
which matters given Risk R9 forces HF `generate()` rollouts on Gemma-4.

Serves both DPO and ORPO from one config shape (`stage: dpo`, `dpo.method: dpo
| orpo`) rather than a second entry point, mirroring how `sft.py` serves Cat A/
B/C from one module. The two differ only in trainer/config class and whether a
reference model is used (ORPO needs none; DPO with a PEFT checkpoint and no
explicit `ref_model` uses TRL's `model.disable_adapter()` fallback instead of
loading a second full copy of the base model).

Consumes the pair sets built by `scripts/build_preference_pairs.py`
(SYNTHETIC corruptions of gold turns) and `scripts/mine_model_negatives.py`
(on-distribution negatives mined from the checkpoint's own greedy errors),
both in TRL's conversational preference format:

    {"prompt": [...], "chosen": [{"role": "assistant", ...}],
     "rejected": [{"role": "assistant", ...}], "source": "synthetic" | "model"}

The held-out guardrail (Risk R5) and the Gemma-4 KV-shared-proxy workaround are
shared with `grpo.py` via `reward_utils.py` / `_utils.py` rather than
redefined here — see those modules' docstrings for why duplicating either was
what caused past regressions to go unnoticed.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from llm_workflow_agents.training._utils import (
    unwrap_unsloth_gemma4_kv_zero_proxy,
)
from llm_workflow_agents.training.reward_utils import (
    heldout_composite_score,
    is_reward_hacking,
)

if TYPE_CHECKING:
    from datasets import Dataset

logger = structlog.get_logger(__name__)

_VALID_METHODS = ("dpo", "orpo")


@dataclass(frozen=True)
class DPOResult:
    """Result of a DPO/ORPO preference-learning run."""

    checkpoint_path: Path | None = None
    method: str = "dpo"
    best_eval_loss: float | None = None
    total_steps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    held_out_scores: list[float] = field(default_factory=list)
    early_stopped: bool = False
    error: str | None = None


def _load_dpo_config(config_path: Path) -> dict[str, Any]:
    """Load a DPO/ORPO YAML config (v3 format with a `stage` field)."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if config.get("stage") != "dpo":
        raise ValueError(f"Expected stage='dpo', got '{config.get('stage')}'")
    return config


def _resolve_method(dpo_cfg: dict[str, Any]) -> str:
    """Return the requested algorithm, defaulting to plain DPO."""
    method = str(dpo_cfg.get("method", "dpo")).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"dpo.method must be one of {_VALID_METHODS}, got {method!r}"
        )
    return method


#: Trailing run stamp appended by a `run_phase2_*.sh`-style launcher to the
#: patched config filename (see CLAUDE.md R13), e.g. "dpo_cat_a_20260817T101500Z".
_RUN_STAMP_RE = re.compile(r"_\d{8}T\d{6}Z$")


def _resolve_output_dir(
    config: dict[str, Any], config_path: Path, model_name: str
) -> Path:
    """Resolve the checkpoint output directory for a DPO/ORPO run.

    Precedence:
      1. Explicit ``output_dir`` in the config — the only way to give a run a
         distinct directory (otherwise a second run overwrites the first).
      2. The config filename stem, with any trailing run stamp removed.

    Mirrors ``sft.py::_resolve_output_dir`` / ``grpo.py::_resolve_output_dir``
    (CLAUDE.md R13): a run-stamped patched config must not silently relocate
    checkpoints to a per-run path a DVC stage does not track. Provenance
    belongs in the config filename; the checkpoint path stays stable unless
    asked to change.
    """
    explicit = config.get("output_dir")
    run_name = str(explicit) if explicit else _RUN_STAMP_RE.sub("", config_path.stem)
    return Path("checkpoints") / run_name / Path(model_name).name


def _validate_pair_row(row: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``row`` is not TRL conversational-preference shape.

    Checked eagerly at load time rather than left to TRL/Datasets, which would
    otherwise surface a malformed row as an opaque failure deep inside the
    trainer's tokenization step, well after the (expensive) model load.
    """
    for key in ("prompt", "chosen", "rejected"):
        if key not in row:
            raise ValueError(f"preference row missing {key!r}: {row}")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"'prompt' must be a non-empty message list: {row}")
    for key in ("chosen", "rejected"):
        msgs = row[key]
        if (
            not isinstance(msgs, list)
            or not msgs
            or msgs[0].get("role") != "assistant"
        ):
            raise ValueError(
                f"{key!r} must be a message list starting with an assistant "
                f"turn: {row}"
            )


def _read_preference_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and validate one preference-pair JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preference pair file missing: {path}")
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            _validate_pair_row(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"Preference pair file is empty: {path}")
    return rows


def _mix_preference_sources(
    synthetic_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    model_share: float | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Combine synthetic pairs with mined on-distribution negatives.

    R18 recommends complementing the synthetic corruptions
    (`scripts/build_preference_pairs.py`) with negatives mined from the
    checkpoint's own errors (`scripts/mine_model_negatives.py`): synthetic-only
    negatives teach the model to discriminate against the corruption function,
    not to get the task right (R15 is the corpus-level precedent for that
    trap). But mined negatives are scarce by construction — the first mining
    run yielded 51 rows against 29,256 synthetic ones, so a plain concat would
    make them well under 1% of the training signal despite being the more
    valuable, on-distribution rows.

    ``model_share`` sets the target fraction of the merged set that is
    mined-negative rows, reached by sampling mined rows **with replacement**
    (there are too few to reach a useful share otherwise). ``None`` is a
    passthrough plain concat. A share at or below the natural rate is also a
    passthrough — this never downsamples either source to hit a number the
    data already beats, mirroring ``grpo.py::_tool_bearing_mix_indices``.
    """
    if not model_rows:
        return list(synthetic_rows)
    if model_share is None:
        return list(synthetic_rows) + list(model_rows)
    if not 0.0 <= model_share < 1.0:
        raise ValueError(
            f"model_negative_share must be in [0, 1), got {model_share}"
        )

    n_synthetic = len(synthetic_rows)
    total_natural = n_synthetic + len(model_rows)
    natural_share = len(model_rows) / total_natural if total_natural else 0.0
    if model_share <= natural_share:
        return list(synthetic_rows) + list(model_rows)

    # Solve n_model (sampled with replacement) for
    # n_model / (n_synthetic + n_model) == model_share.
    n_model = round(n_synthetic * model_share / (1.0 - model_share))
    rng = random.Random(seed)
    oversampled = rng.choices(model_rows, k=n_model) if n_model else []
    merged = list(synthetic_rows) + oversampled
    rng.shuffle(merged)
    return merged


def _load_dpo_dataset(
    data_cfg: dict[str, Any], seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Load, mix and shape the train/validation preference datasets.

    ``data.train_sources`` lists one or more preference-pair JSONL files;
    rows are split by their ``source`` field ("synthetic" vs "model") and
    recombined via :func:`_mix_preference_sources`. ``data.validation_source``
    is a single file — never mixed, so held-out numbers stay comparable run to
    run regardless of ``model_negative_share``.

    Only the three columns TRL's conversational-preference format needs
    (``prompt``, ``chosen``, ``rejected``) are kept; bookkeeping fields
    (``rejected_type``, ``source``, ``prompt_fingerprint``) are dropped rather
    than handed to the trainer.
    """
    from datasets import Dataset

    train_sources = data_cfg.get("train_sources") or []
    if not train_sources:
        raise ValueError(
            "data.train_sources must list at least one preference-pair "
            "JSONL file"
        )
    validation_source = data_cfg.get("validation_source")
    if not validation_source:
        raise ValueError("data.validation_source must be set")

    synthetic_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for src in train_sources:
        for row in _read_preference_jsonl(Path(src)):
            (model_rows if row.get("source") == "model" else synthetic_rows).append(
                row
            )

    merged = _mix_preference_sources(
        synthetic_rows, model_rows, data_cfg.get("model_negative_share"), seed=seed
    )
    logger.info(
        "dpo_dataset_loaded",
        n_synthetic=len(synthetic_rows),
        n_model=len(model_rows),
        n_merged=len(merged),
        model_negative_share_requested=data_cfg.get("model_negative_share"),
    )

    eval_rows = _read_preference_jsonl(Path(validation_source))

    def _to_dataset(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                }
                for r in rows
            ]
        )

    return _to_dataset(merged), _to_dataset(eval_rows)


def _filter_dpo_config_kwargs(
    kwargs: dict[str, Any], method: str
) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs the installed TRL ``DPOConfig``/``ORPOConfig`` does not accept.

    Same defect class as ``grpo.py::_filter_grpo_config_kwargs`` (R16): TRL
    moves fields between releases, and an unfiltered kwarg dict can raise
    ``TypeError`` at ``DPOConfig(**kwargs)`` after the model and both dataset
    splits have already loaded. Returns ``(kept, dropped)``; the caller must
    log ``dropped`` — a silently dropped length or weighting knob is exactly
    how R16 went unnoticed for months.
    """
    import dataclasses

    if method == "orpo":
        from trl import ORPOConfig as ConfigCls
    elif method == "dpo":
        from trl import DPOConfig as ConfigCls
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {_VALID_METHODS}")

    supported = {f.name for f in dataclasses.fields(ConfigCls)}
    kept = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(kept))
    return kept, dropped


def train_dpo(config_path: Path) -> DPOResult:
    """Run the Unsloth DPO/ORPO preference-learning pipeline.

    Pipeline:
      1. Load the SFT/C2 checkpoint via ``FastLanguageModel.from_pretrained()``
         — the checkpoint already carries a LoRA adapter, so (unlike ``sft.py``)
         no separate ``get_peft_model`` call is made.
      2. Load and mix the preference-pair train/validation datasets.
      3. Configure ``DPOTrainer``/``ORPOTrainer`` per ``dpo.method``. DPO with a
         PEFT model and no explicit ``ref_model`` uses TRL's
         ``model.disable_adapter()`` fallback rather than a second full model
         copy; ORPO needs no reference model at all.
      4. Train for the configured steps. Optionally run the held-out
         reward-hacking guardrail (Risk R5) every
         ``monitoring.eval_held_out_every`` steps.
      5. Return ``DPOResult``.
    """
    from unsloth import FastLanguageModel

    # Same ordering rationale as grpo.py: importing unsloth installs
    # unsloth_zoo's Gemma-4 KV-zero proxy, which must be disarmed before any
    # Gemma-4 AutoConfig resolution or FastLanguageModel.from_pretrained call.
    unwrap_unsloth_gemma4_kv_zero_proxy()

    # Unsloth ships no trainer of its own for preference learning — DPO/ORPO
    # still run through TRL's DPOTrainer/ORPOTrainer. What Unsloth provides is
    # `PatchDPOTrainer()`, which replaces TRL's loss computation with a fused,
    # memory-efficient kernel (~2x faster, ~50% less VRAM per Unsloth's DPO
    # notebooks) — it must run before a DPOTrainer/ORPOTrainer is
    # *constructed*, so this early in `train_dpo` is safe even though
    # `_filter_dpo_config_kwargs` below imports `DPOConfig`/`ORPOConfig` first.
    # Best-effort: an older/newer Unsloth without this symbol should not block
    # training, only lose the speedup.
    try:
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()
        logger.info("dpo_unsloth_patch_applied")
    except ImportError:
        logger.warning(
            "dpo_unsloth_patch_unavailable",
            note=(
                "unsloth.PatchDPOTrainer not found on this Unsloth version — "
                "falling back to unpatched TRL DPOTrainer/ORPOTrainer. "
                "Training still runs, just slower and at higher VRAM."
            ),
        )

    config = _load_dpo_config(config_path)
    dpo_cfg = config.get("dpo", {})
    data_cfg = config.get("data", {})
    monitoring_cfg = config.get("monitoring", {})

    method = _resolve_method(dpo_cfg)

    sft_checkpoint = config.get("model", {}).get("sft_checkpoint")
    if not sft_checkpoint:
        return DPOResult(method=method, error="model.sft_checkpoint not set in DPO config")

    logger.info(
        "dpo_starting",
        method=method,
        sft_checkpoint=sft_checkpoint,
        training_steps=dpo_cfg.get("training_steps", 500),
        beta=dpo_cfg.get("beta", 0.1),
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_checkpoint,
        max_seq_length=dpo_cfg.get("max_seq_length", 8192),
        dtype=None,
        load_in_4bit=True,
    )

    # Re-arm the unwrap. FastLanguageModel.from_pretrained re-applies
    # unsloth_zoo's temporary patches, which reinstalls the proxy on top of
    # ours (see grpo.py::train_grpo — this bit every Gemma-4 held-out eval
    # silently until it was re-armed here too).
    unwrap_unsloth_gemma4_kv_zero_proxy()

    # Gemma-4's RoPE forward needs the same cuBLAS-stride patch SFT applies;
    # DPO/ORPO run the same forward path (policy, and for DPO an implicit
    # reference pass via `model.disable_adapter()`), so it needs it too.
    from llm_workflow_agents.training.sft import _patch_gemma4_rope_stride

    _patch_gemma4_rope_stride(model)

    train_ds, eval_ds = _load_dpo_dataset(data_cfg, seed=dpo_cfg.get("seed", 42))

    model_basename = Path(sft_checkpoint).parent.name

    trainer_kwargs: dict[str, Any] = dict(
        output_dir=str(_resolve_output_dir(config, Path(config_path), model_basename)),
        max_steps=dpo_cfg.get("training_steps", 500),
        learning_rate=dpo_cfg.get("learning_rate", 5e-6),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.05),
        save_steps=int(dpo_cfg.get("save_steps", 100)),
        save_total_limit=int(dpo_cfg.get("save_total_limit", 3)),
        eval_strategy="steps",
        eval_steps=int(dpo_cfg.get("eval_steps", 100)),
        bf16=True,
        report_to="wandb",
    )
    if method == "dpo":
        trainer_kwargs["beta"] = dpo_cfg.get("beta", 0.1)
        # No explicit ref_model: TRL falls back to `model.disable_adapter()`
        # on a PEFT model, avoiding a second full copy of a 26B+ checkpoint.

    trainer_kwargs, dropped_kwargs = _filter_dpo_config_kwargs(trainer_kwargs, method)
    if dropped_kwargs:
        logger.warning(
            "dpo_config_kwargs_unsupported",
            method=method,
            dropped=dropped_kwargs,
            note=(
                "These keys are set but do not exist on the installed TRL's "
                "config class, so they have NO effect. If a dropped key "
                "bounds a length, a rate or the KL weight, verify the run is "
                "still within budget."
            ),
        )

    if method == "orpo":
        from trl import ORPOConfig, ORPOTrainer

        trainer_config = ORPOConfig(**trainer_kwargs)
        trainer = ORPOTrainer(
            model=model,
            args=trainer_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
    else:
        from trl import DPOConfig, DPOTrainer

        trainer_config = DPOConfig(**trainer_kwargs)
        trainer = DPOTrainer(
            model=model,
            args=trainer_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

    callbacks = []
    if monitoring_cfg.get("reward_hacking_detector", False):
        callbacks.append(
            _build_heldout_callback(
                model=model,
                tokenizer=tokenizer,
                data_cfg=data_cfg,
                monitoring_cfg=monitoring_cfg,
                max_new_tokens=int(dpo_cfg.get("max_completion_length", 512)),
            )
        )
        for cb in callbacks:
            trainer.add_callback(cb)

    output_dir = Path(trainer_config.output_dir)
    resume_from: str | None = None
    if output_dir.is_dir():
        existing = sorted(
            (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if existing:
            resume_from = str(existing[-1])
            logger.info("dpo_resuming", from_checkpoint=resume_from)
        else:
            logger.info("dpo_starting_fresh", output_dir=str(output_dir))

    result = trainer.train(resume_from_checkpoint=resume_from)
    eval_metrics = trainer.evaluate()

    held_out_scores: list[float] = []
    early_stopped = False
    for cb in callbacks:
        if hasattr(cb, "held_out_history"):
            held_out_scores = cb.held_out_history
            early_stopped = bool(
                len(held_out_scores) >= 2 and held_out_scores[-1] < held_out_scores[-2]
            )

    return DPOResult(
        checkpoint_path=output_dir,
        method=method,
        best_eval_loss=eval_metrics.get("eval_loss"),
        total_steps=result.global_step,
        metrics={**result.metrics, **eval_metrics},
        held_out_scores=held_out_scores,
        early_stopped=early_stopped,
    )


def _build_heldout_callback(
    *,
    model: Any,
    tokenizer: Any,
    data_cfg: dict[str, Any],
    monitoring_cfg: dict[str, Any],
    max_new_tokens: int,
) -> Any:
    """Build the R5 held-out reward-hacking guardrail callback.

    Structurally identical to ``grpo.py::_HeldOutEvalCallback``, adapted to
    a preference-loss run: "reward" there is whatever scalar the training loop
    reports per log step (a DPO/ORPO accuracy or margin metric here, a GRPO
    reward there) — ``is_reward_hacking`` is objective-agnostic. Reuses
    ``grpo.py::_load_grpo_jsonl`` for the held-out prompt source: preference
    pairs are derived from the same underlying GRPO-format corpus
    (``data/output/grpo/task_a``), and that loader already applies the ground
    truth JSON parsing this callback needs — reimplementing it here would be
    exactly the duplication ``_utils.py``'s docstring warns against.
    """
    from transformers import TrainerCallback

    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    heldout_data_source = monitoring_cfg.get(
        "heldout_data_source", data_cfg.get("heldout_data_source", "data/output/grpo/task_a")
    )
    eval_held_out_every = int(monitoring_cfg.get("eval_held_out_every", 50))
    n_held_out = int(monitoring_cfg.get("eval_held_out_num_prompts", 50))

    held_out_rows: list[dict[str, Any]] = []
    try:
        val_ds = _load_grpo_jsonl(Path(heldout_data_source), split="validation")
        held_out_rows = [val_ds[i] for i in range(min(n_held_out, len(val_ds)))]
        logger.info("dpo_heldout_loaded", n_prompts=len(held_out_rows))
    except FileNotFoundError:
        logger.warning(
            "dpo_heldout_split_missing",
            note="validation split not found; held-out guardrail disabled",
            data_source=heldout_data_source,
        )

    class _DPOHeldOutEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.metric_history: list[float] = []
            self.held_out_history: list[float] = []

        def _evaluate(self) -> float | None:
            if not held_out_rows:
                return None
            import torch

            was_training = model.training
            model.eval()
            completions: list[str] = []
            gts: list[dict[str, Any]] = []
            try:
                with torch.no_grad():
                    for row in held_out_rows:
                        text = tokenizer.apply_chat_template(
                            row["prompt"], tokenize=False, add_generation_prompt=True
                        )
                        enc = tokenizer(
                            text, return_tensors="pt", truncation=True, max_length=7680
                        ).to(model.device)
                        out = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            generation_config=getattr(model, "generation_config", None),
                        )
                        gen = tokenizer.decode(
                            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
                        )
                        completions.append(gen)
                        gt_raw = row.get("ground_truth")
                        gts.append(
                            json.loads(gt_raw) if isinstance(gt_raw, str) else (gt_raw or {})
                        )
            except Exception as exc:
                logger.warning("dpo_heldout_eval_failed", error=str(exc))
                return None
            finally:
                if was_training:
                    model.train()
            return heldout_composite_score(completions, gts)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            metric = logs.get("rewards/accuracies", logs.get("loss"))
            if metric is not None:
                self.metric_history.append(metric)
            if self.held_out_history:
                logs["eval/held_out_composite"] = self.held_out_history[-1]

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % eval_held_out_every == 0:
                score = self._evaluate()
                if score is None:
                    return
                self.held_out_history.append(score)
                logger.info(
                    "dpo_heldout_eval", step=state.global_step, held_out_composite=score
                )
                if is_reward_hacking(self.metric_history, self.held_out_history):
                    logger.warning(
                        "dpo_reward_hacking_detected",
                        step=state.global_step,
                        held_out_recent=self.held_out_history[-1],
                        held_out_prev=self.held_out_history[-2],
                    )
                    control.should_training_stop = True

    return _DPOHeldOutEvalCallback()
