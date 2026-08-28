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

from llm_workflow_agents.data.heldout_clean_set import (
    reserve_guardrail_slice,
    user_turn_fingerprint,
)
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


def _dpo_trainer_kwargs(
    dpo_cfg: dict[str, Any], method: str, output_dir: str
) -> dict[str, Any]:
    """Build the kwargs for TRL's ``DPOConfig``/``ORPOConfig``.

    Two of these bounds are load-bearing and neither has a safe TRL default.

    ``max_length`` — R16, one objective further down the pipeline. TRL's
    ``DataCollatorForPreference`` concatenates ``prompt_ids + chosen_ids`` (and
    the rejected counterpart) and slices to ``max_length``, which defaults to
    **1024** with ``truncation_mode='keep_start'``. Cat A prompts are median
    ~4,400 tokens and never shorter than ~2,400, while ``chosen`` and
    ``rejected`` differ ONLY in the trailing assistant turn. At the default the
    retained window is pure system prompt, both sequences truncate to identical
    token ids, ``completion_mask`` is all zeros, the implicit margin is exactly
    0 and the gradient is exactly 0 — the DPO analogue of the tied-group
    failure that made GRPO unlearnable (R18), and just as silent.

    ``per_device_eval_batch_size`` — TRL defaults it to 8, and DPO scores
    chosen and rejected both, so that is 16 sequences of up to ``max_length``
    against a 262,144-token vocab. It only ever survived because the collator
    was truncating to 1024; pinning ``max_length`` without pinning this trades
    a silent no-op run for an OOM at the first eval. Follows the train batch
    size unless set explicitly, as ``sft.py::_sft_eval_batch_size`` does.

    ``precompute_ref_log_probs`` — Unsloth's compiled DPO trainer materializes
    four full ``[2, S, 262144]`` **fp32** logits tensors per step: policy
    (line 1568) and reference (line 1605), each ``.contiguous()``-copied while
    the original is still live. That is ~33 GiB at this corpus's median prompt
    length and ~64 GiB at ``max_length=8192`` — an OOM on an 80 GB H100 before
    step 1. Precomputing makes the trainer read cached reference logps from the
    batch, dropping the reference forward and its two tensors. It front-loads
    one pass over the train split, so pair it with a train set sized to what
    the run will actually consume (``training_steps`` x effective batch).

    ``use_liger_kernel`` — plumbed through but **not usable on this project's
    checkpoints**. It would be the real fix (a fused chunked loss that never
    builds ``[2, S, 262144]`` in either direction), but
    ``_compute_loss_liger`` raises ``NotImplementedError: Liger DPO loss is not
    implemented for PEFT models.`` and every Cat A DPO run starts from a LoRA
    adapter. It is also mutually exclusive with ``precompute_ref_log_probs``
    (TRL raises at trainer ``__init__``). Kept wired so it flips on the day
    upstream supports PEFT; ``liger-kernel`` is not a declared dependency until
    then.

    Note ``max_prompt_length`` and ``max_completion_length`` are NOT fields of
    TRL 1.0.0's ``DPOConfig``; ``dpo.max_completion_length`` bounds the
    held-out guardrail's generation only, never training.
    """
    per_device_bs = int(dpo_cfg.get("per_device_train_batch_size", 1))
    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        max_steps=dpo_cfg.get("training_steps", 500),
        learning_rate=dpo_cfg.get("learning_rate", 5e-6),
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=int(
            dpo_cfg.get("per_device_eval_batch_size", per_device_bs)
        ),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.05),
        max_length=int(dpo_cfg.get("max_seq_length", 8192)),
        # Cache the reference logps up front instead of re-deriving them every
        # step. See the note below on why this is the difference between an
        # OOM and a run.
        precompute_ref_log_probs=bool(
            dpo_cfg.get("precompute_ref_log_probs", False)
        ),
        # Fused chunked DPO loss — never materializes the logits tensor at all.
        use_liger_kernel=bool(dpo_cfg.get("use_liger_kernel", False)),
        save_steps=int(dpo_cfg.get("save_steps", 100)),
        save_total_limit=int(dpo_cfg.get("save_total_limit", 3)),
        eval_strategy="steps",
        eval_steps=int(dpo_cfg.get("eval_steps", 100)),
        bf16=True,
        report_to="wandb",
    )
    if dpo_cfg.get("precompute_ref_batch_size") is not None:
        kwargs["precompute_ref_batch_size"] = int(
            dpo_cfg["precompute_ref_batch_size"]
        )
    if method == "dpo":
        kwargs["beta"] = dpo_cfg.get("beta", 0.1)
        # No explicit ref_model: TRL falls back to `model.disable_adapter()`
        # on a PEFT model, avoiding a second full copy of a 26B+ checkpoint.
    return kwargs


def _assert_dpo_row_processing_support(trainer_cls: type) -> None:
    """Fail fast if the installed TRL's ``DPOTrainer`` assumes a processor.

    TRL 0.23.1 chooses its row-processing path from
    ``model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``
    — a property of the MODEL, not of the ``processing_class`` it was actually
    handed — and then dereferences ``processing_class.tokenizer``
    unconditionally (``trl/trainer/dpo_trainer.py:739``). Gemma-4 is a
    SigLIP+Gemma4 stack (R9), so it takes that vision path while Unsloth hands
    the trainer a plain tokenizer. On transformers 5.x that is a
    ``GemmaTokenizer``/``TokenizersBackend``, which exposes ``_tokenizer`` and
    not ``tokenizer``, so the run dies with ``AttributeError: TokenizersBackend
    has no attribute tokenizer`` — but only after the 26B checkpoint has
    loaded and the dataset has been tokenized, i.e. ~35 minutes in.

    TRL 1.0.0 branches on ``isinstance(processing_class, ProcessorMixin)``
    instead and takes the text path for a plain tokenizer, which is what Cat A
    preference pairs need — they carry no images.

    This is an environment guard, not a code-compatibility shim. It fires when
    ``.venv-train`` is incomplete: that venv shadows the base image's
    transformers with 5.x but, if ``scripts/install_train.sh`` has not been
    run, ships no ``trl`` of its own, so ``import trl`` silently falls through
    to the image's older copy. Restoring the pin is the fix.

    Follows ``trajectory_rollout.assert_trajectory_rollout_support()``:
    inspect the installed, possibly Unsloth-patched source rather than trust a
    version string. Unsloth wraps ``__init__``, so the whole MRO is scanned —
    an unreadable or fully-wrapped source is treated as "cannot tell" and
    allowed through, because blocking a run on a failed introspection would be
    worse than the failure it guards.

    Raises:
        RuntimeError: if the resolved trainer selects its row-processing path
            from the model type (the TRL 0.23.x behaviour).
    """
    import inspect

    sources: list[str] = []
    for klass in getattr(trainer_cls, "__mro__", (trainer_cls,)):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sources.append(inspect.getsource(init))
        except (OSError, TypeError):  # C-level or source unavailable
            continue

    if not any("is_vision_model" in s for s in sources):
        return

    try:
        import trl

        version = getattr(trl, "__version__", "unknown")
    except ImportError:  # pragma: no cover - trl is imported by the caller
        version = "unknown"

    raise RuntimeError(
        f"TRL {version}: DPOTrainer selects its row-processing path from the "
        "model type (`is_vision_model`) and then dereferences "
        "`processing_class.tokenizer` unconditionally. Gemma-4 takes that "
        "vision path while Unsloth supplies a plain tokenizer, so training "
        "would die with `AttributeError: ... has no attribute tokenizer` "
        "after the 26B load. Install the pinned trl==1.0.0 into .venv-train "
        "(scripts/install_train.sh), which branches on the processing class "
        "instead. A partially-built .venv-train is the usual cause: it "
        "shadows transformers but not trl, so `import trl` falls through to "
        "the base image's older copy."
    )


def _resolve_trl_classes(method: str) -> tuple[type, type]:
    """Return the ``(Config, Trainer)`` pair the installed TRL provides.

    Called before the model loads. TRL 1.0.0 ships ``DPOConfig``/``DPOTrainer``
    but no ``ORPOConfig``/``ORPOTrainer``, and these used to be imported lazily
    at trainer-construction time — i.e. after a 26B checkpoint had been pulled
    onto the GPU and ~650 MB of preference pairs read. Resolving up front turns
    a ~10-minute walk to a bare ``ImportError`` into an immediate one that says
    what to do about it.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"dpo.method must be one of {_VALID_METHODS}, got {method!r}"
        )
    try:
        if method == "orpo":
            from trl import ORPOConfig, ORPOTrainer

            return ORPOConfig, ORPOTrainer
        from trl import DPOConfig, DPOTrainer

        _assert_dpo_row_processing_support(DPOTrainer)
        return DPOConfig, DPOTrainer
    except ImportError as exc:
        import trl

        raise RuntimeError(
            f"dpo.method={method!r} is not available: the installed TRL "
            f"({getattr(trl, '__version__', 'unknown')}) provides no "
            f"{method.upper()}Config/{method.upper()}Trainer. Set "
            f"dpo.method: \"dpo\" (supported on this TRL), or pin a TRL "
            f"release that still ships {method.upper()}."
        ) from exc


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

    config_cls, _trainer_cls = _resolve_trl_classes(method)

    supported = {f.name for f in dataclasses.fields(config_cls)}
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
    # Before the 26B load: an unavailable algorithm must not cost a model load.
    config_cls, trainer_cls = _resolve_trl_classes(method)

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

    trainer_kwargs = _dpo_trainer_kwargs(
        dpo_cfg,
        method,
        output_dir=str(
            _resolve_output_dir(config, Path(config_path), model_basename)
        ),
    )

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

    trainer_config = config_cls(**trainer_kwargs)
    trainer = trainer_cls(
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
    guardrail_reserved_fraction = float(data_cfg.get("guardrail_reserved_fraction", 0.2))
    guardrail_reserved_seed = int(data_cfg.get("guardrail_reserved_seed", 42))

    held_out_rows: list[dict[str, Any]] = []
    try:
        reserved_fps = reserve_guardrail_slice(
            Path(heldout_data_source),
            split="validation",
            reserved_fraction=guardrail_reserved_fraction,
            seed=guardrail_reserved_seed,
        )
        val_ds = _load_grpo_jsonl(Path(heldout_data_source), split="validation")
        # Restricted to the reserved slice, NOT "first N rows of validation" —
        # this is what keeps the guardrail independent of anything mined as a
        # DPO negative from the rest of validation (see the reserved-slice
        # design in docs/superpowers/specs/2026-08-17-mining-yield-investigation-design.md).
        held_out_rows = [
            row
            for row in val_ds
            if user_turn_fingerprint({"messages": row["prompt"]}) in reserved_fps
        ][:n_held_out]
        logger.info(
            "dpo_heldout_loaded",
            n_prompts=len(held_out_rows),
            reserved_fraction=guardrail_reserved_fraction,
            reserved_seed=guardrail_reserved_seed,
        )
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
            self.held_out_rows = held_out_rows

        def _evaluate(self) -> float | None:
            if not held_out_rows:
                return None
            import torch

            was_training = model.training
            # HF Trainer sets this False at train start under gradient
            # checkpointing; model.generate() below flips it back True to
            # speed up decoding and, left alone, leaves it there — so the
            # training step right after this eval would allocate a KV cache
            # gradient checkpointing had disabled (R19 / see the finally
            # block below).
            use_cache_before_eval = getattr(model.config, "use_cache", None)
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
                # Undo generate()'s use_cache=True before training resumes —
                # see the note above _evaluate. Untested prior to this fix;
                # leading hypothesis for the OOM that always hit the training
                # step immediately after a guardrail eval
                # (docs/dpo_memory_ceiling_investigation.md §5).
                model.config.use_cache = use_cache_before_eval
                # Release the per-prompt KV caches before the optimizer runs
                # again. They are freed by refcount but stay in PyTorch's
                # caching allocator, so without this the step after an eval
                # starts with less memory than the step before it — measured
                # as an OOM asking for 52 MiB with 31.9 MiB free and 525 MiB
                # reserved-but-unallocated (checkpoints/dpo_cat_a_smoke7/).
                enc = out = None
                torch.cuda.empty_cache()
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
