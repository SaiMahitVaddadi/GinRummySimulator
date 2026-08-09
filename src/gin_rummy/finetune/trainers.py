"""Trainer wrappers.

Each function takes a ``FineTuneRun`` + dataset, imports ``trl`` lazily,
raises a helpful error if the extra isn't installed, and hands off to the
corresponding trl trainer. Kept intentionally short — the config
dataclass is the reproducibility artifact, not the wrapper.

All four local trainers (SFT/DPO/KTO/GRPO) share the same skeleton:

1. Lazy-import ``transformers``/``peft``/``trl``/``datasets`` — surface a
   clean :class:`MissingFineTuneExtras` if anything is absent.
2. Load base model + tokenizer, wrap with a LoRA/QLoRA adapter via
   :class:`peft.LoraConfig` per ``run.peft``.
3. Build a :class:`datasets.Dataset` from the input rows.
4. Configure :class:`transformers.TrainingArguments` from
   ``run.optimizer`` + ``run.scheduler``.
5. Launch the matching trl trainer, save LoRA weights to
   ``run.output_dir``, and return that path.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from gin_rummy.finetune.config import FineTuneRun, PEFTConfig


class MissingFineTuneExtras(RuntimeError):
    """Raised when the caller tries to run a trainer without the heavy deps."""


def _require(name: str, hint: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise MissingFineTuneExtras(
            f"'{name}' is required for this trainer.\n"
            f"Install the fine-tuning extra:\n    uv sync --extra finetune\n"
            f"({hint})"
        ) from exc


# ---------------------------------------------------------------------------
# Shared helpers — every local trainer runs through these.
# ---------------------------------------------------------------------------

def _load_base(run: FineTuneRun) -> tuple[Any, Any]:
    """Load base model + tokenizer, honouring QLoRA 4-bit loading if requested."""
    transformers = _require("transformers", "for the base model")
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {}
    if run.peft.load_in_4bit:
        try:
            BitsAndBytesConfig = transformers.BitsAndBytesConfig
        except AttributeError as exc:  # pragma: no cover — bnb wiring
            raise MissingFineTuneExtras(
                "BitsAndBytesConfig missing — install `bitsandbytes` for 4-bit."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(run.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(run.base_model, **model_kwargs)
    return model, tokenizer


def _apply_peft(model: Any, peft_cfg: PEFTConfig) -> Any:
    """Wrap ``model`` in a LoRA/QLoRA adapter (no-op if method == 'none')."""
    if peft_cfg.method == "none":
        return model
    peft = _require("peft", "for LoRA adapters")
    LoraConfig = peft.LoraConfig
    get_peft_model = peft.get_peft_model

    lora_cfg = LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        target_modules=list(peft_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if peft_cfg.load_in_4bit:
        prepare = getattr(peft, "prepare_model_for_kbit_training", None)
        if prepare is not None:
            model = prepare(model)
    return get_peft_model(model, lora_cfg)


def _training_args(run: FineTuneRun) -> Any:
    """Translate our config dataclasses into a ``TrainingArguments`` instance."""
    transformers = _require("transformers", "for TrainingArguments")
    return transformers.TrainingArguments(
        output_dir=run.output_dir,
        per_device_train_batch_size=run.batch_size,
        num_train_epochs=run.num_epochs,
        gradient_accumulation_steps=run.optimizer.gradient_accumulation_steps,
        learning_rate=run.optimizer.lr,
        weight_decay=run.optimizer.weight_decay,
        max_grad_norm=run.optimizer.max_grad_norm,
        optim=run.optimizer.name,
        lr_scheduler_type=run.scheduler.name,
        warmup_ratio=run.scheduler.warmup_ratio,
        seed=run.seed,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )


def _rows_to_dataset(rows: Sequence[dict[str, Any]]) -> Any:
    datasets = _require("datasets", "for building the trainer's Dataset")
    return datasets.Dataset.from_list(list(rows))


def _save(trainer: Any, run: FineTuneRun) -> str:
    """Persist the LoRA weights (or full model) and return the output dir."""
    trainer.save_model(run.output_dir)
    return run.output_dir


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def run_sft(run: FineTuneRun, dataset: Sequence[dict[str, str]]) -> str:
    """SFT via ``trl.SFTTrainer``. Returns the output directory.

    ``dataset`` rows must have ``prompt`` + ``completion`` fields — the same
    shape produced by :meth:`DataCollector.to_sft_rows`.
    """
    trl = _require("trl", "for SFTTrainer")
    model, tokenizer = _load_base(run)
    model = _apply_peft(model, run.peft)
    ds = _rows_to_dataset(
        [{"text": r["prompt"] + r["completion"]} for r in dataset]
    )
    args = _training_args(run)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": ds,
        "args": args,
        "dataset_text_field": "text",
        "max_seq_length": run.max_seq_len,
    }
    trainer_kwargs.update(run.method_kwargs)
    trainer = trl.SFTTrainer(**trainer_kwargs)
    trainer.train()
    return _save(trainer, run)


def run_dpo(run: FineTuneRun, pairs: Sequence[dict[str, str]]) -> str:
    """DPO via ``trl.DPOTrainer``.

    ``pairs`` must have ``prompt`` / ``chosen`` / ``rejected`` fields — the
    shape produced by :meth:`DataCollector.to_dpo_pairs`. The DPO temperature
    ``beta`` is read from ``run.method_kwargs.get("beta", 0.1)`` per Rafailov
    et al. (arXiv:2305.18290) — 0.1 is the paper's default.
    """
    trl = _require("trl", "for DPOTrainer")
    model, tokenizer = _load_base(run)
    model = _apply_peft(model, run.peft)
    ds = _rows_to_dataset(list(pairs))
    args = _training_args(run)
    beta = float(run.method_kwargs.get("beta", 0.1))
    extra_kwargs = {k: v for k, v in run.method_kwargs.items() if k != "beta"}
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,  # peft path: reference = base weights (adapter off)
        args=args,
        beta=beta,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_length=run.max_seq_len,
        **extra_kwargs,
    )
    trainer.train()
    return _save(trainer, run)


def run_kto(run: FineTuneRun, rows: Sequence[dict[str, Any]]) -> str:
    """KTO via ``trl.KTOTrainer`` — natural fit for binary game outcomes.

    ``rows`` must have ``prompt`` / ``completion`` / ``label`` (bool) fields
    — the shape produced by :meth:`DataCollector.to_kto_rows`.
    """
    trl = _require("trl", "for KTOTrainer")
    model, tokenizer = _load_base(run)
    model = _apply_peft(model, run.peft)
    ds = _rows_to_dataset(list(rows))
    args = _training_args(run)
    beta = float(run.method_kwargs.get("beta", 0.1))
    extra_kwargs = {k: v for k, v in run.method_kwargs.items() if k != "beta"}
    trainer = trl.KTOTrainer(
        model=model,
        ref_model=None,
        args=args,
        beta=beta,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_length=run.max_seq_len,
        **extra_kwargs,
    )
    trainer.train()
    return _save(trainer, run)


def run_grpo(
    run: FineTuneRun,
    prompts: Sequence[str],
    reward_fn: Callable[[list[str]], list[float]],
    *,
    group_size: int = 8,
) -> str:
    """GRPO via ``trl.GRPOTrainer`` (Shao et al. 2024, DeepSeekMath).

    ``reward_fn`` takes a batch of completions and returns per-completion
    scalar rewards — in the rummy setting this rolls out the game to end
    and returns win/loss (or deadwood delta).
    """
    trl = _require("trl", "for GRPOTrainer")
    model, tokenizer = _load_base(run)
    model = _apply_peft(model, run.peft)
    ds = _rows_to_dataset([{"prompt": p} for p in prompts])
    args = _training_args(run)

    # trl's GRPOTrainer expects a reward function that takes the batch's
    # completions and returns rewards; adapt our simpler contract.
    def _reward_wrapper(completions, **_kwargs):  # pragma: no cover — infra
        return reward_fn(list(completions))

    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=[_reward_wrapper],
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        num_generations=group_size,
        **run.method_kwargs,
    )
    trainer.train()
    return _save(trainer, run)


# ---------- API-side trainers ----------

def run_openai_sft(run: FineTuneRun, jsonl_path: str) -> dict[str, Any]:
    """Fire an OpenAI SFT job. Returns the job metadata."""
    openai = _require("openai", "for OpenAI fine-tuning")
    client = openai.OpenAI()
    file = client.files.create(file=open(jsonl_path, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model=run.base_model,
        hyperparameters={"n_epochs": run.num_epochs},
    )
    return job.model_dump() if hasattr(job, "model_dump") else dict(job)


def run_openai_dpo(run: FineTuneRun, jsonl_path: str) -> dict[str, Any]:
    """OpenAI DPO fine-tune (GA on gpt-4o-2024-08-06+ since late 2024)."""
    openai = _require("openai", "for OpenAI fine-tuning")
    client = openai.OpenAI()
    file = client.files.create(file=open(jsonl_path, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model=run.base_model,
        method={"type": "dpo", "dpo": {"hyperparameters": {"n_epochs": run.num_epochs}}},
    )
    return job.model_dump() if hasattr(job, "model_dump") else dict(job)
