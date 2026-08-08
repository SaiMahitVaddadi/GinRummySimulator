"""Thin trainer wrappers.

Each function takes a ``FineTuneRun`` + dataset, imports ``trl`` lazily,
raises a helpful error if the extra isn't installed, and hands off to the
corresponding trl trainer. Kept intentionally short — the config
dataclass is the reproducibility artifact, not the wrapper.
"""

from __future__ import annotations

from typing import Any, Sequence

from gin_rummy.finetune.config import FineTuneRun


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


def run_sft(run: FineTuneRun, dataset: Sequence[dict[str, str]]) -> str:
    """SFT via ``trl.SFTTrainer``. Returns the output directory."""
    trl = _require("trl", "for SFTTrainer")
    _ = _require("transformers", "for the base model")
    _ = _require("peft", "for LoRA adapters")
    _config = run.to_dict()
    raise NotImplementedError(
        "Trainer bodies are intentionally scaffolded; wire trl.SFTTrainer "
        "here per your infra. Config supplied: %s. Dataset rows: %d."
        % (_config, len(dataset))
    )


def run_dpo(run: FineTuneRun, pairs: Sequence[dict[str, str]]) -> str:
    """DPO via ``trl.DPOTrainer``."""
    _require("trl", "for DPOTrainer")
    raise NotImplementedError(
        "Wire trl.DPOTrainer here. pairs=%d, run=%s" % (len(pairs), run.name)
    )


def run_kto(run: FineTuneRun, rows: Sequence[dict[str, Any]]) -> str:
    """KTO via ``trl.KTOTrainer`` — natural fit for binary game outcomes."""
    _require("trl", "for KTOTrainer")
    raise NotImplementedError(
        "Wire trl.KTOTrainer here. rows=%d, run=%s" % (len(rows), run.name)
    )


def run_grpo(
    run: FineTuneRun,
    prompts: Sequence[str],
    reward_fn: Any,
    *,
    group_size: int = 8,
) -> str:
    """GRPO via ``trl.GRPOTrainer`` (Shao et al. 2024, DeepSeekMath).

    ``reward_fn`` takes a batch of completions and returns per-completion
    scalar rewards — in the rummy setting this rolls out the game to end
    and returns win/loss (or deadwood delta).
    """
    _require("trl", "for GRPOTrainer")
    raise NotImplementedError(
        "Wire trl.GRPOTrainer here. group_size=%d, prompts=%d, run=%s"
        % (group_size, len(prompts), run.name)
    )


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
