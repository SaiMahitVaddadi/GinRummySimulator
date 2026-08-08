"""LLM fine-tuning scaffold.

The heavy libraries (``transformers``, ``peft``, ``trl``, ``bitsandbytes``,
``unsloth``) are **optional** — install the ``finetune`` extra to pull
them in::

    uv sync --extra finetune

What ships here without any heavy deps:

* ``collector`` — a ``DataCollector`` that hooks the game loop and dumps
  trajectories as JSONL. The dataset formats it produces (SFT rows,
  DPO/KTO preference pairs, GRPO group-reward tuples) are the input
  contracts every trainer honours.
* ``config`` — dataclasses describing PEFT configs, optimizer/scheduler
  choices, and a ``FineTuneRun`` that composes them into a reproducible
  recipe. No trainer is invoked here; the config is just a manifest.
* ``sft``, ``dpo``, ``kto``, ``grpo`` — thin wrappers that import ``trl``
  lazily and raise a helpful error if the extra is not installed.
* ``api`` — wrappers for provider-side fine-tuning (OpenAI / Together /
  Fireworks) so the framework is symmetric between local and API-side.

Design references:
  * Rafailov et al. NeurIPS 2023 (DPO)     — arXiv:2305.18290
  * Ethayarajh et al. 2024 (KTO)
  * Shao et al. 2024 (GRPO / DeepSeekMath) — arXiv:2402.03300
  * Chen et al. ICML 2024 (SPIN)           — arXiv:2401.01335
  * Dettmers et al. NeurIPS 2023 (QLoRA)   — arXiv:2305.14314
"""

from gin_rummy.finetune.collector import DataCollector, TrajectorySample
from gin_rummy.finetune.config import (
    FineTuneRun,
    OptimizerConfig,
    PEFTConfig,
    SchedulerConfig,
)

__all__ = [
    "DataCollector",
    "FineTuneRun",
    "OptimizerConfig",
    "PEFTConfig",
    "SchedulerConfig",
    "TrajectorySample",
]
