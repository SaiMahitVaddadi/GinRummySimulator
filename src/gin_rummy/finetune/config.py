"""Fine-tuning configuration dataclasses.

Serialisable, YAML/JSON-round-trippable configs that describe a single
fine-tuning run end-to-end. Trainer wrappers consume these — no side
effects here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PEFTMethod = Literal["lora", "qlora", "dora", "adalora", "ia3", "pissa", "none"]
Method = Literal["sft", "dpo", "kto", "orpo", "grpo", "rloo", "simpo"]


@dataclass
class PEFTConfig:
    """Parameter-efficient-fine-tuning knobs.

    Defaults are the community-standard "safe" LoRA recipe: rank 16,
    alpha 32, dropout 0.05, applied to attention + MLP projections.
    Switch ``method`` to ``"qlora"`` to enable 4-bit base model loading.
    """

    method: PEFTMethod = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = False  # auto-set to True when method == "qlora"

    def __post_init__(self) -> None:
        if self.method == "qlora":
            self.load_in_4bit = True


@dataclass
class OptimizerConfig:
    """Optimizer + LR + weight-decay + accumulation settings."""

    name: Literal["adamw", "adamw_8bit", "paged_adamw_8bit", "lion", "adafactor", "schedulefree_adamw"] = "paged_adamw_8bit"
    lr: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


@dataclass
class SchedulerConfig:
    """Learning-rate schedule."""

    name: Literal["cosine", "linear", "constant", "wsd", "constant_with_warmup"] = "cosine"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1  # final LR = min_lr_ratio × lr (cosine)
    stable_steps: int | None = None  # WSD plateau length
    decay_ratio: float = 0.1  # WSD cooldown fraction


@dataclass
class FineTuneRun:
    """A single fine-tuning experiment.

    A ``FineTuneRun`` is a pure specification — writing it to disk gives
    you full reproducibility. The trainer wrappers in this package accept
    a run and materialise the actual training loop.
    """

    name: str
    base_model: str  # HF hub id or local path
    method: Method
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 0
    batch_size: int = 4
    num_epochs: int = 1
    max_seq_len: int = 2048
    output_dir: str = "./checkpoints"
    # Method-specific knobs — kept loose so we can add new trainers cheaply.
    method_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
