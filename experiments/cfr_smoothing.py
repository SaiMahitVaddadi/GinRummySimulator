"""Smoothed / regularised MCCFR ablation on abstracted Classic Gin.

Addresses PAPER.md §6 prediction #3: as vanilla MCCFR trains against the
coarse Classic-Gin abstraction, its average policy concentrates near-
deterministic per info-set, and a 1-step-lookahead sampled BR exploits
that concentration harder than it exploits a diffuse uniform baseline —
so sampled exploitability *rises* with more iterations.

This script measures four fixed-budget (5000 outcome-sampling MCCFR
iterations) configurations:

  * baseline           — no smoothing
  * lambda=0.1         — KL-to-uniform blend on the average strategy
  * min_prob=0.02      — post-hoc floor on average action probabilities
  * both               — combine the two

For each we report:
  * distinct info-sets in the trained tables
  * sampled exploitability (1-step lookahead sampled BR, per module)
  * head-to-head win rate vs uniform (a low-variance robustness signal)
  * average Shannon entropy (nats) of the trained strategy over
    encountered info-sets

Usage:
    uv run python experiments/cfr_smoothing.py

Writes JSONL to ``experiments/results/cfr_smoothing.jsonl`` and prints
a Markdown table.

Not invoked from the test suite — the ablation is a one-shot experiment
and runtime is ~2 minutes total on a laptop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from gin_rummy.solvers.cfr import AverageStrategy
from gin_rummy.solvers.gin_cfr import (
    average_policy_entropy,
    head_to_head_score,
    sample_exploitability,
    train_gin_cfr,
)


ITERATIONS: int = 5_000
NUM_DEALS_EXPL: int = 32
NUM_DEALS_H2H: int = 200
SEED: int = 0


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    regularisation_lambda: float
    min_prob: float


CONFIGS: tuple[Config, ...] = (
    Config("baseline", 0.0, 0.0),
    Config("lambda=0.1", 0.1, 0.0),
    Config("min_prob=0.02", 0.0, 0.02),
    Config("both", 0.1, 0.02),
)


def _row(cfg: Config, iters: int, seed: int) -> dict[str, float | int | str]:
    t0 = time.time()
    strat = train_gin_cfr(
        iters,
        seed=seed,
        regularisation_lambda=cfg.regularisation_lambda,
        min_prob=cfg.min_prob,
    )
    train_dt = time.time() - t0

    t0 = time.time()
    expl = sample_exploitability(strat, num_deals=NUM_DEALS_EXPL, seed=seed)
    expl_dt = time.time() - t0

    t0 = time.time()
    h2h = head_to_head_score(
        strat, AverageStrategy(), num_deals=NUM_DEALS_H2H, seed=seed
    )
    h2h_dt = time.time() - t0

    ent = average_policy_entropy(strat)
    n_iset = strat.num_info_sets()

    return {
        "name": cfg.name,
        "iterations": iters,
        "regularisation_lambda": cfg.regularisation_lambda,
        "min_prob": cfg.min_prob,
        "info_sets": n_iset,
        "exploitability": expl,
        "h2h_vs_uniform": h2h,
        "entropy": ent,
        "elapsed_s": train_dt + expl_dt + h2h_dt,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cfr_smoothing.jsonl"

    records: list[dict[str, float | int | str]] = []
    with out_path.open("w") as f:
        for cfg in CONFIGS:
            print(
                f"== {cfg.name}: lambda={cfg.regularisation_lambda},"
                f" min_prob={cfg.min_prob} =="
            )
            rec = _row(cfg, ITERATIONS, SEED)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"  info_sets={rec['info_sets']} "
                f"expl={rec['exploitability']:.4f} "
                f"h2h={rec['h2h_vs_uniform']:+.4f} "
                f"entropy={rec['entropy']:.4f} "
                f"({rec['elapsed_s']:.1f}s)"
            )

    print("\n## Smoothing ablation (5k iters, seed=0)\n")
    print(
        "| config          | info_sets | exploitability | h2h_vs_uniform | entropy |"
    )
    print(
        "|-----------------|----------:|---------------:|---------------:|--------:|"
    )
    for r in records:
        print(
            f"| {r['name']:<15} | {r['info_sets']:>9} |"
            f" {r['exploitability']:>14.4f} |"
            f" {r['h2h_vs_uniform']:>+14.4f} |"
            f" {r['entropy']:>7.4f} |"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
