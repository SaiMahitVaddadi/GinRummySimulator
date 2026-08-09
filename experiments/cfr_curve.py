"""Run the sampled exploitability learning curve for abstracted Classic Gin.

Usage:
    uv run python experiments/cfr_curve.py

Writes ``experiments/results/cfr_curve.jsonl`` — one JSON line per
checkpoint with fields ``{iterations, exploitability, info_sets,
h2h_vs_uniform, elapsed_s}``.

Runs at 100 / 500 / 2000 / 10000 iterations. Total wall-clock is ~5
minutes on a laptop (dominated by the 10000-iteration train + the four
sampled-exploitability estimates at 32 deals each). The exploitability
metric is a sampled proxy — see :func:`sample_exploitability` docstring
for its limitations.

This script is intentionally NOT invoked from the test suite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from gin_rummy.solvers.cfr import AverageStrategy, ExternalSamplingMCCFR
from gin_rummy.solvers.gin_cfr import (
    ClassicGinEngine,
    head_to_head_score,
    sample_exploitability,
)


CHECKPOINTS: list[int] = [100, 500, 2000, 10000]
NUM_DEALS: int = 32
H2H_DEALS: int = 200
SEED: int = 0


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cfr_curve.jsonl"

    engine = ClassicGinEngine()
    solver = ExternalSamplingMCCFR(seed=SEED, engine=engine, sampling="outcome")
    uniform = AverageStrategy()

    # Baseline: uniform strategy — an unrelated but useful anchor.
    print("== Baseline (uniform) ==")
    t0 = time.time()
    e_uni = sample_exploitability(uniform, num_deals=NUM_DEALS, seed=SEED)
    dt = time.time() - t0
    print(f"  uniform sampled exploitability: {e_uni:.4f}  ({dt:.1f}s)")

    prev = 0
    records: list[dict[str, float | int]] = []
    with out_path.open("w") as f:
        base = {
            "iterations": 0,
            "exploitability": e_uni,
            "info_sets": 0,
            "h2h_vs_uniform": 0.0,
            "elapsed_s": dt,
            "label": "uniform_baseline",
        }
        f.write(json.dumps(base) + "\n")
        records.append(base)

        for target in CHECKPOINTS:
            step = target - prev
            t0 = time.time()
            print(f"\n== Training +{step} iters (cumulative: {target}) ==")
            solver.train(step, seed=None)
            train_dt = time.time() - t0
            strat = solver.average_strategy()
            n_iset = strat.num_info_sets()
            print(f"  train: {train_dt:.1f}s  info-sets: {n_iset}")

            t0 = time.time()
            expl = sample_exploitability(strat, num_deals=NUM_DEALS, seed=SEED)
            eval_dt = time.time() - t0
            print(f"  sampled exploitability: {expl:.4f}  ({eval_dt:.1f}s)")

            t0 = time.time()
            h2h = head_to_head_score(strat, uniform, num_deals=H2H_DEALS, seed=SEED)
            h2h_dt = time.time() - t0
            print(f"  head-to-head vs uniform: {h2h:+.4f}  ({h2h_dt:.1f}s)")

            rec = {
                "iterations": target,
                "exploitability": expl,
                "info_sets": n_iset,
                "h2h_vs_uniform": h2h,
                "elapsed_s": train_dt + eval_dt + h2h_dt,
                "label": f"cfr_{target}",
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)
            prev = target

    print("\n== Summary ==")
    print(f"{'iters':>8}  {'expl':>8}  {'iset':>6}  {'h2h':>8}")
    for r in records:
        print(
            f"{r['iterations']:>8}  {r['exploitability']:>8.4f}  "
            f"{r['info_sets']:>6}  {r['h2h_vs_uniform']:>+8.4f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
