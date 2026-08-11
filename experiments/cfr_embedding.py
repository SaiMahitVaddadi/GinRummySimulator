"""U x C interaction study: learned embedding bucketing vs hand-crafted HandBucket.

Trains MCCFR on abstracted Classic Gin with the *embedding* hand
fingerprint from
:class:`gin_rummy.solvers.embedding_abstraction.EmbeddingHandBucket` in
place of the hand-crafted :class:`gin_rummy.solvers.abstraction.HandBucket`,
across k in {8, 16, 32, 64}. All other knobs (iterations, seed,
smoothing config) are held constant so any delta is attributable to the
abstraction.

Reports per row:
  * distinct info-sets touched in the trained tables
  * sampled exploitability (32 deals, 1-step lookahead sampled BR)
  * head-to-head win rate vs uniform (200 deals)
  * average Shannon entropy (nats) of the reported strategy

Row 0 is the hand-crafted baseline (existing HandBucket, identical
training config) so the comparison is honest.

Usage:
    uv run python experiments/cfr_embedding.py

Writes ``experiments/results/cfr_embedding.jsonl``. Not exercised by
the test suite — runtime is a few minutes on a laptop.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gin_rummy.solvers.cfr import AverageStrategy
from gin_rummy.solvers.embedding_abstraction import (
    DEFAULT_WALK_CONFIG,
    EmbeddingHandBucket,
    build_embedding_bucketing,
)
from gin_rummy.solvers.gin_cfr import (
    average_policy_entropy,
    embedding_information_set,
    head_to_head_score,
    head_to_head_score_with_engines,
    information_set,
    sample_exploitability,
    sample_exploitability_with_engine,
    train_gin_cfr,
    train_gin_cfr_embedding,
)


ITERATIONS: int = 5_000
NUM_DEALS_EXPL: int = 32
NUM_DEALS_H2H: int = 200
REGULARISATION_LAMBDA: float = 0.1  # winning smoothing config from §5c
MIN_PROB: float = 0.0
CORPUS_SIZE: int = 2_000
K_VALUES: tuple[int, ...] = (8, 16, 32, 64)
SEED: int = 0


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    info_sets: int
    exploitability: float
    h2h_vs_uniform: float
    entropy: float
    elapsed_s: float
    k: int | None = None


def _baseline_row(seed: int) -> Row:
    t0 = time.time()
    strat = train_gin_cfr(
        ITERATIONS,
        seed=seed,
        regularisation_lambda=REGULARISATION_LAMBDA,
        min_prob=MIN_PROB,
    )
    train_dt = time.time() - t0
    expl = sample_exploitability(strat, num_deals=NUM_DEALS_EXPL, seed=seed)
    h2h = head_to_head_score(
        strat, AverageStrategy(), num_deals=NUM_DEALS_H2H, seed=seed
    )
    return Row(
        name="hand_crafted",
        info_sets=strat.num_info_sets(),
        exploitability=expl,
        h2h_vs_uniform=h2h,
        entropy=average_policy_entropy(strat),
        elapsed_s=time.time() - t0,
        k=None,
    )


def _embedding_row(k: int, bucketer: EmbeddingHandBucket, seed: int) -> Row:
    t0 = time.time()
    strat = train_gin_cfr_embedding(
        bucketer,
        ITERATIONS,
        seed=seed,
        regularisation_lambda=REGULARISATION_LAMBDA,
        min_prob=MIN_PROB,
    )
    # Evaluate under the *same* abstraction the strategy was trained under.
    info_fn = embedding_information_set(bucketer)
    expl = sample_exploitability_with_engine(
        strat, info_fn, num_deals=NUM_DEALS_EXPL, seed=seed
    )
    # Head-to-head vs uniform — uniform is bucket-agnostic (empty policy
    # returns uniform-over-legal from any info-set), so we can use the
    # hand-crafted info_fn for the uniform side without loss.
    h2h = head_to_head_score_with_engines(
        strat, info_fn,
        AverageStrategy(), information_set,
        num_deals=NUM_DEALS_H2H, seed=seed,
    )
    return Row(
        name=f"embed_k={k}",
        info_sets=strat.num_info_sets(),
        exploitability=expl,
        h2h_vs_uniform=h2h,
        entropy=average_policy_entropy(strat),
        elapsed_s=time.time() - t0,
        k=k,
    )


def _print_row(r: Row) -> None:
    print(
        f"  {r.name:<14} info_sets={r.info_sets:>6}"
        f"  expl={r.exploitability:.4f}"
        f"  h2h={r.h2h_vs_uniform:+.4f}"
        f"  entropy={r.entropy:.4f}"
        f"  ({r.elapsed_s:.1f}s)"
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cfr_embedding.jsonl"
    rows: list[Row] = []

    with out_path.open("w") as f:
        print("== Baseline (hand-crafted HandBucket) ==")
        base = _baseline_row(SEED)
        rows.append(base)
        f.write(json.dumps(asdict(base)) + "\n"); f.flush()
        _print_row(base)

        for k in K_VALUES:
            print(f"\n== Embedding bucketer k={k} (corpus={CORPUS_SIZE}) ==")
            t0 = time.time()
            bucketer = build_embedding_bucketing(
                k=k,
                corpus_size=CORPUS_SIZE,
                walk_config=DEFAULT_WALK_CONFIG,
                seed=SEED,
            )
            print(
                f"  fit: {time.time() - t0:.1f}s"
                f"  n_train={bucketer.n_train}  dim={bucketer.dim}"
                f"  centroids={len(bucketer.centroids)}"
            )
            row = _embedding_row(k, bucketer, SEED)
            rows.append(row)
            f.write(json.dumps(asdict(row)) + "\n"); f.flush()
            _print_row(row)

    print("\n## U x C interaction (5k iters, lambda=0.1, seed=0)\n")
    print(
        "| config       | info_sets | exploitability | h2h_vs_uniform | entropy |"
    )
    print(
        "|--------------|----------:|---------------:|---------------:|--------:|"
    )
    for r in rows:
        print(
            f"| {r.name:<12} | {r.info_sets:>9} |"
            f" {r.exploitability:>14.4f} |"
            f" {r.h2h_vs_uniform:>+14.4f} |"
            f" {r.entropy:>7.4f} |"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
