"""End-to-end HMM-conditioned cascade study.

Pipeline
--------
1. Train an :class:`OpponentHandHMM` on 200 seeded ClassicGin games
   (two ``GreedyKnockPolicy`` seats).
2. Play 50 held-out games with the same seat wiring, snapshotting each
   game's state right before the target seat's discard on turn 8.
3. Run the cascade comparison on those snapshots: baseline vs. HMM-
   conditioned.
4. Report top-1 predictive accuracy and top-5 continuation coverage,
   with Wilson 95% CIs, and dump per-game records to
   ``experiments/results/hmm_cascade.jsonl``.

Interpretation
--------------
The point isn't to prove HMM conditioning "wins" — the greedy heuristic
seat has almost no information the HMM can lever. It's to *measure*
whether the belief signal moves the response distribution enough to
matter for short-horizon prediction under a fixed, weak opponent
policy. The report prints an honest verdict at the bottom.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from gin_rummy import ClassicGin, GreedyKnockPolicy
from gin_rummy.markov.hmm_cascade import (
    replay_with_snapshots,
    run_cascade_comparison,
)
from gin_rummy.markov.opponent_hand_hmm import OpponentHandHMM
from gin_rummy.stats import wilson_ci


TRAIN_GAMES = 200
HELDOUT_GAMES = 50
TARGET_PLAYER_ID = 1
TARGET_TURN = 8
CASCADE_DEPTH = 3
TOP_K = 5
SEED = 20260810
EM_ITERATIONS = 20
BETA = 0.5
BASE_TEMPERATURE = 1.0


def _play(seed: int):
    rng = random.Random(seed)
    engine = ClassicGin(
        num_players=2,
        policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
        seed=seed,
    )
    return engine.play()


def _factory(seed: int) -> ClassicGin:
    rng = random.Random(seed)
    return ClassicGin(
        num_players=2,
        policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
        seed=seed,
    )


def main() -> None:
    # ---- train
    print(f"[hmm-cascade] training HMM on {TRAIN_GAMES} games (seed={SEED})...")
    train_games = []
    for i in range(TRAIN_GAMES):
        train_games.append(_play(SEED + i))
    hmm = OpponentHandHMM(rng=random.Random(SEED))
    hmm.train(
        train_games,
        target_player_id=TARGET_PLAYER_ID,
        iterations=EM_ITERATIONS,
    )
    print(f"[hmm-cascade] HMM trained ({EM_ITERATIONS} EM iters)")

    # ---- snapshot held-out
    print(
        f"[hmm-cascade] snapshotting {HELDOUT_GAMES} held-out games "
        f"at turn {TARGET_TURN}..."
    )
    snapshots = []
    heldout_base = SEED + 100_000
    for i in range(HELDOUT_GAMES):
        got = replay_with_snapshots(
            _factory,
            game_seed=heldout_base + i,
            target_player_id=TARGET_PLAYER_ID,
            target_turn=TARGET_TURN,
        )
        if got is None:
            continue
        state, result, record = got
        snapshots.append((i, state, result, record))
    print(f"[hmm-cascade] usable snapshots: {len(snapshots)}/{HELDOUT_GAMES}")

    # ---- compare
    report = run_cascade_comparison(
        snapshots,
        target_player_id=TARGET_PLAYER_ID,
        hmm=hmm,
        depth=CASCADE_DEPTH,
        top_k=TOP_K,
        beta=BETA,
        base_temperature=BASE_TEMPERATURE,
    )

    b = report["baseline"]
    h = report["hmm"]

    b_top1_ci = wilson_ci(b["top1_correct"], b["top1_total"])
    h_top1_ci = wilson_ci(h["top1_correct"], h["top1_total"])
    b_cov_ci = wilson_ci(b["top5_cover"], b["top5_total"])
    h_cov_ci = wilson_ci(h["top5_cover"], h["top5_total"])

    print()
    print("=" * 78)
    print(
        f"{'model':<20} {'top1 acc':>16} {'top1 95% CI':>22} {'top5 cov':>10} "
        f"{'top5 95% CI':>22}"
    )
    print("-" * 78)
    print(
        f"{'baseline':<20} "
        f"{b['top1_correct']}/{b['top1_total']} ({b_top1_ci.rate:.3f})".ljust(37)
        + f"[{b_top1_ci.lower:.3f}, {b_top1_ci.upper:.3f}]".rjust(22)
        + f"  {b_cov_ci.rate:.3f}".rjust(10)
        + f"[{b_cov_ci.lower:.3f}, {b_cov_ci.upper:.3f}]".rjust(22)
    )
    print(
        f"{'hmm-conditioned':<20} "
        f"{h['top1_correct']}/{h['top1_total']} ({h_top1_ci.rate:.3f})".ljust(37)
        + f"[{h_top1_ci.lower:.3f}, {h_top1_ci.upper:.3f}]".rjust(22)
        + f"  {h_cov_ci.rate:.3f}".rjust(10)
        + f"[{h_cov_ci.lower:.3f}, {h_cov_ci.upper:.3f}]".rjust(22)
    )
    print("=" * 78)
    print()

    # ---- dump
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "hmm_cascade.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        # First line: summary
        summary = {
            "kind": "summary",
            "train_games": TRAIN_GAMES,
            "heldout_games": HELDOUT_GAMES,
            "target_turn": TARGET_TURN,
            "target_player_id": TARGET_PLAYER_ID,
            "cascade_depth": CASCADE_DEPTH,
            "top_k": TOP_K,
            "beta": BETA,
            "base_temperature": BASE_TEMPERATURE,
            "seed": SEED,
            "baseline": {
                "top1_correct": b["top1_correct"],
                "top1_total": b["top1_total"],
                "top1_rate": b_top1_ci.rate,
                "top1_ci_lower": b_top1_ci.lower,
                "top1_ci_upper": b_top1_ci.upper,
                "top5_cover": b["top5_cover"],
                "top5_total": b["top5_total"],
                "top5_rate": b_cov_ci.rate,
                "top5_ci_lower": b_cov_ci.lower,
                "top5_ci_upper": b_cov_ci.upper,
            },
            "hmm": {
                "top1_correct": h["top1_correct"],
                "top1_total": h["top1_total"],
                "top1_rate": h_top1_ci.rate,
                "top1_ci_lower": h_top1_ci.lower,
                "top1_ci_upper": h_top1_ci.upper,
                "top5_cover": h["top5_cover"],
                "top5_total": h["top5_total"],
                "top5_rate": h_cov_ci.rate,
                "top5_ci_lower": h_cov_ci.lower,
                "top5_ci_upper": h_cov_ci.upper,
            },
        }
        f.write(json.dumps(summary) + os.linesep)
        for row in report["rows"]:
            f.write(
                json.dumps(
                    {
                        "kind": "row",
                        "game_index": row.game_index,
                        "turn": row.turn,
                        "baseline_top1_correct": row.baseline_top1_correct,
                        "hmm_top1_correct": row.hmm_top1_correct,
                        "baseline_top5_covers_observed": row.baseline_top5_covers_observed,
                        "hmm_top5_covers_observed": row.hmm_top5_covers_observed,
                        "baseline_top1_prob": row.baseline_top1_prob,
                        "hmm_top1_prob": row.hmm_top1_prob,
                        "baseline_top1_value": row.baseline_top1_value,
                        "hmm_top1_value": row.hmm_top1_value,
                        "observed_next_discard": row.observed_next_discard,
                    }
                )
                + os.linesep
            )
    print(f"[hmm-cascade] wrote {out_path}")


if __name__ == "__main__":
    main()
