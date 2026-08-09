"""End-to-end GNN SFT experiment: collect -> train -> evaluate.

Runnable as a script:

    uv run experiments/gnn_train.py \
        --num-games 400 --epochs 5 --eval-games 200

Produces:

* ``data/gnn_trajectories.jsonl`` — logged trajectories.
* ``checkpoints/gnn_sft.pt`` — trained model weights.
* ``results/gnn_train.json`` — training report + head-to-head win rates.

Head-to-head evaluation: 200 games each (paired seed) against
``GreedyKnockPolicy`` and ``RandomPolicy``. Reports win rates with
Wilson 95 % CIs — an honest number, even if the model loses.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gin_rummy.models.train_gnn import (
    collect_trajectories,
    examples_from_trajectories,
    save_trajectories,
    train_sft,
)
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import Policy, RandomPolicy
from gin_rummy.stats import wilson_ci
from gin_rummy.variants.classic import ClassicGin


@dataclass
class HeadToHead:
    opponent: str
    games: int
    learner_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    ci_lower: float
    ci_upper: float


def _play_pair(
    learner_seat: int,
    learner: Policy,
    opponent: Policy,
    seed: int,
) -> int:
    """Play one game with the given seat assignment. Returns +1 (learner wins),
    -1 (opponent wins), 0 (draw)."""
    if learner_seat == 0:
        policies = [learner, opponent]
    else:
        policies = [opponent, learner]
    game = ClassicGin(num_players=2, seed=seed, policies=policies)
    result = game.play()
    if result.winner_id is None:
        return 0
    return 1 if result.winner_id == learner_seat else -1


def evaluate_head_to_head(
    learner_factory,
    opponent_factory,
    *,
    num_games: int,
    seed: int,
    opponent_name: str,
) -> HeadToHead:
    """Paired-seat evaluation: each game is played twice (once per seat)
    on the same base seed so the two policies see mirrored deals."""
    rng = random.Random(seed)
    wins = losses = draws = 0
    games_played = 0
    for gi in range(num_games // 2):
        base = rng.randrange(2**31)
        for seat in (0, 1):
            learner = learner_factory(random.Random(base + seat * 7919))
            opponent = opponent_factory(random.Random(base + 31 + seat * 7919))
            outcome = _play_pair(seat, learner, opponent, seed=base + seat)
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1
            games_played += 1
    if games_played == 0:
        games_played = 1
    ci = wilson_ci(wins, games_played)
    return HeadToHead(
        opponent=opponent_name,
        games=games_played,
        learner_wins=wins,
        opponent_wins=losses,
        draws=draws,
        win_rate=ci.rate,
        ci_lower=ci.lower,
        ci_upper=ci.upper,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-games", type=int, default=400,
                        help="self-play games for trajectory collection")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="cap on collected trajectory rows")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-path", type=str,
                        default="data/gnn_trajectories.jsonl")
    parser.add_argument("--checkpoint-path", type=str,
                        default="checkpoints/gnn_sft.pt")
    parser.add_argument("--results-path", type=str,
                        default="results/gnn_train.json")
    args = parser.parse_args()

    # -------- collect --------
    t0 = time.time()
    print(f"[collect] {args.num_games} games, cap {args.max_rows} rows")
    rows = collect_trajectories(
        num_games=args.num_games,
        learner_factory=lambda rng: GreedyKnockPolicy(rng),
        opponent_factory=lambda rng: RandomPolicy(rng),
        seed=args.seed,
        max_rows=args.max_rows,
    )
    save_trajectories(rows, args.trajectory_path)
    print(f"[collect] {len(rows)} rows dumped to {args.trajectory_path} "
          f"({time.time() - t0:.1f}s)")

    # -------- prepare --------
    examples = examples_from_trajectories(rows)
    print(f"[prepare] {len(examples)} training examples")

    # -------- train --------
    t0 = time.time()
    report = train_sft(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        val_frac=args.val_frac,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
    print(f"[train] finished in {time.time() - t0:.1f}s")

    # -------- evaluate --------
    from gin_rummy.models.gnn_policy import GNNPolicy

    def gnn_factory(_rng: random.Random) -> Policy:
        return GNNPolicy(
            model_path=args.checkpoint_path,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
        )

    def greedy_factory(rng: random.Random) -> Policy:
        return GreedyKnockPolicy(rng)

    def random_factory(rng: random.Random) -> Policy:
        return RandomPolicy(rng)

    t0 = time.time()
    vs_greedy = evaluate_head_to_head(
        gnn_factory, greedy_factory,
        num_games=args.eval_games, seed=args.seed + 101,
        opponent_name="greedy",
    )
    vs_random = evaluate_head_to_head(
        gnn_factory, random_factory,
        num_games=args.eval_games, seed=args.seed + 202,
        opponent_name="random",
    )
    print(f"[eval] finished in {time.time() - t0:.1f}s")
    print(f"[eval] vs greedy: {vs_greedy.learner_wins}/{vs_greedy.games} "
          f"= {vs_greedy.win_rate:.3f} "
          f"[{vs_greedy.ci_lower:.3f}, {vs_greedy.ci_upper:.3f}]")
    print(f"[eval] vs random: {vs_random.learner_wins}/{vs_random.games} "
          f"= {vs_random.win_rate:.3f} "
          f"[{vs_random.ci_lower:.3f}, {vs_random.ci_upper:.3f}]")

    # -------- dump --------
    out = {
        "args": vars(args),
        "num_train": report.num_train,
        "num_val": report.num_val,
        "epoch_stats": [asdict(s) for s in report.epoch_stats],
        "checkpoint_path": report.checkpoint_path,
        "head_to_head": {
            "vs_greedy": asdict(vs_greedy),
            "vs_random": asdict(vs_random),
        },
    }
    out_path = Path(args.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
