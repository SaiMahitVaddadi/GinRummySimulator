"""Command-line entry point.

Examples
--------
    gin-rummy                                   # one Classic Gin hand, 2 players
    gin-rummy --variant oklahoma --seed 42
    gin-rummy --variant hollywood --players 4 --hands 3
    gin-rummy --variant indian --players 4 --hand-size 13
    gin-rummy --variant classic --games 1000 --quiet   # batch stats
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Sequence

from gin_rummy.variants.classic import ClassicGin
from gin_rummy.variants.hollywood import HollywoodGin
from gin_rummy.variants.indian import IndianRummy
from gin_rummy.variants.oklahoma import OklahomaGin

VARIANTS = {
    "classic": ClassicGin,
    "oklahoma": OklahomaGin,
    "hollywood": HollywoodGin,
    "indian": IndianRummy,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gin-rummy",
        description="Multi-variant rummy simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="classic",
        help="Which ruleset to play.",
    )
    p.add_argument("--players", type=int, default=2, help="Number of players (2..12).")
    p.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility."
    )
    p.add_argument(
        "--games", type=int, default=1, help="How many independent games to simulate."
    )
    p.add_argument(
        "--hand-size",
        type=int,
        default=None,
        help="Override hand size (Indian Rummy only supports 3/7/10/13/15).",
    )
    p.add_argument(
        "--hands",
        type=int,
        default=3,
        help="Hands per series (Hollywood variant only).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-game output.")
    return p


def _play_one(variant: str, players: int, seed: int | None, args) -> str | None:
    """Return the winning player name (e.g. 'Player 1'), or None for a draw."""
    if variant == "hollywood":
        game = HollywoodGin(players, num_hands=args.hands, seed=seed)
        result = game.play()
        return f"Player {result.winner_id + 1}" if result.winner_id is not None else None
    if variant == "indian":
        hand_size = args.hand_size or 13
        game = IndianRummy(players, hand_size=hand_size, seed=seed)
    else:
        game_cls = VARIANTS[variant]
        kwargs = {}
        if args.hand_size is not None:
            kwargs["hand_size"] = args.hand_size
        game = game_cls(players, seed=seed, **kwargs)
    result = game.play()
    return result.winner_name


def _print_single_game(variant: str, players: int, seed: int | None, args) -> None:
    print(f"\n=== {variant.title()} Rummy — {players} players — seed={seed} ===")
    if variant == "hollywood":
        game = HollywoodGin(players, num_hands=args.hands, seed=seed)
        result = game.play()
        for i, hand in enumerate(result.hands, 1):
            print(f"  Hand {i}: {hand.outcome} in {hand.turns} turns → {hand.scores}")
        print(f"  Totals: {result.totals}")
        print(
            f"  Winner: {'Player ' + str(result.winner_id + 1) if result.winner_id is not None else 'draw'}"
        )
        return

    if variant == "indian":
        hand_size = args.hand_size or 13
        game = IndianRummy(players, hand_size=hand_size, seed=seed)
    else:
        game_cls = VARIANTS[variant]
        kwargs = {}
        if args.hand_size is not None:
            kwargs["hand_size"] = args.hand_size
        game = game_cls(players, seed=seed, **kwargs)
    result = game.play()
    print(f"  Outcome: {result.outcome} in {result.turns} turns")
    print(f"  Scores:  {result.scores}")
    print(f"  Winner:  {result.winner_name or 'draw'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.games == 1 and not args.quiet:
        _print_single_game(args.variant, args.players, args.seed, args)
        return 0

    tally: Counter[str] = Counter()
    for i in range(args.games):
        seed = None if args.seed is None else args.seed + i
        winner = _play_one(args.variant, args.players, seed, args)
        tally[winner or "draw"] += 1
        if not args.quiet:
            print(f"  Game {i + 1:>4}/{args.games}: winner={winner or 'draw'}")

    print(f"\n=== Summary ({args.games} games of {args.variant}) ===")
    width = max(len(k) for k in tally)
    for name, count in tally.most_common():
        pct = 100.0 * count / args.games
        print(f"  {name:<{width}}  {count:>5}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
