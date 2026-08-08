"""Command-line entry point.

Two modes:

    # Play mode (default): one or more games with a running commentary.
    gin-rummy                                          # single Classic Gin hand, 2 players
    gin-rummy --variant oklahoma --seed 42
    gin-rummy --variant hollywood --players 4 --hands 3
    gin-rummy --variant indian --players 4 --hand-size 13
    gin-rummy --variant classic --games 1000 --seed 0 --quiet

    # Benchmark mode: N games between named policies; prints a stats table.
    gin-rummy --bench --games 500 --policies greedy,random --seed 0
    gin-rummy --bench --variant oklahoma --policies greedy,greedy --games 1000
    gin-rummy --bench --policies llm:gpt-4o-mini,greedy --games 20   # needs litellm
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Sequence

from gin_rummy.bench import PolicyFactory, benchmark, build_policy_factory
from gin_rummy.eval.reporters import export_jsonl, print_tournament
from gin_rummy.eval.tournament import PolicyEntry, Tournament
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
    p.add_argument(
        "--bench",
        action="store_true",
        help="Run in benchmark mode (aggregate stats across --games games).",
    )
    p.add_argument(
        "--policies",
        default=None,
        help=(
            "Comma-separated policy specs per seat, e.g. 'greedy,random' or "
            "'llm:gpt-4o-mini,greedy'. Defaults to random for every seat."
        ),
    )
    p.add_argument(
        "--paired",
        action="store_true",
        help=(
            "Paired-hands variance reduction (bench mode, 2 players only): each "
            "seed plays twice with seats swapped so both policies face the same deals."
        ),
    )
    p.add_argument(
        "--tournament",
        action="store_true",
        help=(
            "Run a round-robin tournament instead of a 2-way bench. All policies "
            "in --policies play every other; cross-play matrix + Bradley–Terry Elo "
            "with bootstrap CIs are reported. Ignores --players (always 2-seat)."
        ),
    )
    p.add_argument(
        "--jsonl",
        default=None,
        help="Path to write full tournament results as JSONL for post-hoc analysis.",
    )
    return p


def _parse_policies(spec: str | None, num_players: int) -> list[PolicyFactory]:
    if not spec:
        return [build_policy_factory("random") for _ in range(num_players)]
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if len(parts) != num_players:
        raise SystemExit(
            f"--policies expected {num_players} entries (one per seat), got {len(parts)}"
        )
    return [build_policy_factory(s) for s in parts]


def _play_one(variant: str, players: int, seed: int | None, args) -> str | None:
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
    return game.play().winner_name


def _print_single_game(variant: str, players: int, seed: int | None, args) -> None:
    print(f"\n=== {variant.title()} Rummy — {players} players — seed={seed} ===")
    if variant == "hollywood":
        game = HollywoodGin(players, num_hands=args.hands, seed=seed)
        result = game.play()
        for i, hand in enumerate(result.hands, 1):
            print(f"  Hand {i}: {hand.outcome} in {hand.turns} turns → {hand.scores}")
        print(f"  Totals: {result.totals}")
        winner = (
            f"Player {result.winner_id + 1}" if result.winner_id is not None else "draw"
        )
        print(f"  Winner: {winner}")
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


def _run_bench(args) -> int:
    if args.variant == "hollywood":
        raise SystemExit("--bench doesn't support Hollywood (multi-hand series); pick another variant")
    game_cls = VARIANTS[args.variant]
    factories = _parse_policies(args.policies, args.players)
    kwargs = {}
    if args.hand_size is not None:
        kwargs["hand_size"] = args.hand_size
    result = benchmark(
        game_cls=game_cls,
        policy_factories=factories,
        num_games=args.games,
        seed=args.seed,
        num_players=args.players,
        paired=args.paired,
        **kwargs,
    )
    result.print()
    return 0


def _run_tournament(args) -> int:
    if args.variant == "hollywood":
        raise SystemExit("--tournament doesn't support Hollywood; pick another variant")
    game_cls = VARIANTS[args.variant]
    factories = _parse_policies(args.policies, num_players=len(_split_policies(args.policies)))
    entries = [PolicyEntry(name=f.name, build=f.build) for f in factories]
    kwargs = {}
    if args.hand_size is not None:
        kwargs["hand_size"] = args.hand_size
    t = Tournament(
        game_cls=game_cls,
        entries=entries,
        games_per_pair=args.games,
        seed=args.seed,
        paired=args.paired,
        **kwargs,
    )
    result = t.run()
    print_tournament(result)
    if args.jsonl:
        export_jsonl(result, args.jsonl)
        print(f"[wrote {len(result.matches)} matches + summary to {args.jsonl}]")
    return 0


def _split_policies(spec: str | None) -> list[str]:
    if not spec:
        return ["random", "random"]
    return [s.strip() for s in spec.split(",") if s.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.tournament:
        return _run_tournament(args)

    if args.bench:
        return _run_bench(args)

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
