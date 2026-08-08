"""Batch benchmarking harness with proper statistics.

Runs N independent games between a set of policies, aggregates descriptive
statistics with **Wilson 95% confidence intervals** on every win-rate,
and can pair hands (swap seats on identical deals) to reduce variance —
the classical technique from duplicate poker.

Programmatic usage
------------------
    from gin_rummy.bench import benchmark, PolicyFactory
    from gin_rummy.variants.classic import ClassicGin
    from gin_rummy.policies.heuristic import GreedyKnockPolicy
    from gin_rummy import RandomPolicy

    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[
            PolicyFactory("greedy", lambda rng: GreedyKnockPolicy(rng)),
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
        ],
        num_games=500,
        seed=0,
        paired=True,           # swap seats on each identical deal
    )
    result.print()
"""

from __future__ import annotations

import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from gin_rummy.game import GinRummyGame
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import Policy
from gin_rummy.stats import wilson_ci


@dataclass
class PolicyFactory:
    """Wraps a name + a builder that produces a fresh Policy per game."""

    name: str
    build: Callable[[random.Random], Policy]


@dataclass
class BenchmarkResult:
    variant: str
    num_games: int
    seat_names: list[str]
    wins: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    turns: list[int] = field(default_factory=list)
    losing_deadwood: list[int] = field(default_factory=list)
    game_seconds: list[float] = field(default_factory=list)
    paired: bool = False

    def win_rates(self) -> dict[str, float]:
        return {
            name: self.wins[name] / self.num_games
            for name in self.seat_names + ["draw"]
        }

    def outcome_rates(self) -> dict[str, float]:
        return {k: v / self.num_games for k, v in self.outcomes.items()}

    def summary(self) -> str:
        lines: list[str] = []
        header = f"=== Benchmark: {self.variant} · {self.num_games} games"
        if self.paired:
            header += " (paired seats)"
        header += " ==="
        lines.append(header)
        lines.append("")

        lines.append("Win rates (95% Wilson CI):")
        width = max(len(n) for n in self.seat_names + ["draw"])
        for name in self.seat_names + ["draw"]:
            ci = wilson_ci(self.wins[name], self.num_games)
            lines.append(
                f"  {name:<{width}}  {ci.rate * 100:6.2f}%  "
                f"[{ci.lower * 100:5.2f}, {ci.upper * 100:5.2f}]  "
                f"({self.wins[name]} games)"
            )
        lines.append("")

        lines.append("Outcome distribution (95% Wilson CI):")
        for kind, count in self.outcomes.most_common():
            ci = wilson_ci(count, self.num_games)
            lines.append(
                f"  {kind:<10}  {ci.rate * 100:6.2f}%  "
                f"[{ci.lower * 100:5.2f}, {ci.upper * 100:5.2f}]"
            )
        lines.append("")

        if self.turns:
            mean_t = statistics.fmean(self.turns)
            med_t = statistics.median(self.turns)
            lines.append(f"Turns:            mean={mean_t:5.1f}  median={med_t:5.1f}")
        if self.losing_deadwood:
            mean_dw = statistics.fmean(self.losing_deadwood)
            lines.append(f"Loser deadwood:   mean={mean_dw:5.1f}")
        if self.game_seconds:
            total = sum(self.game_seconds)
            lines.append(
                f"Time:             total={total:6.2f}s  "
                f"per-game={total * 1000 / self.num_games:5.2f}ms"
            )
        return "\n".join(lines)

    def print(self) -> None:
        print(self.summary())


def benchmark(
    *,
    game_cls: type[GinRummyGame],
    policy_factories: Sequence[PolicyFactory],
    num_games: int,
    seed: int | None = None,
    num_players: int | None = None,
    paired: bool = False,
    **game_kwargs,
) -> BenchmarkResult:
    """Run ``num_games`` and collect stats.

    Parameters
    ----------
    paired : bool, default False
        If True (2-player only), each seed is played twice — once with the
        given seating and once with seats swapped — so that both policies
        face identical deals from both sides. Halves the effective seat-bias
        variance. ``num_games`` is the total game count including pairs.
    """
    if num_players is None:
        num_players = len(policy_factories)
    if len(policy_factories) != num_players:
        raise ValueError(
            f"Need one policy factory per seat: {len(policy_factories)} != {num_players}"
        )
    if paired and num_players != 2:
        raise ValueError("Paired mode is only defined for 2-player games")
    if paired and num_games % 2:
        raise ValueError("Paired mode requires an even num_games")

    seat_names = [f"P{i + 1}:{f.name}" for i, f in enumerate(policy_factories)]
    result = BenchmarkResult(
        variant=game_cls.__name__,
        num_games=num_games,
        seat_names=seat_names,
        paired=paired,
    )

    pairs = num_games // 2 if paired else num_games
    for i in range(pairs):
        game_seed = None if seed is None else seed + i
        _run_one(result, game_cls, policy_factories, num_players, game_seed, game_kwargs)
        if paired:
            # Same seed → same shuffle → same deal, but with seats swapped.
            swapped = list(reversed(policy_factories))
            _run_one(result, game_cls, swapped, num_players, game_seed, game_kwargs, swap_names=True, original_names=seat_names)

    return result


def _run_one(
    result: BenchmarkResult,
    game_cls: type[GinRummyGame],
    policy_factories: Sequence[PolicyFactory],
    num_players: int,
    game_seed: int | None,
    game_kwargs: dict,
    *,
    swap_names: bool = False,
    original_names: list[str] | None = None,
) -> None:
    rng = random.Random(game_seed)
    policies = [factory.build(rng) for factory in policy_factories]
    t0 = time.perf_counter()
    game = game_cls(num_players, policies=policies, seed=game_seed, **game_kwargs)
    game_result = game.play()
    result.game_seconds.append(time.perf_counter() - t0)
    result.turns.append(game_result.turns)
    result.outcomes[game_result.outcome] += 1

    if game_result.winner_id is None:
        result.wins["draw"] += 1
    else:
        # When seats are swapped, the physical winner_id refers to the
        # swapped list, but we attribute the win back to the *original*
        # policy identity so seat_names indexing stays coherent.
        if swap_names and original_names is not None:
            winner_original_id = num_players - 1 - game_result.winner_id
            result.wins[original_names[winner_original_id]] += 1
        else:
            result.wins[result.seat_names[game_result.winner_id]] += 1

    if game_result.winner_id is not None:
        for p in game.players:
            if p.player_id == game_result.winner_id:
                continue
            _, _, dv = optimal_decomposition(p.hand)
            result.losing_deadwood.append(dv)


# ---------- CLI-friendly policy specs ----------

def build_policy_factory(spec: str) -> PolicyFactory:
    """Parse a policy spec string into a PolicyFactory.

    Supported specs::

        random
        greedy
        llm:<model>          # e.g. llm:gpt-4o-mini
    """
    from gin_rummy.policies.heuristic import GreedyKnockPolicy
    from gin_rummy.policy import RandomPolicy

    if spec == "random":
        return PolicyFactory("random", lambda rng: RandomPolicy(rng))
    if spec == "greedy":
        return PolicyFactory("greedy", lambda rng: GreedyKnockPolicy(rng))
    if spec.startswith("llm:"):
        model = spec[4:]
        from gin_rummy.policies.llm import LLMPolicy

        def _build(rng: random.Random) -> Policy:
            return LLMPolicy(model=model, fallback=GreedyKnockPolicy(rng))

        return PolicyFactory(f"llm:{model}", _build)
    raise ValueError(f"Unknown policy spec: {spec!r}")
