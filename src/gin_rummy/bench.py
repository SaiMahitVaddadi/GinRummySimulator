"""Batch benchmarking harness.

Runs N independent games between a set of policies, aggregates descriptive
statistics (win rate, outcome distribution, mean turns, mean losing-side
deadwood, wall-clock per game), and prints a clean table. This is the
scaffolding for the descriptive-statistics angle called out in the
research surveys.

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

    def win_rates(self) -> dict[str, float]:
        return {name: self.wins[name] / self.num_games for name in self.seat_names + ["draw"]}

    def outcome_rates(self) -> dict[str, float]:
        return {k: v / self.num_games for k, v in self.outcomes.items()}

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"=== Benchmark: {self.variant} · {self.num_games} games ===")
        lines.append("")
        lines.append("Win rates:")
        for name, rate in self.win_rates().items():
            lines.append(f"  {name:<20} {rate * 100:6.2f}%   ({self.wins[name]} games)")
        lines.append("")
        lines.append("Outcome distribution:")
        for kind, rate in self.outcome_rates().items():
            lines.append(f"  {kind:<20} {rate * 100:6.2f}%")
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
                f"Time:             total={total:6.2f}s  per-game={total * 1000 / self.num_games:5.2f}ms"
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
    **game_kwargs,
) -> BenchmarkResult:
    """Run ``num_games`` and collect stats.

    Each game gets a fresh RNG derived deterministically from ``seed + i`` so
    the run is reproducible. Policies are rebuilt per game (so any internal
    state resets cleanly).
    """
    if num_players is None:
        num_players = len(policy_factories)
    if len(policy_factories) != num_players:
        raise ValueError(
            f"Need one policy factory per seat: {len(policy_factories)} != {num_players}"
        )

    seat_names = [f"P{i + 1}:{f.name}" for i, f in enumerate(policy_factories)]
    result = BenchmarkResult(
        variant=game_cls.__name__,
        num_games=num_games,
        seat_names=seat_names,
    )

    for i in range(num_games):
        game_seed = None if seed is None else seed + i
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
            result.wins[seat_names[game_result.winner_id]] += 1

        # Loser deadwood at end of hand — useful signal.
        if game_result.winner_id is not None:
            for p in game.players:
                if p.player_id == game_result.winner_id:
                    continue
                _, _, dv = optimal_decomposition(p.hand)
                result.losing_deadwood.append(dv)

    return result


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
