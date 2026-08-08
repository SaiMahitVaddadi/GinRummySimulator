"""N-way round-robin tournament with cross-play matrix and telemetry.

Generalises the old 2-way ``bench.benchmark`` into a proper tournament
runner. Every pairing plays a configurable number of games (paired if
requested), telemetry is collected per policy per matchup, and the
result exposes a cross-play matrix ready for Bradley–Terry rating.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

from gin_rummy.eval.stats import BootstrapCI, ProportionCI, bootstrap_ci, wilson_ci
from gin_rummy.eval.telemetry import PolicyTelemetry, TelemetryPolicy
from gin_rummy.game import GinRummyGame
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import Policy

PolicyBuilder = Callable[[random.Random], Policy]


@dataclass
class PolicyEntry:
    name: str
    build: PolicyBuilder


@dataclass
class MatchOutcome:
    seed: int | None
    seat_names: list[str]  # policy names in seat order
    outcome: str  # "gin" | "knock" | "undercut" | "draw"
    winner_seat: int | None
    turns: int
    loser_deadwood: int | None
    seconds: float


@dataclass
class TournamentResult:
    variant: str
    policy_names: list[str]
    games_per_pair: int
    paired: bool
    matches: list[MatchOutcome] = field(default_factory=list)
    telemetry: dict[str, PolicyTelemetry] = field(default_factory=dict)

    # ---------- aggregates ----------

    def wins_between(self, a: str, b: str) -> tuple[int, int, int]:
        """Return (a_wins, b_wins, draws) across all matches where {a, b} played."""
        a_wins = b_wins = draws = 0
        for m in self.matches:
            if a not in m.seat_names or b not in m.seat_names:
                continue
            if m.winner_seat is None:
                draws += 1
                continue
            winner_name = m.seat_names[m.winner_seat]
            if winner_name == a:
                a_wins += 1
            elif winner_name == b:
                b_wins += 1
            else:
                draws += 1
        return a_wins, b_wins, draws

    def crossplay_matrix(self, level: float = 0.95) -> dict[tuple[str, str], ProportionCI]:
        """Row-vs-column win rate with Wilson CI; diagonal is None."""
        out: dict[tuple[str, str], ProportionCI] = {}
        for a, b in combinations(self.policy_names, 2):
            a_wins, b_wins, _ = self.wins_between(a, b)
            trials = a_wins + b_wins  # exclude draws from head-to-head rate
            out[(a, b)] = wilson_ci(a_wins, trials, level=level)
            out[(b, a)] = wilson_ci(b_wins, trials, level=level)
        return out

    def outcome_distribution(self) -> Counter[str]:
        return Counter(m.outcome for m in self.matches)

    def turns_ci(self, level: float = 0.95) -> BootstrapCI:
        return bootstrap_ci(
            [float(m.turns) for m in self.matches],
            level=level,
            n_resamples=2000,
            rng=random.Random(0),
        )

    def deadwood_ci(self, level: float = 0.95) -> BootstrapCI:
        samples = [float(m.loser_deadwood) for m in self.matches if m.loser_deadwood is not None]
        return bootstrap_ci(
            samples, level=level, n_resamples=2000, rng=random.Random(0)
        )


class Tournament:
    """Round-robin runner. All 2-combinations of the registered policies play
    ``games_per_pair`` matches. For a 2-policy tournament this reduces to the
    old pairwise benchmark; for N ≥ 3 you get a cross-play matrix."""

    def __init__(
        self,
        *,
        game_cls: type[GinRummyGame],
        entries: Sequence[PolicyEntry],
        games_per_pair: int,
        seed: int | None = None,
        num_players: int | None = None,
        paired: bool = False,
        **game_kwargs,
    ) -> None:
        if num_players is None:
            num_players = 2
        if num_players != 2:
            # Cross-play matrix only makes sense pairwise; N>2 seats-with-N-agents
            # is a separate protocol.
            raise ValueError(
                "Tournament currently assumes 2-seat games; multi-seat is future work."
            )
        if len(entries) < 2:
            raise ValueError("Need at least two policies to run a tournament")
        if paired and games_per_pair % 2:
            raise ValueError("Paired mode requires an even games_per_pair")

        self.game_cls = game_cls
        self.entries = list(entries)
        self.games_per_pair = games_per_pair
        self.seed = seed
        self.num_players = num_players
        self.paired = paired
        self.game_kwargs = game_kwargs

    def run(self) -> TournamentResult:
        names = [e.name for e in self.entries]
        result = TournamentResult(
            variant=self.game_cls.__name__,
            policy_names=names,
            games_per_pair=self.games_per_pair,
            paired=self.paired,
        )
        for name in names:
            result.telemetry[name] = PolicyTelemetry(name=name)

        seed_counter = 0
        for a, b in combinations(self.entries, 2):
            pairs = self.games_per_pair // 2 if self.paired else self.games_per_pair
            for _ in range(pairs):
                base_seed = None if self.seed is None else self.seed + seed_counter
                seed_counter += 1
                self._play_one(result, a, b, base_seed, swap=False)
                if self.paired:
                    self._play_one(result, a, b, base_seed, swap=True)

        return result

    def _play_one(
        self,
        result: TournamentResult,
        a: PolicyEntry,
        b: PolicyEntry,
        game_seed: int | None,
        *,
        swap: bool,
    ) -> None:
        rng = random.Random(game_seed)
        first, second = (b, a) if swap else (a, b)
        wrapped: list[TelemetryPolicy] = []
        seat_names: list[str] = []
        for entry in (first, second):
            inner = entry.build(rng)
            wrapped.append(TelemetryPolicy(inner, result.telemetry[entry.name]))
            seat_names.append(entry.name)

        t0 = time.perf_counter()
        game = self.game_cls(
            self.num_players,
            policies=wrapped,
            seed=game_seed,
            **self.game_kwargs,
        )
        game_result = game.play()
        elapsed = time.perf_counter() - t0

        loser_dw: int | None = None
        if game_result.winner_id is not None:
            for p in game.players:
                if p.player_id == game_result.winner_id:
                    continue
                _, _, dv = optimal_decomposition(p.hand)
                loser_dw = dv
                break

        result.matches.append(
            MatchOutcome(
                seed=game_seed,
                seat_names=seat_names,
                outcome=game_result.outcome,
                winner_seat=game_result.winner_id,
                turns=game_result.turns,
                loser_deadwood=loser_dw,
                seconds=elapsed,
            )
        )
