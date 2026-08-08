"""Hollywood Gin — a series of hands with cumulative scoring.

Each hand runs a fresh ``GinRummyGame``; per-hand scores are summed to a
running total across the series. Winner = highest total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gin_rummy.game import GameResult, GinRummyGame


@dataclass
class HollywoodResult:
    hands: list[GameResult] = field(default_factory=list)
    winner_id: int | None = None
    totals: dict[str, int] = field(default_factory=dict)


class HollywoodGin:
    def __init__(
        self,
        num_players: int = 2,
        *,
        num_hands: int = 3,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        self.num_players = num_players
        self.num_hands = num_hands
        self.seed = seed
        self.kwargs = kwargs

    def play(self) -> HollywoodResult:
        totals = {f"Player {i + 1}": 0 for i in range(self.num_players)}
        hands: list[GameResult] = []
        for h in range(self.num_hands):
            hand_seed = None if self.seed is None else self.seed + h
            g = GinRummyGame(self.num_players, seed=hand_seed, **self.kwargs)
            r = g.play()
            hands.append(r)
            for name, s in r.scores.items():
                totals[name] += s

        best_score = max(totals.values())
        winners = [name for name, s in totals.items() if s == best_score]
        winner_id = (
            int(winners[0].split()[-1]) - 1 if best_score > 0 and len(winners) == 1 else None
        )
        return HollywoodResult(hands=hands, winner_id=winner_id, totals=totals)
