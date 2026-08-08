"""Policy protocol and baseline implementations.

A ``Policy`` is anything that, given an ``Observation``, decides where to
draw from, whether to knock (when legal), and what to discard. Games are
policy-agnostic — the same engine runs the built-in ``RandomPolicy``, a
future LLM-backed policy, an RL agent, or a hand-crafted heuristic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Protocol

from gin_rummy.cards import Card

DrawSource = Literal["deck", "discard"]


@dataclass(frozen=True)
class Observation:
    """The information visible to a player at decision time."""

    hand: tuple[Card, ...]
    top_discard: Card | None
    discard_pile_size: int
    deck_size: int
    turn_number: int
    knock_limit: int
    player_id: int
    num_players: int
    other_hand_sizes: tuple[int, ...]


class Policy(Protocol):
    def choose_draw_source(self, obs: Observation) -> DrawSource: ...
    def choose_discard(self, obs: Observation) -> Card: ...
    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool: ...


class RandomPolicy:
    """Uniform-random baseline. Reproduces the original simulator's behaviour."""

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        always_knock_when_legal: bool = True,
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._always_knock = always_knock_when_legal

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        if obs.top_discard is None:
            return "deck"
        return "discard" if self._rng.random() < 0.5 else "deck"

    def choose_discard(self, obs: Observation) -> Card:
        return self._rng.choice(obs.hand)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._always_knock
