"""A simple hand-crafted heuristic — useful as a reference opponent.

Strategy:
  * Draw from the discard pile only when the top card would land in a meld
    or reduce deadwood.
  * Discard the highest-value card that doesn't belong to any meld.
  * Knock as soon as it's legal (matches the base engine's assumption).

This is deliberately unsophisticated. The point is to have *something*
that beats ``RandomPolicy`` so ``bench`` runs are informative.
"""

from __future__ import annotations

import random

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation


class GreedyKnockPolicy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        if obs.top_discard is None:
            return "deck"
        _, _, current_dw = optimal_decomposition(list(obs.hand))
        hypothetical = list(obs.hand) + [obs.top_discard]
        _, _, next_dw = optimal_decomposition(hypothetical)
        # Adding a card can only decrease deadwood if it slots into a meld,
        # so any strict decrease means the discard is worth taking.
        return "discard" if next_dw < current_dw else "deck"

    def choose_discard(self, obs: Observation) -> Card:
        _, deadwood, _ = optimal_decomposition(list(obs.hand))
        if not deadwood:
            # Every card is melded — arbitrary pick (rare edge case).
            return self._rng.choice(obs.hand)
        return max(deadwood, key=lambda c: c.value)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return True
