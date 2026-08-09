"""Hand-crafted heuristic policies — useful reference opponents.

Two flavours ship here:

* :class:`ComposableHeuristicPolicy` — a general policy that dispatches
  each of the three heuristic decisions (draw source, discard choice,
  knock timing) to an independently swappable rule. This is the shape
  ablation-study code should target.
* :class:`GreedyKnockPolicy` — the historical monolith. Now a thin
  backwards-compatible subclass that plugs the original defaults into
  ``ComposableHeuristicPolicy``. Existing call sites keep working
  unchanged.

The original defaults were:

  * Draw from the discard pile only when it strictly reduces optimal
    deadwood (i.e. slots into a meld).
  * Discard the highest-value deadwood card.
  * Knock as soon as it's legal.
"""

from __future__ import annotations

import random

from gin_rummy.cards import Card
from gin_rummy.policies.heuristic_components import (
    DiscardRule,
    DrawRule,
    KnockRule,
    discard_highest_deadwood,
    draw_if_reduces_deadwood,
    knock_asap,
)
from gin_rummy.policy import DrawSource, Observation


class ComposableHeuristicPolicy:
    """A Policy that delegates each decision to an injectable rule.

    Parameters
    ----------
    draw_rule : DrawRule
        ``(obs, rng) -> "deck"|"discard"``.
    discard_rule : DiscardRule
        ``(obs, rng) -> Card``.
    knock_rule : KnockRule
        ``(obs, deadwood_value, rng) -> bool``.
    rng : random.Random, optional
        Shared RNG passed to every rule so seeded runs are reproducible.
    """

    def __init__(
        self,
        draw_rule: DrawRule,
        discard_rule: DiscardRule,
        knock_rule: KnockRule,
        rng: random.Random | None = None,
    ) -> None:
        self._draw_rule = draw_rule
        self._discard_rule = discard_rule
        self._knock_rule = knock_rule
        self._rng = rng if rng is not None else random.Random()

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        return self._draw_rule(obs, self._rng)

    def choose_discard(self, obs: Observation) -> Card:
        return self._discard_rule(obs, self._rng)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._knock_rule(obs, deadwood_value, self._rng)


class GreedyKnockPolicy(ComposableHeuristicPolicy):
    """Backwards-compatible name for the original heuristic triple.

    Composes ``draw_if_reduces_deadwood`` + ``discard_highest_deadwood`` +
    ``knock_asap``. Existing tests and benchmarks that import this class
    keep working unchanged.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(
            draw_rule=draw_if_reduces_deadwood,
            discard_rule=discard_highest_deadwood,
            knock_rule=knock_asap,
            rng=rng,
        )
