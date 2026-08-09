"""Orthogonal heuristic components for ablation studies.

The monolithic ``GreedyKnockPolicy`` bundles three independent decisions:

  1. **Draw source** — deck vs. discard pile.
  2. **Discard choice** — which card to shed.
  3. **Knock timing** — knock now, or hold for gin.

This module factors each dimension into small named callables. A
``ComposableHeuristicPolicy`` (see :mod:`gin_rummy.policies.heuristic`)
consumes one rule from each dimension, letting an ablation framework
enumerate factor levels across the cartesian product.

All rules are plain callables — no shared state — so they compose freely
and are trivially picklable/serialisable for parallel sweeps.
"""

from __future__ import annotations

import random
from typing import Callable

from gin_rummy.cards import RANK_ORDER, Card
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation

# ---------- rule signatures ----------

DrawRule = Callable[[Observation, random.Random], DrawSource]
DiscardRule = Callable[[Observation, random.Random], Card]
KnockRule = Callable[[Observation, int, random.Random], bool]


# ============================================================
# Draw-source rules: (obs, rng) -> "deck" | "discard"
# ============================================================


def draw_random(obs: Observation, rng: random.Random) -> DrawSource:
    """Uniform coin flip. Falls back to deck if the discard pile is empty."""
    if obs.top_discard is None:
        return "deck"
    return "discard" if rng.random() < 0.5 else "deck"


def draw_if_reduces_deadwood(obs: Observation, rng: random.Random) -> DrawSource:
    """Take the discard iff adding it strictly reduces optimal deadwood.

    This is the default from the original ``GreedyKnockPolicy``: since
    adding a card can only lower deadwood if it slots into a meld, a strict
    drop confirms the discard is worth taking.
    """
    if obs.top_discard is None:
        return "deck"
    _, _, current_dw = optimal_decomposition(list(obs.hand))
    hypothetical = list(obs.hand) + [obs.top_discard]
    _, _, next_dw = optimal_decomposition(hypothetical)
    return "discard" if next_dw < current_dw else "deck"


def draw_always_deck(obs: Observation, rng: random.Random) -> DrawSource:
    """Never take from the discard pile. Deliberately conservative extreme."""
    return "deck"


# ============================================================
# Discard rules: (obs, rng) -> Card
# ============================================================


def discard_random(obs: Observation, rng: random.Random) -> Card:
    """Uniform-random pick over the current hand."""
    return rng.choice(obs.hand)


def discard_highest_deadwood(obs: Observation, rng: random.Random) -> Card:
    """Shed the highest-value card that isn't part of any optimal meld."""
    _, deadwood, _ = optimal_decomposition(list(obs.hand))
    if not deadwood:
        # Every card is melded — arbitrary pick (rare edge case).
        return rng.choice(obs.hand)
    return max(deadwood, key=lambda c: c.value)


def discard_lowest_deadwood(obs: Observation, rng: random.Random) -> Card:
    """Shed the lowest-value deadwood card.

    Deliberately bad — kept as a null baseline so ablation sweeps can
    distinguish "the rule matters" from "any rule works".
    """
    _, deadwood, _ = optimal_decomposition(list(obs.hand))
    if not deadwood:
        return rng.choice(obs.hand)
    return min(deadwood, key=lambda c: c.value)


def _adjacent_to(card: Card, target: Card) -> bool:
    """Two cards are 'adjacent' if same rank, or same suit and |rank_order| <= 1."""
    if card.rank == target.rank:
        return True
    if card.suit == target.suit:
        a, b = RANK_ORDER.get(card.rank, -99), RANK_ORDER.get(target.rank, 99)
        if a >= 0 and b >= 0 and abs(a - b) <= 1:
            return True
    return False


def discard_safest(obs: Observation, rng: random.Random) -> Card:
    """Highest-value deadwood card that is NOT adjacent to the top discard.

    The engine's ``Observation`` does not expose full discard-pile history,
    so we use the current ``top_discard`` as a proxy for the opponent's
    most recent activity. If it's ``None``, fall back to
    :func:`discard_highest_deadwood`.
    """
    if obs.top_discard is None:
        return discard_highest_deadwood(obs, rng)

    _, deadwood, _ = optimal_decomposition(list(obs.hand))
    if not deadwood:
        return rng.choice(obs.hand)

    safe = [c for c in deadwood if not _adjacent_to(c, obs.top_discard)]
    pool = safe if safe else deadwood
    return max(pool, key=lambda c: c.value)


# ============================================================
# Knock rules: (obs, deadwood_value, rng) -> bool
# ============================================================


def knock_never(obs: Observation, deadwood_value: int, rng: random.Random) -> bool:
    """Never voluntarily knock; hold out for gin every time."""
    return False


def knock_asap(obs: Observation, deadwood_value: int, rng: random.Random) -> bool:
    """Knock the moment it's legally allowed."""
    return True


def knock_below(threshold: int) -> KnockRule:
    """Factory: knock only when deadwood is at or below ``threshold``.

    Useful for reducing undercut risk — e.g. ``knock_below(5)`` waits until
    the hand is compact enough that being undercut is unlikely.
    """

    def _rule(obs: Observation, deadwood_value: int, rng: random.Random) -> bool:
        return deadwood_value <= threshold

    _rule.__name__ = f"knock_below_{threshold}"
    return _rule


def knock_after_turn(turn: int) -> KnockRule:
    """Factory: hold gin attempts early, knock once the game has run past ``turn``.

    Models a mixed strategy: seek gin bonus in the opening, cash in later.
    """

    def _rule(obs: Observation, deadwood_value: int, rng: random.Random) -> bool:
        return obs.turn_number >= turn

    _rule.__name__ = f"knock_after_turn_{turn}"
    return _rule


__all__ = [
    "DrawRule",
    "DiscardRule",
    "KnockRule",
    "draw_random",
    "draw_if_reduces_deadwood",
    "draw_always_deck",
    "discard_random",
    "discard_highest_deadwood",
    "discard_lowest_deadwood",
    "discard_safest",
    "knock_never",
    "knock_asap",
    "knock_below",
    "knock_after_turn",
]
