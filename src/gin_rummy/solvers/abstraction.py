"""Hand and public-state abstraction for tabular CFR on 10-card Classic Gin.

Full 2-player Classic Gin has an astronomical information-set count — a
private hand alone is ``C(52, 10) ~ 1.6e10`` values, before mixing in the
public discard history and deck cardinality. Even after standard perfect-
recall-lite compression (drop private action history, keep the current
hand + top-of-discard + deck size) the raw info-set space is far outside
tabular MCCFR's reach.

To make MCCFR tractable on Classic Gin we compose two lossy encodings:

* :class:`HandBucket` — a coarse fingerprint of the acting player's
  private hand (deadwood value bucket, longest run length, number of
  sets, top-card rank class).
* :class:`PublicBucket` — a coarse fingerprint of the public state
  visible to both players (top-of-discard rank class, deck-size bucket).

Combined with a binary draw action and 4-way discard-value bucketed
discards (see :func:`abstract_action`), the *product* of the bucket
spaces has ``6 * 11 * 4 * 4 * 5 * 4 = 21_120`` reachable info-set slots
in principle (many are structurally impossible). The design target is
:math:`\\le 30\\text{k}` reachable info-sets on random 100-game trajectories;
see :func:`estimate_infoset_count` and ``tests/test_gin_cfr.py`` for the
empirical check.

This is a **coarse** abstraction: any two hands that share the same
``HandBucket`` and see the same ``PublicBucket`` will play the same
strategy under the resulting solver. Finer abstractions (e.g. bucketing
by "hand potential" as in Ganzfried & Sandholm's *Potential-Aware
Automated Abstraction*, or opponent-aware discard classes) would recover
signal at the cost of a larger table.

References
----------
* Sandholm & Singh. *Lossy Stochastic Game Abstraction with Bounds.*
  EC 2012. (Bounds on regret from abstraction.)
* Ganzfried & Sandholm. *Potential-Aware Automated Abstraction of
  Sequential Games*. AAMAS 2012.
* Johanson et al. *Evaluating State-Space Abstractions in Extensive-Form
  Games.* AAMAS 2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition


# ---------------------------------------------------------------------------
# Deadwood bucketing
# ---------------------------------------------------------------------------


# 6 deadwood buckets covering [0, 1-5, 6-10, 11-20, 21-40, 41+].
DEADWOOD_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 40),
    (41, 10_000),
)


def deadwood_bucket(value: int) -> int:
    """Return the bucket index (0..5) for a deadwood point total."""
    for i, (lo, hi) in enumerate(DEADWOOD_BUCKETS):
        if lo <= value <= hi:
            return i
    return len(DEADWOOD_BUCKETS) - 1  # defensive


# ---------------------------------------------------------------------------
# Top-card / rank classes
# ---------------------------------------------------------------------------


# Four rank classes: A / 2-5 / 6-10 / J-Q-K. Used both for the top card of
# the acting player's hand and for the top-of-discard.
RANK_CLASS_NAMES: tuple[str, ...] = ("A", "2-5", "6-10", "JQK")
_RANK_CLASS: dict[str, int] = {
    "A": 0,
    "2": 1, "3": 1, "4": 1, "5": 1,
    "6": 2, "7": 2, "8": 2, "9": 2, "10": 2,
    "J": 3, "Q": 3, "K": 3,
    "JKR": 3,  # jokers ride with face cards for bucketing purposes only
}


def rank_class(card: Card) -> int:
    return _RANK_CLASS[card.rank]


# ---------------------------------------------------------------------------
# Deck-size bucketing
# ---------------------------------------------------------------------------


# 4 deck-size buckets: >30, 15-30, 5-15, <=5 (mirrors "early / mid /
# late / endgame" phases of a Classic Gin hand).
def deck_size_bucket(deck_size: int) -> int:
    if deck_size > 30:
        return 0
    if deck_size >= 15:
        return 1
    if deck_size >= 5:
        return 2
    return 3


DECK_SIZE_BUCKET_NAMES: tuple[str, ...] = (">30", "15-30", "5-15", "<=5")


# ---------------------------------------------------------------------------
# HandBucket
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandBucket:
    """Coarse, lossy encoding of a 10-card hand.

    Fields:
        deadwood_bucket: index into :data:`DEADWOOD_BUCKETS` (0..5).
        longest_run: 0, 3, 4, or 5+ (capped at 5).
        num_sets: number of size-3+ sets in the optimal decomposition, capped
            at 3.
        top_card_class: 0..3 (A / 2-5 / 6-10 / J-Q-K) for the highest-value
            unmelded card. If the hand has no deadwood we use 0.
    """

    deadwood_bucket: int
    longest_run: int
    num_sets: int
    top_card_class: int

    def key(self) -> tuple[int, int, int, int]:
        return (self.deadwood_bucket, self.longest_run, self.num_sets, self.top_card_class)

    def __str__(self) -> str:
        return (
            f"HB(dw={DEADWOOD_BUCKETS[self.deadwood_bucket]},"
            f"run={self.longest_run},sets={self.num_sets},"
            f"top={RANK_CLASS_NAMES[self.top_card_class]})"
        )


def _longest_run_length(melds: Sequence[Sequence[Card]]) -> int:
    """Return the longest run (consecutive same-suit) length among ``melds``.

    A meld is a "run" iff it has 3+ cards, all same suit, contiguous by
    order. Everything else (sets) contributes 0.
    """
    best = 0
    for meld in melds:
        if len(meld) < 3:
            continue
        suits = {c.suit for c in meld}
        if len(suits) != 1:
            continue
        orders = sorted(c.order for c in meld)
        if all(orders[i + 1] == orders[i] + 1 for i in range(len(orders) - 1)):
            best = max(best, len(meld))
    return min(best, 5)


def _num_sets(melds: Sequence[Sequence[Card]]) -> int:
    """Count melds that are sets (3+ same rank). Capped at 3."""
    n = 0
    for meld in melds:
        if len(meld) < 3:
            continue
        if len({c.rank for c in meld}) == 1:
            n += 1
    return min(n, 3)


def _top_deadwood_class(deadwood: Sequence[Card]) -> int:
    if not deadwood:
        return 0
    top = max(deadwood, key=lambda c: c.value)
    return rank_class(top)


@lru_cache(maxsize=200_000)
def _hand_bucket_cached(hand_tuple: tuple[Card, ...]) -> HandBucket:
    melds, deadwood_cards, dv = optimal_decomposition(list(hand_tuple))
    return HandBucket(
        deadwood_bucket=deadwood_bucket(dv),
        longest_run=_longest_run_length(melds),
        num_sets=_num_sets(melds),
        top_card_class=_top_deadwood_class(deadwood_cards),
    )


def hand_bucket(hand: Iterable[Card]) -> HandBucket:
    """Compute the :class:`HandBucket` for a Classic Gin hand (cached)."""
    return _hand_bucket_cached(tuple(hand))


# ---------------------------------------------------------------------------
# PublicBucket
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicBucket:
    """Coarse public state observable to both players.

    Fields:
        top_discard_class: 0..3 rank class of the top of the discard pile;
            -1 if the pile is empty.
        deck_size_bucket: index in {0, 1, 2, 3} from
            :func:`deck_size_bucket`.
    """

    top_discard_class: int
    deck_size_bucket: int

    def key(self) -> tuple[int, int]:
        return (self.top_discard_class, self.deck_size_bucket)

    def __str__(self) -> str:
        top = "empty" if self.top_discard_class < 0 else RANK_CLASS_NAMES[self.top_discard_class]
        return f"PB(top={top},deck={DECK_SIZE_BUCKET_NAMES[self.deck_size_bucket]})"


def public_bucket(top_discard: Card | None, deck_size: int) -> PublicBucket:
    return PublicBucket(
        top_discard_class=rank_class(top_discard) if top_discard is not None else -1,
        deck_size_bucket=deck_size_bucket(deck_size),
    )


# ---------------------------------------------------------------------------
# Abstract actions
# ---------------------------------------------------------------------------


# Draw: two abstract labels.
DRAW_DECK = ("draw", 0)
DRAW_DISCARD = ("draw", 1)


# Discard-value classes: 0=Ace (1pt), 1=low (2-5), 2=mid (6-9), 3=high
# (10/J/Q/K, all worth 10 pts). Four classes as required.
def _discard_value_class_for_value(value: int, rank: str) -> int:
    if rank == "A" or value == 1:
        return 0
    if value <= 5:
        return 1
    if value <= 9:
        return 2
    return 3  # 10, J, Q, K (all worth 10)


def discard_value_class(card: Card) -> int:
    return _discard_value_class_for_value(card.value, card.rank)


DISCARD_CLASS_NAMES: tuple[str, ...] = ("A", "2-5", "6-9", "10JQK")


def make_discard(cls: int) -> tuple[str, int]:
    return ("discard", cls)


def abstract_action(action: tuple[str, int | Card]) -> tuple[str, int]:
    """Coarsen a concrete action into an abstract-action key.

    ``action`` may be:
        * ``("draw", 0)`` / ``("draw", 1)`` — pass through.
        * ``("discard", <Card>)`` — bucketed to its value class.

    Returns a hashable tuple ``(kind, int)``.
    """
    kind, payload = action
    if kind == "draw":
        return ("draw", int(payload))  # 0 or 1
    if kind == "discard":
        if isinstance(payload, Card):
            return ("discard", discard_value_class(payload))
        return ("discard", int(payload))
    raise ValueError(f"Unknown action kind: {kind!r}")


# ---------------------------------------------------------------------------
# Info-set count estimation (sanity check helper)
# ---------------------------------------------------------------------------


def estimate_infoset_count(num_games: int = 100, seed: int = 0) -> int:
    """Run ``num_games`` random-play rollouts and return the number of
    distinct abstracted info-set keys encountered.

    Kept here so tests and experiments share one honest counter.
    """
    # Local import: avoid a circular import at module load. ``gin_cfr``
    # imports :mod:`abstraction` (this module).
    from gin_rummy.solvers.gin_cfr import ClassicGinEngine  # noqa: WPS433

    engine = ClassicGinEngine()
    seen: set[str] = set()
    import random as _random
    rng = _random.Random(seed)
    for _ in range(num_games):
        state = engine.sample_deal(rng)
        steps = 0
        while not engine.is_terminal(state) and steps < 400:
            cur = engine.current_player(state)
            seen.add(engine.information_set(state, cur))
            legal = engine.legal_actions(state)
            state = engine.apply(state, rng.choice(legal))
            steps += 1
    return len(seen)


__all__ = [
    "DEADWOOD_BUCKETS",
    "DECK_SIZE_BUCKET_NAMES",
    "DISCARD_CLASS_NAMES",
    "DRAW_DECK",
    "DRAW_DISCARD",
    "HandBucket",
    "PublicBucket",
    "RANK_CLASS_NAMES",
    "abstract_action",
    "deadwood_bucket",
    "deck_size_bucket",
    "discard_value_class",
    "estimate_infoset_count",
    "hand_bucket",
    "make_discard",
    "public_bucket",
    "rank_class",
]
