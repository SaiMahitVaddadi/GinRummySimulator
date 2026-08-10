"""Deck-depletion Markov process.

Models the *rank-class distribution* of the remaining deck as a discrete
process indexed by turn. The state we track is which rank class the
next-drawn card belongs to. We group the 13 ranks into 4 "rank classes"
so the state space is small and the empirical vs. predicted comparison
in ``experiments/markov_demo.py`` has enough samples per bucket to be
readable:

* ``low``  — A, 2, 3, 4 (low-value; useful in low-end runs and sets)
* ``mid``  — 5, 6, 7      (mid-value)
* ``high`` — 8, 9, 10     (high-value pip cards)
* ``face`` — J, Q, K      (10-point face cards)

Independence approximation
--------------------------
Under a *without-replacement* draw process the exact top-card
distribution at turn t depends on the empirical composition of what has
already been drawn — a hypergeometric with 4-way marginals. That's
tractable but blows up the state space and, more importantly, makes the
Markov abstraction a lie: the "state" would need to be the full
multivariate composition, which defeats the point of a compact model.

We instead adopt an *independent-draws approximation*: each drawn card
is i.i.d. from the initial rank-class distribution. This is exact for
turn 1 and asymptotically off by a small amount as the shoe depletes
(the shoe's residual composition drifts from the initial proportions
only by the sampling noise of what has been drawn). The
``markov_demo`` script empirically checks the approximation error at
turn 30 across 500 seeded games; on a 1-deck 2-player game the
approximation is typically within a few percentage points per class.

For high-fidelity simulation the ``.simulate`` method draws exactly
from the current shoe without replacement, so callers who need the
ground truth (rather than the approximation) still have it here.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from gin_rummy.cards import RANKS


RANK_CLASSES: dict[str, str] = {
    "A": "low", "2": "low", "3": "low", "4": "low",
    "5": "mid", "6": "mid", "7": "mid",
    "8": "high", "9": "high", "10": "high",
    "J": "face", "Q": "face", "K": "face",
}
CLASS_NAMES: tuple[str, ...] = ("low", "mid", "high", "face")


def _class_count(num_decks: int, cls: str) -> int:
    """Cards of class ``cls`` per multi-deck shoe (4 suits per rank)."""
    ranks_in_class = [r for r in RANKS if RANK_CLASSES[r] == cls]
    return len(ranks_in_class) * 4 * num_decks


@dataclass
class DeckDepletionModel:
    """Markov abstraction of a well-shuffled shoe under the independent-draws
    approximation.

    Parameters
    ----------
    num_decks : int
        Number of stacked 52-card decks in the shoe.
    """

    num_decks: int = 1

    def __post_init__(self) -> None:
        if self.num_decks < 1:
            raise ValueError("num_decks must be >= 1")
        self._total = 52 * self.num_decks
        self._counts: dict[str, int] = {
            cls: _class_count(self.num_decks, cls) for cls in CLASS_NAMES
        }
        # Sanity: sum of class counts equals full shoe.
        assert sum(self._counts.values()) == self._total

    # ---- initial marginal distribution ----

    def initial_distribution(self) -> dict[str, float]:
        """Return P(top card in class c) at turn 0.

        Under the independence approximation this same distribution holds
        at every turn.
        """
        return {c: self._counts[c] / self._total for c in CLASS_NAMES}

    def probability_of_class_at_turn(self, class_name: str, turn: int) -> float:
        """Independent-draws approximation: P(class == class_name) is
        time-invariant and equals the initial marginal.

        Parameters
        ----------
        class_name : str
            One of ``CLASS_NAMES``.
        turn : int
            0-indexed draw number. Kept for API symmetry — the returned
            value is turn-independent under this approximation.

        Notes
        -----
        The exact (hypergeometric) distribution at turn ``t`` conditional
        on what has been drawn will drift from this constant; the size
        of the drift is bounded above by the sampling variance of the
        empirical draw composition, which for a 52-card shoe is small
        for t < ~30. The ``experiments/markov_demo.py`` script quantifies
        this drift empirically.
        """
        if class_name not in self._counts:
            raise KeyError(
                f"Unknown class {class_name!r}; expected one of {CLASS_NAMES}"
            )
        if turn < 0:
            raise ValueError("turn must be >= 0")
        return self._counts[class_name] / self._total

    # ---- ground-truth Monte-Carlo depletion ----

    def simulate(self, n_turns: int, rng: random.Random) -> list[str]:
        """Draw ``n_turns`` cards *without replacement* and return the
        sequence of rank-class labels.

        This is the ground truth against which
        :meth:`probability_of_class_at_turn` (the approximation) can be
        empirically checked.
        """
        if n_turns < 0:
            raise ValueError("n_turns must be >= 0")
        shoe: list[str] = []
        for r in RANKS:
            cls = RANK_CLASSES[r]
            shoe.extend([cls] * (4 * self.num_decks))
        rng.shuffle(shoe)
        n = min(n_turns, len(shoe))
        return shoe[:n]

    # ---- expected time to see a rank ----

    def expected_draws_to_rank(self, rank: str) -> float:
        """Expected number of draws (from a full shoe) until the first card
        of the given ``rank`` appears.

        Uses the exact expectation for sampling without replacement:
        ``E[T] = (N + 1) / (k + 1)`` where ``N`` is the shoe size and
        ``k`` the number of cards of the target rank. Derivation: T is
        the position of the first "success" in a uniformly random
        ordering, whose expectation is a standard result. Under the
        independent-draws approximation this simplifies to ``N / k``
        (geometric with success prob k/N); we return the exact
        without-replacement value here because it costs the same to
        compute and is more useful.
        """
        if rank not in RANK_CLASSES:
            raise KeyError(f"Unknown rank {rank!r}")
        k = 4 * self.num_decks  # copies of that rank in the shoe
        N = self._total
        return (N + 1) / (k + 1)


__all__ = [
    "CLASS_NAMES",
    "DeckDepletionModel",
    "RANK_CLASSES",
]
