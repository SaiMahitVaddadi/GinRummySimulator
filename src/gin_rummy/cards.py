"""Cards and decks. Standard 52-card deck, extensible to multi-deck."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

SUITS: tuple[str, ...] = ("♠", "♥", "♦", "♣")  # ♠ ♥ ♦ ♣
RANKS: tuple[str, ...] = (
    "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
)
RANK_VALUE: dict[str, int] = {
    r: 1 if r == "A" else 10 if r in ("J", "Q", "K") else int(r) for r in RANKS
}
RANK_VALUE["JKR"] = 0  # Jokers carry no penalty; games that use them handle wildcards.
RANK_ORDER: dict[str, int] = {r: i for i, r in enumerate(RANKS)}
RANK_ORDER["JKR"] = -1  # Sentinel; jokers never participate in runs.


@dataclass(frozen=True, slots=True)
class Card:
    rank: str
    suit: str
    deck_id: int = 0

    @property
    def value(self) -> int:
        return RANK_VALUE[self.rank]

    @property
    def order(self) -> int:
        return RANK_ORDER[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


class Deck:
    """A shuffled stack of cards. Draws pop from the top."""

    __slots__ = ("num_decks", "_rng", "_cards")

    def __init__(self, num_decks: int = 1, *, rng: random.Random | None = None) -> None:
        if num_decks < 1:
            raise ValueError("num_decks must be >= 1")
        self.num_decks = num_decks
        self._rng = rng if rng is not None else random.Random()
        self._cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Rebuild the deck with num_decks × 52 cards and shuffle."""
        self._cards = [
            Card(rank, suit, deck_id)
            for deck_id in range(self.num_decks)
            for suit in SUITS
            for rank in RANKS
        ]
        self._rng.shuffle(self._cards)

    def draw(self) -> Card | None:
        return self._cards.pop() if self._cards else None

    def deal(self, n: int) -> list[Card]:
        out: list[Card] = []
        for _ in range(n):
            c = self.draw()
            if c is None:
                break
            out.append(c)
        return out

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    @staticmethod
    def optimal_deck_count(num_players: int) -> int:
        """Pick a deck count that leaves comfortable headroom for the hands."""
        if num_players <= 2:
            return 1
        if num_players <= 4:
            return 2
        if num_players <= 6:
            return 3
        return min(num_players // 2 + 1, 8)
