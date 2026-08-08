"""Player state: name, id, and a hand of cards."""

from __future__ import annotations

from dataclasses import dataclass, field

from gin_rummy.cards import Card


@dataclass
class Player:
    name: str
    player_id: int
    hand: list[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        self.hand.append(card)

    def remove(self, card: Card) -> None:
        self.hand.remove(card)

    def __str__(self) -> str:
        return f"{self.name}: {' '.join(str(c) for c in self.hand)}"
