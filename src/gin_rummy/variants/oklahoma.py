"""Oklahoma Gin — the first upcard's rank sets the knock threshold.

Ace ⇒ only gin ends the hand (knock_limit=0); face cards ⇒ 10;
otherwise the pip value.
"""

from __future__ import annotations

from gin_rummy.game import GinRummyGame


def _limit_from_upcard(rank: str) -> int:
    if rank == "A":
        return 0
    if rank in ("J", "Q", "K"):
        return 10
    return int(rank)


class OklahomaGin(GinRummyGame):
    def __init__(self, num_players: int = 2, **kwargs) -> None:
        kwargs.pop("knock_limit", None)
        super().__init__(num_players, knock_limit=10, **kwargs)

    def _deal(self) -> None:
        super()._deal()
        self.knock_limit = _limit_from_upcard(self.discard_pile[0].rank)
