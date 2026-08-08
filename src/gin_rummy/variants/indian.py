"""Indian Rummy — configurable hand size (3/7/10/13/15) on the shared engine.

**Caveat.** This is a *rummy-family* variant, not a rules-perfect Indian
Rummy implementation. Real Indian Rummy declaration rules (2 sequences
including one pure, jokers as wildcards, 13-card standard) are more
elaborate than what the base engine models. IRumAI (arXiv:2606.21975)
is the reference implementation for canonical Indian Rummy. What this
class *does* provide is a knob for hand size — useful for studying how
rummy dynamics change with hand length.
"""

from __future__ import annotations

from gin_rummy.game import GinRummyGame


class IndianRummy(GinRummyGame):
    SUPPORTED_HAND_SIZES = (3, 7, 10, 13, 15)

    def __init__(
        self,
        num_players: int = 4,
        *,
        hand_size: int = 13,
        knock_limit: int = 10,
        **kwargs,
    ) -> None:
        if hand_size not in self.SUPPORTED_HAND_SIZES:
            raise ValueError(
                f"hand_size must be one of {self.SUPPORTED_HAND_SIZES}, got {hand_size}"
            )
        super().__init__(
            num_players,
            hand_size=hand_size,
            knock_limit=knock_limit,
            **kwargs,
        )
