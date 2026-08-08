"""Classic Gin Rummy — 10-card hand, knock at ≤10 deadwood, gin bonus 25."""

from __future__ import annotations

from gin_rummy.game import GinRummyGame


class ClassicGin(GinRummyGame):
    """Uses the base engine's defaults verbatim."""
