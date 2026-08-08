"""Rummy variants — each wraps or subclasses the base engine."""

from gin_rummy.variants.classic import ClassicGin
from gin_rummy.variants.hollywood import HollywoodGin, HollywoodResult
from gin_rummy.variants.indian import IndianRummy
from gin_rummy.variants.oklahoma import OklahomaGin

__all__ = [
    "ClassicGin",
    "HollywoodGin",
    "HollywoodResult",
    "IndianRummy",
    "OklahomaGin",
]
