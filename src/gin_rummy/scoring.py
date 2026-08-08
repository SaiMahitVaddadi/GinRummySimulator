"""Scoring rule bundle. Constants live here so variants can override cleanly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoringRules:
    knock_bonus: int = 10
    gin_bonus: int = 25
    undercut_bonus: int = 20
