"""A fast, dependency-free multi-variant rummy simulator."""

from gin_rummy.cards import Card, Deck, RANKS, SUITS
from gin_rummy.meld import optimal_decomposition
from gin_rummy.player import Player
from gin_rummy.policy import Observation, Policy, RandomPolicy
from gin_rummy.game import GinRummyGame, GameResult, TurnRecord
from gin_rummy.scoring import ScoringRules
from gin_rummy.variants.classic import ClassicGin
from gin_rummy.variants.oklahoma import OklahomaGin
from gin_rummy.variants.hollywood import HollywoodGin, HollywoodResult
from gin_rummy.variants.indian import IndianRummy

__version__ = "0.2.0"

__all__ = [
    "Card",
    "ClassicGin",
    "Deck",
    "GameResult",
    "GinRummyGame",
    "HollywoodGin",
    "HollywoodResult",
    "IndianRummy",
    "Observation",
    "OklahomaGin",
    "Player",
    "Policy",
    "RANKS",
    "RandomPolicy",
    "SUITS",
    "ScoringRules",
    "TurnRecord",
    "optimal_decomposition",
]
