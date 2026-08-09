"""CFR solvers for the mini-gin toy variant.

This subpackage provides the first public CFR solve of a rummy-family
variant, as motivated in ``minigin.py``. See ``cfr.py`` for the trainer,
``best_response.py`` for exploitability, and ``minigin.py`` for the game.
"""

from gin_rummy.solvers.best_response import (
    best_response_value,
    expected_value,
    exploitability,
)
from gin_rummy.solvers.cfr import (
    AverageStrategy,
    ExternalSamplingMCCFR,
    train,
    uniform_strategy,
)
from gin_rummy.solvers.minigin import (
    Action,
    DRAW_DECK,
    DRAW_DISCARD,
    HAND_SIZE,
    MAX_TURNS,
    MiniGinState,
    NUM_CARDS,
    Phase,
    apply,
    information_set,
    initial_state,
    is_gin_hand,
    is_terminal,
    legal_actions,
    make_discard,
    returns,
    sample_deal,
)

__all__ = [
    "Action",
    "AverageStrategy",
    "DRAW_DECK",
    "DRAW_DISCARD",
    "ExternalSamplingMCCFR",
    "HAND_SIZE",
    "MAX_TURNS",
    "MiniGinState",
    "NUM_CARDS",
    "Phase",
    "apply",
    "best_response_value",
    "expected_value",
    "exploitability",
    "information_set",
    "initial_state",
    "is_gin_hand",
    "is_terminal",
    "legal_actions",
    "make_discard",
    "returns",
    "sample_deal",
    "train",
    "uniform_strategy",
]
