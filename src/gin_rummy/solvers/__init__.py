"""CFR solvers for rummy-family variants.

* ``minigin.py`` — a 12-card mini-gin toy variant used for the exact
  tabular CFR solve.
* ``cfr.py`` — the shared MCCFR trainer (external + outcome sampling).
* ``best_response.py`` — sampled exploitability for mini-gin.
* ``abstraction.py`` — hand/public-state abstraction for 10-card Classic
  Gin.
* ``gin_cfr.py`` — MCCFR on abstracted Classic Gin.
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
