"""External-sampling Monte Carlo CFR for mini-gin.

We implement the *external-sampling* variant of Monte Carlo Counterfactual
Regret Minimization from Lanctot, Waugh, Zinkevich & Bowling
(NeurIPS 2009). At each iteration a single ``traverser`` is chosen: for the
traverser we enumerate *all* legal actions and compute counterfactual
values; for the opponent and chance we sample a single action according to
the current strategy / uniform chance distribution. Regrets and average
strategy accumulators are updated only for the traverser's information sets.

Regret matching (Hart & Mas-Colell 2000) converts cumulative regret into a
per-node action distribution:

    sigma_t(a) = R^+_t(a) / sum_a R^+_t(a)   (uniform if all zero)

The **average** strategy across iterations, not the current one, converges
to a Nash equilibrium of the (abstracted) two-player zero-sum game.

References
----------
* Zinkevich et al. *Regret Minimization in Games with Incomplete
  Information.* NeurIPS 2007. (Original CFR.)
* Lanctot, Waugh, Zinkevich, Bowling. *Monte Carlo Sampling for Regret
  Minimization in Extensive Games.* NeurIPS 2009. (External sampling.)
* Hart & Mas-Colell. *A Simple Adaptive Procedure Leading to Correlated
  Equilibrium.* Econometrica 2000. (Regret matching.)
* OpenSpiel implementation reference:
  https://openspiel.readthedocs.io/en/latest/algorithms.html#cfr-family
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from gin_rummy.solvers.minigin import (
    Action,
    MiniGinState,
    Phase,
    apply,
    information_set,
    initial_state,
    is_terminal,
    legal_actions,
    returns,
    sample_deal,
)


# ---------------------------------------------------------------------------
# Strategy container
# ---------------------------------------------------------------------------


@dataclass
class AverageStrategy:
    """Average strategy across all training iterations.

    ``policy[info_set]`` maps to a dict[Action -> probability].
    Missing keys default to uniform-random over the queried legal actions.
    """

    policy: dict[str, dict[Action, float]] = field(default_factory=dict)

    def action_probs(self, info_set: str, legal: list[Action]) -> dict[Action, float]:
        dist = self.policy.get(info_set)
        if not dist:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        # Restrict to legal and renormalize.
        masked = {a: dist.get(a, 0.0) for a in legal}
        s = sum(masked.values())
        if s <= 0:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: v / s for a, v in masked.items()}

    def num_info_sets(self) -> int:
        return len(self.policy)


# ---------------------------------------------------------------------------
# MCCFR trainer
# ---------------------------------------------------------------------------


class ExternalSamplingMCCFR:
    """Regret & strategy tables for external-sampling MCCFR on mini-gin.

    Only the tables and the recursive traversal live here; the game logic is
    entirely in ``minigin``.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        # info_set -> {action: cumulative regret}
        self.regrets: dict[str, dict[Action, float]] = {}
        # info_set -> {action: cumulative strategy weight}
        self.strategy_sum: dict[str, dict[Action, float]] = {}

    # ------------------------------------------------------------------
    # Regret matching
    # ------------------------------------------------------------------

    def _current_strategy(self, info_set: str, legal: list[Action]) -> dict[Action, float]:
        table = self.regrets.setdefault(info_set, {})
        pos = {a: max(table.get(a, 0.0), 0.0) for a in legal}
        s = sum(pos.values())
        if s <= 0:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: v / s for a, v in pos.items()}

    # ------------------------------------------------------------------
    # Main traversal
    # ------------------------------------------------------------------

    def _traverse(self, state: MiniGinState, traverser: int) -> float:
        """Return the counterfactual value for ``traverser`` from ``state``.

        Implements external sampling: the traverser branches on all actions;
        opponent and chance nodes sample one action each.
        """
        if is_terminal(state):
            return returns(state)[traverser]

        # Chance node: sample one deal outcome.
        if state.phase == Phase.DEAL:
            child = sample_deal(self.rng)
            return self._traverse(child, traverser)

        legal = legal_actions(state)
        info_set = information_set(state, state.current_player)
        strategy = self._current_strategy(info_set, legal)

        if state.current_player == traverser:
            # Enumerate all actions, accumulate weighted sum, update regrets.
            utils: dict[Action, float] = {}
            node_util = 0.0
            for a in legal:
                utils[a] = self._traverse(apply(state, a), traverser)
                node_util += strategy[a] * utils[a]
            regret_table = self.regrets.setdefault(info_set, {})
            strat_table = self.strategy_sum.setdefault(info_set, {})
            for a in legal:
                regret_table[a] = regret_table.get(a, 0.0) + (utils[a] - node_util)
                strat_table[a] = strat_table.get(a, 0.0) + strategy[a]
            return node_util
        else:
            # Opponent: sample one action from their current strategy.
            a = _sample_from(strategy, self.rng)
            # Accumulate opponent's strategy too so the average strategy has
            # entries for every reached info-set (helps best-response).
            strat_table = self.strategy_sum.setdefault(info_set, {})
            for act, prob in strategy.items():
                strat_table[act] = strat_table.get(act, 0.0) + prob
            return self._traverse(apply(state, a), traverser)

    # ------------------------------------------------------------------
    # Public train
    # ------------------------------------------------------------------

    def train(
        self,
        iterations: int,
        seed: int | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> AverageStrategy:
        if seed is not None:
            self.rng = random.Random(seed)
        for it in range(iterations):
            for traverser in (0, 1):
                root = initial_state()
                self._traverse(root, traverser)
            if progress is not None and (it + 1) % max(1, iterations // 10) == 0:
                progress(it + 1)
        return self.average_strategy()

    def average_strategy(self) -> AverageStrategy:
        avg = AverageStrategy()
        for iset, table in self.strategy_sum.items():
            total = sum(table.values())
            if total <= 0:
                avg.policy[iset] = {a: 1.0 / len(table) for a in table}
            else:
                avg.policy[iset] = {a: v / total for a, v in table.items()}
        return avg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_from(dist: dict[Action, float], rng: random.Random) -> Action:
    r = rng.random()
    cum = 0.0
    last: Action | None = None
    for a, p in dist.items():
        cum += p
        last = a
        if r <= cum:
            return a
    assert last is not None
    return last


def uniform_strategy(iterations: int = 0, seed: int = 0) -> AverageStrategy:
    """Return an empty AverageStrategy that defaults to uniform-random."""
    return AverageStrategy()


def train(iterations: int, seed: int = 0) -> AverageStrategy:
    """Convenience wrapper: build a solver, train, return average strategy."""
    solver = ExternalSamplingMCCFR(seed=seed)
    return solver.train(iterations, seed=seed)


__all__ = [
    "AverageStrategy",
    "ExternalSamplingMCCFR",
    "train",
    "uniform_strategy",
]
