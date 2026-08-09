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
from typing import Any, Callable, Protocol

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
# Engine adapter protocol
# ---------------------------------------------------------------------------


class MCCFREngine(Protocol):
    """Minimal callable bundle the MCCFR trainer needs to drive any game.

    The default engine is mini-gin (see ``_minigin_engine`` below). Any other
    game (e.g. abstracted 10-card Classic Gin) supplies its own concrete
    implementations of these functions and its own ``DEAL_PHASE`` sentinel.

    Contract:
        * ``initial_state()`` -> root state (phase == DEAL_PHASE at the root).
        * ``sample_deal(rng)`` -> post-deal state (chance step).
        * ``is_terminal(state)`` -> bool.
        * ``returns(state)`` -> tuple[float, float] for terminal states.
        * ``legal_actions(state)`` -> non-empty list at any non-terminal
          non-chance decision node.
        * ``apply(state, action)`` -> next state.
        * ``information_set(state, player)`` -> hashable str key.
        * ``is_chance(state)`` -> True iff this is a chance node the trainer
          should resolve via ``sample_deal``.
        * ``current_player(state)`` -> int in {0, 1}.
    """

    def initial_state(self) -> Any: ...
    def sample_deal(self, rng: random.Random) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def returns(self, state: Any) -> tuple[float, float]: ...
    def legal_actions(self, state: Any) -> list[Any]: ...
    def apply(self, state: Any, action: Any) -> Any: ...
    def information_set(self, state: Any, player: int) -> str: ...
    def is_chance(self, state: Any) -> bool: ...
    def current_player(self, state: Any) -> int: ...


@dataclass(frozen=True)
class _CallableEngine:
    """Wrap a bag of callables into an ``MCCFREngine``. Used internally."""

    _initial_state: Callable[[], Any]
    _sample_deal: Callable[[random.Random], Any]
    _is_terminal: Callable[[Any], bool]
    _returns: Callable[[Any], tuple[float, float]]
    _legal_actions: Callable[[Any], list[Any]]
    _apply: Callable[[Any, Any], Any]
    _information_set: Callable[[Any, int], str]
    _is_chance: Callable[[Any], bool]
    _current_player: Callable[[Any], int]

    def initial_state(self) -> Any: return self._initial_state()
    def sample_deal(self, rng: random.Random) -> Any: return self._sample_deal(rng)
    def is_terminal(self, state: Any) -> bool: return self._is_terminal(state)
    def returns(self, state: Any) -> tuple[float, float]: return self._returns(state)
    def legal_actions(self, state: Any) -> list[Any]: return self._legal_actions(state)
    def apply(self, state: Any, action: Any) -> Any: return self._apply(state, action)
    def information_set(self, state: Any, player: int) -> str: return self._information_set(state, player)
    def is_chance(self, state: Any) -> bool: return self._is_chance(state)
    def current_player(self, state: Any) -> int: return self._current_player(state)


def _minigin_engine() -> MCCFREngine:
    return _CallableEngine(
        _initial_state=initial_state,
        _sample_deal=sample_deal,
        _is_terminal=is_terminal,
        _returns=returns,
        _legal_actions=legal_actions,
        _apply=apply,
        _information_set=information_set,
        _is_chance=lambda s: s.phase == Phase.DEAL,
        _current_player=lambda s: s.current_player,
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
    """Regret & strategy tables for external-sampling MCCFR.

    Only the tables and the recursive traversal live here; the game logic is
    delegated to an ``MCCFREngine`` (mini-gin by default). To run MCCFR on
    another 2p zero-sum imperfect-information game (e.g. abstracted 10-card
    Classic Gin), pass a custom engine.
    """

    def __init__(
        self,
        seed: int = 0,
        engine: MCCFREngine | None = None,
        sampling: str = "external",
        regularisation_lambda: float = 0.0,
        min_prob: float = 0.0,
    ) -> None:
        """Trainer.

        Parameters
        ----------
        seed : int
            RNG seed.
        engine : MCCFREngine, optional
            Game adapter. Defaults to mini-gin.
        sampling : {"external", "outcome"}
            Sampling scheme. External sampling enumerates all actions at
            the traverser's decision nodes (Lanctot et al. 2009); it is
            the default and matches the mini-gin behaviour. Outcome
            sampling (see Lanctot et al. 2009, §3.2) samples one action
            per node — used for games where the game tree is too deep
            for external sampling (e.g. abstracted Classic Gin with 4-way
            discard branching and 12+ half-turns).
        regularisation_lambda : float
            KL-to-uniform regularisation weight in ``[0, 1]``. When
            ``> 0``, the *average-strategy accumulator* is updated with a
            blended distribution
            ``pi_reg = (1 - lambda) * regret_matching(regrets) + lambda * uniform``
            instead of the raw regret-matching strategy. This prevents
            the average policy from becoming near-deterministic and
            therefore easier to exploit by a state-aware best responder
            (see PAPER.md §6 prediction #3). Regret updates still use
            the un-blended regret-matching strategy so convergence
            properties of MCCFR are preserved; only the *reported*
            average strategy is softened.
        min_prob : float
            After averaging strategies, floor every action probability
            at ``min_prob`` and renormalise. A cheap alternative /
            complement to KL regularisation. ``0.0`` disables. Applied
            in :meth:`average_strategy`.
        """
        self.rng = random.Random(seed)
        self.engine: MCCFREngine = engine if engine is not None else _minigin_engine()
        if sampling not in ("external", "outcome"):
            raise ValueError(f"Unknown sampling scheme: {sampling!r}")
        if not 0.0 <= regularisation_lambda <= 1.0:
            raise ValueError(
                f"regularisation_lambda must be in [0, 1], got {regularisation_lambda}"
            )
        if min_prob < 0.0:
            raise ValueError(f"min_prob must be >= 0, got {min_prob}")
        self.sampling = sampling
        self.regularisation_lambda = float(regularisation_lambda)
        self.min_prob = float(min_prob)
        # info_set -> {action: cumulative regret}
        self.regrets: dict[str, dict[Any, float]] = {}
        # info_set -> {action: cumulative strategy weight}
        self.strategy_sum: dict[str, dict[Any, float]] = {}

    # ------------------------------------------------------------------
    # Regret matching
    # ------------------------------------------------------------------

    def _current_strategy(self, info_set: str, legal: list[Any]) -> dict[Any, float]:
        table = self.regrets.setdefault(info_set, {})
        pos = {a: max(table.get(a, 0.0), 0.0) for a in legal}
        s = sum(pos.values())
        if s <= 0:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: v / s for a, v in pos.items()}

    def _reg_blended(self, strategy: dict[Any, float]) -> dict[Any, float]:
        """Blend a regret-matching strategy toward uniform for the average.

        When ``regularisation_lambda == 0`` this is the identity. When
        ``> 0`` we return
        ``(1 - lambda) * strategy + lambda * uniform_over_legal``.
        Only affects what is accumulated into ``strategy_sum`` (i.e.
        the reported average). Regret updates still use ``strategy``.
        """
        lam = self.regularisation_lambda
        if lam <= 0.0 or not strategy:
            return strategy
        u = 1.0 / len(strategy)
        return {a: (1.0 - lam) * p + lam * u for a, p in strategy.items()}

    # ------------------------------------------------------------------
    # Main traversal
    # ------------------------------------------------------------------

    def _traverse(self, state: Any, traverser: int) -> float:
        """Return the counterfactual value for ``traverser`` from ``state``.

        Implements external sampling: the traverser branches on all actions;
        opponent and chance nodes sample one action each.
        """
        eng = self.engine
        if eng.is_terminal(state):
            return eng.returns(state)[traverser]

        # Chance node: sample one deal outcome.
        if eng.is_chance(state):
            child = eng.sample_deal(self.rng)
            return self._traverse(child, traverser)

        legal = eng.legal_actions(state)
        cur = eng.current_player(state)
        info_set = eng.information_set(state, cur)
        strategy = self._current_strategy(info_set, legal)

        if cur == traverser:
            # Enumerate all actions, accumulate weighted sum, update regrets.
            utils: dict[Any, float] = {}
            node_util = 0.0
            for a in legal:
                utils[a] = self._traverse(eng.apply(state, a), traverser)
                node_util += strategy[a] * utils[a]
            regret_table = self.regrets.setdefault(info_set, {})
            strat_table = self.strategy_sum.setdefault(info_set, {})
            accum = self._reg_blended(strategy)
            for a in legal:
                regret_table[a] = regret_table.get(a, 0.0) + (utils[a] - node_util)
                strat_table[a] = strat_table.get(a, 0.0) + accum[a]
            return node_util
        else:
            # Opponent: sample one action from their current strategy.
            a = _sample_from(strategy, self.rng)
            # Accumulate opponent's strategy too so the average strategy has
            # entries for every reached info-set (helps best-response).
            strat_table = self.strategy_sum.setdefault(info_set, {})
            accum = self._reg_blended(strategy)
            for act, prob in accum.items():
                strat_table[act] = strat_table.get(act, 0.0) + prob
            return self._traverse(eng.apply(state, a), traverser)

    # ------------------------------------------------------------------
    # Outcome-sampling MCCFR (single-trajectory variant)
    # ------------------------------------------------------------------

    def _traverse_outcome(
        self,
        state: Any,
        traverser: int,
        pi_traverser: float,
        pi_opponent: float,
        sample_prob: float,
    ) -> float:
        """Outcome-sampling traversal (Lanctot et al. 2009, §3.2).

        Samples a single trajectory per iteration. At every decision node
        we sample one action and update regrets using importance-weighted
        counterfactual values. Costs O(game length) per iteration versus
        external sampling's O(branching^depth). Convergence is slower per
        iteration but iteration cost is bounded, making this the practical
        choice for deep games.

        Parameters:
            pi_traverser: reach probability contribution of the traverser
                to reach this node under sigma.
            pi_opponent: reach probability contribution of the opponent
                (and chance) to reach this node.
            sample_prob: probability that the current trajectory samples
                its way to this node.
        """
        eng = self.engine
        if eng.is_terminal(state):
            return eng.returns(state)[traverser] / max(sample_prob, 1e-12)

        if eng.is_chance(state):
            child = eng.sample_deal(self.rng)
            return self._traverse_outcome(
                child, traverser, pi_traverser, pi_opponent, sample_prob
            )

        legal = eng.legal_actions(state)
        cur = eng.current_player(state)
        info_set = eng.information_set(state, cur)
        strategy = self._current_strategy(info_set, legal)

        # Epsilon-greedy sampling policy for the traverser (exploration).
        eps = 0.6 if cur == traverser else 0.0
        n = len(legal)
        sampling_dist = {
            a: (eps / n) + (1.0 - eps) * strategy[a] for a in legal
        }
        # Sample one action from sampling_dist.
        a = _sample_from(sampling_dist, self.rng)
        p_a = sampling_dist[a]
        s_a = strategy[a]

        if cur == traverser:
            child_util = self._traverse_outcome(
                eng.apply(state, a),
                traverser,
                pi_traverser * s_a,
                pi_opponent,
                sample_prob * p_a,
            )
            # Standard OS regret update: for the sampled action, unit
            # weight (1 - sigma(a)); for all other actions, -sigma(a).
            # Multiplied by counterfactual reach and the tail utility.
            w = child_util * pi_opponent
            regret_table = self.regrets.setdefault(info_set, {})
            strat_table = self.strategy_sum.setdefault(info_set, {})
            accum = self._reg_blended(strategy)
            reach_wt = pi_traverser / max(sample_prob, 1e-12)
            for act in legal:
                if act == a:
                    regret_table[act] = regret_table.get(act, 0.0) + w * (1.0 - s_a)
                else:
                    regret_table[act] = regret_table.get(act, 0.0) - w * s_a
                # Weighted-average strategy: use pi_traverser as the
                # reach weight (OS variant of CFR+).
                strat_table[act] = strat_table.get(act, 0.0) + reach_wt * accum[act]
            return child_util
        else:
            # Opponent decision node: sample one action, accumulate its
            # strategy for the average.
            strat_table = self.strategy_sum.setdefault(info_set, {})
            accum = self._reg_blended(strategy)
            for act, prob in accum.items():
                strat_table[act] = strat_table.get(act, 0.0) + prob
            return self._traverse_outcome(
                eng.apply(state, a),
                traverser,
                pi_traverser,
                pi_opponent * s_a,
                sample_prob * p_a,
            )

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
                root = self.engine.initial_state()
                if self.sampling == "external":
                    self._traverse(root, traverser)
                else:
                    self._traverse_outcome(root, traverser, 1.0, 1.0, 1.0)
            if progress is not None and (it + 1) % max(1, iterations // 10) == 0:
                progress(it + 1)
        return self.average_strategy()

    def average_strategy(self) -> AverageStrategy:
        avg = AverageStrategy()
        floor = self.min_prob
        for iset, table in self.strategy_sum.items():
            total = sum(table.values())
            if total <= 0:
                dist = {a: 1.0 / len(table) for a in table}
            else:
                dist = {a: v / total for a, v in table.items()}
            avg.policy[iset] = _apply_min_prob(dist, floor)
        return avg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_min_prob(
    dist: dict[Any, float], floor: float
) -> dict[Any, float]:
    """Guarantee every action prob is ``>= floor`` and the sum is 1.

    Uses the standard convex construction
    ``p'_a = floor + (1 - n * floor) * p_a`` where ``p_a`` is the
    original (already normalised) probability. This is equivalent to
    mixing ``dist`` with ``uniform`` at weight ``n * floor``.

    If ``floor <= 0`` or the distribution is empty, returns the input
    unchanged. If ``floor * n >= 1`` the result is uniform.
    """
    if floor <= 0.0 or not dist:
        return dist
    n = len(dist)
    if floor * n >= 1.0:
        p = 1.0 / n
        return {a: p for a in dist}
    total = sum(dist.values())
    if total <= 0.0:
        p = 1.0 / n
        return {a: p for a in dist}
    # Normalise defensively, then mix with uniform.
    norm = {a: p / total for a, p in dist.items()}
    keep = 1.0 - n * floor  # in (0, 1)
    return {a: floor + keep * p for a, p in norm.items()}


def _sample_from(dist: dict[Any, float], rng: random.Random) -> Any:
    r = rng.random()
    cum = 0.0
    last: Any = None
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
    "MCCFREngine",
    "train",
    "uniform_strategy",
]
