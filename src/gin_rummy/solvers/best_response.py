"""Best-response value and exploitability (NashConv) for mini-gin.

For two-player zero-sum games the standard exploitability metric is
``NashConv(sigma) = sum_p [BR_p(sigma_{-p}) - v_p(sigma)]`` where ``BR_p`` is
the value of player ``p``'s best response against the opponent playing
``sigma_{-p}``. A Nash equilibrium has NashConv == 0.

Exact best-response requires enumerating the imperfect-information game
tree with belief propagation over hidden state (see Johanson et al.
*Accelerating Best Response Calculation in Large Extensive Games*, IJCAI
2011). For mini-gin the *abstracted* tree is small enough that a
depth-first expectimax over sampled deals gives a tight enough estimate
for exploitability tracking during CFR development.

Approach
--------
We compute a *sampled* best-response by rolling out ``num_deals`` random
deals. For each deal we recursively evaluate:

* At the best-responder's decision node we branch over all legal actions
  and take the ``max`` of the resulting subgame values.
* At the opponent's decision node we take the expectation under
  ``sigma`` over their legal actions (full enumeration, since branching is
  small).
* Chance nodes below the root are deterministic in mini-gin (deals are
  the only chance move), so the recursion is deterministic once a deal is
  fixed.

Because the best-responder observes hidden state during evaluation, this
gives an *upper bound* on the true best-response value (an "omniscient BR"
in OpenSpiel terminology). It is therefore an upper bound on NashConv and
is monotone in true exploitability, which is all we need for verifying
that MCCFR training is reducing exploitability.

References
----------
* Johanson, Waugh, Bowling, Zinkevich. *Accelerating Best Response
  Calculation in Large Extensive Games.* IJCAI 2011.
* Zinkevich et al. NeurIPS 2007 (CFR / exploitability).
* OpenSpiel best-response utilities:
  https://openspiel.readthedocs.io/en/latest/algorithms.html
"""

from __future__ import annotations

import random

from gin_rummy.solvers.cfr import AverageStrategy
from gin_rummy.solvers.minigin import (
    Action,
    MiniGinState,
    Phase,
    apply,
    information_set,
    is_terminal,
    legal_actions,
    returns,
    sample_deal,
)


# ---------------------------------------------------------------------------
# Subgame value (deal is fixed)
# ---------------------------------------------------------------------------


def _subgame_value(
    state: MiniGinState,
    responder: int,
    sigma: AverageStrategy,
    memo: dict[MiniGinState, float] | None = None,
) -> float:
    """Expected value for ``responder`` from ``state`` assuming they play a
    best-response and the opponent plays ``sigma``.

    ``memo`` caches subgame values keyed on the (hashable) frozen state so
    repeated subtrees inside a single BR computation are visited once.
    """
    if memo is None:
        memo = {}
    if state in memo:
        return memo[state]
    if is_terminal(state):
        v = returns(state)[responder]
        memo[state] = v
        return v
    if state.phase == Phase.DEAL:  # pragma: no cover - deal handled above
        raise RuntimeError("Chance node inside subgame")

    legal = legal_actions(state)
    if not legal:  # pragma: no cover - defensive
        raise RuntimeError("Non-terminal state with no legal actions")

    if state.current_player == responder:
        best = float("-inf")
        for a in legal:
            v = _subgame_value(apply(state, a), responder, sigma, memo)
            if v > best:
                best = v
        memo[state] = best
        return best
    # Opponent decision: expectation under sigma.
    info_set = information_set(state, state.current_player)
    dist = sigma.action_probs(info_set, legal)
    expected = 0.0
    for a in legal:
        expected += dist[a] * _subgame_value(apply(state, a), responder, sigma, memo)
    memo[state] = expected
    return expected


# ---------------------------------------------------------------------------
# Public best-response API
# ---------------------------------------------------------------------------


def best_response_value(
    sigma: AverageStrategy,
    responder: int,
    num_deals: int = 200,
    seed: int = 0,
) -> float:
    """Estimate ``BR_{responder}(sigma_{-responder})`` via ``num_deals`` random deals."""
    rng = random.Random(seed)
    total = 0.0
    memo: dict[MiniGinState, float] = {}
    for _ in range(num_deals):
        deal = sample_deal(rng)
        total += _subgame_value(deal, responder, sigma, memo)
    return total / num_deals


def expected_value(
    sigma: AverageStrategy,
    player: int,
    num_deals: int = 200,
    seed: int = 0,
) -> float:
    """Estimate ``v_player(sigma, sigma)`` via ``num_deals`` random deals."""
    rng = random.Random(seed)
    total = 0.0
    memo: dict[MiniGinState, float] = {}
    for _ in range(num_deals):
        deal = sample_deal(rng)
        total += _expected_value_recursive(deal, player, sigma, memo)
    return total / num_deals


def _expected_value_recursive(
    state: MiniGinState,
    player: int,
    sigma: AverageStrategy,
    memo: dict[MiniGinState, float] | None = None,
) -> float:
    if memo is None:
        memo = {}
    if state in memo:
        return memo[state]
    if is_terminal(state):
        v = returns(state)[player]
        memo[state] = v
        return v
    legal = legal_actions(state)
    info_set = information_set(state, state.current_player)
    dist = sigma.action_probs(info_set, legal)
    expected = 0.0
    for a in legal:
        expected += dist[a] * _expected_value_recursive(apply(state, a), player, sigma, memo)
    memo[state] = expected
    return expected


def exploitability(
    sigma: AverageStrategy,
    num_deals: int = 200,
    seed: int = 0,
) -> float:
    """Sampled NashConv estimate: sum over players of best-response gain.

    For a symmetric zero-sum game a Nash strategy has ``exploitability == 0``
    (and by symmetry ``v_p(sigma, sigma) == 0``). For arbitrary strategies
    we compute ``BR_p(sigma) - v_p(sigma)`` per player and sum.
    """
    br0 = best_response_value(sigma, 0, num_deals=num_deals, seed=seed)
    br1 = best_response_value(sigma, 1, num_deals=num_deals, seed=seed + 1)
    v0 = expected_value(sigma, 0, num_deals=num_deals, seed=seed + 2)
    v1 = expected_value(sigma, 1, num_deals=num_deals, seed=seed + 3)
    return (br0 - v0) + (br1 - v1)


__all__ = [
    "best_response_value",
    "expected_value",
    "exploitability",
]
