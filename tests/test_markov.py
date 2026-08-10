"""Tests for the general Markov machinery.

Scope: :mod:`gin_rummy.markov.chain`, :mod:`gin_rummy.markov.absorbing`,
:mod:`gin_rummy.markov.deck_process`, :mod:`gin_rummy.markov.cascade`.
The HMM subpackage is exercised by a sibling test module.
"""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy.cards import Card
from gin_rummy.markov.absorbing import AbsorbingChain, expected_turns_to_gin
from gin_rummy.markov.cascade import (
    CascadeState,
    cascade_value,
    greedy_response_model,
    probabilistic_cascade,
)
from gin_rummy.markov.chain import MarkovChain, estimate_chain_from_sequences
from gin_rummy.markov.deck_process import CLASS_NAMES, DeckDepletionModel


# --------------------------------------------------------------------- chain


def _two_state_chain(p: float, q: float) -> MarkovChain:
    """Classic 2-state chain: 0 -> 1 w.p. p, 1 -> 0 w.p. q."""
    return MarkovChain(
        states=["0", "1"],
        transition={
            "0": {"0": 1 - p, "1": p},
            "1": {"0": q, "1": 1 - q},
        },
    )


def test_markov_step_uses_transition_probabilities() -> None:
    """Empirical step distribution matches the row over 10k samples."""
    chain = _two_state_chain(p=0.3, q=0.7)
    chain.validate()
    rng = random.Random(0xC0FFEE)
    n = 10_000
    from_0 = [chain.step("0", rng) for _ in range(n)]
    from_1 = [chain.step("1", rng) for _ in range(n)]
    p_hat_01 = from_0.count("1") / n
    p_hat_10 = from_1.count("0") / n
    assert abs(p_hat_01 - 0.3) < 0.02
    assert abs(p_hat_10 - 0.7) < 0.02


def test_stationary_distribution_of_2_state_chain() -> None:
    """Closed form: pi_0 = q / (p + q), pi_1 = p / (p + q)."""
    p, q = 0.2, 0.5
    chain = _two_state_chain(p=p, q=q)
    pi = chain.stationary_distribution(tol=1e-12, max_iter=10_000)
    assert abs(pi["0"] - q / (p + q)) < 1e-6
    assert abs(pi["1"] - p / (p + q)) < 1e-6
    # Also: check the row-sum invariant is preserved.
    assert abs(sum(pi.values()) - 1.0) < 1e-9


def test_transition_matrix_row_stochastic() -> None:
    chain = _two_state_chain(p=0.4, q=0.6)
    P = chain.transition_matrix()
    for row in P:
        assert abs(sum(row) - 1.0) < 1e-12


def test_expected_hits_symmetric_2_state() -> None:
    """For the symmetric chain p=q, expected hits from 0 to 1 equals 1/p."""
    p = 0.25
    chain = _two_state_chain(p=p, q=p)
    h = chain.expected_hits("0", "1")
    assert abs(h - 1 / p) < 1e-9
    # Same-state hit is zero.
    assert chain.expected_hits("0", "0") == 0.0


def test_mean_first_passage_matrix_shape() -> None:
    chain = _two_state_chain(p=0.3, q=0.4)
    M = chain.mean_first_passage_matrix()
    assert set(M) == {"0", "1"}
    for s in M:
        assert set(M[s]) == {"0", "1"}
        assert M[s][s] == 0.0
        assert M[s][("1" if s == "0" else "0")] > 0


def test_estimate_chain_from_sequences_smoothing() -> None:
    seqs = [["a", "a", "b"], ["b", "a"]]
    chain = estimate_chain_from_sequences(seqs, ["a", "b"], laplace_alpha=1.0)
    chain.validate()
    # Every entry should be strictly positive after add-one smoothing.
    for row in chain.transition.values():
        for p in row.values():
            assert p > 0.0


def test_validate_rejects_bad_row_sum() -> None:
    bad = MarkovChain(
        states=["a", "b"],
        transition={"a": {"a": 0.3, "b": 0.3}, "b": {"a": 0.5, "b": 0.5}},
    )
    with pytest.raises(ValueError):
        bad.validate()


# ------------------------------------------------------------------ absorbing


def _gambler_chain(a: int, b: int, p: float = 0.5) -> AbsorbingChain:
    """Gambler's ruin: start with wealth ``a``, absorb at 0 or ``b``.

    On each step, wealth goes up 1 with probability p, down 1 with 1-p.
    """
    states = [str(i) for i in range(b + 1)]
    trans: dict[str, dict[str, float]] = {}
    for i in range(b + 1):
        if i == 0 or i == b:
            trans[str(i)] = {str(i): 1.0}
        else:
            trans[str(i)] = {str(i + 1): p, str(i - 1): 1 - p}
    return AbsorbingChain(
        states=states,
        transition=trans,
        absorbing_states=["0", str(b)],
    )


def test_absorbing_expected_absorption_of_gambler() -> None:
    """Classic p=0.5 result: expected steps from wealth a in game of size b
    is ``a * (b - a)``."""
    a, b = 3, 7
    chain = _gambler_chain(a=a, b=b, p=0.5)
    chain.validate()
    chain.validate_absorbing()
    expected = a * (b - a)
    got = chain.expected_absorption(str(a))
    assert abs(got - expected) < 0.1


def test_absorption_probabilities_gambler_fair() -> None:
    """For fair p=0.5 the probability of reaching wealth b starting from a
    is a / b."""
    a, b = 4, 10
    chain = _gambler_chain(a=a, b=b, p=0.5)
    B = chain.absorption_probabilities()
    # Row ordering: transient states in original order = "1".."b-1".
    # Absorbing column ordering: ["0", str(b)].
    row = B[a - 1]  # transient index a-1 == wealth a
    assert abs(row[0] - (1 - a / b)) < 1e-9
    assert abs(row[1] - (a / b)) < 1e-9


def test_fundamental_matrix_dims() -> None:
    chain = _gambler_chain(a=2, b=5, p=0.5)
    N = chain.fundamental_matrix()
    # 4 transient states (1..4).
    assert len(N) == 4
    assert all(len(row) == 4 for row in N)
    # Diagonal entries >= 1 (state visits itself at least once).
    for i in range(4):
        assert N[i][i] >= 1.0 - 1e-9


def test_expected_turns_to_gin_positive() -> None:
    val = expected_turns_to_gin({"start_bucket": "mid"})
    assert val > 0
    # Starting from a low-deadwood bucket should be faster.
    lo = expected_turns_to_gin({"start_bucket": "lo"})
    hi = expected_turns_to_gin({"start_bucket": "hi"})
    assert lo < hi


# ---------------------------------------------------------------- deck_process


def test_deck_depletion_probability_sums_to_one() -> None:
    model = DeckDepletionModel(num_decks=1)
    for turn in (0, 5, 20, 51):
        s = sum(
            model.probability_of_class_at_turn(c, turn) for c in CLASS_NAMES
        )
        assert abs(s - 1.0) < 1e-9


def test_deck_depletion_simulate_length() -> None:
    model = DeckDepletionModel(num_decks=1)
    rng = random.Random(0)
    seq = model.simulate(30, rng)
    assert len(seq) == 30
    for cls in seq:
        assert cls in CLASS_NAMES


def test_expected_draws_to_rank_bounds() -> None:
    model = DeckDepletionModel(num_decks=1)
    # 4 copies of each rank in a 52-card shoe -> E[T] = (52+1)/(4+1) = 10.6
    e = model.expected_draws_to_rank("A")
    assert abs(e - 53 / 5) < 1e-9


def test_deck_depletion_matches_simulation_at_turn_1() -> None:
    """At turn 1 the approximation is exact (well-shuffled uniform draw)."""
    model = DeckDepletionModel(num_decks=1)
    rng = random.Random(42)
    n_trials = 5_000
    counter = {c: 0 for c in CLASS_NAMES}
    for _ in range(n_trials):
        first = model.simulate(1, rng)[0]
        counter[first] += 1
    predicted = model.initial_distribution()
    for c in CLASS_NAMES:
        empirical = counter[c] / n_trials
        assert abs(empirical - predicted[c]) < 0.03


# --------------------------------------------------------------------- cascade


def _make_cards() -> list[Card]:
    """Ten-card hand with a set and a run plus some deadwood."""
    hand = [
        Card("A", "♠"), Card("A", "♥"), Card("A", "♦"),   # set of aces
        Card("5", "♣"), Card("6", "♣"), Card("7", "♣"),   # run of clubs
        Card("K", "♠"), Card("Q", "♥"), Card("9", "♦"), Card("3", "♣"),
    ]
    return hand


def test_greedy_response_model_returns_valid_distribution() -> None:
    hand = _make_cards()
    state = CascadeState(hands={0: hand}, top_discard=None, turn=0)
    dist = greedy_response_model(state, actor_id=0, temperature=1.0)
    assert dist
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9
    for p in dist.values():
        assert 0.0 < p <= 1.0


def test_greedy_response_model_temperature_zero_picks_argmin() -> None:
    hand = _make_cards()
    state = CascadeState(hands={0: hand}, top_discard=None, turn=0)
    dist = greedy_response_model(state, actor_id=0, temperature=0.0)
    # Should place all mass on the argmin-deadwood discard (the K).
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9
    winners = [c for c, p in dist.items() if p > 0]
    assert Card("K", "♠") in winners


def test_cascade_top_k_are_valid_probabilities() -> None:
    hand0 = _make_cards()
    hand1 = _make_cards()
    state = CascadeState(hands={0: hand0, 1: hand1}, top_discard=None, turn=0)

    def model(s: CascadeState, actor_id: int) -> dict[Card, float]:
        return greedy_response_model(s, actor_id, temperature=1.0)

    paths = probabilistic_cascade(
        state, model, depth=3, top_k=5, actor_order=[0, 1]
    )
    assert 0 < len(paths) <= 5
    for path in paths:
        assert 1 <= len(path) <= 3
        joint = 1.0
        for step in path:
            assert 0.0 < step.probability <= 1.0
            joint *= step.probability
        assert 0.0 < joint <= 1.0


def test_cascade_value_without_initial_hand_is_joint_prob() -> None:
    hand0 = _make_cards()
    hand1 = _make_cards()
    state = CascadeState(hands={0: hand0, 1: hand1}, top_discard=None, turn=0)

    def model(s: CascadeState, actor_id: int) -> dict[Card, float]:
        return greedy_response_model(s, actor_id, temperature=1.0)

    paths = probabilistic_cascade(state, model, depth=2, top_k=3)
    assert paths
    v = cascade_value(paths[0])
    expected = math.prod(step.probability for step in paths[0])
    assert abs(v - expected) < 1e-12


def test_cascade_value_with_initial_hand_uses_deadwood_delta() -> None:
    hand0 = _make_cards()
    hand1 = _make_cards()
    state = CascadeState(hands={0: hand0, 1: hand1}, top_discard=None, turn=0)

    def model(s: CascadeState, actor_id: int) -> dict[Card, float]:
        return greedy_response_model(s, actor_id, temperature=0.5)

    paths = probabilistic_cascade(
        state, model, depth=2, top_k=3, actor_order=[0, 1]
    )
    assert paths
    # Actor 0 discards deadwood -> delta >= 0 -> value >= 0.
    v = cascade_value(
        paths[0], perspective_actor_id=0, initial_hand=hand0
    )
    assert v >= 0.0
