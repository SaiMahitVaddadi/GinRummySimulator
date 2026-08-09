"""Tests for the mini-gin external-sampling MCCFR solver."""

from __future__ import annotations

import random

import pytest

from gin_rummy.solvers import (
    AverageStrategy,
    ExternalSamplingMCCFR,
    MiniGinState,
    apply,
    best_response_value,
    exploitability,
    information_set,
    is_gin_hand,
    is_terminal,
    legal_actions,
    returns,
    sample_deal,
    train,
    uniform_strategy,
)
from gin_rummy.solvers.minigin import (
    DRAW_DECK,
    DRAW_DISCARD,
    HAND_SIZE,
    Phase,
    card_str,
    make_discard,
)


# ---------------------------------------------------------------------------
# Game-mechanics sanity
# ---------------------------------------------------------------------------


def test_gin_hand_detection_set_and_run():
    # Set: all Aces (rank 0) across three different suits: cards 0,1,2
    assert is_gin_hand((0, 1, 2)) is True
    # Run: A-S, 2-S, 3-S  -> card ids 0, 4, 8
    assert is_gin_hand((0, 4, 8)) is True
    # Not a meld
    assert is_gin_hand((0, 5, 10)) is False


def test_sample_deal_shape():
    rng = random.Random(0)
    s = sample_deal(rng)
    assert len(s.hands[0]) == HAND_SIZE
    assert len(s.hands[1]) == HAND_SIZE
    # 12 cards total = 3 + 3 + 1 upcard + 5 in deck
    assert len(s.deck) == 5
    assert len(s.discard) == 1
    assert s.phase == Phase.DRAW
    assert s.current_player == 0


def test_information_set_is_deterministic():
    """Same state and player must always yield the identical info-set key."""
    rng = random.Random(42)
    state = sample_deal(rng)
    k1 = information_set(state, 0)
    k2 = information_set(state, 0)
    assert k1 == k2
    # And different players get different keys (different private hands).
    assert information_set(state, 0) != information_set(state, 1)


def test_apply_draw_then_discard_flow():
    rng = random.Random(1)
    state = sample_deal(rng)
    # Draw from deck.
    s2 = apply(state, DRAW_DECK)
    assert s2.phase == Phase.DISCARD
    assert len(s2.hands[0]) == HAND_SIZE + 1
    # Discard one card back.
    discard_card = s2.hands[0][0]
    s3 = apply(s2, make_discard(discard_card))
    assert len(s3.hands[0]) == HAND_SIZE
    # Turn passes (unless gin, which is unlikely for a random deal).
    if not is_terminal(s3):
        assert s3.current_player == 1


def test_terminal_returns_are_zero_sum():
    """For any terminal state the two players' returns sum to zero."""
    rng = random.Random(2)
    # Force a terminal by rolling out a random game.
    state = sample_deal(rng)
    steps = 0
    while not is_terminal(state) and steps < 200:
        legal = legal_actions(state)
        state = apply(state, rng.choice(legal))
        steps += 1
    assert is_terminal(state)
    r0, r1 = returns(state)
    assert r0 + r1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MCCFR trainer
# ---------------------------------------------------------------------------


def test_mccfr_runs_1000_iterations_without_error():
    solver = ExternalSamplingMCCFR(seed=0)
    avg = solver.train(iterations=1000, seed=0)
    assert isinstance(avg, AverageStrategy)
    # Should have discovered at least a few dozen information sets.
    assert avg.num_info_sets() > 20


def test_train_convenience_function():
    avg = train(iterations=100, seed=1)
    assert isinstance(avg, AverageStrategy)
    assert avg.num_info_sets() >= 1


def test_average_strategy_probs_sum_to_one():
    solver = ExternalSamplingMCCFR(seed=0)
    avg = solver.train(iterations=200, seed=0)
    for iset, dist in avg.policy.items():
        s = sum(dist.values())
        assert s == pytest.approx(1.0, abs=1e-6), f"{iset}: sum={s}"


# ---------------------------------------------------------------------------
# Exploitability regression
# ---------------------------------------------------------------------------


def test_exploitability_decreases_from_uniform_to_trained():
    """After MCCFR training the average strategy must be no more
    exploitable (within noise) than the uniform-random baseline. On a
    reasonable-sized run trained exploitability should be strictly lower.
    """
    uniform = uniform_strategy()
    trained = train(iterations=1500, seed=0)
    e_uniform = exploitability(uniform, num_deals=60, seed=100)
    e_trained = exploitability(trained, num_deals=60, seed=100)
    # Both should be non-negative in expectation; trained should not blow up.
    assert e_trained <= e_uniform + 0.05, (
        f"trained exploitability {e_trained} not <= uniform {e_uniform}"
    )
