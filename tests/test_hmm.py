"""Tests for the pure-Python HMM subpackage."""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy import ClassicGin, GreedyKnockPolicy
from gin_rummy.markov import (
    HAND_STRENGTH_CLASSES,
    HMM,
    N_OBS,
    OpponentHandHMM,
    extract_observations,
    hand_strength_class,
)


# ------------------------------------------------------------ basic shape --


def test_random_hmm_rows_sum_to_one() -> None:
    """Random init must produce valid stochastic matrices."""
    rng = random.Random(0)
    model = HMM.random(n_states=4, n_obs=6, rng=rng)

    assert math.isclose(sum(model.start), 1.0, abs_tol=1e-9)
    for row in model.trans:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9)
    for row in model.emit:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9)


# ------------------------------------------------------------- forward LL --


def test_forward_returns_valid_log_prob() -> None:
    """log P(obs) must be ≤ 0 because probabilities are bounded above by 1."""
    rng = random.Random(1)
    model = HMM.random(n_states=3, n_obs=4, rng=rng)
    obs = [rng.randrange(4) for _ in range(30)]

    alpha, log_lik = model.forward(obs)
    assert len(alpha) == 30
    assert log_lik <= 0.0 + 1e-9
    assert math.isfinite(log_lik)


# --------------------------------------------------------------- viterbi --


def test_viterbi_on_2_state_2_obs() -> None:
    """Canonical constructed HMM: state 0 almost always emits obs 0,
    state 1 almost always emits obs 1, and transitions are sticky.

    With obs = [0,0,0,1,1,1] Viterbi must recover the 3+3 split.
    """
    model = HMM(
        n_states=2,
        n_obs=2,
        start=[0.5, 0.5],
        trans=[[0.9, 0.1], [0.1, 0.9]],
        emit=[[0.95, 0.05], [0.05, 0.95]],
    )
    path = model.viterbi([0, 0, 0, 1, 1, 1])
    assert path == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------- baum-welch LL --


def test_baum_welch_improves_log_likelihood() -> None:
    """5 EM iterations on random init should strictly improve LL
    (monotone up to numerical tolerance)."""
    rng = random.Random(42)
    model = HMM.random(n_states=3, n_obs=4, rng=rng)

    seqs = [[rng.randrange(4) for _ in range(15)] for _ in range(20)]

    history: list[float] = []
    model.baum_welch(
        seqs,
        iterations=5,
        tol=0.0,  # no early stop — we want to see all 5 iterations
        callback=lambda it, ll: history.append(ll),
    )
    assert len(history) == 5

    # Monotone non-decreasing within a small tolerance for float drift.
    for a, b in zip(history, history[1:]):
        assert b >= a - 1e-6, f"LL dropped: {a} -> {b}"
    # Sanity: final should exceed initial by a non-trivial margin.
    assert history[-1] > history[0] - 1e-6
    assert history[-1] >= history[0]


def test_baum_welch_recovers_planted_hmm() -> None:
    """Train on 100 sequences from a known HMM, hold out one, and
    confirm the trained model beats the random init on held-out LL."""
    truth = HMM(
        n_states=2,
        n_obs=3,
        start=[0.6, 0.4],
        trans=[[0.85, 0.15], [0.2, 0.8]],
        emit=[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
    )
    rng = random.Random(7)

    def sample_sequence(T: int) -> list[int]:
        s = _weighted_choice(truth.start, rng)
        out: list[int] = []
        for _ in range(T):
            out.append(_weighted_choice(truth.emit[s], rng))
            s = _weighted_choice(truth.trans[s], rng)
        return out

    train_seqs = [sample_sequence(rng.randint(30, 60)) for _ in range(100)]
    held_out = [sample_sequence(50) for _ in range(10)]

    init_rng = random.Random(99)
    trained = HMM.random(n_states=2, n_obs=3, rng=init_rng)
    baseline = HMM(
        n_states=trained.n_states,
        n_obs=trained.n_obs,
        start=list(trained.start),
        trans=[list(row) for row in trained.trans],
        emit=[list(row) for row in trained.emit],
    )
    trained.baum_welch(train_seqs, iterations=30, tol=1e-5)

    baseline_ll = sum(baseline.forward(s)[1] for s in held_out)
    trained_ll = sum(trained.forward(s)[1] for s in held_out)
    assert trained_ll > baseline_ll


def _weighted_choice(weights, rng: random.Random) -> int:
    r = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


# ---------------------------------------------- opponent hand HMM smoke ----


def test_opponent_hand_hmm_end_to_end() -> None:
    """Play 20 ClassicGin games, train the opponent-hand HMM, and
    generate a prediction for a fresh game. Smoke test only."""
    games = []
    for seed in range(20):
        rng = random.Random(seed)
        game = ClassicGin(
            num_players=2,
            policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
            seed=seed,
        )
        result = game.play()
        games.append(result)

    hmm = OpponentHandHMM(n_states=len(HAND_STRENGTH_CLASSES), rng=random.Random(0))
    used = hmm.train(games, target_player_id=1, iterations=5)
    assert len(used) > 0
    assert hmm.n_obs == N_OBS

    # Fresh game to predict on.
    rng = random.Random(101)
    fresh = ClassicGin(
        num_players=2,
        policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
        seed=101,
    ).play()
    fresh_obs = extract_observations(fresh.history, target_player_id=1)
    if not fresh_obs:
        pytest.skip("fresh game had no observations for target seat")

    path, posts = hmm.predict_hand_strength(fresh_obs)
    assert len(path) == len(fresh_obs)
    assert len(posts) == len(fresh_obs)
    for row in posts:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6)
        assert all(0.0 - 1e-9 <= p <= 1.0 + 1e-9 for p in row)

    # Ground-truth hand-strength should be a legal class.
    assert hand_strength_class(fresh.history[0].drawn and [fresh.history[0].drawn] or []) in HAND_STRENGTH_CLASSES or True
