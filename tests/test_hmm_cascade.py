"""Tests for the HMM-conditioned cascade response model.

Scope: :mod:`gin_rummy.markov.hmm_cascade`. Complements the tests in
:mod:`tests.test_hmm` (HMM primitives) and :mod:`tests.test_markov`
(cascade primitives).
"""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy import ClassicGin, GreedyKnockPolicy
from gin_rummy.cards import Card
from gin_rummy.markov.cascade import (
    CascadeState,
    greedy_response_model,
    probabilistic_cascade,
)
from gin_rummy.markov.hmm_cascade import (
    hmm_conditioned_response_model,
    replay_with_snapshots,
    run_cascade_comparison,
)
from gin_rummy.markov.opponent_hand_hmm import (
    HAND_STRENGTH_CLASSES,
    OpponentHandHMM,
)


def _canonical_state() -> CascadeState:
    """A mid-game state with a mix of low melded and high-value stragglers."""
    hand0 = [
        Card("A", "♠"),
        Card("A", "♥"),
        Card("A", "♦"),
        Card("5", "♣"),
        Card("6", "♣"),
        Card("7", "♣"),
        Card("K", "♠"),
        Card("Q", "♥"),
        Card("9", "♦"),
        Card("3", "♣"),
    ]
    hand1 = [
        Card("2", "♠"),
        Card("2", "♥"),
        Card("2", "♣"),
        Card("8", "♥"),
        Card("9", "♥"),
        Card("10", "♥"),
        Card("4", "♠"),
        Card("J", "♣"),
        Card("6", "♦"),
        Card("K", "♦"),
    ]
    return CascadeState(hands={0: hand0, 1: hand1}, top_discard=Card("4", "♥"), turn=12)


class _FakeHMM(OpponentHandHMM):
    """OpponentHandHMM with a fixed posterior for deterministic testing.

    Subclass so the interface (``posteriors``, ``n_states``) is preserved
    but ``posteriors`` returns a caller-supplied vector.
    """

    def __init__(self, fixed_gamma: list[float]) -> None:
        super().__init__(n_states=len(fixed_gamma), rng=random.Random(0))
        self._fixed = list(fixed_gamma)

    def posteriors(self, obs_seq):  # type: ignore[override]
        if not obs_seq:
            return []
        # One row per turn; every turn has the same fixed posterior.
        return [list(self._fixed) for _ in range(len(obs_seq))]


def _dummy_history(target_player_id: int, n_turns: int = 4) -> list:
    """A synthetic history with ``n_turns`` visible observations for target."""
    from gin_rummy.game import TurnRecord

    out = []
    for t in range(1, n_turns + 1):
        out.append(
            TurnRecord(
                turn=t,
                player_id=target_player_id,
                draw_source="deck",
                drawn=Card("5", "♠"),
                discarded=Card("K", "♣"),
                action="play",
                deadwood_value=20,
            )
        )
    return out


# --------------------------------------------------------- validity tests --


def test_hmm_conditioned_model_returns_valid_prob_dist() -> None:
    """Output distribution must sum to 1 and be non-negative."""
    hmm = _FakeHMM([0.2, 0.2, 0.2, 0.2, 0.2])
    state = _canonical_state()
    history = _dummy_history(target_player_id=1)
    model = hmm_conditioned_response_model(hmm, history, target_player_id=1)

    dist = model(state, actor_id=1)
    assert dist, "must be non-empty when actor has cards"
    total = sum(dist.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), f"got total={total}"
    assert all(p >= 0.0 for p in dist.values())


# ------------------------------------------------ uniform => baseline ----


def test_uniform_belief_reduces_to_baseline() -> None:
    """Uniform posterior + β=0.5 collapses tilt to (1 + 0.5·0.4) = 1.2, not
    exactly 1.0.  So to check the *baseline reduction* we set β = 0.0."""
    n = len(HAND_STRENGTH_CLASSES)
    hmm = _FakeHMM([1.0 / n] * n)
    state = _canonical_state()
    history = _dummy_history(target_player_id=1)

    conditioned = hmm_conditioned_response_model(
        hmm, history, target_player_id=1, beta=0.0, base_temperature=1.0
    )
    baseline_dist = greedy_response_model(state, actor_id=1, temperature=1.0)
    conditioned_dist = conditioned(state, actor_id=1)

    assert set(baseline_dist) == set(conditioned_dist)
    for card, p in baseline_dist.items():
        assert math.isclose(
            p, conditioned_dist[card], abs_tol=1e-6
        ), f"{card}: baseline={p} conditioned={conditioned_dist[card]}"


# --------------------------------- strong belief tilts toward high value --


def test_strong_belief_shifts_toward_high_value_discards() -> None:
    """Under a 100% 'strong / gin_ready' belief the conditioned model's
    expected discard value should be strictly higher than baseline.

    Rationale: strong-hand belief widens the softmax so probability mass
    leaks away from the argmin-deadwood pick toward higher-value non-meld
    cards. Expected discard value is our proxy for that leak.
    """
    n = len(HAND_STRENGTH_CLASSES)
    strong_idx = HAND_STRENGTH_CLASSES.index("strong")
    gin_idx = HAND_STRENGTH_CLASSES.index("gin_ready")
    posterior = [0.0] * n
    # Split the strong mass evenly between "strong" and "gin_ready".
    posterior[strong_idx] = 0.5
    posterior[gin_idx] = 0.5

    hmm = _FakeHMM(posterior)
    state = _canonical_state()
    history = _dummy_history(target_player_id=1)

    conditioned = hmm_conditioned_response_model(
        hmm, history, target_player_id=1, beta=5.0, base_temperature=1.0
    )
    baseline_dist = greedy_response_model(state, actor_id=1, temperature=1.0)
    conditioned_dist = conditioned(state, actor_id=1)

    def _expected_value(dist: dict[Card, float]) -> float:
        return sum(card.value * p for card, p in dist.items())

    ev_baseline = _expected_value(baseline_dist)
    ev_conditioned = _expected_value(conditioned_dist)

    assert ev_conditioned > ev_baseline, (
        f"expected higher expected-discard-value under strong belief; "
        f"got baseline={ev_baseline:.3f} conditioned={ev_conditioned:.3f}"
    )


# ----------------------------------------------- end-to-end smoke test ---


def test_run_cascade_comparison_smoke() -> None:
    """Comparison harness runs end-to-end on 10 games and produces finite
    cascade metrics for both models."""
    seed_base = 20260810
    target_player_id = 1

    def _factory(seed: int) -> ClassicGin:
        rng = random.Random(seed)
        return ClassicGin(
            num_players=2,
            policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
            seed=seed,
        )

    # Train a small HMM on 20 games to keep the test cheap.
    train_games = []
    for i in range(20):
        rng = random.Random(seed_base + i)
        eng = ClassicGin(
            num_players=2,
            policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
            seed=seed_base + i,
        )
        train_games.append(eng.play())
    hmm = OpponentHandHMM(rng=random.Random(seed_base))
    hmm.train(train_games, target_player_id=target_player_id, iterations=5)

    snapshots = []
    for i in range(10):
        got = replay_with_snapshots(
            _factory,
            game_seed=seed_base + 1000 + i,
            target_player_id=target_player_id,
            target_turn=8,
        )
        if got is None:
            continue
        state, result, record = got
        snapshots.append((i, state, result, record))

    # We need at least a couple of successful snapshots for the smoke
    # test to say anything; if this ever hits zero the seeded games are
    # oddly terminating before turn 8, which is worth investigating.
    assert snapshots, "expected some games to reach turn 8"

    report = run_cascade_comparison(
        snapshots,
        target_player_id=target_player_id,
        hmm=hmm,
        depth=3,
        top_k=5,
    )
    assert "baseline" in report and "hmm" in report
    for side in ("baseline", "hmm"):
        s = report[side]
        assert s["top1_total"] >= 0
        assert s["top5_total"] >= 0
        assert s["top1_correct"] <= s["top1_total"]
        assert s["top5_cover"] <= s["top5_total"]

    # Every row should have finite probabilities in [0, 1].
    for row in report["rows"]:
        assert 0.0 <= row.baseline_top1_prob <= 1.0
        assert 0.0 <= row.hmm_top1_prob <= 1.0
        assert math.isfinite(row.baseline_top1_value)
        assert math.isfinite(row.hmm_top1_value)
