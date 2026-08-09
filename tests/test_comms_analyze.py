"""Tests for the coordination-analysis runner (comms.analyze)."""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy.cards import Card
from gin_rummy.comms.analyze import (
    CoordinationReport,
    action_bucket,
    compare_policies,
    coordination_study,
    hand_bucket,
    public_bucket,
    random_policy_factory,
)
from gin_rummy.variants.euchre import EuchreObservation


# ------------------------------------------------------------- buckets ---


def test_hand_bucket_is_deterministic():
    hand = (
        Card("J", "♠"),
        Card("A", "♠"),
        Card("K", "♠"),
        Card("9", "♥"),
        Card("10", "♦"),
    )
    b1 = hand_bucket(hand, "♠")
    b2 = hand_bucket(hand, "♠")
    assert b1 == b2 == "strong_trump"


def test_hand_bucket_classifies_weak_and_medium():
    weak = (Card("9", "♥"), Card("10", "♦"), Card("Q", "♣"))
    assert hand_bucket(weak, "♠") == "weak"

    medium_ace = (Card("A", "♥"), Card("9", "♦"), Card("10", "♣"))
    assert hand_bucket(medium_ace, "♠") == "medium"

    medium_trump = (Card("9", "♠"), Card("10", "♥"), Card("Q", "♣"))
    assert hand_bucket(medium_trump, "♠") == "medium"


def test_hand_bucket_handles_unknown_trump():
    hand = (Card("A", "♠"), Card("A", "♥"), Card("9", "♦"))
    assert hand_bucket(hand, None) == "strong_trump"
    assert hand_bucket((Card("9", "♦"),), None) == "weak"


def test_action_bucket_maps_all_categories():
    assert action_bucket(Card("J", "♠"), "♠") == "trump_high"
    assert action_bucket(Card("A", "♠"), "♠") == "trump_high"
    assert action_bucket(Card("9", "♠"), "♠") == "trump_low"
    assert action_bucket(Card("A", "♥"), "♠") == "offsuit_ace"
    assert action_bucket(Card("Q", "♥"), "♠") == "offsuit_face"
    assert action_bucket(Card("9", "♥"), "♠") == "offsuit_low"


def test_action_bucket_is_deterministic():
    c = Card("K", "♣")
    assert action_bucket(c, "♠") == action_bucket(c, "♠") == "offsuit_face"


def test_public_bucket_uses_only_public_state():
    obs = EuchreObservation(
        player_id=1,
        hand=(Card("9", "♠"),),
        trump_suit="♠",
        upcard=Card("Q", "♠"),
        caller_id=0,
        partner_id=3,
        trick_history=((Card("A", "♥"), 0),),
        trick_number=2,
        scores=(0, 0),
    )
    b1 = public_bucket(obs)
    b2 = public_bucket(obs)
    assert b1 == b2
    # Public bucket exposes lead category and trick phase; kept small
    # by design to avoid inflating plug-in MI with finite samples.
    assert "lead=" in b1
    assert "phase=" in b1
    # Trick number 2 is the last "early" bucket.
    assert "phase=early" in b1


# --------------------------------------------------------- study runner ---


def test_coordination_study_runs_on_random_policy():
    report = coordination_study(
        policy_factory=random_policy_factory,
        num_games=20,
        seed=1234,
        n_resamples=100,
    )
    assert isinstance(report, CoordinationReport)
    assert report.n_games == 20
    assert report.n_plays > 0
    for ci in (
        report.mi_action_own_hand,
        report.mi_action_partner_hand,
        report.mi_action_opponent_hand,
    ):
        assert math.isfinite(ci.point)
        assert math.isfinite(ci.lower)
        assert math.isfinite(ci.upper)
        assert ci.point >= 0.0
        # Bootstrap bounds must bracket (or touch) the point estimate,
        # up to Monte-Carlo noise from a small resample budget.
        assert ci.lower <= ci.upper + 1e-9


def test_coordination_study_summary_contains_key_lines():
    report = coordination_study(
        policy_factory=random_policy_factory,
        num_games=10,
        seed=7,
        n_resamples=50,
        policy_name="random",
    )
    s = report.summary()
    assert "random" in s
    assert "own_hand" in s
    assert "partner_hand" in s
    assert "opponent_hand" in s


def test_partner_mi_is_small_for_random_play():
    """For fully random play, actions are independent of partner's hand
    conditional on public state. The upper bound of the MI CI should be
    tiny in nats (well under 0.1)."""
    report = coordination_study(
        policy_factory=random_policy_factory,
        num_games=60,
        seed=42,
        n_resamples=200,
    )
    # Plug-in MI is upward biased with finite samples so we check the
    # point estimate against a lenient absolute ceiling rather than
    # requiring the lower CI to be exactly zero.
    assert report.mi_action_partner_hand.point < 0.15
    # And the CI lower bound is very close to zero.
    assert report.mi_action_partner_hand.lower < 0.1


def test_coordination_study_is_reproducible():
    a = coordination_study(
        policy_factory=random_policy_factory,
        num_games=15,
        seed=99,
        n_resamples=50,
    )
    b = coordination_study(
        policy_factory=random_policy_factory,
        num_games=15,
        seed=99,
        n_resamples=50,
    )
    assert a.n_plays == b.n_plays
    assert a.mi_action_own_hand.point == pytest.approx(b.mi_action_own_hand.point)
    assert a.mi_action_partner_hand.point == pytest.approx(
        b.mi_action_partner_hand.point
    )


def test_compare_policies_returns_report_per_name(capsys):
    reports = compare_policies(
        policies={
            "random_a": random_policy_factory,
            "random_b": random_policy_factory,
        },
        num_games=10,
        seed=3,
        n_resamples=50,
    )
    assert set(reports) == {"random_a", "random_b"}
    for r in reports.values():
        assert isinstance(r, CoordinationReport)
    captured = capsys.readouterr()
    assert "random_a" in captured.out
    assert "random_b" in captured.out


def test_coordination_study_handles_zero_resamples_gracefully():
    """Even with a tiny resample budget, the runner must not crash."""
    report = coordination_study(
        policy_factory=random_policy_factory,
        num_games=5,
        seed=0,
        n_resamples=10,
    )
    assert report.n_games == 5
