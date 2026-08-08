"""End-to-end: each variant plays through and is reproducible with a seed."""

from __future__ import annotations

import pytest

from gin_rummy.variants.classic import ClassicGin
from gin_rummy.variants.hollywood import HollywoodGin
from gin_rummy.variants.indian import IndianRummy
from gin_rummy.variants.oklahoma import OklahomaGin


def test_classic_plays_through():
    result = ClassicGin(num_players=2, seed=7).play()
    assert result.outcome in {"gin", "knock", "undercut", "draw"}
    assert result.turns > 0
    assert set(result.scores) == {"Player 1", "Player 2"}


def test_classic_is_reproducible_with_seed():
    a = ClassicGin(num_players=2, seed=123).play()
    b = ClassicGin(num_players=2, seed=123).play()
    assert a.scores == b.scores
    assert a.outcome == b.outcome
    assert a.turns == b.turns


def test_oklahoma_sets_knock_limit_from_upcard():
    game = OklahomaGin(num_players=2, seed=42)
    game.play()
    # After play(), knock_limit should have been set from the upcard rank.
    upcard = game.discard_pile[0]
    if upcard.rank == "A":
        assert game.knock_limit == 0
    elif upcard.rank in ("J", "Q", "K"):
        assert game.knock_limit == 10
    else:
        assert game.knock_limit == int(upcard.rank)


@pytest.mark.parametrize("players", [2, 3, 4, 6])
def test_classic_scales_to_multiple_players(players):
    result = ClassicGin(num_players=players, seed=1).play()
    assert len(result.scores) == players


def test_hollywood_series_totals_are_finite():
    result = HollywoodGin(num_players=2, num_hands=3, seed=99).play()
    assert len(result.hands) == 3
    assert sum(result.totals.values()) >= 0


@pytest.mark.parametrize("hand_size", [3, 7, 10, 13, 15])
def test_indian_rummy_hand_sizes(hand_size):
    result = IndianRummy(num_players=3, hand_size=hand_size, seed=5).play()
    assert result.turns > 0


def test_indian_rummy_rejects_unsupported_hand_size():
    with pytest.raises(ValueError):
        IndianRummy(num_players=3, hand_size=11, seed=5)


def test_hollywood_is_reproducible_with_seed():
    a = HollywoodGin(num_players=2, num_hands=3, seed=55).play()
    b = HollywoodGin(num_players=2, num_hands=3, seed=55).play()
    assert a.totals == b.totals
