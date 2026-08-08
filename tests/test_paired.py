"""Paired-hands variance reduction — the duplicate-poker trick."""

from __future__ import annotations

import pytest

from gin_rummy.bench import PolicyFactory, benchmark
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import RandomPolicy
from gin_rummy.variants.classic import ClassicGin


def _random_factory():
    return PolicyFactory("random", lambda rng: RandomPolicy(rng))


def _greedy_factory():
    return PolicyFactory("greedy", lambda rng: GreedyKnockPolicy(rng))


def test_paired_requires_two_players():
    with pytest.raises(ValueError):
        benchmark(
            game_cls=ClassicGin,
            policy_factories=[_random_factory(), _random_factory(), _random_factory()],
            num_games=10,
            seed=0,
            num_players=3,
            paired=True,
        )


def test_paired_requires_even_games():
    with pytest.raises(ValueError):
        benchmark(
            game_cls=ClassicGin,
            policy_factories=[_random_factory(), _random_factory()],
            num_games=11,
            seed=0,
            paired=True,
        )


def test_paired_runs_num_games_total():
    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[_greedy_factory(), _random_factory()],
        num_games=20,
        seed=0,
        paired=True,
    )
    assert result.num_games == 20
    assert sum(result.wins.values()) == 20
    assert result.paired is True


def test_paired_credits_wins_to_original_policy_across_swap():
    """Greedy should dominate whether seated first or second — pairing must
    credit both halves to the greedy label, not the physical seat."""
    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[_greedy_factory(), _random_factory()],
        num_games=100,
        seed=0,
        paired=True,
    )
    # P1 is the greedy label. In paired mode, both seatings should credit
    # greedy — so its win count should be near-total across the full run.
    assert result.wins["P1:greedy"] > result.wins["P2:random"] * 5
