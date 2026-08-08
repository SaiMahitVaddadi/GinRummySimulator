"""Batch-benchmark harness sanity."""

from __future__ import annotations

from gin_rummy.bench import PolicyFactory, benchmark, build_policy_factory
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import RandomPolicy
from gin_rummy.variants.classic import ClassicGin


def test_benchmark_random_vs_random():
    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
        ],
        num_games=25,
        seed=0,
    )
    assert result.num_games == 25
    assert sum(result.wins.values()) == 25
    assert sum(result.outcomes.values()) == 25
    assert len(result.turns) == 25


def test_benchmark_is_reproducible():
    def make() -> "BenchmarkResult":  # type: ignore[name-defined]
        return benchmark(
            game_cls=ClassicGin,
            policy_factories=[
                PolicyFactory("random", lambda rng: RandomPolicy(rng)),
                PolicyFactory("random", lambda rng: RandomPolicy(rng)),
            ],
            num_games=25,
            seed=42,
        )

    a = make()
    b = make()
    assert a.wins == b.wins
    assert a.outcomes == b.outcomes


def test_greedy_beats_random_over_a_batch():
    """Sanity: our heuristic should crush random over enough games."""
    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[
            PolicyFactory("greedy", lambda rng: GreedyKnockPolicy(rng)),
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
        ],
        num_games=200,
        seed=0,
    )
    greedy_wins = result.wins["P1:greedy"]
    random_wins = result.wins["P2:random"]
    assert greedy_wins > random_wins, (
        f"expected greedy to beat random; got greedy={greedy_wins} random={random_wins}"
    )


def test_build_policy_factory_random_and_greedy():
    r = build_policy_factory("random")
    g = build_policy_factory("greedy")
    assert r.name == "random"
    assert g.name == "greedy"


def test_build_policy_factory_llm_spec():
    f = build_policy_factory("llm:fake-model")
    assert f.name == "llm:fake-model"


def test_build_policy_factory_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        build_policy_factory("not-a-real-thing")


def test_benchmark_summary_formats_cleanly():
    result = benchmark(
        game_cls=ClassicGin,
        policy_factories=[
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
            PolicyFactory("random", lambda rng: RandomPolicy(rng)),
        ],
        num_games=5,
        seed=0,
    )
    summary = result.summary()
    assert "Win rates:" in summary
    assert "Outcome distribution:" in summary
    assert "ClassicGin" in summary
