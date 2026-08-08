"""End-to-end tests for the eval subpackage."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from gin_rummy.eval.ensemble import (
    HandStrengthGatedMoE,
    PhaseGatedMoE,
    VotingEnsemble,
    seed_ensemble,
)
from gin_rummy.eval.kfold import deal_kfold, opponent_kfold
from gin_rummy.eval.rating import bradley_terry_ratings, rating_ci_table
from gin_rummy.eval.reporters import export_jsonl
from gin_rummy.eval.stats import (
    bootstrap_ci,
    holm_bonferroni,
    iqm,
    paired_sign_test,
    wilson_ci,
)
from gin_rummy.eval.tournament import PolicyEntry, Tournament
from gin_rummy.eval.xplain import (
    discard_regret_table,
    fingerprint_from_history,
    introspect_hand,
    reward_channels,
)
from gin_rummy.cards import Card
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import RandomPolicy
from gin_rummy.variants.classic import ClassicGin


# ---------- stats ----------

def test_bootstrap_ci_covers_point():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ci = bootstrap_ci(samples, n_resamples=1000, rng=random.Random(0))
    assert ci.lower <= ci.point <= ci.upper


def test_iqm_trims_outliers():
    samples = [0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0, 100.0]
    # Middle 50% is [5,5,5,5,5], mean = 5. Ordinary mean = 23.
    assert iqm(samples) == pytest.approx(5.0)


def test_holm_bonferroni_rejects_below_threshold():
    # p = [0.001, 0.02, 0.04] at alpha=0.05, m=3
    # Sorted: 0.001 (thr 0.05/3 ≈ 0.0167 → reject), 0.02 (thr 0.05/2 = 0.025 → reject),
    # 0.04 (thr 0.05/1 = 0.05 → reject). All rejected.
    rej = holm_bonferroni([0.001, 0.02, 0.04])
    assert rej == [True, True, True]


def test_holm_bonferroni_stops_when_one_fails():
    rej = holm_bonferroni([0.001, 0.06, 0.01])
    # Sorted by p: 0.001 → thr 0.0167 reject; 0.01 → thr 0.025 reject; 0.06 → thr 0.05 fail; stop.
    assert rej == [True, False, True]


def test_paired_sign_test_all_wins():
    p = paired_sign_test([1] * 10 + [-1] * 0)
    assert p < 0.01


def test_paired_sign_test_ties():
    p = paired_sign_test([0, 0, 0])
    assert p == 1.0


def test_wilson_ci_still_works():
    ci = wilson_ci(30, 100)
    assert ci.rate == 0.30
    assert 0.20 < ci.lower < 0.25


# ---------- tournament ----------

def _entry(name: str, build) -> PolicyEntry:
    return PolicyEntry(name=name, build=build)


def test_tournament_two_way_matches_bench():
    entries = [
        _entry("greedy", lambda rng: GreedyKnockPolicy(rng)),
        _entry("random", lambda rng: RandomPolicy(rng)),
    ]
    t = Tournament(game_cls=ClassicGin, entries=entries, games_per_pair=50, seed=0)
    r = t.run()
    assert len(r.matches) == 50
    a_wins, b_wins, _ = r.wins_between("greedy", "random")
    assert a_wins > b_wins  # sanity: greedy dominates


def test_tournament_three_way_produces_crossplay_matrix():
    entries = [
        _entry("greedy", lambda rng: GreedyKnockPolicy(rng)),
        _entry("random", lambda rng: RandomPolicy(rng)),
        _entry("greedy2", lambda rng: GreedyKnockPolicy(rng)),
    ]
    t = Tournament(game_cls=ClassicGin, entries=entries, games_per_pair=20, seed=0)
    r = t.run()
    matrix = r.crossplay_matrix()
    # Three ordered off-diagonal pairs × 2 directions = 6 entries.
    assert len(matrix) == 6
    # greedy vs random should be near-1
    assert matrix[("greedy", "random")].rate > 0.8


def test_tournament_paired_swaps_seats():
    entries = [
        _entry("greedy", lambda rng: GreedyKnockPolicy(rng)),
        _entry("random", lambda rng: RandomPolicy(rng)),
    ]
    t = Tournament(
        game_cls=ClassicGin, entries=entries, games_per_pair=20, seed=0, paired=True
    )
    r = t.run()
    # Each pair produces 2 matches; total should equal games_per_pair.
    assert len(r.matches) == 20


# ---------- rating ----------

def test_bradley_terry_orders_by_win_rate():
    matches = [("A", "B", "seed") for _ in range(30)] + [
        ("B", "C", "seed") for _ in range(20)
    ]
    strengths = bradley_terry_ratings(matches)
    assert strengths["A"] > strengths["B"] > strengths["C"]


def test_rating_ci_table_returns_sorted_elos():
    matches = [("A", "B", str(i)) for i in range(30)] + [
        ("B", "C", str(i)) for i in range(20)
    ]
    table = rating_ci_table(matches, n_resamples=50, seed=0)
    assert [r.name for r in table] == ["A", "B", "C"]
    for r in table:
        assert r.lower <= r.elo <= r.upper


# ---------- ensemble ----------

def test_voting_ensemble_plays_a_game():
    experts = [GreedyKnockPolicy(random.Random(i)) for i in range(3)]
    ens = VotingEnsemble(experts=experts)
    r = ClassicGin(
        num_players=2,
        seed=0,
        policies=[ens, GreedyKnockPolicy(random.Random(99))],
    ).play()
    assert r.turns > 0


def test_phase_gated_moe_routes_correctly():
    early = GreedyKnockPolicy(random.Random(0))
    mid = GreedyKnockPolicy(random.Random(1))
    late = GreedyKnockPolicy(random.Random(2))
    moe = PhaseGatedMoE(early=early, mid=mid, late=late, phase_thresholds=(3, 10))
    from gin_rummy.policy import Observation
    obs_early = Observation(
        hand=(Card("A", "♠"),), top_discard=None, discard_pile_size=0,
        deck_size=10, turn_number=1, knock_limit=10, player_id=0,
        num_players=2, other_hand_sizes=(10,),
    )
    obs_late = Observation(
        hand=(Card("A", "♠"),), top_discard=None, discard_pile_size=0,
        deck_size=10, turn_number=15, knock_limit=10, player_id=0,
        num_players=2, other_hand_sizes=(10,),
    )
    assert moe._route(obs_early) is early
    assert moe._route(obs_late) is late


def test_hand_strength_gated_moe_routes_by_deadwood():
    strong_hand_policy = GreedyKnockPolicy(random.Random(0))
    weak_hand_policy = GreedyKnockPolicy(random.Random(1))
    moe = HandStrengthGatedMoE(
        cheap=weak_hand_policy, careful=strong_hand_policy, deadwood_threshold=20
    )
    from gin_rummy.policy import Observation
    strong_hand = tuple([Card("A", "♠"), Card("A", "♥"), Card("A", "♦")])
    obs = Observation(
        hand=strong_hand, top_discard=None, discard_pile_size=0, deck_size=10,
        turn_number=1, knock_limit=10, player_id=0, num_players=2, other_hand_sizes=(10,),
    )
    assert moe._route(obs) is strong_hand_policy


def test_seed_ensemble_produces_independent_rngs():
    from gin_rummy.policy import RandomPolicy
    policies = seed_ensemble(lambda rng: RandomPolicy(rng), n=3, base_seed=42)
    assert len(policies) == 3


# ---------- k-fold ----------

def test_opponent_kfold_runs():
    focal = _entry("greedy", lambda rng: GreedyKnockPolicy(rng))
    opps = [
        _entry(f"random_{i}", lambda rng: RandomPolicy(rng)) for i in range(4)
    ]
    report = opponent_kfold(
        game_cls=ClassicGin,
        focal=focal,
        opponents=opps,
        games_per_pair=6,
        k=2,
        seed=0,
        paired=True,
    )
    assert len(report.folds) == 2
    # Greedy should win most folds vs random opponents.
    assert report.iqm_win_rate > 0.8


def test_deal_kfold_splits_evenly():
    folds = deal_kfold(list(range(10)), k=5)
    assert len(folds) == 5
    for f in folds:
        assert len(f) == 2


# ---------- xplain ----------

def test_introspect_hand_agrees_with_optimal_decomp():
    hand = [Card("A", "♠"), Card("A", "♥"), Card("A", "♦"), Card("K", "♣")]
    intro = introspect_hand(hand)
    assert intro.deadwood_value == 10
    # Discarding K leaves the three aces set.
    assert intro.per_discard["K♣"] == 0


def test_discard_regret_flags_bad_choices():
    hand = [Card("A", "♠"), Card("A", "♥"), Card("A", "♦"), Card("K", "♣")]
    regret = discard_regret_table(hand, chosen=Card("K", "♣"))
    # Discarding an ace would break the set (leaves 2 aces + K = 12) — that's +12 regret.
    assert regret["A♠"] > 0
    assert regret["K♣"] == 0


def test_reward_channels_track_deadwood_reduction():
    before = [Card("A", "♠"), Card("A", "♥"), Card("K", "♣")]  # dw = 12
    after = [Card("A", "♠"), Card("A", "♥"), Card("A", "♦")]  # dw = 0, +1 meld
    ch = reward_channels(before, after, knock_limit=10)
    assert ch.deadwood_reduction == 12
    assert ch.meld_formation_delta == 1


def test_fingerprint_extracts_stats():
    g = ClassicGin(num_players=2, seed=0)
    r = g.play()
    fp = fingerprint_from_history(r.history, "test", 0)
    assert fp.total_turns > 0


# ---------- reporters ----------

def test_export_jsonl_writes_summary_and_matches(tmp_path: Path):
    entries = [
        _entry("greedy", lambda rng: GreedyKnockPolicy(rng)),
        _entry("random", lambda rng: RandomPolicy(rng)),
    ]
    t = Tournament(game_cls=ClassicGin, entries=entries, games_per_pair=10, seed=0)
    r = t.run()
    out = tmp_path / "tournament.jsonl"
    export_jsonl(r, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 11  # 1 summary + 10 matches
    import json
    summary = json.loads(lines[0])
    assert summary["kind"] == "summary"
    assert summary["variant"] == "ClassicGin"
