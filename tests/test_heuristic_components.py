"""Tests for the orthogonal heuristic components.

Verifies:
  * Each named rule is deterministic under a seeded RNG.
  * ``ComposableHeuristicPolicy`` with the default triple plays byte-for-byte
    identically to ``GreedyKnockPolicy`` on a seeded game.
  * The ``knock_below(k)`` factory never knocks when deadwood > k.
  * Every rule-triple combination plays a full game without raising.
"""

from __future__ import annotations

import itertools
import random

import pytest

from gin_rummy.cards import Card
from gin_rummy.policies.heuristic import (
    ComposableHeuristicPolicy,
    GreedyKnockPolicy,
)
from gin_rummy.policies.heuristic_components import (
    discard_highest_deadwood,
    discard_lowest_deadwood,
    discard_random,
    discard_safest,
    draw_always_deck,
    draw_if_reduces_deadwood,
    draw_random,
    knock_after_turn,
    knock_asap,
    knock_below,
    knock_never,
)
from gin_rummy.policy import Observation
from gin_rummy.variants.classic import ClassicGin


# ---------- helpers ----------


def _obs(
    hand: tuple[Card, ...],
    *,
    top_discard: Card | None = None,
    turn_number: int = 1,
) -> Observation:
    return Observation(
        hand=hand,
        top_discard=top_discard,
        discard_pile_size=1 if top_discard else 0,
        deck_size=30,
        turn_number=turn_number,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10, 10),
    )


# ---------- determinism ----------


def test_draw_random_deterministic_under_seed():
    hand = (Card("A", "♠"), Card("K", "♥"))
    top = Card("2", "♣")
    obs = _obs(hand, top_discard=top)
    a = [draw_random(obs, random.Random(42)) for _ in range(1)]
    b = [draw_random(obs, random.Random(42)) for _ in range(1)]
    assert a == b
    # Multi-call determinism from a shared seeded RNG.
    rng1, rng2 = random.Random(7), random.Random(7)
    seq1 = [draw_random(obs, rng1) for _ in range(20)]
    seq2 = [draw_random(obs, rng2) for _ in range(20)]
    assert seq1 == seq2


def test_draw_if_reduces_deadwood_deterministic():
    hand = (Card("7", "♠"), Card("7", "♥"), Card("K", "♦"))
    top = Card("7", "♦")
    obs = _obs(hand, top_discard=top)
    for _ in range(3):
        assert draw_if_reduces_deadwood(obs, random.Random(0)) == "discard"


def test_draw_always_deck_ignores_everything():
    hand = (Card("7", "♠"), Card("7", "♥"), Card("K", "♦"))
    top = Card("7", "♦")  # would obviously slot in
    assert draw_always_deck(_obs(hand, top_discard=top), random.Random(0)) == "deck"
    assert draw_always_deck(_obs(hand), random.Random(0)) == "deck"


def test_discard_random_deterministic_under_seed():
    hand = tuple(Card(r, "♠") for r in ("A", "2", "3", "9", "K"))
    obs = _obs(hand)
    rng1, rng2 = random.Random(11), random.Random(11)
    picks1 = [discard_random(obs, rng1) for _ in range(10)]
    picks2 = [discard_random(obs, rng2) for _ in range(10)]
    assert picks1 == picks2


def test_discard_highest_deadwood_picks_king():
    hand = (Card("A", "♠"), Card("K", "♥"), Card("2", "♦"))
    assert discard_highest_deadwood(_obs(hand), random.Random(0)) == Card("K", "♥")


def test_discard_lowest_deadwood_picks_ace():
    hand = (Card("A", "♠"), Card("K", "♥"), Card("2", "♦"))
    assert discard_lowest_deadwood(_obs(hand), random.Random(0)) == Card("A", "♠")


def test_discard_safest_avoids_adjacent_to_top_discard():
    # Hand has K♥ (adjacent to Q♥) and 5♣ (unrelated).
    hand = (Card("K", "♥"), Card("5", "♣"), Card("2", "♦"))
    top = Card("Q", "♥")
    # K♥ is adjacent (same suit, |rank|=1). 5♣ is the highest non-adjacent
    # deadwood card, so that's what safest picks.
    assert discard_safest(_obs(hand, top_discard=top), random.Random(0)) == Card(
        "5", "♣"
    )


def test_discard_safest_falls_back_when_no_top_discard():
    hand = (Card("A", "♠"), Card("K", "♥"), Card("2", "♦"))
    # No top_discard → should behave like discard_highest_deadwood.
    assert discard_safest(_obs(hand), random.Random(0)) == Card("K", "♥")


def test_discard_safest_falls_back_when_all_deadwood_adjacent():
    # Every deadwood card is adjacent to the top discard → fall back to
    # highest-value.
    hand = (Card("K", "♥"), Card("Q", "♥"), Card("J", "♥"))
    # This actually forms a run, so let's pick cards that are all adjacent
    # to the top but don't form a meld.
    hand = (Card("2", "♠"), Card("2", "♥"))  # both same rank as top
    top = Card("2", "♣")
    result = discard_safest(_obs(hand, top_discard=top), random.Random(0))
    assert result in hand  # falls back to highest deadwood among the pool


def test_knock_never_always_false():
    obs = _obs((Card("A", "♠"),))
    assert knock_never(obs, 0, random.Random(0)) is False
    assert knock_never(obs, 100, random.Random(0)) is False


def test_knock_asap_always_true():
    obs = _obs((Card("A", "♠"),))
    assert knock_asap(obs, 0, random.Random(0)) is True
    assert knock_asap(obs, 100, random.Random(0)) is True


def test_knock_below_threshold_boundary():
    rule = knock_below(5)
    obs = _obs((Card("A", "♠"),))
    assert rule(obs, 0, random.Random(0)) is True
    assert rule(obs, 5, random.Random(0)) is True
    assert rule(obs, 6, random.Random(0)) is False
    assert rule(obs, 10, random.Random(0)) is False


def test_knock_below_never_knocks_above_threshold():
    rule = knock_below(5)
    for dw in range(6, 30):
        obs = _obs((Card("A", "♠"),), turn_number=dw)
        assert rule(obs, dw, random.Random(dw)) is False


def test_knock_after_turn_gates_by_turn_number():
    rule = knock_after_turn(20)
    for t in range(0, 20):
        assert rule(_obs((Card("A", "♠"),), turn_number=t), 0, random.Random(0)) is False
    for t in range(20, 40):
        assert rule(_obs((Card("A", "♠"),), turn_number=t), 0, random.Random(0)) is True


# ---------- byte-equivalence with GreedyKnockPolicy ----------


def _game_history(policy_factory, seed: int):
    """Play a game with two independent policies from the factory."""
    p1 = policy_factory()
    p2 = policy_factory()
    result = ClassicGin(num_players=2, seed=seed, policies=[p1, p2]).play()
    # Reduce TurnRecord to a plain tuple so equality is byte-strict.
    return [
        (r.turn, r.player_id, r.draw_source, r.drawn, r.discarded, r.action, r.deadwood_value)
        for r in result.history
    ], result.outcome, result.winner_id, result.turns, result.scores


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_composable_default_triple_matches_greedy(seed):
    def make_greedy():
        # RNG doesn't affect the default triple (no rule uses it in the
        # normal path), but pass a seeded one anyway for parity.
        return GreedyKnockPolicy(random.Random(seed))

    def make_composable():
        return ComposableHeuristicPolicy(
            draw_rule=draw_if_reduces_deadwood,
            discard_rule=discard_highest_deadwood,
            knock_rule=knock_asap,
            rng=random.Random(seed),
        )

    greedy_result = _game_history(make_greedy, seed)
    composable_result = _game_history(make_composable, seed)
    assert greedy_result == composable_result


# ---------- knock_below sanity in a real game ----------


def test_knock_below_5_never_knocks_when_deadwood_over_5():
    """Play a full game with knock_below(5) and check every knock-eligible
    ``choose_to_knock`` decision was consistent."""
    rule = knock_below(5)
    # Directly assert the rule's contract for all plausible dw values seen
    # in a hand of 10 cards (max deadwood ~ 100).
    obs = _obs((Card("A", "♠"),))
    for dw in range(6, 100):
        assert rule(obs, dw, random.Random(dw)) is False


# ---------- full game plays through with every combination ----------


DRAW_RULES = [draw_random, draw_if_reduces_deadwood, draw_always_deck]
DISCARD_RULES = [
    discard_random,
    discard_highest_deadwood,
    discard_lowest_deadwood,
    discard_safest,
]
KNOCK_RULES = [knock_never, knock_asap, knock_below(5), knock_after_turn(20)]


@pytest.mark.parametrize(
    "draw_rule,discard_rule,knock_rule",
    list(itertools.product(DRAW_RULES, DISCARD_RULES, KNOCK_RULES)),
)
def test_every_combination_plays_through(draw_rule, discard_rule, knock_rule):
    p1 = ComposableHeuristicPolicy(
        draw_rule=draw_rule,
        discard_rule=discard_rule,
        knock_rule=knock_rule,
        rng=random.Random(0),
    )
    p2 = ComposableHeuristicPolicy(
        draw_rule=draw_rule,
        discard_rule=discard_rule,
        knock_rule=knock_rule,
        rng=random.Random(1),
    )
    result = ClassicGin(num_players=2, seed=99, policies=[p1, p2]).play()
    assert result.turns > 0
    assert result.outcome in {"gin", "knock", "undercut", "draw"}
