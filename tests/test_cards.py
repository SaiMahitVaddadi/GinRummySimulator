"""Cards and deck fundamentals."""

from __future__ import annotations

import random

import pytest

from gin_rummy.cards import Card, Deck, RANKS, SUITS


def test_card_value_rules():
    assert Card("A", "♠").value == 1
    assert Card("10", "♥").value == 10
    assert Card("J", "♦").value == 10
    assert Card("K", "♣").value == 10
    assert Card("5", "♠").value == 5


def test_card_equality_and_hash_respect_deck_id():
    a = Card("A", "♠", deck_id=0)
    b = Card("A", "♠", deck_id=0)
    c = Card("A", "♠", deck_id=1)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2


def test_deck_has_52n_cards():
    for n in (1, 2, 3, 4):
        assert len(Deck(n)) == 52 * n


def test_deck_seeded_shuffle_is_reproducible():
    d1 = Deck(1, rng=random.Random(42))
    d2 = Deck(1, rng=random.Random(42))
    assert [d1.draw() for _ in range(52)] == [d2.draw() for _ in range(52)]


def test_deck_draw_returns_none_when_empty():
    d = Deck(1)
    while d.draw() is not None:
        pass
    assert d.draw() is None


def test_deck_reset_reshuffles():
    d = Deck(1, rng=random.Random(0))
    first = [d.draw() for _ in range(52)]
    d.reset()
    second = [d.draw() for _ in range(52)]
    assert sorted(map(str, first)) == sorted(map(str, second))


@pytest.mark.parametrize(
    "players,expected", [(2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (12, 7)]
)
def test_optimal_deck_count(players, expected):
    assert Deck.optimal_deck_count(players) == expected


def test_ranks_and_suits_constants():
    assert len(RANKS) == 13
    assert len(SUITS) == 4
