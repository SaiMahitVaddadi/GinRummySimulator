"""Optimal meld decomposition — the correctness bar for the core algorithm."""

from __future__ import annotations

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition


def _hand(*specs: str) -> list[Card]:
    out: list[Card] = []
    for spec in specs:
        # Format: "A♠" or "10♥" — last char is suit, rest is rank
        suit = spec[-1]
        rank = spec[:-1]
        out.append(Card(rank, suit))
    return out


def test_empty_hand():
    melds, deadwood, dv = optimal_decomposition([])
    assert melds == []
    assert deadwood == []
    assert dv == 0


def test_pure_gin_hand():
    # Two sets of 3 and one run of 4 → all melded.
    hand = _hand(
        "7♠", "7♥", "7♦",
        "K♠", "K♥", "K♦",
        "2♣", "3♣", "4♣", "5♣",
    )
    melds, deadwood, dv = optimal_decomposition(hand)
    assert dv == 0
    assert deadwood == []
    total_cards = sum(len(m) for m in melds)
    assert total_cards == 10


def test_optimal_beats_greedy_when_set_and_run_compete_for_a_card():
    # 7♠ 8♠ 9♠ is a run; 9♠ 9♥ 9♦ is a set; they conflict on 9♠.
    # Taking the SET leaves 7♠ + 8♠ + 2♣ = 17 deadwood.
    # Taking the RUN leaves 9♥ + 9♦ + 2♣ = 20 deadwood.
    # Optimal chooses whichever wins → 17.
    hand = _hand("7♠", "8♠", "9♠", "9♥", "9♦", "2♣")
    _, _, dv = optimal_decomposition(hand)
    assert dv == 17


def test_greedy_would_miss_optimal():
    # A♣ 2♣ 3♣ (run, deadwood saved: 6) vs.
    # 3♣ 3♥ 3♦ (set, deadwood saved: 9) — can't take both because 3♣ collides.
    # If we take the set, deadwood = A♣(1) + 2♣(2) = 3.
    # If we take the run, deadwood = 3♥(3) + 3♦(3) = 6.
    # Optimal chooses set → 3.
    hand = _hand("A♣", "2♣", "3♣", "3♥", "3♦")
    _, _, dv = optimal_decomposition(hand)
    assert dv == 3


def test_run_of_five_beats_run_of_three_plus_pair():
    # 4♠ 5♠ 6♠ 7♠ 8♠ — optimal takes the whole run, 0 deadwood.
    hand = _hand("4♠", "5♠", "6♠", "7♠", "8♠")
    _, _, dv = optimal_decomposition(hand)
    assert dv == 0


def test_pair_is_deadwood():
    # Two queens don't form a meld (need 3+); both count as deadwood.
    hand = _hand("Q♠", "Q♥")
    _, _, dv = optimal_decomposition(hand)
    assert dv == 20


def test_jokers_never_form_melds():
    hand = _hand("A♠", "A♥", "A♦") + [Card("JKR", "🃏")]
    melds, _, dv = optimal_decomposition(hand)
    # Three aces form a set. Joker sits in deadwood with value 0.
    assert len(melds) == 1
    assert dv == 0
