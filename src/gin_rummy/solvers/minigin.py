"""Mini-gin: a self-contained micro-variant of gin rummy for tabular CFR.

Motivation
----------
The full 52-card gin rummy game has an astronomical information-set count
(easily 10^15+); no public CFR solve of any rummy variant exists to date
(see the RL/equilibrium survey included with this repo). To make tabular
external-sampling MCCFR (Lanctot et al. 2009) tractable on a laptop we
define ``mini-gin``: a truly small imperfect-information zero-sum card
game that preserves the essential structure of gin (draw / discard /
knock-on-gin) but reduces the state space to something a dictionary-based
regret table can hold.

Game specification
------------------
* Deck: **12 cards** = 3 ranks {A, 2, 3} x 4 suits {S, H, D, C}.
* Two players; each dealt a hand of **3 cards**; one upcard is turned face-up
  on the discard pile; the remaining 5 cards form the stock (deck).
* On each turn the acting player:
    1. **Draw source:** ``DRAW_DECK`` or ``DRAW_DISCARD``.
    2. **Discard:** pick any card from the resulting 4-card hand and place it
       on top of the discard pile.
    3. **Auto-knock rule:** if the post-discard 3-card hand has *deadwood == 0*
       (i.e. it is a valid meld), the player wins with **gin**.
* Melds of size 3:
    * A **set** of 3 = three cards of the same rank (16 such sets: 3 ranks x
      C(4,3)=4 per rank).
    * A **run** of 3 = A-2-3 in the same suit (4 such runs, one per suit).
    * Total: **20 gin hands** out of C(12,3)=220 possible 3-card hands.
* No knock threshold: **gin is the only way to win**. No undercut or bonus.
* Game ends when (a) a player gins, (b) the deck is exhausted, or (c) the
  turn cap (``MAX_TURNS``) is reached. Ties yield 0 utility for both.
* Payoffs: +1 to the winner, -1 to the loser, 0 on a tie. Zero-sum.

Why these numbers?
------------------
With hand size 3 and 12 distinct cards, the acting player's private hand has
C(12,3)=220 possible values (only 165 are reachable given the fixed 3-card
opponent hand and public upcard, but 220 upper-bounds it). The public state
per turn is (top-of-discard in {12 cards or empty}, deck size 0..5, turn
counter 0..MAX_TURNS, phase in {DRAW, DISCARD}). Cross-multiplying with
canonicalization gives an information-set count well under 10^5, allowing
tabular external-sampling MCCFR to converge in <60s on a laptop.

Information-set abstraction (perfect-recall-lite)
-------------------------------------------------
A fully perfect-recall info-set would encode the entire action history the
acting player has observed. For tractability we use a standard rummy
abstraction: the info-set key is

    (player_id, phase, turn_number, sorted(hand), top_of_discard, deck_size)

This drops the private sequence of past draws/discards but keeps every
observable variable that a strong human would base a decision on. This
abstraction is analogous to the "bucketed" info-sets used in poker CFR
solvers; MCCFR still converges to a Nash equilibrium of the *abstracted*
game (see Sandholm & Waugh 2010 on abstraction).

References
----------
* Zinkevich, Bowling, Johanson, Piccione. *Regret Minimization in Games
  with Incomplete Information*. NeurIPS 2007.
* Lanctot, Waugh, Zinkevich, Bowling. *Monte Carlo Sampling for Regret
  Minimization in Extensive Games*. NeurIPS 2009 (introduces external
  sampling MCCFR).
* OpenSpiel CFR docs: https://openspiel.readthedocs.io/en/latest/algorithms.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import Iterable

# ---------------------------------------------------------------------------
# Card representation
# ---------------------------------------------------------------------------

RANKS: tuple[str, ...] = ("A", "2", "3")
SUITS: tuple[str, ...] = ("S", "H", "D", "C")

# Each card is an int in [0, 12). Layout: card_id = rank_idx * 4 + suit_idx.
NUM_CARDS: int = 12
HAND_SIZE: int = 3
# Five half-turns is enough for CFR to learn non-trivial strategy (draw
# source choice, discard choice, and gin recognition) while keeping the
# game tree small enough for exact best-response computation on a laptop.
MAX_TURNS: int = 5


def card_rank(c: int) -> int:
    return c // 4


def card_suit(c: int) -> int:
    return c % 4


def card_str(c: int) -> str:
    return f"{RANKS[card_rank(c)]}{SUITS[card_suit(c)]}"


def hand_str(hand: tuple[int, ...]) -> str:
    return "[" + ",".join(card_str(c) for c in hand) + "]"


# ---------------------------------------------------------------------------
# Meld detection & deadwood
# ---------------------------------------------------------------------------


def is_gin_hand(hand: tuple[int, ...]) -> bool:
    """Return True iff a 3-card hand is a single meld (set or A-2-3 run)."""
    if len(hand) != HAND_SIZE:
        return False
    ranks = sorted(card_rank(c) for c in hand)
    suits = sorted(card_suit(c) for c in hand)
    # Set: all same rank.
    if ranks[0] == ranks[1] == ranks[2]:
        return True
    # Run: A, 2, 3 all same suit.
    if ranks == [0, 1, 2] and suits[0] == suits[1] == suits[2]:
        return True
    return False


def deadwood(hand: tuple[int, ...]) -> int:
    """Sum of card values not covered by any meld.

    In mini-gin we only care about the binary distinction gin vs. not-gin, so
    we return 0 iff ``is_gin_hand``; otherwise we return an arbitrary positive
    number (the count of unmatched cards, which is 3 when no meld).
    """
    if is_gin_hand(hand):
        return 0
    return len(hand)  # placeholder positive value


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class Phase(IntEnum):
    DRAW = 0
    DISCARD = 1
    DEAL = 2  # chance node: initial deal
    TERMINAL = 3


class Action:
    """Base class for actions; equality by (kind, payload)."""

    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload: int) -> None:
        self.kind = kind
        self.payload = payload

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, Action) and self.kind == other.kind and self.payload == other.payload

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash((self.kind, self.payload))

    def __repr__(self) -> str:
        if self.kind == "draw":
            return "DRAW_DECK" if self.payload == 0 else "DRAW_DISCARD"
        return f"DISCARD({card_str(self.payload)})"


DRAW_DECK = Action("draw", 0)
DRAW_DISCARD = Action("draw", 1)


def make_discard(card_id: int) -> Action:
    return Action("discard", card_id)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiniGinState:
    """Immutable snapshot of a mini-gin game.

    Fields:
        hands: tuple(hand_p0, hand_p1) where each hand is a sorted tuple of
            card ids. During the DISCARD phase the acting player's hand has
            length HAND_SIZE + 1.
        deck: tuple of card ids remaining in the stock (top of deck is the
            LAST element).
        discard: tuple of card ids in the discard pile (top is the LAST
            element).
        current_player: 0 or 1; ignored for DEAL/TERMINAL phases.
        phase: Phase enum.
        turn: number of full turns completed (increments after each DISCARD).
        winner: -1 if not decided, 0/1 if a player gins, 2 for draw.
    """

    hands: tuple[tuple[int, ...], tuple[int, ...]]
    deck: tuple[int, ...]
    discard: tuple[int, ...]
    current_player: int
    phase: Phase
    turn: int
    winner: int = -1

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def top_discard(self) -> int:
        return self.discard[-1] if self.discard else -1


# ---------------------------------------------------------------------------
# Deal / chance
# ---------------------------------------------------------------------------


def initial_state() -> MiniGinState:
    """Return the pre-deal chance node."""
    return MiniGinState(
        hands=((), ()),
        deck=tuple(range(NUM_CARDS)),
        discard=(),
        current_player=0,
        phase=Phase.DEAL,
        turn=0,
        winner=-1,
    )


def deal_outcomes() -> Iterable[tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]]:
    """Enumerate all distinct deals as (hand0, hand1, upcard, deck).

    Order of dealing does not matter for CFR because we treat the deal as one
    chance move with uniform probability over ordered partitions. To keep the
    tree small we sample a canonical partition: player 0's hand is any 3 of
    12, player 1's hand any 3 of the remaining 9, and the upcard any 1 of the
    remaining 6 (the other 5 form the stock).
    """
    all_cards = list(range(NUM_CARDS))
    for h0 in combinations(all_cards, HAND_SIZE):
        rem1 = [c for c in all_cards if c not in h0]
        for h1 in combinations(rem1, HAND_SIZE):
            rem2 = [c for c in rem1 if c not in h1]
            for up in rem2:
                deck = tuple(c for c in rem2 if c != up)
                yield tuple(sorted(h0)), tuple(sorted(h1)), up, deck


def sample_deal(rng) -> MiniGinState:
    """Sample a uniformly-random deal using ``rng`` (random.Random)."""
    cards = list(range(NUM_CARDS))
    rng.shuffle(cards)
    h0 = tuple(sorted(cards[0:HAND_SIZE]))
    h1 = tuple(sorted(cards[HAND_SIZE : 2 * HAND_SIZE]))
    up = cards[2 * HAND_SIZE]
    deck = tuple(cards[2 * HAND_SIZE + 1 :])
    return MiniGinState(
        hands=(h0, h1),
        deck=deck,
        discard=(up,),
        current_player=0,
        phase=Phase.DRAW,
        turn=0,
        winner=-1,
    )


# ---------------------------------------------------------------------------
# Legal actions & transitions
# ---------------------------------------------------------------------------


def legal_actions(state: MiniGinState) -> list[Action]:
    if state.phase == Phase.TERMINAL:
        return []
    if state.phase == Phase.DEAL:
        return []  # handled as chance
    if state.phase == Phase.DRAW:
        acts: list[Action] = []
        if state.deck:
            acts.append(DRAW_DECK)
        if state.discard:
            acts.append(DRAW_DISCARD)
        return acts
    # DISCARD phase: any card in hand may be discarded.
    hand = state.hands[state.current_player]
    return [make_discard(c) for c in hand]


def apply(state: MiniGinState, action: Action) -> MiniGinState:
    """Apply a *player* action. Chance moves use ``sample_deal``."""
    if state.phase == Phase.DRAW:
        return _apply_draw(state, action)
    if state.phase == Phase.DISCARD:
        return _apply_discard(state, action)
    raise ValueError(f"Cannot apply player action in phase {state.phase}")


def _apply_draw(state: MiniGinState, action: Action) -> MiniGinState:
    p = state.current_player
    hand = list(state.hands[p])
    if action.kind != "draw":
        raise ValueError("Expected draw action")
    if action.payload == 0:  # DRAW_DECK
        if not state.deck:
            raise ValueError("Deck empty")
        drawn = state.deck[-1]
        new_deck = state.deck[:-1]
        new_discard = state.discard
    else:  # DRAW_DISCARD
        if not state.discard:
            raise ValueError("Discard pile empty")
        drawn = state.discard[-1]
        new_discard = state.discard[:-1]
        new_deck = state.deck
    hand.append(drawn)
    hand.sort()
    new_hands = list(state.hands)
    new_hands[p] = tuple(hand)
    return MiniGinState(
        hands=(new_hands[0], new_hands[1]),
        deck=new_deck,
        discard=new_discard,
        current_player=p,
        phase=Phase.DISCARD,
        turn=state.turn,
        winner=-1,
    )


def _apply_discard(state: MiniGinState, action: Action) -> MiniGinState:
    p = state.current_player
    if action.kind != "discard":
        raise ValueError("Expected discard action")
    card = action.payload
    hand = list(state.hands[p])
    if card not in hand:
        raise ValueError(f"Cannot discard {card_str(card)}; not in hand")
    hand.remove(card)
    hand.sort()
    new_hands = list(state.hands)
    new_hands[p] = tuple(hand)
    new_discard = state.discard + (card,)
    # Auto-knock on gin.
    if is_gin_hand(tuple(hand)):
        return MiniGinState(
            hands=(new_hands[0], new_hands[1]),
            deck=state.deck,
            discard=new_discard,
            current_player=p,
            phase=Phase.TERMINAL,
            turn=state.turn + 1,
            winner=p,
        )
    # Pass turn to the other player.
    next_turn = state.turn + 1
    next_player = 1 - p
    # Termination on deck exhaustion or turn cap: draw.
    if next_turn >= MAX_TURNS or not state.deck and not new_discard:
        return MiniGinState(
            hands=(new_hands[0], new_hands[1]),
            deck=state.deck,
            discard=new_discard,
            current_player=next_player,
            phase=Phase.TERMINAL,
            turn=next_turn,
            winner=2,  # draw
        )
    # If the next player has no legal draw (both deck and discard empty)
    # something is wrong; treat as draw.
    if not state.deck and not new_discard:
        return MiniGinState(
            hands=(new_hands[0], new_hands[1]),
            deck=state.deck,
            discard=new_discard,
            current_player=next_player,
            phase=Phase.TERMINAL,
            turn=next_turn,
            winner=2,
        )
    return MiniGinState(
        hands=(new_hands[0], new_hands[1]),
        deck=state.deck,
        discard=new_discard,
        current_player=next_player,
        phase=Phase.DRAW,
        turn=next_turn,
        winner=-1,
    )


# ---------------------------------------------------------------------------
# Terminal / returns
# ---------------------------------------------------------------------------


def is_terminal(state: MiniGinState) -> bool:
    return state.phase == Phase.TERMINAL


def returns(state: MiniGinState) -> tuple[float, float]:
    if not is_terminal(state):
        raise ValueError("Non-terminal state has no returns")
    if state.winner == 2 or state.winner == -1:
        return (0.0, 0.0)
    if state.winner == 0:
        return (1.0, -1.0)
    return (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------


def information_set(state: MiniGinState, player: int) -> str:
    """Return a canonical string key for the information set observed by
    ``player`` at ``state``.

    We include: player id, phase, turn, the player's own hand (sorted), the
    top of the discard pile, deck size. We deliberately drop private action
    history for tractability (see module docstring).
    """
    hand = state.hands[player]
    top = state.top_discard()
    top_s = card_str(top) if top >= 0 else "-"
    hand_s = hand_str(hand)
    return f"P{player}|ph{int(state.phase)}|t{state.turn}|h{hand_s}|d{top_s}|k{len(state.deck)}"


__all__ = [
    "Action",
    "DRAW_DECK",
    "DRAW_DISCARD",
    "HAND_SIZE",
    "MAX_TURNS",
    "MiniGinState",
    "NUM_CARDS",
    "Phase",
    "apply",
    "card_str",
    "deadwood",
    "deal_outcomes",
    "hand_str",
    "information_set",
    "initial_state",
    "is_gin_hand",
    "is_terminal",
    "legal_actions",
    "make_discard",
    "returns",
    "sample_deal",
]
