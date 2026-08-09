"""Euchre — 4-player partnership trick-taking on a 24-card deck.

Simplified rules (enough for coordination studies; not a rules-perfect
tournament implementation):

* 24 cards: 9, 10, J, Q, K, A × four suits.
* Four players in fixed cross-partnerships: seats 0 and 2 vs. 1 and 3.
* Deal 5 cards each; the top of the remaining deck flips as the upcard.
* One round of trump-calling: each player in order (dealer last) can
  accept the upcard's suit as trump ("order it up") or pass. If everyone
  passes, a second round lets players name a different suit or pass out
  the hand.
* No going alone, no left-bower promotion (the Jack of the trump suit is
  still the highest trump; suit ordering follows the pinochle-style rank
  order below).
* Five tricks: winning 3 = 1 point, all 5 = 2 points; "euchre" (defenders
  win 3+ when opponents called trump) = 2 points to defenders.

The point of this class is to give ``comms`` and ``eval`` a partnership
substrate where an information-theoretic coordination score is
meaningful.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Sequence

from gin_rummy.cards import Card

EUCHRE_RANKS: tuple[str, ...] = ("9", "10", "J", "Q", "K", "A")
EUCHRE_SUITS: tuple[str, ...] = ("♠", "♥", "♦", "♣")
RANK_ORDER_EUCHRE = {r: i for i, r in enumerate(EUCHRE_RANKS)}

Suit = str
CallDecision = Literal["accept", "pass"]


def build_euchre_deck(rng: random.Random | None = None) -> list[Card]:
    rng = rng if rng is not None else random.Random()
    cards = [Card(r, s) for s in EUCHRE_SUITS for r in EUCHRE_RANKS]
    rng.shuffle(cards)
    return cards


def is_trump(card: Card, trump_suit: Suit) -> bool:
    """The Jack of the trump suit (right bower) is trump; simplified rules
    keep the left bower in its native suit."""
    return card.suit == trump_suit


def card_beats(a: Card, b: Card, *, lead_suit: Suit, trump: Suit) -> bool:
    """Return True iff card ``a`` beats card ``b`` in a trick led by ``lead_suit``.

    Simplified precedence: trump beats non-trump; within trump, the Jack of
    trump (right bower) is highest, otherwise rank order; within lead suit,
    plain rank order; off-suit non-trump does not beat.
    """
    a_trump = is_trump(a, trump)
    b_trump = is_trump(b, trump)
    if a_trump and not b_trump:
        return True
    if b_trump and not a_trump:
        return False
    if a_trump and b_trump:
        return _trump_rank(a, trump) > _trump_rank(b, trump)
    # Both non-trump.
    a_lead = a.suit == lead_suit
    b_lead = b.suit == lead_suit
    if a_lead and not b_lead:
        return True
    if b_lead and not a_lead:
        return False
    if a_lead and b_lead:
        return RANK_ORDER_EUCHRE[a.rank] > RANK_ORDER_EUCHRE[b.rank]
    return False  # off-suit doesn't beat off-suit


def _trump_rank(card: Card, trump: Suit) -> int:
    if card.rank == "J" and card.suit == trump:
        return 100  # right bower: strictly highest
    return RANK_ORDER_EUCHRE[card.rank]


@dataclass
class EuchreObservation:
    """What a player sees at decision time."""

    player_id: int
    hand: tuple[Card, ...]
    trump_suit: Suit | None
    upcard: Card | None
    caller_id: int | None
    partner_id: int
    trick_history: tuple[tuple[Card, int], ...]  # (card, seat) per play
    trick_number: int
    scores: tuple[int, int]  # team 0, team 1


class EuchrePolicy:
    """Protocol for Euchre agents. Distinct from Gin ``Policy`` because the
    action space is different (call trump, play card, pass)."""

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return "pass"

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return obs.upcard.suit if obs.upcard else EUCHRE_SUITS[0]

    def choose_play(self, obs: EuchreObservation, legal: Sequence[Card]) -> Card:
        return legal[0]


class RandomEuchrePolicy(EuchrePolicy):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return "accept" if self._rng.random() < 0.4 else "pass"

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return self._rng.choice(EUCHRE_SUITS)

    def choose_play(self, obs: EuchreObservation, legal: Sequence[Card]) -> Card:
        return self._rng.choice(list(legal))


@dataclass
class TrickPlay:
    trick_number: int
    plays: list[tuple[int, Card]] = field(default_factory=list)
    winner_id: int | None = None
    lead_suit: Suit | None = None


@dataclass
class EuchreResult:
    outcome: str  # "made" | "euchre" | "sweep" | "passout"
    caller_id: int | None
    trump_suit: Suit | None
    tricks: list[TrickPlay]
    scores: tuple[int, int]
    winner_team: int | None


class EuchreGame:
    """A single deal of Euchre. Fixed partnerships: {0,2} vs. {1,3}."""

    def __init__(
        self,
        *,
        policies: Sequence[EuchrePolicy] | None = None,
        seed: int | None = None,
        dealer: int = 3,
    ) -> None:
        if policies is not None and len(policies) != 4:
            raise ValueError("Euchre needs exactly 4 policies")
        self.rng = random.Random(seed)
        self.policies: list[EuchrePolicy] = list(policies) if policies else [
            RandomEuchrePolicy(self.rng) for _ in range(4)
        ]
        self.dealer = dealer
        self.hands: list[list[Card]] = [[] for _ in range(4)]
        self.upcard: Card | None = None
        self.trump_suit: Suit | None = None
        self.caller_id: int | None = None
        self.tricks: list[TrickPlay] = []
        self.tricks_won = [0, 0, 0, 0]

    # ---------- setup ----------

    def _deal(self) -> None:
        deck = build_euchre_deck(self.rng)
        for i in range(5):
            for seat in range(4):
                self.hands[seat].append(deck.pop())
        self.upcard = deck.pop() if deck else None

    def _left_of(self, seat: int) -> int:
        return (seat + 1) % 4

    def _partner_of(self, seat: int) -> int:
        return (seat + 2) % 4

    def _team_of(self, seat: int) -> int:
        return seat % 2

    # ---------- trump calling ----------

    def _call_trump(self) -> bool:
        """Return True if trump was called; False for a pass-out."""
        assert self.upcard is not None
        order = [(self.dealer + 1 + i) % 4 for i in range(4)]
        # First round: accept the upcard's suit or pass.
        for seat in order:
            policy = self.policies[seat]
            obs = self._make_obs(seat)
            if policy.choose_call(obs) == "accept":
                self.trump_suit = self.upcard.suit
                self.caller_id = seat
                return True
        # Second round: name a different suit or pass.
        for seat in order:
            policy = self.policies[seat]
            obs = self._make_obs(seat)
            if policy.choose_call(obs) == "accept":
                pick = policy.choose_trump_suit(obs)
                if pick != self.upcard.suit:
                    self.trump_suit = pick
                    self.caller_id = seat
                    return True
        return False

    # ---------- observation ----------

    def _make_obs(self, seat: int) -> EuchreObservation:
        return EuchreObservation(
            player_id=seat,
            hand=tuple(self.hands[seat]),
            trump_suit=self.trump_suit,
            upcard=self.upcard,
            caller_id=self.caller_id,
            partner_id=self._partner_of(seat),
            trick_history=tuple(
                (card, s) for tr in self.tricks for s, card in tr.plays
            ),
            trick_number=len(self.tricks),
            scores=(0, 0),  # per-hand scoring computed at the end
        )

    # ---------- play ----------

    def _legal_plays(self, seat: int, lead_suit: Suit | None) -> list[Card]:
        hand = self.hands[seat]
        if lead_suit is None:
            return list(hand)
        follows = [c for c in hand if c.suit == lead_suit]
        return follows if follows else list(hand)

    def _play_trick(self, leader: int) -> TrickPlay:
        trick = TrickPlay(trick_number=len(self.tricks))
        lead_suit: Suit | None = None
        for i in range(4):
            seat = (leader + i) % 4
            legal = self._legal_plays(seat, lead_suit)
            obs = self._make_obs(seat)
            card = self.policies[seat].choose_play(obs, legal)
            if card not in legal:
                card = legal[0]  # bad policies are silently corrected
            self.hands[seat].remove(card)
            trick.plays.append((seat, card))
            if lead_suit is None:
                lead_suit = card.suit
        # Determine winner.
        assert self.trump_suit is not None
        assert lead_suit is not None
        trick.lead_suit = lead_suit
        best_seat, best_card = trick.plays[0]
        for seat, card in trick.plays[1:]:
            if card_beats(card, best_card, lead_suit=lead_suit, trump=self.trump_suit):
                best_seat, best_card = seat, card
        trick.winner_id = best_seat
        return trick

    # ---------- main ----------

    def play(self) -> EuchreResult:
        self._deal()
        called = self._call_trump()
        if not called:
            return EuchreResult(
                outcome="passout",
                caller_id=None,
                trump_suit=None,
                tricks=[],
                scores=(0, 0),
                winner_team=None,
            )

        leader = self._left_of(self.dealer)
        for _ in range(5):
            trick = self._play_trick(leader)
            self.tricks.append(trick)
            assert trick.winner_id is not None
            self.tricks_won[trick.winner_id] += 1
            leader = trick.winner_id

        team_tricks = [
            self.tricks_won[0] + self.tricks_won[2],
            self.tricks_won[1] + self.tricks_won[3],
        ]
        assert self.caller_id is not None
        caller_team = self._team_of(self.caller_id)
        callers = team_tricks[caller_team]
        defenders = team_tricks[1 - caller_team]

        scores = [0, 0]
        if callers >= 3:
            scores[caller_team] = 2 if callers == 5 else 1
            outcome = "sweep" if callers == 5 else "made"
        else:
            scores[1 - caller_team] = 2  # euchre
            outcome = "euchre"

        winner_team = 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else None
        return EuchreResult(
            outcome=outcome,
            caller_id=self.caller_id,
            trump_suit=self.trump_suit,
            tricks=list(self.tricks),
            scores=(scores[0], scores[1]),
            winner_team=winner_team,
        )
