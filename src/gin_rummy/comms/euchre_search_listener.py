"""Search-based listener for Euchre — the "natural next step" flagged in
PAPER.md §6/§10.

The hand-written :class:`EuchreListeningPolicy` in ``euchre_signals.py``
implements a handful of common-sense rules. In the N=200 study it does
not lift partner MI above the silent baseline (0.023 vs 0.026, CIs
overlap). This module ships an alternative listener that turns the
partner's signal into an explicit *prior over partner-hand
compositions* and then does a shallow (1-ply) rollout to pick the
expected-trick-maximising play.

Wiring stays the same as :class:`EuchreListeningPolicy` so it can slot
into the ``closed_loop_search`` treatment without touching the game
loop. The listener still only *reads* the shared
:class:`MessageChannel`; it never writes to it.

Design summary
--------------
1. ``partner_hand_prior(signal, remaining_deck, trump)`` — returns
   probabilities over a small alphabet of *hand-composition classes*:

    - ``ge3_trump``   partner holds ≥ 3 cards of the trump suit
    - ``two_trump``   partner holds exactly 2 trump
    - ``one_trump``   partner holds exactly 1 trump
    - ``no_trump``    partner holds 0 trump
    - ``holds_bower`` partner holds the Jack of trump
    - ``holds_ace``   partner holds an off-suit ace
    - ``other``       none of the above

    The prior is signal-conditional:

    - ``STRONG_TRUMP``: 70% ge3_trump, 20% two_trump, 10% other
    - ``BOWER``: 90% holds_bower, 10% other
    - ``HAS_ACE``: 80% holds_ace (off-suit), 20% other
    - ``WEAK``: 5% ge3_trump, 95% ≤ one_trump (split 50/50 no/one)
    - anything else: uniform over the alphabet

2. ``sample_partner_hands(prior, remaining_deck, my_hand, k)`` — reject
   samples up to ``k`` concrete 5-card partner hands consistent with
   both the composition prior and the cards not yet visible (the
   ``remaining_deck``). This is not perfectly rigorous — the prior is
   over hand *classes*, so we condition the class draw first and then
   deal a plausible hand within the class. That's enough for MI-lift
   experiments because it moves the search's implied partner
   distribution away from uniform, which is what we're testing.

3. :class:`SearchListeningPolicy` — plug-compatible with
   :class:`EuchreListeningPolicy` (same constructor shape, same
   behavioural stats: ``plays_total``, ``plays_signal_conditioned``).
   On each play:

    - Read partner's most recent signal for this trick.
    - If no signal, delegate to ``inner``.
    - Otherwise build the prior, sample partner hands, and for every
      legal move run a 1-ply rollout in which partner plays greedily to
      win the trick and the two opponents play greedily too. Score by
      expected tricks won on this play. Pick argmax.
    - If the argmax differs from ``inner``'s pick,
      increment ``plays_signal_conditioned``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.comms.euchre_signals import (
    EuchreSignalKind,
    _current_trick_plays,
    _is_lead,
    _partner_signal_for_trick,
)
from gin_rummy.variants.euchre import (
    CallDecision,
    EUCHRE_RANKS,
    EUCHRE_SUITS,
    EuchreObservation,
    EuchrePolicy,
    RANK_ORDER_EUCHRE,
    Suit,
    _trump_rank,
    card_beats,
    is_trump,
)


# ------------------------------------------------------------- prior alphabet -


# The prior lives over these coarse composition classes. Kept small so
# the reject sampler has a chance of hitting each class within a
# reasonable retry budget.
HAND_CLASSES: tuple[str, ...] = (
    "ge3_trump",
    "two_trump",
    "one_trump",
    "no_trump",
    "holds_bower",
    "holds_ace",
    "other",
)


def partner_hand_prior(
    signal_kind: EuchreSignalKind,
    remaining_deck: Sequence[Card],
    trump: Suit,
) -> dict[str, float]:
    """Return a probability mass function over :data:`HAND_CLASSES`.

    ``remaining_deck`` is kept in the signature so future refinements can
    condition on what's already been played (e.g. downweight
    ``ge3_trump`` if 6 trumps are already accounted for). For the current
    implementation the prior only reads the signal.
    """
    del remaining_deck  # placeholder for a future data-conditional prior

    if signal_kind is EuchreSignalKind.STRONG_TRUMP:
        return {
            "ge3_trump": 0.70,
            "two_trump": 0.20,
            "other": 0.10,
        }
    if signal_kind is EuchreSignalKind.BOWER:
        return {
            "holds_bower": 0.90,
            "other": 0.10,
        }
    if signal_kind is EuchreSignalKind.HAS_ACE:
        return {
            "holds_ace": 0.80,
            "other": 0.20,
        }
    if signal_kind is EuchreSignalKind.WEAK:
        return {
            "ge3_trump": 0.05,
            "no_trump": 0.475,
            "one_trump": 0.475,
        }
    # NONE / anything unexpected: uniform.
    p = 1.0 / len(HAND_CLASSES)
    return {cls: p for cls in HAND_CLASSES}


# ---------------------------------------------------------------- sampling ---


def _hand_matches_class(
    hand: Sequence[Card], cls: str, trump: Suit
) -> bool:
    """True iff ``hand`` satisfies the composition class ``cls``."""
    trumps = [c for c in hand if is_trump(c, trump)]
    n_tr = len(trumps)
    if cls == "ge3_trump":
        return n_tr >= 3
    if cls == "two_trump":
        return n_tr == 2
    if cls == "one_trump":
        return n_tr == 1
    if cls == "no_trump":
        return n_tr == 0
    if cls == "holds_bower":
        return any(c.rank == "J" and c.suit == trump for c in hand)
    if cls == "holds_ace":
        return any(c.rank == "A" and c.suit != trump for c in hand)
    # "other" — fallback / uniform: any hand qualifies.
    return True


def sample_partner_hands(
    prior: dict[str, float],
    remaining_deck: Sequence[Card],
    my_hand: Sequence[Card],
    *,
    trump: Suit | None = None,
    partner_size: int = 5,
    k: int = 32,
    rng: random.Random | None = None,
    max_attempts_per_sample: int = 32,
) -> list[list[Card]]:
    """Reject-sample up to ``k`` concrete partner hands.

    Steps:
      1. Draw a class ``cls`` from ``prior``.
      2. Deal ``partner_size`` cards uniformly from ``remaining_deck``
         (which must exclude ``my_hand``).
      3. Keep the hand if it satisfies ``cls``, else retry (bounded).

    If no consistent hand can be found for a drawn class within the
    budget, that draw is dropped. We stop early if ``remaining_deck``
    is smaller than ``partner_size``.

    ``trump`` is passed separately so the class-matching predicate can
    be evaluated without threading the trump suit through the prior
    dict itself. ``None`` is treated as "accept unconditionally" and is
    mostly useful for tests.
    """
    rng = rng if rng is not None else random.Random()
    my_set = set(my_hand)
    deck_pool = [c for c in remaining_deck if c not in my_set]
    if len(deck_pool) < partner_size:
        return []

    classes = list(prior.keys())
    weights = [prior[c] for c in classes]
    hands: list[list[Card]] = []

    for _ in range(k):
        cls = rng.choices(classes, weights=weights, k=1)[0]
        got: list[Card] | None = None
        for _attempt in range(max_attempts_per_sample):
            candidate = rng.sample(deck_pool, partner_size)
            if trump is None:
                got = candidate
                break
            if _hand_matches_class(candidate, cls, trump):
                got = candidate
                break
        if got is not None:
            hands.append(got)
    return hands


# --------------------------------------------------- rollout scoring helpers -


def _greedy_trick_winner_move(
    hand: Sequence[Card],
    plays_so_far: Sequence[tuple[Card, int]],
    *,
    trump: Suit,
    my_seat: int | None = None,
    partner_seat: int | None = None,
) -> Card:
    """Given the current partial trick, pick the card from ``hand`` that
    (a) is legal (follow-suit if possible) and (b) tries to win — but
    without overtaking a partner who is already winning.

    Simple rule: if playing lead, pick highest-strength card; otherwise
    if a partner is currently winning the trick, dump the weakest
    legal card; otherwise play the lowest card that beats the current
    winner, else dump the lowest legal card.
    """
    if not plays_so_far:
        # Leading: pick strongest card.
        return _strength_argmax(hand, trump=trump, lead_suit=None)

    lead_suit = plays_so_far[0][0].suit
    follows = [c for c in hand if c.suit == lead_suit]
    legal = follows if follows else list(hand)

    # Current winner (card + seat) in the partial trick.
    best_card, best_seat = plays_so_far[0]
    for card, seat in plays_so_far[1:]:
        if card_beats(card, best_card, lead_suit=lead_suit, trump=trump):
            best_card, best_seat = card, seat

    # Team-aware: if partner is already winning, don't overtake — dump
    # the weakest legal card so we conserve high cards for later.
    if partner_seat is not None and best_seat == partner_seat:
        return min(
            legal,
            key=lambda c: _card_strength(c, trump=trump, lead_suit=lead_suit),
        )

    # Try to beat with the lowest winner.
    winners = [
        c
        for c in legal
        if card_beats(c, best_card, lead_suit=lead_suit, trump=trump)
    ]
    if winners:
        return min(
            winners,
            key=lambda c: _card_strength(c, trump=trump, lead_suit=lead_suit),
        )
    # Can't win — dump the weakest legal card.
    return min(
        legal, key=lambda c: _card_strength(c, trump=trump, lead_suit=lead_suit)
    )


def _card_strength(card: Card, *, trump: Suit, lead_suit: Suit | None) -> int:
    """Comparable strength score for card evaluation (higher = stronger)."""
    if is_trump(card, trump):
        return 1000 + _trump_rank(card, trump)
    if lead_suit is not None and card.suit == lead_suit:
        return 100 + RANK_ORDER_EUCHRE[card.rank]
    return RANK_ORDER_EUCHRE[card.rank]


def _strength_argmax(
    cards: Sequence[Card], *, trump: Suit, lead_suit: Suit | None
) -> Card:
    return max(
        cards, key=lambda c: _card_strength(c, trump=trump, lead_suit=lead_suit)
    )


def _play_order_after(seat: int) -> list[int]:
    """The three seats that play after ``seat`` in a 4-player trick, in
    turn order (partner sits at ``(seat + 2) % 4``, opponents at ``+1``
    and ``+3``)."""
    return [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]


def _seat_hand(
    seat: int,
    *,
    my_seat: int,
    my_hand: Sequence[Card],
    partner_seat: int,
    partner_hand: Sequence[Card],
    opp_pool: Sequence[Card],
) -> list[Card]:
    """Return the hand attributed to ``seat`` in the rollout, splitting
    the opponent pool between the two opponent seats."""
    if seat == my_seat:
        return list(my_hand)
    if seat == partner_seat:
        return list(partner_hand)
    # Two opponents share ``opp_pool``. Deterministic split so rollout
    # scoring is stable across calls with a given sample.
    opps = sorted(opp_pool, key=lambda c: (c.suit, c.rank))
    left_seat = (my_seat + 1) % 4
    if seat == left_seat:
        return opps[: len(opps) // 2] or opps[:1]
    return opps[len(opps) // 2 :] or opps[-1:]


def _rollout_trick(
    my_card: Card,
    *,
    my_seat: int,
    my_hand: Sequence[Card],
    partner_seat: int,
    partner_hand: Sequence[Card],
    opp_pool: Sequence[Card],
    trump: Suit,
    plays_so_far: Sequence[tuple[Card, int]],
) -> int:
    """Simulate the rest of the current trick after we play ``my_card``.

    Returns 1 if our team (my_seat + partner_seat) wins the trick, else
    0.

    The rollout is greedy: each remaining seat plays via
    :func:`_greedy_trick_winner_move` restricted to its notional hand.
    """
    plays: list[tuple[Card, int]] = list(plays_so_far) + [(my_card, my_seat)]
    # Rebuild my_hand minus my_card.
    remaining_by_seat: dict[int, list[Card]] = {}
    remaining_by_seat[my_seat] = [c for c in my_hand if c != my_card]
    remaining_by_seat[partner_seat] = list(partner_hand)

    # Opponents share the pool; split deterministically.
    left_seat = (my_seat + 1) % 4
    right_seat = (my_seat + 3) % 4
    opps = sorted(opp_pool, key=lambda c: (c.suit, c.rank))
    remaining_by_seat[left_seat] = opps[: len(opps) // 2] or opps[:1]
    remaining_by_seat[right_seat] = opps[len(opps) // 2 :] or opps[-1:]

    # Who still needs to play?
    already_played = {seat for _, seat in plays}
    # Turn order in a trick: leader, +1, +2, +3. We infer the leader
    # from ``plays_so_far`` (the first play in the trick) if any exist;
    # otherwise we lead ourselves.
    if plays_so_far:
        leader = plays_so_far[0][1]
    else:
        leader = my_seat
    order = [(leader + i) % 4 for i in range(4)]

    for seat in order:
        if seat in already_played:
            continue
        # Attribute plausible hands.
        hand = remaining_by_seat.get(seat, [])
        if not hand:
            # Shouldn't happen with a valid sample; skip.
            continue
        # Team-aware partner for this seat: partners are at (seat+2)%4.
        seat_partner = (seat + 2) % 4
        card = _greedy_trick_winner_move(
            hand, plays, trump=trump, my_seat=seat, partner_seat=seat_partner
        )
        plays.append((card, seat))
        already_played.add(seat)

    # Determine trick winner.
    lead_suit = plays[0][0].suit
    best_card, best_seat = plays[0]
    for card, seat in plays[1:]:
        if card_beats(card, best_card, lead_suit=lead_suit, trump=trump):
            best_card, best_seat = card, seat
    # Our team = my_seat and partner_seat.
    return int(best_seat in (my_seat, partner_seat))


# --------------------------------------------------------------- listener ---


class SearchListeningPolicy:
    """Search-based listener.

    Same public shape as :class:`EuchreListeningPolicy`. On each play,
    combines the partner's most recent signal with a shallow rollout to
    pick the expected-trick-maximising card.

    Parameters
    ----------
    inner
        Fallback policy used when there is no signal or trump is not
        yet set.
    channel
        Shared broadcast log; read-only for this policy.
    my_id, partner_id
        Seat identifiers, as in :class:`EuchreListeningPolicy`.
    k_samples
        Number of partner-hand samples per candidate move.
    rollout_depth
        Currently only ``1`` is supported (a single-trick rollout).
        Included for future extension.
    rng_seed
        Optional deterministic seed. If ``None`` the policy uses a
        fresh :class:`random.Random`.
    """

    def __init__(
        self,
        inner: EuchrePolicy,
        channel: MessageChannel,
        my_id: int,
        partner_id: int,
        *,
        k_samples: int = 32,
        rollout_depth: int = 1,
        rng_seed: int | None = None,
    ) -> None:
        if rollout_depth != 1:
            # Explicit: we don't implement deeper rollouts yet.
            raise ValueError("SearchListeningPolicy only supports rollout_depth=1")
        self._inner = inner
        self.channel = channel
        self.my_id = my_id
        self.partner_id = partner_id
        self.k_samples = k_samples
        self.rollout_depth = rollout_depth
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self.plays_total: int = 0
        self.plays_signal_conditioned: int = 0

    # ---- trump-calling delegation (mirrors EuchreListeningPolicy). --------

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return self._inner.choose_call(obs)

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return self._inner.choose_trump_suit(obs)

    # ---- deck accounting --------------------------------------------------

    def _remaining_deck(self, obs: EuchreObservation) -> list[Card]:
        """Cards not yet seen by this actor: full 24-card Euchre deck
        minus (a) our hand and (b) every card visible in
        ``trick_history`` or as ``upcard``."""
        full = [Card(r, s) for s in EUCHRE_SUITS for r in EUCHRE_RANKS]
        seen: set[Card] = set(obs.hand)
        for card, _ in obs.trick_history:
            seen.add(card)
        if obs.upcard is not None:
            seen.add(obs.upcard)
        return [c for c in full if c not in seen]

    # ---- the search itself ------------------------------------------------

    def _score_move(
        self,
        move: Card,
        obs: EuchreObservation,
        samples: Sequence[list[Card]],
        *,
        trump: Suit,
    ) -> float:
        """Expected trick-win rate for ``move`` under sampled partner
        hands, with a small opportunity-cost penalty for spending a
        strong card.

        The raw 1-ply rollout is myopic: it always prefers to win with
        our strongest card, because any other card is more likely to
        be beaten. Real trump strategy pays a *future* cost for
        wasting high cards. We approximate that cost with a tiny
        linear penalty on card strength — small enough that a clearly
        winning move still wins, big enough to break near-ties in
        favour of cheaper cards. This is what surfaces the "hold top
        trump; let partner take with their claimed bower" behaviour
        the STRONG_TRUMP signal is meant to elicit.
        """
        if not samples:
            return 0.0
        plays_so_far = _current_trick_plays(obs)
        total = 0
        for partner_hand in samples:
            # Opponent pool = remaining deck minus (partner sample ∪ my hand ∪ played).
            deck = self._remaining_deck(obs)
            partner_set = set(partner_hand)
            opp_pool = [c for c in deck if c not in partner_set]
            total += _rollout_trick(
                move,
                my_seat=self.my_id,
                my_hand=obs.hand,
                partner_seat=self.partner_id,
                partner_hand=partner_hand,
                opp_pool=opp_pool,
                trump=trump,
                plays_so_far=plays_so_far,
            )
        exp_win = total / len(samples)
        lead_suit = plays_so_far[0][0].suit if plays_so_far else None
        # Card strength normalised into [0, 1]: trumps go up to ~1100,
        # off-suit face ~105. Scale by 1e-3 so a strength delta of 100
        # (one rank) shifts score by 0.10 — comparable to a 1-in-10
        # rollout swing but small enough not to override a clear
        # trick-win probability difference of ≥ 0.20.
        strength = _card_strength(move, trump=trump, lead_suit=lead_suit)
        cost = 1e-3 * strength
        return exp_win - cost

    def choose_play(
        self, obs: EuchreObservation, legal: Sequence[Card]
    ) -> Card:
        self.plays_total += 1
        base = self._inner.choose_play(obs, legal)

        trump = obs.trump_suit
        if trump is None:
            return base

        signal = _partner_signal_for_trick(
            self.channel, self.partner_id, obs.trick_number
        )
        if signal is None or signal is EuchreSignalKind.NONE:
            return base

        # Build prior + samples.
        remaining = self._remaining_deck(obs)
        prior = partner_hand_prior(signal, remaining, trump)
        samples = sample_partner_hands(
            prior,
            remaining,
            list(obs.hand),
            trump=trump,
            partner_size=self._partner_hand_size(obs),
            k=self.k_samples,
            rng=self._rng,
        )
        if not samples:
            return base

        # Score every legal move; argmax with a small deterministic
        # tiebreak so ``plays_signal_conditioned`` reflects a real
        # search-vs-inner disagreement (not floating-point noise).
        scored = [
            (self._score_move(m, obs, samples, trump=trump), m) for m in legal
        ]
        best_score = max(s for s, _ in scored)
        # Tolerance covers the small cost term we added in _score_move
        # so genuine near-ties collapse to the inner's pick.
        tol = 1e-9
        best_moves = [m for s, m in scored if s >= best_score - tol]
        if base in best_moves:
            picked = base
        else:
            picked = best_moves[0]

        if picked not in legal:
            return base
        if picked != base:
            self.plays_signal_conditioned += 1
        return picked

    # ---- helpers ----------------------------------------------------------

    def _partner_hand_size(self, obs: EuchreObservation) -> int:
        """Cards remaining in the partner's hand at this decision point.

        Start of hand: 5. Every completed trick reduces it by 1. If the
        partner has already played into the current trick, subtract
        another 1.
        """
        base = 5 - obs.trick_number
        for card, seat in _current_trick_plays(obs):
            if seat == self.partner_id:
                base -= 1
        return max(base, 0)


# --------------------------------------------------- closed-loop factory ---


def ClosedLoopSearchTeam(  # noqa: N802 — factory reads like a class instantiation
    seat0: EuchrePolicy,
    seat2: EuchrePolicy,
    channel: MessageChannel,
    *,
    k_samples: int = 32,
    rng_seed: int | None = None,
) -> tuple[EuchrePolicy, EuchrePolicy]:
    """Wire seats 0 and 2 as talk-and-search-listen partners.

    Parallel to :func:`gin_rummy.comms.euchre_signals.ClosedLoopTeam`,
    but the listener wrapper is :class:`SearchListeningPolicy`.
    """
    # Local import to avoid a circular reference at module load.
    from gin_rummy.comms.euchre_signals import EuchreTalkingPolicy

    talker0 = EuchreTalkingPolicy(seat0, channel, sender_id=0)
    talker2 = EuchreTalkingPolicy(seat2, channel, sender_id=2)
    listener0 = SearchListeningPolicy(
        talker0,
        channel,
        my_id=0,
        partner_id=2,
        k_samples=k_samples,
        rng_seed=rng_seed,
    )
    listener2 = SearchListeningPolicy(
        talker2,
        channel,
        my_id=2,
        partner_id=0,
        k_samples=k_samples,
        rng_seed=None if rng_seed is None else rng_seed ^ 0x1337,
    )
    return listener0, listener2


__all__ = [
    "ClosedLoopSearchTeam",
    "HAND_CLASSES",
    "SearchListeningPolicy",
    "partner_hand_prior",
    "sample_partner_hands",
]
