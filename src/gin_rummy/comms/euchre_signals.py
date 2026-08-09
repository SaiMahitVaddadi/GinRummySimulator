"""Structured signalling for Euchre partnerships.

Companion to ``comms/signals.py`` — that module wraps a *Gin* ``Policy``;
this one wraps an :class:`EuchrePolicy` because Euchre has a different
action space (call trump, then play a card per trick).

Signal alphabet (broadcast once trump is known, at the top of each trick
lead the wrapped policy sees):

* ``STRONG_TRUMP`` — hand holds ≥ 3 trumps.
* ``BOWER``        — hand holds the Jack of trump (right bower).
* ``HAS_ACE``      — hand holds any off-suit ace.
* ``WEAK``         — hand holds zero trumps.
* ``NONE``         — none of the above (no message emitted).

The talker/listener split mirrors ``comms/signals.py``:

* :class:`EuchreTalkingPolicy` broadcasts one structured signal per
  trick (on the lead) and delegates all play decisions to its inner
  policy. Purely a producer.
* :class:`EuchreListeningPolicy` reads the shared channel for the
  partner's most recent signal *for the current trick* and applies
  small, interpretable modifications to its inner policy's play. The
  listener is deterministic conditional on the observation + signal:
  no random tie-breaking beyond what the inner policy does. When there
  is no relevant signal, it delegates byte-for-byte to the inner. This
  is what closes the coordination loop so that partner-MI can move.
* :class:`EuchreByzantinePolicy` wraps an :class:`EuchreTalkingPolicy`
  and inverts its signals (strong→weak, bower→weak, ace→weak,
  weak→strong). Play behaviour is delegated to the honest wrapper's
  inner policy — the deception is purely on the channel.

The concrete listening rules (kept simple so their behavioural effect
is auditable):

* On ``STRONG_TRUMP`` or ``BOWER`` from partner + it's your lead:
  lead an off-suit low card so partner can ruff / take the trick with
  a top trump. (Follows: delegate.)
* On ``WEAK`` from partner: if the actor is currently winning the
  trick, hold high trump (play the lowest non-trump legal card);
  otherwise dump the lowest legal card. On lead, lead low.
* On ``HAS_ACE`` from partner + it's your lead: prefer leading a trump
  (any trump) if legal — clear trumps so partner's off-suit ace lives.
* On ``NONE`` / no message: delegate to inner.

``ClosedLoopTeam`` is a convenience factory that wires seats 0 and 2
as talk-and-listen (each seat is both a talker and a listener), which
is what the ``closed_loop`` treatment in ``experiments/signalling.py``
uses.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.variants.euchre import (
    CallDecision,
    EuchreObservation,
    EuchrePolicy,
    Suit,
    is_trump,
)


class EuchreSignalKind(Enum):
    NONE = "none"
    STRONG_TRUMP = "strong_trump"  # ≥ 3 trumps
    BOWER = "bower"                # right bower in hand
    HAS_ACE = "has_ace"            # any off-suit ace
    WEAK = "weak"                  # zero trumps


def classify_hand(hand: Sequence[Card], trump: Suit | None) -> EuchreSignalKind:
    """Deterministic mapping from (hand, trump) to a single signal kind.

    Ordered by informativeness — the strongest applicable label wins.
    Emitted only after trump is set; before that we return ``NONE`` so
    the wrapper is a no-op during trump-calling.
    """
    if trump is None:
        return EuchreSignalKind.NONE
    trumps = [c for c in hand if is_trump(c, trump)]
    has_bower = any(c.rank == "J" and c.suit == trump for c in trumps)
    if has_bower:
        return EuchreSignalKind.BOWER
    if len(trumps) >= 3:
        return EuchreSignalKind.STRONG_TRUMP
    if any(c.rank == "A" and c.suit != trump for c in hand):
        return EuchreSignalKind.HAS_ACE
    if len(trumps) == 0:
        return EuchreSignalKind.WEAK
    return EuchreSignalKind.NONE


class EuchreTalkingPolicy:
    """Wraps an :class:`EuchrePolicy`, emitting one structured signal per
    trick that this seat leads (i.e. once trump is known and it is this
    seat's turn to make the very first play of a trick).

    The wrapped policy's call/trump/play decisions are unchanged.
    """

    def __init__(
        self,
        inner: EuchrePolicy,
        channel: MessageChannel,
        sender_id: int,
    ) -> None:
        self._inner = inner
        self.channel = channel
        self.sender_id = sender_id
        # Track the trick number we most recently broadcast for, so we
        # emit at most one message per seat per trick even if
        # ``choose_play`` is called multiple times.
        self._last_signalled_trick: int = -1

    # ---- delegation ------------------------------------------------------

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return self._inner.choose_call(obs)

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return self._inner.choose_trump_suit(obs)

    # ---- signal emission -------------------------------------------------

    def _signal_for(self, obs: EuchreObservation) -> EuchreSignalKind:
        return classify_hand(obs.hand, obs.trump_suit)

    def _emit(self, obs: EuchreObservation, *, kind_str: str = "signal") -> None:
        if obs.trump_suit is None:
            return
        # One signal per (seat, trick).
        if obs.trick_number == self._last_signalled_trick:
            return
        # Only emit on the lead — inspect trick_history for cards from
        # the current trick. We reconstruct this by counting plays whose
        # index in trick_history is ≥ 4 * trick_number.
        plays_in_current_trick = len(obs.trick_history) - 4 * obs.trick_number
        if plays_in_current_trick != 0:
            return
        kind = self._signal_for(obs)
        if kind is EuchreSignalKind.NONE:
            self._last_signalled_trick = obs.trick_number
            return
        self.channel.broadcast(
            self.sender_id, obs.trick_number, kind_str, kind.value
        )
        self._last_signalled_trick = obs.trick_number

    def choose_play(
        self, obs: EuchreObservation, legal: Sequence[Card]
    ) -> Card:
        self._emit(obs)
        return self._inner.choose_play(obs, legal)


_INVERT: dict[EuchreSignalKind, EuchreSignalKind] = {
    EuchreSignalKind.STRONG_TRUMP: EuchreSignalKind.WEAK,
    EuchreSignalKind.BOWER: EuchreSignalKind.WEAK,
    EuchreSignalKind.HAS_ACE: EuchreSignalKind.WEAK,
    EuchreSignalKind.WEAK: EuchreSignalKind.STRONG_TRUMP,
    EuchreSignalKind.NONE: EuchreSignalKind.NONE,
}


def euchre_byzantine_swap(kind: EuchreSignalKind) -> EuchreSignalKind:
    """Adversarial inversion used by :class:`EuchreByzantinePolicy`."""
    return _INVERT.get(kind, kind)


class EuchreByzantinePolicy:
    """Wraps an :class:`EuchreTalkingPolicy` and inverts its signals.

    Play behaviour is delegated to the honest wrapper's inner policy, so
    the deception is purely on the message channel — matching
    :class:`ByzantinePolicy` for Gin.
    """

    def __init__(self, wrapped: EuchreTalkingPolicy) -> None:
        self._wrapped = wrapped

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return self._wrapped.choose_call(obs)

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return self._wrapped.choose_trump_suit(obs)

    def _emit_lie(self, obs: EuchreObservation) -> None:
        if obs.trump_suit is None:
            return
        if obs.trick_number == self._wrapped._last_signalled_trick:
            return
        plays_in_current_trick = len(obs.trick_history) - 4 * obs.trick_number
        if plays_in_current_trick != 0:
            return
        truth = self._wrapped._signal_for(obs)
        lie = euchre_byzantine_swap(truth)
        if lie is not EuchreSignalKind.NONE:
            self._wrapped.channel.broadcast(
                self._wrapped.sender_id,
                obs.trick_number,
                "byzantine",
                lie.value,
            )
        self._wrapped._last_signalled_trick = obs.trick_number

    def choose_play(
        self, obs: EuchreObservation, legal: Sequence[Card]
    ) -> Card:
        self._emit_lie(obs)
        return self._wrapped._inner.choose_play(obs, legal)


def _is_lead(obs: EuchreObservation) -> bool:
    """True iff the actor is about to lead the current trick.

    Mirrors :meth:`EuchreTalkingPolicy._emit` — a lead is when no card
    has yet been played in the current trick, i.e.
    ``len(trick_history) == 4 * trick_number``.
    """
    plays_in_current_trick = len(obs.trick_history) - 4 * obs.trick_number
    return plays_in_current_trick == 0


def _current_trick_plays(
    obs: EuchreObservation,
) -> tuple[tuple[Card, int], ...]:
    """The (card, seat) plays that have landed in the current trick so
    far, in play order. Empty when the actor is on lead."""
    start = 4 * obs.trick_number
    return obs.trick_history[start:]


def _current_winner_seat(
    obs: EuchreObservation, trump: Suit
) -> int | None:
    """Best (winning-so-far) seat in the current trick, or None if the
    actor is on lead."""
    plays = _current_trick_plays(obs)
    if not plays:
        return None
    # Import locally to avoid circular imports at module load time.
    from gin_rummy.variants.euchre import card_beats

    lead_suit = plays[0][0].suit
    best_card, best_seat = plays[0]
    for card, seat in plays[1:]:
        if card_beats(card, best_card, lead_suit=lead_suit, trump=trump):
            best_card, best_seat = card, seat
    return best_seat


def _partner_signal_for_trick(
    channel: MessageChannel, partner_id: int, trick_number: int
) -> EuchreSignalKind | None:
    """Most recent message from ``partner_id`` for ``trick_number``.

    Returns ``None`` if the partner has not signalled for this trick
    yet, or if the recorded payload is not a recognised
    :class:`EuchreSignalKind`. Payloads produced by
    :class:`EuchreByzantinePolicy` are honoured as-is — a listener
    trusts the channel; the byzantine treatment's whole point is to
    show what that trust costs."""
    latest: EuchreSignalKind | None = None
    for msg in channel.messages:
        if msg.sender_id != partner_id:
            continue
        if msg.turn != trick_number:
            continue
        # Only "signal" and "byzantine" kinds carry a EuchreSignalKind
        # payload. Anything else (e.g. free-form chat) is ignored.
        if msg.kind not in ("signal", "byzantine"):
            continue
        try:
            latest = EuchreSignalKind(msg.payload)
        except ValueError:
            continue
    return latest


def _lowest_by_rank(cards: Sequence[Card]) -> Card:
    """Pick the lowest-rank card by :data:`RANK_ORDER_EUCHRE`."""
    from gin_rummy.variants.euchre import RANK_ORDER_EUCHRE

    return min(cards, key=lambda c: RANK_ORDER_EUCHRE[c.rank])


def _highest_by_rank(cards: Sequence[Card], trump: Suit) -> Card:
    """Pick the highest-value trump-aware card in ``cards``."""
    from gin_rummy.variants.euchre import _trump_rank, RANK_ORDER_EUCHRE

    def key(c: Card) -> int:
        return _trump_rank(c, trump) if is_trump(c, trump) else RANK_ORDER_EUCHRE[c.rank]

    return max(cards, key=key)


class EuchreListeningPolicy:
    """Wraps an :class:`EuchrePolicy` and modifies its play using the
    partner's most recent signal on the shared :class:`MessageChannel`.

    Parameters
    ----------
    inner
        The base policy to delegate to when there is no signal, no
        trump yet, or the signal does not match a listening rule.
    channel
        Shared message log. This policy only *reads* it.
    my_id
        Seat of this policy (0..3). Used to filter observations.
    partner_id
        Seat of this policy's partner (``(my_id + 2) % 4``). Kept as a
        constructor arg so callers can be explicit / unit-test in
        isolation.

    Behavioural stats:
        * :attr:`plays_total` — total ``choose_play`` invocations.
        * :attr:`plays_signal_conditioned` — invocations where a
          partner signal was present *and* changed the play relative
          to what the inner policy would have chosen. Used by the
          experiment runner to report listener fidelity.
    """

    def __init__(
        self,
        inner: EuchrePolicy,
        channel: MessageChannel,
        my_id: int,
        partner_id: int,
    ) -> None:
        self._inner = inner
        self.channel = channel
        self.my_id = my_id
        self.partner_id = partner_id
        self.plays_total: int = 0
        self.plays_signal_conditioned: int = 0

    # ---- delegation of trump-calling (listener is not a caller override).

    def choose_call(self, obs: EuchreObservation) -> CallDecision:
        return self._inner.choose_call(obs)

    def choose_trump_suit(self, obs: EuchreObservation) -> Suit:
        return self._inner.choose_trump_suit(obs)

    # ---- the listening rules.

    def _rule_pick(
        self,
        obs: EuchreObservation,
        legal: Sequence[Card],
        signal: EuchreSignalKind,
    ) -> Card | None:
        """Return a card selected by the listening rule for ``signal``,
        or ``None`` to delegate to the inner policy."""
        trump = obs.trump_suit
        if trump is None:
            return None

        lead = _is_lead(obs)

        if signal in (EuchreSignalKind.STRONG_TRUMP, EuchreSignalKind.BOWER):
            # Partner has top trump. On lead, feed them: play an
            # off-suit low so they can ruff or take the trick with a
            # bower/ace of trumps.
            if lead:
                offsuit = [c for c in legal if not is_trump(c, trump)]
                if offsuit:
                    return _lowest_by_rank(offsuit)
                # All trumps: just play the lowest trump.
                return _lowest_by_rank(list(legal))
            # Following: delegate — the inner heuristic (RandomEuchrePolicy
            # here) is as good as anything without more state.
            return None

        if signal is EuchreSignalKind.WEAK:
            # Partner has no trumps. On lead, keep it cheap — lead low
            # off-suit if available (don't waste our own trumps).
            if lead:
                offsuit = [c for c in legal if not is_trump(c, trump)]
                pool = offsuit if offsuit else list(legal)
                return _lowest_by_rank(pool)
            # Following: check the current trick winner.
            winner = _current_winner_seat(obs, trump)
            if winner is None:
                return None
            if winner == self.partner_id:
                # We can't help — but WEAK partner won't be the winner
                # by ruffing, so this branch is uncommon. Play cheap.
                return _lowest_by_rank(list(legal))
            # Winner is an opponent (or ourselves already). Play the
            # lowest legal card — conserve trump for later tricks.
            return _lowest_by_rank(list(legal))

        if signal is EuchreSignalKind.HAS_ACE:
            # Partner has an off-suit ace. On lead, clear trumps if we
            # have any (so partner's ace runs later). Otherwise
            # delegate — we can't know which off-suit is safe to lead.
            if lead:
                trumps = [c for c in legal if is_trump(c, trump)]
                if trumps:
                    return _lowest_by_rank(trumps)
            return None

        return None

    def choose_play(
        self, obs: EuchreObservation, legal: Sequence[Card]
    ) -> Card:
        self.plays_total += 1
        base = self._inner.choose_play(obs, legal)
        signal = _partner_signal_for_trick(
            self.channel, self.partner_id, obs.trick_number
        )
        if signal is None or signal is EuchreSignalKind.NONE:
            return base
        picked = self._rule_pick(obs, legal, signal)
        if picked is None:
            return base
        if picked not in legal:
            # Defensive: should never trigger because rules only pick
            # from ``legal``, but keeps behaviour identical to
            # ``EuchreGame._play_trick``'s silent-correction fallback.
            return base
        if picked != base:
            self.plays_signal_conditioned += 1
        return picked


# --------------------------------------------------- closed-loop factory --


def ClosedLoopTeam(  # noqa: N802 — factory reads like a class instantiation
    seat0: EuchrePolicy,
    seat2: EuchrePolicy,
    channel: MessageChannel,
) -> tuple[EuchrePolicy, EuchrePolicy]:
    """Wire ``seat0`` and ``seat2`` as talk-and-listen partners.

    Each returned policy is an :class:`EuchreListeningPolicy` whose
    inner is the corresponding :class:`EuchreTalkingPolicy`. That means
    every play both (a) may broadcast a signal for the upcoming trick
    (via the inner talker's ``choose_play``) and (b) reads the
    partner's most recent signal for the current trick.

    The talkers/listeners share ``channel``. Callers that only want a
    one-sided listener can construct ``EuchreListeningPolicy`` directly.
    """
    talker0 = EuchreTalkingPolicy(seat0, channel, sender_id=0)
    talker2 = EuchreTalkingPolicy(seat2, channel, sender_id=2)
    listener0 = EuchreListeningPolicy(
        talker0, channel, my_id=0, partner_id=2
    )
    listener2 = EuchreListeningPolicy(
        talker2, channel, my_id=2, partner_id=0
    )
    return listener0, listener2


__all__ = [
    "ClosedLoopTeam",
    "EuchreByzantinePolicy",
    "EuchreListeningPolicy",
    "EuchreSignalKind",
    "EuchreTalkingPolicy",
    "classify_hand",
    "euchre_byzantine_swap",
]
