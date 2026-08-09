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

Semantics mirror ``TalkingPolicy``: the wrapped policy plays normally;
the wrapper only broadcasts on the shared :class:`MessageChannel` so
that the sample-collection machinery in ``comms.analyze`` can pick up
the messages as a side-channel history if needed. The wrapper does
*not* itself read the channel to change behaviour — the coordination
benefit, if any, must show up in the plug-in MI estimator.

``EuchreByzantinePolicy`` wraps an :class:`EuchreTalkingPolicy` and
inverts the emitted signal (strong→weak, bower→weak, ace→weak,
weak→strong). Play behaviour is untouched — the deception is purely on
the channel — matching the pattern in ``comms/byzantine.py``.
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


__all__ = [
    "EuchreByzantinePolicy",
    "EuchreSignalKind",
    "EuchreTalkingPolicy",
    "classify_hand",
    "euchre_byzantine_swap",
]
