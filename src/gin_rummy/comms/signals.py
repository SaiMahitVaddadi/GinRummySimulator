"""Structured signal vocabulary.

Enum-style cheap talk: a small, finite alphabet lets you isolate the
*content* of communication from the *representation* (natural language
adds prompt/parse noise on top). Following the arXiv:2510.05748 pattern
where a single-word Stag-Hunt signal raised cooperation from 0% → 96.7%.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation, Policy


class SignalKind(Enum):
    NONE = "none"
    WEAK_HAND = "weak_hand"          # deadwood > threshold
    CLOSE_TO_KNOCK = "close_to_knock"  # deadwood ≤ knock_limit + slack
    GIN_READY = "gin_ready"          # deadwood == 0 (rarely broadcast honestly)
    HELP_ME = "help_me"              # narrative: needs partner support (Euchre-style)


@dataclass(frozen=True)
class StructuredSignal:
    kind: SignalKind
    turn: int
    sender_id: int


class TalkingPolicy:
    """Wraps a base policy and emits a structured signal each turn based on
    the current hand strength. The signal is broadcast on the shared
    ``MessageChannel`` so partners can observe it.
    """

    def __init__(
        self,
        inner: Policy,
        channel: MessageChannel,
        sender_id: int,
        *,
        weak_threshold: int = 30,
        close_slack: int = 3,
    ) -> None:
        self._inner = inner
        self.channel = channel
        self.sender_id = sender_id
        self.weak_threshold = weak_threshold
        self.close_slack = close_slack

    def _signal_for(self, obs: Observation) -> SignalKind:
        _, _, dv = optimal_decomposition(list(obs.hand))
        if dv == 0:
            return SignalKind.GIN_READY
        if dv <= obs.knock_limit + self.close_slack:
            return SignalKind.CLOSE_TO_KNOCK
        if dv >= self.weak_threshold:
            return SignalKind.WEAK_HAND
        return SignalKind.NONE

    def _emit(self, obs: Observation) -> None:
        kind = self._signal_for(obs)
        if kind is SignalKind.NONE:
            return
        self.channel.broadcast(
            self.sender_id, obs.turn_number, "signal", kind.value
        )

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        self._emit(obs)
        return self._inner.choose_draw_source(obs)

    def choose_discard(self, obs: Observation) -> Card:
        return self._inner.choose_discard(obs)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._inner.choose_to_knock(obs, deadwood_value)
