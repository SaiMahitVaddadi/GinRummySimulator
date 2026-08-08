"""Byzantine cheap-talk: one partner emits deceptive signals.

Wraps any ``TalkingPolicy`` and inverts the emitted signal — e.g. broadcast
``WEAK_HAND`` when actually strong. Reproduces the adversarial-signal
pattern from El Mir, Takáč & Lahlou (arXiv:2606.07790, NETYS 2026).

Usage
-----
    channel = MessageChannel()
    honest = TalkingPolicy(GreedyKnockPolicy(rng), channel, sender_id=0)
    liar = ByzantinePolicy(TalkingPolicy(GreedyKnockPolicy(rng), channel, 1))
    # run game — inspect channel.messages to see the deception log.
"""

from __future__ import annotations

from gin_rummy.cards import Card
from gin_rummy.comms.signals import SignalKind, TalkingPolicy
from gin_rummy.policy import DrawSource, Observation


_INVERT = {
    SignalKind.WEAK_HAND: SignalKind.CLOSE_TO_KNOCK,
    SignalKind.CLOSE_TO_KNOCK: SignalKind.WEAK_HAND,
    SignalKind.GIN_READY: SignalKind.WEAK_HAND,
    SignalKind.NONE: SignalKind.NONE,
    SignalKind.HELP_ME: SignalKind.CLOSE_TO_KNOCK,
}


def byzantine_swap(kind: SignalKind) -> SignalKind:
    return _INVERT.get(kind, kind)


class ByzantinePolicy:
    """Adversarial variant of ``TalkingPolicy``: inverts its signal, plays
    honestly. The inner policy's own action distribution is untouched."""

    def __init__(self, wrapped: TalkingPolicy) -> None:
        self._wrapped = wrapped

    def _emit_deceptive(self, obs: Observation) -> None:
        true_kind = self._wrapped._signal_for(obs)
        lie = byzantine_swap(true_kind)
        if lie is SignalKind.NONE:
            return
        self._wrapped.channel.broadcast(
            self._wrapped.sender_id, obs.turn_number, "byzantine", lie.value
        )

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        self._emit_deceptive(obs)
        return self._wrapped._inner.choose_draw_source(obs)

    def choose_discard(self, obs: Observation) -> Card:
        return self._wrapped._inner.choose_discard(obs)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._wrapped._inner.choose_to_knock(obs, deadwood_value)
