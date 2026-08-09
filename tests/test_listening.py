"""Tests for :class:`EuchreListeningPolicy` and the ``closed_loop`` treatment.

These lock in the read-side of the Euchre signalling channel — the
half that ``PAPER.md §6`` was missing when predictions #4 and #5 were
first written. Kept tight (small seeded games and one-trick synthetic
observations) so the suite still runs in under a second.
"""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.comms.euchre_signals import (
    ClosedLoopTeam,
    EuchreListeningPolicy,
    EuchreSignalKind,
    EuchreTalkingPolicy,
    _is_lead,
    _partner_signal_for_trick,
)
from gin_rummy.variants.euchre import (
    EuchreGame,
    EuchreObservation,
    EuchrePolicy,
    RandomEuchrePolicy,
    is_trump,
)


# ------------------------------------------------------------ full game ---


def test_listening_policy_plays_a_full_game_without_error():
    """A closed_loop team must be able to play a full seeded hand
    without raising, regardless of what deal the RNG produces."""
    channel = MessageChannel()
    rng = random.Random(2026)
    listener0, listener2 = ClosedLoopTeam(
        seat0=RandomEuchrePolicy(rng),
        seat2=RandomEuchrePolicy(rng),
        channel=channel,
    )
    policies = [
        listener0,
        RandomEuchrePolicy(rng),
        listener2,
        RandomEuchrePolicy(rng),
    ]
    game = EuchreGame(policies=policies, seed=2026)
    result = game.play()
    assert result.outcome in {"made", "sweep", "euchre", "passout"}


# --------------------------------------------------- rule: STRONG_TRUMP ---


def test_listener_leads_offsuit_low_on_strong_trump_from_partner():
    """When partner has broadcast STRONG_TRUMP for the current trick
    and it's our lead, the listener must emit an off-suit low card
    (feed partner so partner can ruff)."""
    trump = "♠"
    # Actor at seat 0, partner at seat 2. Actor has a mix; the
    # listener rule must pick the *lowest offsuit* (9♥) even though
    # the inner random policy would pick uniformly.
    hand = (
        Card("A", "♠"),  # trump, high
        Card("K", "♥"),  # offsuit face
        Card("9", "♥"),  # offsuit low  <-- must be picked
        Card("Q", "♦"),
        Card("10", "♣"),
    )
    obs = EuchreObservation(
        player_id=0,
        hand=hand,
        trump_suit=trump,
        upcard=Card("A", "♠"),
        caller_id=0,
        partner_id=2,
        trick_history=(),  # empty ⇒ on lead
        trick_number=0,
        scores=(0, 0),
    )
    assert _is_lead(obs)

    channel = MessageChannel()
    channel.broadcast(
        sender_id=2, turn=0, kind="signal",
        payload=EuchreSignalKind.STRONG_TRUMP.value,
    )

    # Inner policy that always picks the *first* legal card so the
    # rule change is observable and deterministic.
    class _FirstLegal(EuchrePolicy):
        def choose_play(self, obs, legal):  # type: ignore[override]
            return legal[0]

    listener = EuchreListeningPolicy(
        _FirstLegal(), channel, my_id=0, partner_id=2
    )
    chosen = listener.choose_play(obs, list(hand))
    # Rule fired: must be the lowest offsuit non-trump card (9♥).
    assert chosen == Card("9", "♥")
    assert not is_trump(chosen, trump)
    assert listener.plays_signal_conditioned == 1


def test_listener_leads_offsuit_low_on_bower_from_partner():
    """BOWER should trigger the same lead-offsuit-low behaviour as
    STRONG_TRUMP."""
    trump = "♠"
    hand = (
        Card("A", "♠"),
        Card("K", "♥"),
        Card("9", "♦"),  # lowest offsuit
        Card("Q", "♣"),
    )
    obs = EuchreObservation(
        player_id=0, hand=hand, trump_suit=trump,
        upcard=Card("A", "♠"), caller_id=0, partner_id=2,
        trick_history=(), trick_number=0, scores=(0, 0),
    )
    channel = MessageChannel()
    channel.broadcast(
        sender_id=2, turn=0, kind="signal",
        payload=EuchreSignalKind.BOWER.value,
    )

    class _AlwaysTrump(EuchrePolicy):
        def choose_play(self, obs, legal):  # type: ignore[override]
            return next(c for c in legal if is_trump(c, "♠"))

    listener = EuchreListeningPolicy(
        _AlwaysTrump(), channel, my_id=0, partner_id=2
    )
    chosen = listener.choose_play(obs, list(hand))
    assert chosen == Card("9", "♦")


# ------------------------------------------------- rule: no signal ⇒ delegate


def test_listener_matches_inner_when_channel_is_empty():
    """With no messages on the channel, the listener must produce
    byte-identical plays to its inner policy, on identically seeded
    games. This is the load-bearing property that lets the ``silent``
    baseline stay comparable to any listener-wrapped treatment."""
    seed = 1234
    # Bare inner run.
    rng_a = random.Random(seed ^ 0xAA)
    inner_only = [RandomEuchrePolicy(rng_a) for _ in range(4)]
    bare = EuchreGame(policies=inner_only, seed=seed)
    result_bare = bare.play()

    # Listener-wrapped run — same RNG stream, empty channel throughout.
    channel = MessageChannel()
    rng_b = random.Random(seed ^ 0xAA)
    wrapped = [
        EuchreListeningPolicy(
            RandomEuchrePolicy(rng_b), channel, my_id=0, partner_id=2
        ),
        RandomEuchrePolicy(rng_b),
        EuchreListeningPolicy(
            RandomEuchrePolicy(rng_b), channel, my_id=2, partner_id=0
        ),
        RandomEuchrePolicy(rng_b),
    ]
    wrapped_game = EuchreGame(policies=wrapped, seed=seed)
    result_wrapped = wrapped_game.play()

    # Same deal (same seed), same policy RNG stream, no messages ever
    # written ⇒ trick sequences must match card-for-card.
    def trick_cards(res):
        return [(seat, card) for tr in res.tricks for seat, card in tr.plays]

    assert trick_cards(result_bare) == trick_cards(result_wrapped)
    # Rule was never triggered.
    listener0 = wrapped[0]
    listener2 = wrapped[2]
    assert isinstance(listener0, EuchreListeningPolicy)
    assert isinstance(listener2, EuchreListeningPolicy)
    assert listener0.plays_signal_conditioned == 0
    assert listener2.plays_signal_conditioned == 0


# ---------------------------------------- partner signal parsing helper ---


def test_partner_signal_for_trick_returns_latest_on_this_trick():
    channel = MessageChannel()
    channel.broadcast(
        sender_id=2, turn=0, kind="signal",
        payload=EuchreSignalKind.WEAK.value,
    )
    channel.broadcast(
        sender_id=2, turn=1, kind="signal",
        payload=EuchreSignalKind.STRONG_TRUMP.value,
    )
    # Trick 0 lookup gets WEAK, not the later STRONG_TRUMP.
    got = _partner_signal_for_trick(channel, partner_id=2, trick_number=0)
    assert got is EuchreSignalKind.WEAK
    # Trick 1 lookup gets STRONG_TRUMP.
    got = _partner_signal_for_trick(channel, partner_id=2, trick_number=1)
    assert got is EuchreSignalKind.STRONG_TRUMP
    # Wrong partner ⇒ None.
    assert (
        _partner_signal_for_trick(channel, partner_id=1, trick_number=0)
        is None
    )


# ---------------------------------------------- closed_loop smoke @ N=6 ---


def test_closed_loop_treatment_smoke_n6():
    """End-to-end smoke: the closed_loop treatment must produce
    finite stats and at least *some* listener activity across 6 games
    (i.e. at least one signal broadcast, at least one listener call —
    the rule may or may not have fired, depending on the deal)."""
    from experiments.signalling import run_treatment

    r = run_treatment("closed_loop", n_games=6, seed=0, n_resamples=20)
    assert r.treatment == "closed_loop"
    assert r.n_games == 6
    assert math.isfinite(r.partner_mi_point)
    assert 0.0 <= r.team0_win_rate <= 1.0
    # Both talker and listener are wired.
    assert r.listener_plays_total > 0
    # Signal-conditioned fraction is a valid probability.
    assert 0.0 <= r.listener_signal_conditioned_fraction <= 1.0


# ---------------------------------------- listener honours byzantine kind


def test_listener_reads_byzantine_kind_messages_too():
    """The listener trusts the channel — a message tagged ``byzantine``
    from the partner seat must be treated as an equally-valid signal.
    This is what makes the ``byzantine`` arm meaningful once a
    listener is in place: lies now change plays."""
    trump = "♠"
    hand = (
        Card("A", "♠"),  # trump high (a full-trumps hand)
        Card("K", "♠"),
        Card("9", "♠"),
        Card("Q", "♠"),
        Card("10", "♠"),
    )
    obs = EuchreObservation(
        player_id=0, hand=hand, trump_suit=trump,
        upcard=Card("A", "♠"), caller_id=0, partner_id=2,
        trick_history=(), trick_number=0, scores=(0, 0),
    )
    channel = MessageChannel()
    # A "lie": byzantine partner says STRONG_TRUMP.
    channel.broadcast(
        sender_id=2, turn=0, kind="byzantine",
        payload=EuchreSignalKind.STRONG_TRUMP.value,
    )

    class _AlwaysHigh(EuchrePolicy):
        def choose_play(self, obs, legal):  # type: ignore[override]
            return legal[0]

    listener = EuchreListeningPolicy(
        _AlwaysHigh(), channel, my_id=0, partner_id=2
    )
    chosen = listener.choose_play(obs, list(hand))
    # STRONG_TRUMP rule: on lead + all-trump hand, fall back to lowest
    # trump. That's the "9♠" in our seeded hand.
    assert chosen == Card("9", "♠")
