"""Tests for :class:`SearchListeningPolicy` — the search-based Euchre
listener.

Kept small and deterministic:

1. A full seeded closed_loop_search game runs without raising.
2. The signal-conditioned play counter fires at least once across a
   handful of games (the search actually diverges from the inner
   heuristic on some deals).
3. On lead + STRONG_TRUMP + all-trump hand, the search never picks
   the strongest trump card (it should hold top trump so partner —
   who is claiming ≥ 3 trumps — can win the trick when we drop
   something small).
"""

from __future__ import annotations

import random

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.comms.euchre_search_listener import (
    ClosedLoopSearchTeam,
    HAND_CLASSES,
    SearchListeningPolicy,
    partner_hand_prior,
    sample_partner_hands,
)
from gin_rummy.comms.euchre_signals import (
    EuchreSignalKind,
    EuchreTalkingPolicy,
)
from gin_rummy.variants.euchre import (
    EuchreGame,
    EuchreObservation,
    EuchrePolicy,
    RandomEuchrePolicy,
    is_trump,
    _trump_rank,
)


# --------------------------------------------------------------- priors ---


def test_prior_shapes_match_signal_semantics():
    """Each signal maps to a normalised prior with the documented
    mass on the "matching" class."""
    remaining: list[Card] = []  # ignored by the current implementation
    trump = "♠"

    p = partner_hand_prior(EuchreSignalKind.STRONG_TRUMP, remaining, trump)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["ge3_trump"] == 0.70

    p = partner_hand_prior(EuchreSignalKind.BOWER, remaining, trump)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["holds_bower"] == 0.90

    p = partner_hand_prior(EuchreSignalKind.HAS_ACE, remaining, trump)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["holds_ace"] == 0.80

    p = partner_hand_prior(EuchreSignalKind.WEAK, remaining, trump)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["ge3_trump"] == 0.05

    p = partner_hand_prior(EuchreSignalKind.NONE, remaining, trump)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert set(p.keys()) == set(HAND_CLASSES)


def test_sample_partner_hands_respects_class():
    """A STRONG_TRUMP prior should mostly yield partner samples with
    ≥ 2 trumps (mass on ge3_trump + two_trump = 0.90)."""
    trump = "♠"
    prior = partner_hand_prior(EuchreSignalKind.STRONG_TRUMP, [], trump)
    # Full 24-card deck minus a plausible "my hand" (5 off-suit cards).
    from gin_rummy.variants.euchre import EUCHRE_RANKS, EUCHRE_SUITS

    full = [Card(r, s) for s in EUCHRE_SUITS for r in EUCHRE_RANKS]
    my_hand = [c for c in full if c.suit == "♥"][:5]
    remaining = [c for c in full if c not in my_hand]
    rng = random.Random(42)
    samples = sample_partner_hands(
        prior, remaining, my_hand, trump=trump, k=32, rng=rng
    )
    assert samples, "sampler produced no partner hands"
    # 90% of prior mass is on ≥2 trumps; sample means ≥ 60% is a safe
    # lower bound for k=32.
    two_plus = sum(
        1 for h in samples if sum(1 for c in h if is_trump(c, trump)) >= 2
    )
    assert two_plus / len(samples) >= 0.6


# --------------------------------------------------------------- game --


def test_search_listener_plays_a_full_game_without_error():
    """A closed_loop_search team must survive a full seeded hand."""
    channel = MessageChannel()
    rng = random.Random(2027)
    listener0, listener2 = ClosedLoopSearchTeam(
        seat0=RandomEuchrePolicy(rng),
        seat2=RandomEuchrePolicy(rng),
        channel=channel,
        k_samples=8,
        rng_seed=2027,
    )
    policies = [
        listener0,
        RandomEuchrePolicy(rng),
        listener2,
        RandomEuchrePolicy(rng),
    ]
    game = EuchreGame(policies=policies, seed=2027)
    result = game.play()
    assert result.outcome in {"made", "sweep", "euchre", "passout"}


def test_search_listener_fires_across_multiple_games():
    """Across a handful of seeded closed_loop_search games, the
    signal-conditioned play counter should tick above zero. This is a
    necessary condition for the treatment to have any behavioural
    difference vs. the silent baseline."""
    fired = 0
    plays = 0
    for seed in range(6):
        channel = MessageChannel()
        rng = random.Random(seed ^ 0xC0)
        listener0, listener2 = ClosedLoopSearchTeam(
            seat0=RandomEuchrePolicy(rng),
            seat2=RandomEuchrePolicy(rng),
            channel=channel,
            k_samples=8,
            rng_seed=seed,
        )
        policies = [
            listener0,
            RandomEuchrePolicy(rng),
            listener2,
            RandomEuchrePolicy(rng),
        ]
        game = EuchreGame(policies=policies, seed=seed)
        game.play()
        for p in (listener0, listener2):
            assert isinstance(p, SearchListeningPolicy)
            fired += p.plays_signal_conditioned
            plays += p.plays_total
    assert plays > 0
    assert fired > 0, (
        "search listener never diverged from inner across 6 games "
        f"(plays={plays})"
    )


# --------------------------- STRONG_TRUMP + all-trump lead -----------------


def test_search_never_leads_strongest_trump_on_strong_partner():
    """When partner signals STRONG_TRUMP and it's our lead and every
    legal play is trump, the search should not lead our strongest
    trump. Partner is claiming ≥3 trumps — likely including the right
    bower — so dumping a low trump lets partner take the trick with a
    top trump and keeps our own top trump for a later trick.

    We use a diamond trump here and a mixed-trump hand where the
    remaining deck actually has enough trumps outside our hand for
    the STRONG_TRUMP prior to admit valid partner samples. Then we
    restrict ``legal`` to just the trumps we hold (simulating a
    following-suit case where trump is what's playable, or a
    scenario where our non-trump has been played out) so the
    argmax is forced to choose within trumps.
    """
    trump = "♦"
    # Our hand has 3 trumps out of 6 total; partner can plausibly hold
    # J♦ + 2 others under the STRONG_TRUMP prior.
    hand = (
        Card("A", "♦"),
        Card("K", "♦"),
        Card("9", "♦"),
        Card("A", "♥"),
        Card("K", "♠"),
    )
    obs = EuchreObservation(
        player_id=0,
        hand=hand,
        trump_suit=trump,
        upcard=Card("A", "♦"),
        caller_id=0,
        partner_id=2,
        trick_history=(),
        trick_number=0,
        scores=(0, 0),
    )
    channel = MessageChannel()
    channel.broadcast(
        sender_id=2, turn=0, kind="signal",
        payload=EuchreSignalKind.STRONG_TRUMP.value,
    )

    class _AlwaysStrongestTrump(EuchrePolicy):
        def choose_play(self, obs, legal):  # type: ignore[override]
            trumps = [c for c in legal if is_trump(c, "♦")]
            pool = trumps if trumps else list(legal)
            return max(pool, key=lambda c: _trump_rank(c, "♦"))

    # Restrict legal to *only the trumps in hand* — mimics the case
    # from the task spec ("only-trump-in-hand legal moves"). Under
    # this restriction the search must still not pick the strongest
    # trump when partner claims STRONG_TRUMP.
    legal_trumps = [c for c in hand if is_trump(c, trump)]
    listener = SearchListeningPolicy(
        _AlwaysStrongestTrump(),
        channel,
        my_id=0,
        partner_id=2,
        k_samples=64,
        rng_seed=7,
    )
    chosen = listener.choose_play(obs, legal_trumps)

    strongest = max(legal_trumps, key=lambda c: _trump_rank(c, trump))
    assert chosen != strongest, (
        f"search picked strongest trump {chosen} (partner claimed STRONG_TRUMP)"
    )
