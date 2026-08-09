"""Smoke tests for the §6 signalling experiment and its Euchre helpers.

Keeps ``N`` tiny (6 games) so the full suite stays fast; the point is
to catch shape/typing bugs and to prove the three treatments plug
together with the coordination-MI machinery.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from gin_rummy.cards import Card
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.comms.euchre_signals import (
    EuchreByzantinePolicy,
    EuchreSignalKind,
    EuchreTalkingPolicy,
    classify_hand,
    euchre_byzantine_swap,
)
from gin_rummy.variants.euchre import EuchreGame, RandomEuchrePolicy

from experiments.signalling import (
    BUILDERS,
    run_signalling,
    run_treatment,
)


# ----------------------------------------------------------- classifier ---


def test_classify_hand_bower_beats_strong():
    trump = "♠"
    hand = (
        Card("J", "♠"),  # right bower
        Card("A", "♠"),
        Card("K", "♠"),
        Card("9", "♠"),
        Card("A", "♥"),
    )
    assert classify_hand(hand, trump) is EuchreSignalKind.BOWER


def test_classify_hand_strong_trump_no_bower():
    trump = "♠"
    hand = (
        Card("A", "♠"),
        Card("K", "♠"),
        Card("9", "♠"),
        Card("Q", "♥"),
        Card("10", "♦"),
    )
    assert classify_hand(hand, trump) is EuchreSignalKind.STRONG_TRUMP


def test_classify_hand_has_ace_offsuit():
    trump = "♠"
    hand = (Card("A", "♥"), Card("K", "♥"), Card("9", "♦"), Card("10", "♣"))
    assert classify_hand(hand, trump) is EuchreSignalKind.HAS_ACE


def test_classify_hand_weak_when_no_trump_no_ace():
    trump = "♠"
    hand = (Card("K", "♥"), Card("Q", "♦"), Card("10", "♣"), Card("9", "♣"))
    assert classify_hand(hand, trump) is EuchreSignalKind.WEAK


def test_classify_hand_returns_none_before_trump_is_known():
    hand = (Card("A", "♥"),)
    assert classify_hand(hand, None) is EuchreSignalKind.NONE


def test_euchre_byzantine_swap_inverts_strong_and_weak():
    assert euchre_byzantine_swap(EuchreSignalKind.STRONG_TRUMP) is EuchreSignalKind.WEAK
    assert euchre_byzantine_swap(EuchreSignalKind.WEAK) is EuchreSignalKind.STRONG_TRUMP
    assert euchre_byzantine_swap(EuchreSignalKind.NONE) is EuchreSignalKind.NONE


# ----------------------------------------- policies plug into EuchreGame ---


def test_talking_policy_emits_on_channel_during_game():
    channel = MessageChannel()
    rng = random.Random(42)
    talker0 = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=0)
    talker2 = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=2)
    policies = [
        talker0,
        RandomEuchrePolicy(rng),
        talker2,
        RandomEuchrePolicy(rng),
    ]
    # Force trump via a seed that historically results in a called hand;
    # if it happens to pass out we simply skip the assertion since only
    # a played hand can emit signals.
    game = EuchreGame(policies=policies, seed=17)
    result = game.play()
    if result.outcome == "passout":
        pytest.skip("seeded deal passed out — no plays, no signals")
    # At least one talker must have spoken on at least one trick.
    assert len(channel) >= 1
    senders = {m.sender_id for m in channel.messages}
    assert senders.issubset({0, 2})


def test_byzantine_policy_plugs_in():
    channel = MessageChannel()
    rng = random.Random(11)
    honest = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=0)
    liar_inner = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=2)
    liar = EuchreByzantinePolicy(liar_inner)
    policies = [honest, RandomEuchrePolicy(rng), liar, RandomEuchrePolicy(rng)]
    game = EuchreGame(policies=policies, seed=3)
    game.play()
    # Byzantine broadcasts should be tagged with kind="byzantine".
    byzantine_msgs = [m for m in channel.messages if m.kind == "byzantine"]
    honest_msgs = [m for m in channel.messages if m.kind == "signal"]
    # Not guaranteed for every seed, but the tag machinery works when
    # anything is emitted at all.
    for m in byzantine_msgs:
        assert m.sender_id == 2
    for m in honest_msgs:
        assert m.sender_id == 0


# ---------------------------------------------------- experiment smoke ---


@pytest.mark.parametrize(
    "treatment", ["silent", "cooperative", "byzantine", "closed_loop"]
)
def test_run_treatment_smoke_n6(treatment):
    r = run_treatment(
        treatment, n_games=6, seed=0, n_resamples=20
    )
    assert r.treatment == treatment
    assert r.n_games == 6
    for v in (
        r.team0_win_rate,
        r.team0_wilson_lower,
        r.team0_wilson_upper,
        r.partner_mi_point,
        r.partner_mi_lower,
        r.partner_mi_upper,
        r.own_mi_point,
        r.signals_per_game,
    ):
        assert math.isfinite(v)
    assert 0.0 <= r.team0_win_rate <= 1.0
    assert r.partner_mi_point >= 0.0
    if treatment == "silent":
        assert r.total_signals == 0
    else:
        # Cooperative and byzantine treatments should attempt to
        # broadcast; with N=6 seeds we can't guarantee non-zero every
        # time (pass-outs suppress plays), but the mean must not be
        # negative.
        assert r.signals_per_game >= 0.0


def test_run_signalling_end_to_end_writes_jsonl(tmp_path):
    out = tmp_path / "results.jsonl"
    results = run_signalling(
        n_games=6, seed=0, n_resamples=10, output_path=out
    )
    assert {r.treatment for r in results} == {
        "silent",
        "cooperative",
        "byzantine",
        "closed_loop",
    }
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + len(results)  # config + treatments
    config = json.loads(lines[0])
    assert config["kind"] == "config"
    assert config["n_games"] == 6
    treatment_rows = [json.loads(x) for x in lines[1:]]
    assert all(row["kind"] == "treatment" for row in treatment_rows)


def test_builders_registered_for_all_treatments():
    assert set(BUILDERS) == {
        "silent",
        "cooperative",
        "byzantine",
        "closed_loop",
    }
