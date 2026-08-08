"""Tests for the communication subpackage."""

from __future__ import annotations

import random

from gin_rummy.cards import Card
from gin_rummy.comms.byzantine import ByzantinePolicy, byzantine_swap
from gin_rummy.comms.channel import ChatMessage, MessageChannel
from gin_rummy.comms.metrics import (
    action_hand_mutual_information,
    signal_cooperation_index,
)
from gin_rummy.comms.orchestrator import BlackboardOrchestrator
from gin_rummy.comms.signals import SignalKind, TalkingPolicy
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import Observation
from gin_rummy.variants.classic import ClassicGin


def _obs(hand, turn=1, knock_limit=10):
    return Observation(
        hand=tuple(hand),
        top_discard=None,
        discard_pile_size=0,
        deck_size=30,
        turn_number=turn,
        knock_limit=knock_limit,
        player_id=0,
        num_players=2,
        other_hand_sizes=(len(hand), len(hand)),
    )


# ---------- channel ----------

def test_channel_broadcast_and_iterate():
    ch = MessageChannel()
    ch.broadcast(0, 1, "signal", "weak_hand")
    ch.broadcast(1, 2, "chat", "hi")
    assert len(ch) == 2
    assert list(ch.by_sender(0)) == [ChatMessage(0, 1, "signal", "weak_hand")]
    assert list(ch.since(2)) == [ChatMessage(1, 2, "chat", "hi")]


# ---------- signals ----------

def test_talking_policy_signals_gin_ready():
    ch = MessageChannel()
    inner = GreedyKnockPolicy(random.Random(0))
    tp = TalkingPolicy(inner, ch, sender_id=0)
    hand = [Card("A", "♠"), Card("A", "♥"), Card("A", "♦")]  # dw = 0
    tp.choose_draw_source(_obs(hand))
    assert any(m.payload == SignalKind.GIN_READY.value for m in ch.messages)


def test_talking_policy_signals_weak_hand():
    ch = MessageChannel()
    inner = GreedyKnockPolicy(random.Random(0))
    tp = TalkingPolicy(inner, ch, sender_id=0)
    # Assorted high cards, no melds → dw = 10 + 10 + 10 + 10 = 40 > 30 threshold
    hand = [Card("K", "♠"), Card("Q", "♥"), Card("J", "♦"), Card("10", "♣")]
    tp.choose_draw_source(_obs(hand))
    assert any(m.payload == SignalKind.WEAK_HAND.value for m in ch.messages)


def test_talking_policy_no_signal_in_middle_range():
    ch = MessageChannel()
    inner = GreedyKnockPolicy(random.Random(0))
    tp = TalkingPolicy(inner, ch, sender_id=0)
    # dw = 15, above knock_limit=10 but below weak_threshold=30
    hand = [Card("7", "♠"), Card("8", "♥")]
    tp.choose_draw_source(_obs(hand))
    assert len(ch) == 0


# ---------- byzantine ----------

def test_byzantine_inverts_signal():
    assert byzantine_swap(SignalKind.WEAK_HAND) == SignalKind.CLOSE_TO_KNOCK
    assert byzantine_swap(SignalKind.CLOSE_TO_KNOCK) == SignalKind.WEAK_HAND
    assert byzantine_swap(SignalKind.GIN_READY) == SignalKind.WEAK_HAND


def test_byzantine_policy_emits_lie():
    ch = MessageChannel()
    inner = GreedyKnockPolicy(random.Random(0))
    talker = TalkingPolicy(inner, ch, sender_id=1)
    liar = ByzantinePolicy(talker)
    hand = [Card("A", "♠"), Card("A", "♥"), Card("A", "♦")]  # actually gin_ready
    liar.choose_draw_source(_obs(hand))
    # Deceptive broadcast should NOT be gin_ready; it should be weak_hand.
    kinds = [m.payload for m in ch.messages]
    assert SignalKind.GIN_READY.value not in kinds
    assert SignalKind.WEAK_HAND.value in kinds


# ---------- metrics ----------

def test_mutual_information_captures_dependence():
    # Perfectly correlated action ↔ hand under a single public bucket.
    samples = [("draw", "strong", "p"), ("skip", "weak", "p")] * 100
    mi = action_hand_mutual_information(samples)
    assert mi > 0.5  # near ln(2) ≈ 0.69


def test_mutual_information_is_zero_when_independent():
    samples = [("draw", "strong", "p"), ("draw", "weak", "p")] * 100
    mi = action_hand_mutual_information(samples)
    assert mi < 0.05  # near zero


def test_signal_cooperation_index_matches():
    signals = [(1, "close_to_knock"), (3, "weak_hand")]
    responses = [(2, "hold"), (4, "attack")]
    expected = {"close_to_knock": "hold", "weak_hand": "attack"}
    assert signal_cooperation_index(signals, responses, expected) == 1.0


# ---------- orchestrator ----------

def test_orchestrator_runs_nodes_and_stops_on_done():
    orch = BlackboardOrchestrator(state={"i": 0})

    def increment(state):
        state["i"] += 1
        if state["i"] >= 5:
            state["done"] = True

    orch.add_node("inc", increment)
    orch.run(max_steps=100)
    assert orch.state["i"] == 5


# ---------- end-to-end: talking policies play a game ----------

def test_talking_policies_play_through():
    ch = MessageChannel()
    p1 = TalkingPolicy(GreedyKnockPolicy(random.Random(0)), ch, sender_id=0)
    p2 = TalkingPolicy(GreedyKnockPolicy(random.Random(1)), ch, sender_id=1)
    r = ClassicGin(num_players=2, seed=0, policies=[p1, p2]).play()
    assert r.turns > 0
    # Some structured signals should have been emitted during play.
    assert len(ch) > 0
