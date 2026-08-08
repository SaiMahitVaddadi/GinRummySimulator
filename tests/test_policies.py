"""Tests for LLMPolicy and GreedyKnockPolicy — no network calls."""

from __future__ import annotations

import json
import random
from typing import Any

from gin_rummy.cards import Card
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policies.llm import LLMPolicy, _extract_json, _resolve_card
from gin_rummy.policy import Observation
from gin_rummy.variants.classic import ClassicGin


def _obs(hand: tuple[Card, ...], top_discard: Card | None = None) -> Observation:
    return Observation(
        hand=hand,
        top_discard=top_discard,
        discard_pile_size=1 if top_discard else 0,
        deck_size=30,
        turn_number=1,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10, 10),
    )


# ---------- extract_json ----------

def test_extract_json_plain():
    assert _extract_json('{"knock": true}') == {"knock": True}


def test_extract_json_with_code_fence():
    raw = '```json\n{"source": "discard"}\n```'
    assert _extract_json(raw) == {"source": "discard"}


def test_extract_json_with_leading_prose():
    raw = 'Sure — I think I should knock.\n{"knock": true}'
    assert _extract_json(raw) == {"knock": True}


def test_extract_json_invalid_returns_none():
    assert _extract_json("nope, no json here") is None
    assert _extract_json("") is None


# ---------- resolve_card ----------

def test_resolve_card_matches_exact():
    hand = (Card("A", "♠"), Card("K", "♥"))
    assert _resolve_card("A♠", hand) is hand[0]
    assert _resolve_card("K♥", hand) is hand[1]


def test_resolve_card_rejects_unknown():
    hand = (Card("A", "♠"),)
    assert _resolve_card("Q♦", hand) is None
    assert _resolve_card(None, hand) is None
    assert _resolve_card(42, hand) is None


# ---------- LLMPolicy with injected completion_fn ----------

class _ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[Any] = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("exhausted replies")
        return self._replies.pop(0)


def test_llm_policy_uses_valid_response():
    fake = _ScriptedLLM(['{"source": "discard"}'])
    policy = LLMPolicy(model="fake", completion_fn=fake)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))
    assert policy.choose_draw_source(obs) == "discard"
    assert len(policy.trace) == 1
    assert not policy.trace[0].fell_back


def test_llm_policy_falls_back_on_bad_json():
    fake = _ScriptedLLM(["I refuse to output JSON"])
    fb = GreedyKnockPolicy(random.Random(0))
    policy = LLMPolicy(model="fake", completion_fn=fake, fallback=fb)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))
    _ = policy.choose_draw_source(obs)
    assert len(policy.trace) == 1
    assert policy.trace[0].fell_back
    assert policy.trace[0].reason == "parse_error"


def test_llm_policy_falls_back_on_illegal_discard():
    # LLM picks a card not in the hand → parse succeeds, resolution fails.
    fake = _ScriptedLLM(['{"discard": "Q♦"}'])
    policy = LLMPolicy(model="fake", completion_fn=fake)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)
    _ = policy.choose_discard(obs)
    assert policy.trace[0].fell_back
    assert policy.trace[0].reason == "illegal_card"


def test_llm_policy_falls_back_on_api_exception():
    def boom(*args, **kwargs):
        raise ConnectionError("simulated network failure")
    policy = LLMPolicy(model="fake", completion_fn=boom)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))
    _ = policy.choose_draw_source(obs)
    assert policy.trace[0].fell_back
    assert policy.trace[0].reason and policy.trace[0].reason.startswith("api_error")


def test_llm_policy_plays_full_game_with_scripted_llm():
    # Feed enough JSON responses to keep the game going without falling back
    # too often. Random fallback is fine when we run out.
    replies = [json.dumps({"source": "deck"})] * 200 + [json.dumps({"discard": ""})] * 200
    fake_a = _ScriptedLLM(replies)
    fake_b = _ScriptedLLM(list(replies))
    p1 = LLMPolicy(model="fake", completion_fn=fake_a)
    p2 = LLMPolicy(model="fake", completion_fn=fake_b)
    result = ClassicGin(num_players=2, seed=0, policies=[p1, p2]).play()
    assert result.turns > 0
    assert p1.trace or p2.trace


# ---------- GreedyKnockPolicy ----------

def test_greedy_takes_discard_when_it_reduces_deadwood():
    hand = (Card("7", "♠"), Card("7", "♥"), Card("K", "♦"))  # deadwood = 27
    top = Card("7", "♦")  # completes the set
    policy = GreedyKnockPolicy(random.Random(0))
    assert policy.choose_draw_source(_obs(hand, top_discard=top)) == "discard"


def test_greedy_skips_discard_when_useless():
    hand = (Card("A", "♠"), Card("K", "♥"))
    top = Card("2", "♣")
    policy = GreedyKnockPolicy(random.Random(0))
    assert policy.choose_draw_source(_obs(hand, top_discard=top)) == "deck"


def test_greedy_discards_highest_deadwood():
    hand = (Card("A", "♠"), Card("K", "♥"), Card("2", "♦"))
    policy = GreedyKnockPolicy(random.Random(0))
    assert policy.choose_discard(_obs(hand)) == Card("K", "♥")
