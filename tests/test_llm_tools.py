"""Tests for LLMPolicy tool-calling and the canonical meld_analyzer tool."""

from __future__ import annotations

import json
from typing import Any

from gin_rummy.cards import Card
from gin_rummy.policies.llm import LLMPolicy
from gin_rummy.policies.tools import Tool, meld_analyzer_tool
from gin_rummy.policy import Observation


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


class _ScriptedLLM:
    """Replay canned assistant messages in order; record every messages list."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, model, messages, **kwargs):
        # Store a defensive copy so we can inspect the conversation history.
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise RuntimeError("exhausted replies")
        return self._replies.pop(0)


# ---------- meld_analyzer_tool ----------

def test_meld_analyzer_tool_wraps_optimal_decomposition():
    tool = meld_analyzer_tool()
    hand = ["7♠", "7♥", "7♦", "K♣", "2♠"]
    out = tool({"hand": hand})
    assert "melds" in out and "deadwood" in out
    assert out["deadwood_value"] == 12  # K(10) + 2
    # The 7-set should be recognised.
    assert any(sorted(m) == sorted(["7♠", "7♥", "7♦"]) for m in out["melds"])
    # per_discard_deadwood spans every card in the hand.
    assert set(out["per_discard_deadwood"].keys()) == set(hand)


def test_meld_analyzer_accepts_ascii_suit_letters():
    tool = meld_analyzer_tool()
    out = tool({"hand": ["7S", "7H", "7D", "KC", "2S"]})
    assert out["deadwood_value"] == 12


def test_meld_analyzer_reports_error_on_bad_input():
    tool = meld_analyzer_tool()
    assert "error" in tool({})
    assert "error" in tool({"hand": "not a list"})
    assert "error" in tool({"hand": ["not a card"]})


def test_meld_analyzer_is_json_serialisable():
    tool = meld_analyzer_tool()
    out = tool({"hand": ["A♠", "K♥", "Q♦"]})
    json.dumps(out)  # must not raise


# ---------- LLMPolicy tool round-trip ----------

def test_llm_policy_dispatches_tool_call_then_final_answer():
    tool = meld_analyzer_tool()
    tool_call = json.dumps(
        {"tool_call": {"name": "meld_analyzer", "args": {"hand": ["A♠", "K♥"]}}}
    )
    final = json.dumps({"source": "discard"})
    fake = _ScriptedLLM([tool_call, final])
    policy = LLMPolicy(model="fake", completion_fn=fake, tools=[tool])
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))

    assert policy.choose_draw_source(obs) == "discard"
    # Two completion calls: initial + post-tool re-ask.
    assert len(fake.calls) == 2
    # Trace: tool_call entry, then the final "draw" entry.
    kinds = [c.kind for c in policy.trace]
    assert kinds == ["tool_call", "draw"]
    tool_entry = policy.trace[0]
    assert tool_entry.fell_back is False
    assert tool_entry.parsed["call"]["name"] == "meld_analyzer"
    assert "deadwood_value" in tool_entry.parsed["result"]

    # System prompt should have gained the tools preamble.
    system = fake.calls[0][0]["content"]
    assert "meld_analyzer" in system
    assert "tool_call" in system

    # Second call must have the tool_result echoed back to the model.
    second_msgs = fake.calls[1]
    assert any('"tool_result"' in m["content"] for m in second_msgs if m["role"] == "user")


def test_llm_policy_respects_max_tool_calls():
    tool = meld_analyzer_tool()
    # Model keeps requesting the tool forever.
    tool_call = json.dumps(
        {"tool_call": {"name": "meld_analyzer", "args": {"hand": ["A♠"]}}}
    )
    # 4 tool-call attempts then plenty of buffer; policy should stop at the
    # 4th, treat it as a protocol failure, and fall back.
    fake = _ScriptedLLM([tool_call] * 10)
    policy = LLMPolicy(
        model="fake", completion_fn=fake, tools=[tool], max_tool_calls=3
    )
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))

    _ = policy.choose_draw_source(obs)
    # The last trace entry is the "draw" fallback with reason tool_call_limit.
    draw_entry = [c for c in policy.trace if c.kind == "draw"][-1]
    assert draw_entry.fell_back is True
    assert draw_entry.reason == "tool_call_limit"
    # Three tool_call entries got logged before the cap tripped.
    tool_entries = [c for c in policy.trace if c.kind == "tool_call"]
    assert len(tool_entries) == 3


def test_llm_policy_handles_unknown_tool_gracefully():
    tool = meld_analyzer_tool()
    tool_call = json.dumps({"tool_call": {"name": "nonexistent", "args": {}}})
    final = json.dumps({"discard": "K♥"})
    fake = _ScriptedLLM([tool_call, final])
    policy = LLMPolicy(model="fake", completion_fn=fake, tools=[tool])
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)

    result = policy.choose_discard(obs)
    assert result == Card("K", "♥")
    tool_entry = [c for c in policy.trace if c.kind == "tool_call"][0]
    assert tool_entry.fell_back is True
    assert tool_entry.reason and tool_entry.reason.startswith("unknown_tool")
    # And the error was surfaced back to the model.
    second_user_msg = [
        m for m in fake.calls[1] if m["role"] == "user"
    ][-1]["content"]
    assert "tool_result" in second_user_msg
    assert "unknown tool" in second_user_msg


def test_llm_policy_handles_tool_exception():
    def boom(_args):
        raise RuntimeError("kaboom")
    tool = Tool(
        name="explode",
        description="always raises",
        schema={"type": "object", "properties": {}},
        fn=boom,
    )
    tool_call = json.dumps({"tool_call": {"name": "explode", "args": {}}})
    final = json.dumps({"knock": True})
    fake = _ScriptedLLM([tool_call, final])
    policy = LLMPolicy(model="fake", completion_fn=fake, tools=[tool])
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)

    assert policy.choose_to_knock(obs, deadwood_value=5) is True
    tool_entry = [c for c in policy.trace if c.kind == "tool_call"][0]
    assert tool_entry.fell_back is True
    assert tool_entry.reason and tool_entry.reason.startswith("tool_error")


def test_llm_policy_without_tools_matches_prior_behaviour():
    """A policy constructed without tools must not add a tools preamble."""
    fake = _ScriptedLLM(['{"source": "deck"}'])
    policy = LLMPolicy(model="fake", completion_fn=fake)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))
    _ = policy.choose_draw_source(obs)
    system = fake.calls[0][0]["content"]
    assert "tool_call" not in system
    assert "Available tools" not in system


def test_llm_policy_ignores_tool_call_when_no_tools_configured():
    """If the model emits a tool_call but no tools are configured, it must
    not be treated as a tool invocation. Behaviour is up to the caller —
    here the request has no 'source' key so the policy correctly falls
    back."""
    tool_call = json.dumps(
        {"tool_call": {"name": "meld_analyzer", "args": {}}}
    )
    fake = _ScriptedLLM([tool_call])
    policy = LLMPolicy(model="fake", completion_fn=fake)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand, top_discard=Card("2", "♣"))
    _ = policy.choose_draw_source(obs)
    # Only one completion — no re-ask — and it fell back because there was
    # no 'source' key.
    assert len(fake.calls) == 1
    assert policy.trace[0].fell_back is True


def test_llm_policy_two_tool_calls_then_final():
    tool = meld_analyzer_tool()
    call1 = json.dumps({"tool_call": {"name": "meld_analyzer", "args": {"hand": ["A♠"]}}})
    call2 = json.dumps({"tool_call": {"name": "meld_analyzer", "args": {"hand": ["K♥"]}}})
    final = json.dumps({"discard": "K♥"})
    fake = _ScriptedLLM([call1, call2, final])
    policy = LLMPolicy(model="fake", completion_fn=fake, tools=[tool], max_tool_calls=3)
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)

    assert policy.choose_discard(obs) == Card("K", "♥")
    tool_entries = [c for c in policy.trace if c.kind == "tool_call"]
    assert len(tool_entries) == 2
    assert all(not e.fell_back for e in tool_entries)


def test_tool_and_meld_analyzer_reexported_at_package_level():
    import gin_rummy
    assert hasattr(gin_rummy, "Tool")
    assert hasattr(gin_rummy, "meld_analyzer_tool")
    assert gin_rummy.meld_analyzer_tool().name == "meld_analyzer"
