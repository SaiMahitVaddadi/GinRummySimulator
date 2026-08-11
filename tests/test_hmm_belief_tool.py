"""Tests for the opponent-hand-strength HMM belief tool.

Covers:
* Shape/semantics of the tool payload (posterior sums to 1, viterbi label
  is a valid class name).
* Bad-history fallback (empty history -> ``{"error": "insufficient_history"}``
  and no exception).
* End-to-end LLMPolicy round-trip using a scripted ``completion_fn`` — the
  model asks for the HMM belief, receives the posterior, then makes its
  knock decision.
* Combined round-trip using both the ``meld_analyzer_tool`` and the
  belief tool with the ``max_tool_calls`` cap respected.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from gin_rummy.cards import Card
from gin_rummy.game import TurnRecord
from gin_rummy.markov.opponent_hand_hmm import (
    HAND_STRENGTH_CLASSES,
    OpponentHandHMM,
)
from gin_rummy.policies.llm import LLMPolicy
from gin_rummy.policies.tools import (
    bind_hmm_belief_tool,
    meld_analyzer_tool,
    opponent_hmm_belief_tool,
)
from gin_rummy.policy import Observation


# --------------------------------------------------------- helpers ---------


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


def _make_hmm(seed: int = 20260810) -> OpponentHandHMM:
    """A deterministic OpponentHandHMM (untrained but valid stochastic)."""
    return OpponentHandHMM(rng=random.Random(seed))


def _fake_history_for_seat(seat: int, num_turns: int = 4) -> list[TurnRecord]:
    """Build a small list of TurnRecords alternating between two seats,
    ensuring at least one record belongs to ``seat``."""
    records: list[TurnRecord] = []
    for t in range(num_turns):
        pid = t % 2
        # Alternate draw sources and discard ranks so extract_observations
        # produces at least two distinct symbols.
        source = "deck" if t % 2 == 0 else "discard"
        discarded = Card("K", "♥") if t % 2 == 0 else Card("2", "♣")
        records.append(
            TurnRecord(
                turn=t + 1,
                player_id=pid,
                draw_source=source,
                drawn=Card("A", "♠"),
                discarded=discarded,
                action="play",
                deadwood_value=8,
            )
        )
    # Force the last-added record to be for the target seat so we know
    # extract_observations yields at least 2 entries for that seat.
    if seat == 1 and records[-1].player_id != 1:
        records.append(
            TurnRecord(
                turn=num_turns + 1,
                player_id=1,
                draw_source="deck",
                drawn=Card("A", "♠"),
                discarded=Card("7", "♦"),
                action="play",
                deadwood_value=6,
            )
        )
    return records


class _ScriptedLLM:
    """Replay canned assistant messages in order; record every messages list."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise RuntimeError("exhausted replies")
        return self._replies.pop(0)


# --------------------------------------------- shape / semantics -----------


def test_tool_returns_valid_posterior_shape() -> None:
    """With a valid (untrained but stochastic) HMM and non-empty history,
    the tool returns a posterior that sums to 1 (±1e-9), a viterbi label
    from the canonical alphabet, and confidence == max(posterior)."""
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1, num_turns=6)
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )
    out = tool({})
    assert "error" not in out, f"unexpected error payload: {out}"

    assert set(out["posterior"].keys()) == set(HAND_STRENGTH_CLASSES)
    total = sum(out["posterior"].values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), total

    assert out["viterbi_current_state"] in HAND_STRENGTH_CLASSES
    assert isinstance(out["confidence"], float)
    expected_conf = max(out["posterior"].values())
    assert math.isclose(out["confidence"], expected_conf, abs_tol=1e-9)


def test_tool_error_on_empty_history() -> None:
    """When history_provider returns an empty list, the tool must not
    raise — it must return ``{"error": "insufficient_history"}``."""
    hmm = _make_hmm()
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: [], get_target_id=lambda: 1
    )
    out = tool({})
    assert out == {"error": "insufficient_history"}


def test_tool_error_when_seat_has_no_turns() -> None:
    """A history that only records seat 0 turns must yield the same
    error when we ask about seat 1 — nothing observable => nothing to
    condition on."""
    hmm = _make_hmm()
    # Every record belongs to seat 0.
    history = [
        TurnRecord(
            turn=i + 1,
            player_id=0,
            draw_source="deck",
            drawn=Card("A", "♠"),
            discarded=Card("K", "♥"),
            action="play",
            deadwood_value=8,
        )
        for i in range(3)
    ]
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )
    assert tool({}) == {"error": "insufficient_history"}


def test_tool_result_is_json_serialisable() -> None:
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1, num_turns=5)
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )
    out = tool({})
    json.dumps(out)  # must not raise


def test_opponent_hmm_belief_tool_alias_matches_binder() -> None:
    """The two factory names in the spec point to the same tool wiring."""
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1)
    a = bind_hmm_belief_tool(hmm, lambda: history, lambda: 1)
    b = opponent_hmm_belief_tool(hmm, lambda: history, lambda: 1)
    assert a.name == b.name == "opponent_hmm_belief"
    assert a.schema == b.schema


# ------------------------------------------- LLMPolicy round-trips ---------


def test_llm_policy_can_call_hmm_belief_tool() -> None:
    """Scripted LLM: first message calls the HMM belief tool, second
    replies with a knock decision. Verify:
      * the returned knock value matches the scripted decision,
      * the trace contains one tool_call entry and one knock entry,
      * the tool_call entry logged the posterior in its ``result``.
    """
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1, num_turns=6)
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )

    tool_call = json.dumps({"tool_call": {"name": "opponent_hmm_belief", "args": {}}})
    final = json.dumps({"knock": True})
    fake = _ScriptedLLM([tool_call, final])

    policy = LLMPolicy(model="fake", completion_fn=fake, tools=[tool])
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)

    assert policy.choose_to_knock(obs, deadwood_value=6) is True
    assert len(fake.calls) == 2

    kinds = [c.kind for c in policy.trace]
    assert kinds == ["tool_call", "knock"]

    tool_entry = policy.trace[0]
    assert tool_entry.fell_back is False
    assert tool_entry.parsed["call"]["name"] == "opponent_hmm_belief"
    result = tool_entry.parsed["result"]
    assert set(result["posterior"].keys()) == set(HAND_STRENGTH_CLASSES)
    assert result["viterbi_current_state"] in HAND_STRENGTH_CLASSES

    # Second turn of the LLM conversation must have echoed the tool_result
    # back to the model.
    second_msgs = fake.calls[1]
    user_msgs = [m["content"] for m in second_msgs if m["role"] == "user"]
    assert any("tool_result" in c for c in user_msgs)
    assert any("posterior" in c for c in user_msgs)


def test_llm_policy_uses_both_meld_and_hmm_tools() -> None:
    """Scripted LLM calls the meld tool, then the HMM tool, then answers.
    Verify both tool round-trips complete and the ``max_tool_calls`` cap is
    respected (we set it to exactly 2 and expect no protocol failure)."""
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1, num_turns=6)
    meld_tool = meld_analyzer_tool()
    hmm_tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )

    call1 = json.dumps(
        {
            "tool_call": {
                "name": "meld_analyzer",
                "args": {"hand": ["7♠", "7♥", "7♦", "K♣", "2♠"]},
            }
        }
    )
    call2 = json.dumps({"tool_call": {"name": "opponent_hmm_belief", "args": {}}})
    final = json.dumps({"knock": False})
    fake = _ScriptedLLM([call1, call2, final])

    policy = LLMPolicy(
        model="fake",
        completion_fn=fake,
        tools=[meld_tool, hmm_tool],
        max_tool_calls=2,
    )
    hand = (
        Card("7", "♠"),
        Card("7", "♥"),
        Card("7", "♦"),
        Card("K", "♣"),
        Card("2", "♠"),
    )
    obs = _obs(hand)

    assert policy.choose_to_knock(obs, deadwood_value=12) is False
    tool_entries = [c for c in policy.trace if c.kind == "tool_call"]
    assert [e.parsed["call"]["name"] for e in tool_entries] == [
        "meld_analyzer",
        "opponent_hmm_belief",
    ]
    assert all(not e.fell_back for e in tool_entries)

    # And the final knock entry did not fall back.
    knock_entries = [c for c in policy.trace if c.kind == "knock"]
    assert len(knock_entries) == 1
    assert knock_entries[0].fell_back is False
    # Three LLM completions total: two tool round-trips + one final.
    assert len(fake.calls) == 3


def test_llm_policy_tool_call_limit_still_respected_with_hmm_tool() -> None:
    """If the model keeps hammering the HMM tool past the cap, the policy
    should fall back gracefully — same guarantee as for other tools."""
    hmm = _make_hmm()
    history = _fake_history_for_seat(seat=1, num_turns=6)
    tool = bind_hmm_belief_tool(
        hmm, get_history=lambda: history, get_target_id=lambda: 1
    )
    tool_call = json.dumps({"tool_call": {"name": "opponent_hmm_belief", "args": {}}})
    fake = _ScriptedLLM([tool_call] * 5)
    policy = LLMPolicy(
        model="fake", completion_fn=fake, tools=[tool], max_tool_calls=2
    )
    hand = (Card("A", "♠"), Card("K", "♥"))
    obs = _obs(hand)

    _ = policy.choose_to_knock(obs, deadwood_value=6)
    knock_entry = [c for c in policy.trace if c.kind == "knock"][-1]
    assert knock_entry.fell_back is True
    assert knock_entry.reason == "tool_call_limit"
    assert len([c for c in policy.trace if c.kind == "tool_call"]) == 2
