"""LLM-backed policy.

Uses `LiteLLM <https://github.com/BerriAI/litellm>`_ as the transport so any
provider (OpenAI, Anthropic, Google, OpenRouter, self-hosted, ...) works with
the same model string. LiteLLM is an *optional* dependency — install with
``pip install litellm`` or ``uv add litellm``.

Design notes
------------
* The policy renders a compact prompt from the ``Observation`` and asks the
  LLM to return a strict JSON object. If parsing or the API call fails, we
  fall back to a supplied ``fallback`` policy (defaults to ``RandomPolicy``)
  and log the failure. Games never crash because the LLM misbehaved.
* All prompt/response pairs are appended to ``self.trace`` so you can later
  audit what the model saw and said. This is the primary research asset.
* Legal-move filtering happens client-side: if the model names a card it
  doesn't hold, we treat that as a parse failure and fall back.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation, Policy, RandomPolicy

logger = logging.getLogger(__name__)


@dataclass
class LLMCall:
    turn: int
    kind: str  # "draw", "discard", "knock"
    prompt: str
    response: str
    parsed: Any
    fell_back: bool
    reason: str | None = None


@dataclass
class LLMPolicy:
    """Policy that delegates decisions to an LLM via LiteLLM.

    Parameters
    ----------
    model : str
        LiteLLM model identifier, e.g. ``"gpt-4o-mini"``,
        ``"anthropic/claude-haiku-4-5"``, ``"openrouter/anthropic/claude-3.5-sonnet"``.
    system_prompt : str, optional
        Overrides the default rummy-strategist system prompt.
    fallback : Policy, optional
        Used when the LLM call fails or returns unparseable output. Defaults
        to a fresh ``RandomPolicy``.
    temperature, max_tokens : passed through to LiteLLM.
    completion_fn : callable, optional
        Injection seam for tests — pass a fake to avoid network calls.
        Signature: ``completion_fn(model, messages, **kwargs) -> str``.
    """

    model: str
    system_prompt: str | None = None
    fallback: Policy | None = None
    temperature: float = 0.2
    max_tokens: int = 200
    completion_fn: Callable[..., str] | None = None
    trace: list[LLMCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = RandomPolicy(random.Random())
        if self.system_prompt is None:
            self.system_prompt = _DEFAULT_SYSTEM_PROMPT

    # ---------- Policy protocol ----------

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        if obs.top_discard is None:
            return "deck"
        prompt = _render_draw_prompt(obs)
        parsed, raw, reason = self._ask(prompt, "draw")
        if parsed and parsed.get("source") in ("deck", "discard"):
            self._log(obs, "draw", prompt, raw, parsed, fell_back=False)
            return parsed["source"]
        self._log(obs, "draw", prompt, raw, parsed, fell_back=True, reason=reason)
        assert self.fallback is not None
        return self.fallback.choose_draw_source(obs)

    def choose_discard(self, obs: Observation) -> Card:
        prompt = _render_discard_prompt(obs)
        parsed, raw, reason = self._ask(prompt, "discard")
        chosen = _resolve_card(parsed.get("discard") if parsed else None, obs.hand)
        if chosen is not None:
            self._log(obs, "discard", prompt, raw, parsed, fell_back=False)
            return chosen
        self._log(
            obs,
            "discard",
            prompt,
            raw,
            parsed,
            fell_back=True,
            reason=reason or "illegal_card",
        )
        assert self.fallback is not None
        return self.fallback.choose_discard(obs)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        prompt = _render_knock_prompt(obs, deadwood_value)
        parsed, raw, reason = self._ask(prompt, "knock")
        if parsed and isinstance(parsed.get("knock"), bool):
            self._log(obs, "knock", prompt, raw, parsed, fell_back=False)
            return parsed["knock"]
        self._log(obs, "knock", prompt, raw, parsed, fell_back=True, reason=reason)
        assert self.fallback is not None
        return self.fallback.choose_to_knock(obs, deadwood_value)

    # ---------- internals ----------

    def _ask(
        self, user_prompt: str, kind: str
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self._completion(messages)
        except Exception as exc:  # noqa: BLE001 — we want any error to trigger fallback
            logger.warning("LLM call failed (%s): %s", kind, exc)
            return None, "", f"api_error:{type(exc).__name__}"

        parsed = _extract_json(raw)
        if parsed is None:
            return None, raw, "parse_error"
        return parsed, raw, None

    def _completion(self, messages: list[dict[str, str]]) -> str:
        if self.completion_fn is not None:
            return self.completion_fn(
                self.model, messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LLMPolicy needs `litellm` installed (or a completion_fn injection)."
            ) from exc
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def _log(
        self,
        obs: Observation,
        kind: str,
        prompt: str,
        raw: str,
        parsed: Any,
        *,
        fell_back: bool,
        reason: str | None = None,
    ) -> None:
        self.trace.append(
            LLMCall(
                turn=obs.turn_number,
                kind=kind,
                prompt=prompt,
                response=raw,
                parsed=parsed,
                fell_back=fell_back,
                reason=reason,
            )
        )


# ---------- prompt rendering ----------

_DEFAULT_SYSTEM_PROMPT = """You are an expert Gin Rummy player. Play to minimise deadwood and win.

For every decision I ask you, reply with exactly one line of strict JSON — no
prose, no code fences. Use the exact keys and value shapes I request. Card
strings are the rank followed by the suit glyph (examples: "A♠", "10♥", "K♦").
"""


def _describe_hand(hand: tuple[Card, ...]) -> str:
    melds, deadwood, dv = optimal_decomposition(list(hand))
    hand_str = " ".join(str(c) for c in hand)
    meld_str = "; ".join("+".join(str(c) for c in m) for m in melds) if melds else "none"
    dw_str = " ".join(str(c) for c in deadwood) if deadwood else "none"
    return (
        f"Hand: {hand_str}\n"
        f"Current melds: {meld_str}\n"
        f"Current deadwood: {dw_str} (value {dv})"
    )


def _shared_state(obs: Observation) -> str:
    top = str(obs.top_discard) if obs.top_discard else "—"
    return (
        f"You are Player {obs.player_id + 1} of {obs.num_players}. "
        f"Turn {obs.turn_number}. Deck: {obs.deck_size} cards. "
        f"Top of discard: {top}. Knock limit: {obs.knock_limit}."
    )


def _render_draw_prompt(obs: Observation) -> str:
    return (
        f"{_shared_state(obs)}\n{_describe_hand(obs.hand)}\n\n"
        'Reply: {"source": "deck"} or {"source": "discard"}.'
    )


def _render_discard_prompt(obs: Observation) -> str:
    hand_list = ", ".join(str(c) for c in obs.hand)
    return (
        f"{_shared_state(obs)}\n{_describe_hand(obs.hand)}\n"
        f"Legal discards: [{hand_list}]\n\n"
        'Reply: {"discard": "<card>"} — the card must be in your hand.'
    )


def _render_knock_prompt(obs: Observation, deadwood_value: int) -> str:
    return (
        f"{_shared_state(obs)}\n{_describe_hand(obs.hand)}\n"
        f"You may knock: your deadwood is {deadwood_value} ≤ limit {obs.knock_limit}.\n\n"
        'Reply: {"knock": true} or {"knock": false}.'
    )


# ---------- parsing ----------

def _extract_json(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of the response. Tolerates code fences."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip common code-fence wrappers.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_card(name: Any, hand: tuple[Card, ...]) -> Card | None:
    if not isinstance(name, str):
        return None
    target = name.strip()
    for c in hand:
        if str(c) == target:
            return c
    return None
