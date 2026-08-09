"""Tools that an :class:`LLMPolicy` can call during a decision.

The design mirrors the OpenAI / Anthropic function-calling convention but
stays transport-agnostic — a :class:`Tool` is just a name, a description, a
JSON-schema-shaped ``schema`` dict, and a callable ``fn(args_dict) -> dict``
that returns a JSON-serialisable result.

The canonical tool shipped here is :func:`meld_analyzer_tool`, which wraps
:func:`gin_rummy.eval.xplain.introspect_hand` (which in turn calls the exact
DP in :func:`gin_rummy.meld.optimal_decomposition`). This implements research
opening #1 from the hybrid-architectures survey: LLM as *meta-controller*,
optimal-decomposition analyzer as *tactical solver*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from gin_rummy.cards import RANKS, SUITS, Card


@dataclass
class Tool:
    """A callable capability exposed to the LLM.

    Attributes
    ----------
    name:
        Identifier the model uses in ``{"tool_call": {"name": ...}}``.
    description:
        One-line human-readable purpose. Rendered into the system prompt.
    schema:
        JSON-schema-shaped dict describing the ``args`` payload. Rendered
        into the system prompt so the model knows the argument shape.
    fn:
        Executes the tool. Receives a plain ``dict`` (the ``args`` payload
        from the model) and must return a JSON-serialisable ``dict``.
    """

    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], dict[str, Any]]

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.fn(args)


# ---------- canonical meld_analyzer tool ----------

_RANK_ALIASES = {r.upper(): r for r in RANKS}
# Convenience aliases so the LLM can spell things loosely.
_RANK_ALIASES.update({"T": "10"})
_SUIT_ALIASES = {
    "S": "♠", "SPADES": "♠", "SPADE": "♠", "♠": "♠",
    "H": "♥", "HEARTS": "♥", "HEART": "♥", "♥": "♥",
    "D": "♦", "DIAMONDS": "♦", "DIAMOND": "♦", "♦": "♦",
    "C": "♣", "CLUBS": "♣", "CLUB": "♣", "♣": "♣",
}
_CARD_RE = re.compile(r"^\s*(10|[A2-9JQKT])\s*([♠♥♦♣SHDCshdc]|SPADES?|HEARTS?|DIAMONDS?|CLUBS?)\s*$", re.IGNORECASE)


def _parse_card(spec: Any) -> Card:
    """Parse a card spec that may use glyphs or ASCII suit letters."""
    if isinstance(spec, Card):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"cannot parse card: {spec!r}")
    m = _CARD_RE.match(spec)
    if not m:
        raise ValueError(f"cannot parse card: {spec!r}")
    rank_raw, suit_raw = m.group(1), m.group(2)
    rank = _RANK_ALIASES.get(rank_raw.upper(), rank_raw.upper())
    suit = _SUIT_ALIASES.get(suit_raw.upper())
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"unknown card: {spec!r}")
    return Card(rank, suit)


def _run_meld_analyzer(args: dict[str, Any]) -> dict[str, Any]:
    hand_spec = args.get("hand")
    if not isinstance(hand_spec, (list, tuple)):
        return {
            "error": "meld_analyzer expects an object with key 'hand' -> list[str].",
        }
    try:
        cards = [_parse_card(c) for c in hand_spec]
    except ValueError as exc:
        return {"error": str(exc)}

    # Imported lazily to avoid a circular import: ``eval`` transitively
    # imports :class:`LLMPolicy`, which imports this module.
    from gin_rummy.eval.xplain import introspect_hand

    intro = introspect_hand(cards)
    return {
        "melds": [[str(c) for c in m] for m in intro.melds],
        "deadwood": [str(c) for c in intro.deadwood],
        "deadwood_value": intro.deadwood_value,
        "per_discard_deadwood": dict(intro.per_discard),
    }


_MELD_ANALYZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hand": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The hand to analyse. Cards use rank + suit glyph, e.g. "
                "'A♠', '10♥', 'K♦'. ASCII 'S/H/D/C' also accepted."
            ),
        }
    },
    "required": ["hand"],
}


def meld_analyzer_tool() -> Tool:
    """Factory for the canonical meld-analyzer tool.

    The tool wraps :func:`optimal_decomposition` (via ``introspect_hand``) and
    lets the LLM ask: "given this hand, what is the optimal meld partition,
    what's my current deadwood, and what deadwood results from discarding
    each card?".
    """
    return Tool(
        name="meld_analyzer",
        description=(
            "Compute the optimal meld decomposition for a rummy hand. "
            "Returns melds, deadwood cards, deadwood value, and a "
            "per-discard-deadwood table so you can pick the best card to "
            "throw."
        ),
        schema=_MELD_ANALYZER_SCHEMA,
        fn=_run_meld_analyzer,
    )


__all__ = ["Tool", "meld_analyzer_tool"]
