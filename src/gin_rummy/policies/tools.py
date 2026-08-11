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

A second tool, :func:`bind_hmm_belief_tool`, exposes a trained
:class:`OpponentHandHMM` as an *inference oracle* — the LLM can query it for
a posterior over the opponent's hand-strength class before committing to a
knock/no-knock decision. Because the HMM needs ambient game context that a
tool call can't carry (the full turn history and the opponent seat id), the
binder pattern injects two zero-arg closures at wiring time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from gin_rummy.cards import RANKS, SUITS, Card

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gin_rummy.game import TurnRecord
    from gin_rummy.markov.opponent_hand_hmm import OpponentHandHMM


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


# ---------- opponent-HMM belief tool ----------


_HMM_BELIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
    "description": (
        "No arguments. The tool reads the ambient game history and the "
        "configured opponent seat id from the closures the LLMPolicy was "
        "wired with."
    ),
}


def _build_hmm_belief_fn(
    hmm: "OpponentHandHMM",
    history_provider: Callable[[], list["TurnRecord"]],
    target_player_id_provider: Callable[[], int],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the callable that the Tool wraps.

    The returned function ignores its ``args`` payload (the schema declares
    no arguments) and pulls the ambient game context from the injected
    closures. This lets tests inject deterministic history/id providers
    without spinning up a full game engine.
    """

    # Import locally so this module stays importable without markov deps at
    # module-load time.
    from gin_rummy.markov.opponent_hand_hmm import (
        HAND_STRENGTH_CLASSES,
        extract_observations,
    )

    def _run(_args: dict[str, Any]) -> dict[str, Any]:
        history = history_provider()
        target_id = target_player_id_provider()
        try:
            observations = extract_observations(history, target_id)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"history_extraction_failed: {exc}"}
        if not observations:
            return {"error": "insufficient_history"}

        viterbi_states, posteriors = hmm.predict_hand_strength(observations)
        if not posteriors:
            return {"error": "insufficient_history"}

        # Posterior at the current (most-recent) turn.
        current_post = posteriors[-1]
        n_states = len(current_post)
        # Guard: only label with semantic class names if the HMM's state
        # count matches the canonical alphabet. Otherwise fall back to
        # "s0", "s1", ... — the caller has intentionally chosen a
        # non-standard state count.
        if n_states == len(HAND_STRENGTH_CLASSES):
            names = HAND_STRENGTH_CLASSES
        else:
            names = tuple(f"s{i}" for i in range(n_states))

        posterior_named = {names[i]: float(current_post[i]) for i in range(n_states)}
        current_viterbi = viterbi_states[-1]
        current_viterbi_name = (
            names[current_viterbi]
            if 0 <= current_viterbi < n_states
            else f"s{current_viterbi}"
        )
        confidence = float(max(current_post)) if current_post else 0.0

        return {
            "posterior": posterior_named,
            "viterbi_current_state": current_viterbi_name,
            "confidence": confidence,
        }

    return _run


def bind_hmm_belief_tool(
    hmm: "OpponentHandHMM",
    get_history: Callable[[], list["TurnRecord"]],
    get_target_id: Callable[[], int],
) -> Tool:
    """Bind an :class:`OpponentHandHMM` and ambient-context closures to a Tool.

    Parameters
    ----------
    hmm:
        A *trained* :class:`OpponentHandHMM`. The tool does not train — it
        only calls :meth:`OpponentHandHMM.predict_hand_strength`.
    get_history:
        Zero-arg closure returning the current ``list[TurnRecord]`` for the
        game in progress. Typically wired to a game-engine attribute that
        accumulates turn records.
    get_target_id:
        Zero-arg closure returning the seat id of the *opponent* whose hand
        strength we want to infer.

    Returns
    -------
    Tool
        A :class:`Tool` named ``"opponent_hmm_belief"`` that takes no
        arguments and returns a posterior over hand-strength classes.
    """
    fn = _build_hmm_belief_fn(hmm, get_history, get_target_id)
    return Tool(
        name="opponent_hmm_belief",
        description=(
            "Query a trained opponent-hand-strength HMM for the current "
            "posterior over the opponent's latent hand-strength class "
            "(very_weak/weak/medium/strong/gin_ready). Returns the "
            "Viterbi single-best current state, the full posterior "
            "distribution, and the max-a-posteriori confidence. Call this "
            "before deciding whether to knock — a high posterior mass on "
            "strong/gin_ready is a reason to hold off."
        ),
        schema=_HMM_BELIEF_SCHEMA,
        fn=fn,
    )


def opponent_hmm_belief_tool(
    hmm: "OpponentHandHMM",
    history_provider: Callable[[], list["TurnRecord"]],
    target_player_id_provider: Callable[[], int],
) -> Tool:
    """Alias for :func:`bind_hmm_belief_tool` matching the name in the
    module docstring / research spec."""
    return bind_hmm_belief_tool(hmm, history_provider, target_player_id_provider)


__all__ = [
    "Tool",
    "bind_hmm_belief_tool",
    "meld_analyzer_tool",
    "opponent_hmm_belief_tool",
]
