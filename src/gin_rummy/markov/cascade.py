"""Probabilistic **chain-reaction cascade** through opponent response space.

The intuition: when I discard a card, my opponent doesn't reply with a
deterministic action — they sample from a distribution over
(draw_source, next_discard) conditioned on what they can see. Their
discard then feeds *my* next-turn distribution, and so on. Propagating
this through ``k`` levels of alternating actors is a probabilistic
cascade. Enumerating the top-``k`` most-probable cascades of length
``depth`` gives a compact "what are the plausible next few plies"
answer, useful both as a planning primitive and as a diagnostic when
studying whether particular discards are systematically inviting bad
downstream states.

The cascade is deliberately *observation-restricted*: the response
model only sees the public state (top discard, deck size, hand sizes,
turn number) and the current actor's own hand. It does not peek at the
non-actor's private hand. That keeps the model honest — a real
opponent doesn't see our hand, and a real *us* doesn't see theirs.

Ships one production-ready response model, :func:`greedy_response_model`,
that uses :func:`gin_rummy.meld.optimal_decomposition` to score discard
candidates by their post-discard deadwood and turns those scores into a
temperature-softmax distribution. Callers who want an LLM-, RL-, or
CFR-derived response model just supply their own callable with the
same signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import nlargest
from typing import Callable, Iterable

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition


# --------------------------------------------------------------- data types


@dataclass(frozen=True)
class CascadeStep:
    """One rung of a cascade: who acted, what they did, with what probability.

    ``action`` is deliberately generic — for the shipped
    :func:`greedy_response_model` it's a ``Card`` (the discard), but a
    caller supplying a richer response model can put any hashable action
    representation here.
    """

    actor_id: int
    action: object
    probability: float


# ------------------------------------------------------------- state ----


@dataclass
class CascadeState:
    """Minimal state carried through the cascade.

    Only fields the shipped response model actually reads live here.
    Callers with richer response models can subclass or pack more data
    into ``extra``.

    Parameters
    ----------
    hands : dict[int, list[Card]]
        actor_id -> hand. Under the "observation-restricted" assumption
        the response model for actor ``a`` only reads ``hands[a]``.
    top_discard : Card | None
        Top of the discard pile (visible to everyone).
    turn : int
        Current turn count; ticks up by 1 per cascade step.
    extra : dict
        Free-form room for callers to plumb through opponent-model
        hyperparameters, deck-remaining estimates, etc.
    """

    hands: dict[int, list[Card]]
    top_discard: Card | None = None
    turn: int = 0
    extra: dict = None  # type: ignore[assignment]

    def clone_after_discard(self, actor_id: int, discard: Card) -> "CascadeState":
        """Return a new state after ``actor_id`` discards ``discard``.

        Removes the card from that actor's hand and sets it as the new
        top of the discard pile. Non-actor hands are shared by reference
        (we don't mutate them) to keep the branching factor cheap in
        memory.
        """
        new_hands = dict(self.hands)
        actor_hand = list(self.hands.get(actor_id, []))
        # Remove one instance of the discard (by equality — Card is frozen).
        for i, c in enumerate(actor_hand):
            if c == discard:
                actor_hand.pop(i)
                break
        new_hands[actor_id] = actor_hand
        return CascadeState(
            hands=new_hands,
            top_discard=discard,
            turn=self.turn + 1,
            extra=self.extra,
        )


ResponseModel = Callable[[CascadeState, int], dict[Card, float]]


# ------------------------------------------------------- greedy response ---


def greedy_response_model(
    state: CascadeState,
    actor_id: int,
    *,
    temperature: float = 1.0,
) -> dict[Card, float]:
    """Temperature-softmax response model over the actor's discard choices.

    For each candidate discard ``c`` in the actor's hand, score:

        score(c) = - post_discard_deadwood(hand \\ {c})

    (lower deadwood is better; the negation makes higher scores prefer
    the greedy discard). Turn those into probabilities via
    softmax(score / T). ``T -> 0`` collapses to argmin-deadwood; large
    ``T`` approaches uniform.

    Notes
    -----
    The score does not simulate whether the actor first drew from the
    discard pile — cascade planners typically care about the *emitted*
    discard, not the intermediate draw. If a caller wants a more
    detailed response model they can chain ``(draw_source, discard)``
    into a joint action space.
    """
    hand = list(state.hands.get(actor_id, []))
    if not hand:
        return {}
    scores: dict[Card, float] = {}
    for c in hand:
        remainder = [x for x in hand if x is not c]
        _, _, dw = optimal_decomposition(remainder)
        scores[c] = -float(dw)
    if not scores:
        return {}
    if temperature <= 0.0:
        # Argmin-deadwood (highest score). Ties get uniform mass.
        best = max(scores.values())
        winners = [c for c, s in scores.items() if s == best]
        share = 1.0 / len(winners)
        return {c: share if c in winners else 0.0 for c in scores}
    # Numerically stable softmax.
    m = max(scores.values())
    exps = {c: math.exp((s - m) / temperature) for c, s in scores.items()}
    z = sum(exps.values())
    return {c: v / z for c, v in exps.items()}


# ----------------------------------------------------------- cascade -------


def probabilistic_cascade(
    state: CascadeState,
    response_model: ResponseModel,
    *,
    depth: int = 3,
    top_k: int = 5,
    actor_order: Iterable[int] | None = None,
) -> list[list[CascadeStep]]:
    """Enumerate the top-``k`` most-probable cascades of length ``depth``.

    At each step the current actor's response model returns a
    distribution over discards; we branch on every discard whose
    probability is > 0, multiply the joint probability so far, and
    advance the state. To keep the frontier finite we prune to the
    ``top_k`` heaviest live paths at each ply (beam search on joint
    probability).

    Parameters
    ----------
    state : CascadeState
        Starting game state.
    response_model : callable
        ``(state, actor_id) -> {action: probability}``.
    depth : int
        Number of plies (each ply = one actor's discard).
    top_k : int
        Beam width and returned-path count.
    actor_order : Iterable[int] | None
        Rotation of actor ids. Defaults to alternating between the
        actor ids present in ``state.hands`` in insertion order.
    """
    if depth <= 0 or top_k <= 0:
        return []
    if actor_order is None:
        actor_order = list(state.hands.keys())
    actors = list(actor_order)
    if not actors:
        return []

    # Beam: list of (joint_prob, path, state).
    beam: list[tuple[float, list[CascadeStep], CascadeState]] = [(1.0, [], state)]

    for step_idx in range(depth):
        actor = actors[step_idx % len(actors)]
        next_beam: list[tuple[float, list[CascadeStep], CascadeState]] = []
        for joint, path, st in beam:
            dist = response_model(st, actor)
            if not dist:
                # Dead branch: hand is empty or no legal action.
                continue
            for action, p in dist.items():
                if p <= 0.0:
                    continue
                new_joint = joint * p
                step = CascadeStep(actor_id=actor, action=action, probability=p)
                new_path = path + [step]
                # Advance state: we only know how to advance for Card discards.
                if isinstance(action, Card):
                    new_state = st.clone_after_discard(actor, action)
                else:
                    # Response model returns non-Card action; leave state
                    # unchanged and let the caller supply richer state
                    # semantics via a subclass if needed.
                    new_state = st
                next_beam.append((new_joint, new_path, new_state))
        if not next_beam:
            break
        # Beam prune: keep the top_k by joint probability.
        beam = nlargest(top_k, next_beam, key=lambda item: item[0])

    # Return paths sorted by joint probability, descending.
    beam.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path, _ in beam[:top_k]]


# ------------------------------------------------------ cascade valuation --


def cascade_value(
    cascade: list[CascadeStep],
    *,
    perspective_actor_id: int = 0,
    initial_hand: list[Card] | None = None,
) -> float:
    """Score a cascade as joint-probability x terminal deadwood delta.

    Higher = better *for* ``perspective_actor_id``. The terminal state's
    "value" is approximated as the perspective actor's deadwood at the
    end of the cascade minus their deadwood before it started (so a
    negative delta = we shed deadwood = good).

    If ``initial_hand`` isn't provided (we don't always know it from the
    cascade alone) the value collapses to just the joint probability,
    which is still useful as a "how confident are we in this branch"
    signal.
    """
    joint = 1.0
    for step in cascade:
        joint *= step.probability
    if initial_hand is None:
        return joint

    # Reconstruct the perspective actor's hand by applying their discards.
    hand = list(initial_hand)
    for step in cascade:
        if step.actor_id != perspective_actor_id:
            continue
        if not isinstance(step.action, Card):
            continue
        for i, c in enumerate(hand):
            if c == step.action:
                hand.pop(i)
                break
    _, _, before_dw = optimal_decomposition(list(initial_hand))
    _, _, after_dw = optimal_decomposition(hand)
    # From actor's perspective, LOWER deadwood is better -> score with
    # the negative delta so "shed 5 points" is +5.
    delta = before_dw - after_dw
    return joint * float(delta)


__all__ = [
    "CascadeState",
    "CascadeStep",
    "ResponseModel",
    "cascade_value",
    "greedy_response_model",
    "probabilistic_cascade",
]
