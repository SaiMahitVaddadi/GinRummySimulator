"""MCCFR on abstracted 10-card 2-player Classic Gin.

Reuses :class:`gin_rummy.solvers.cfr.ExternalSamplingMCCFR` by supplying
an :class:`MCCFREngine` implementation that wraps a fresh functional
Classic Gin state machine (mirroring the structural shape of
``minigin``) and keys information sets through the coarse abstraction
defined in :mod:`gin_rummy.solvers.abstraction`.

Sampling scheme
---------------
By default :func:`train_gin_cfr` uses **outcome sampling** rather than
external sampling. Classic Gin at hand-size 10 has enough game-tree
depth (typically 20+ half-turns before knock/gin/draw) that external
sampling's ``branching ^ depth`` traverser sub-tree becomes intractable
even after action abstraction. Outcome sampling has O(depth) per-
iteration cost and is the standard choice for deep imperfect-
information games (Lanctot, Waugh, Zinkevich & Bowling, NeurIPS 2009).

Terminal scoring densification
------------------------------
Under strict Classic Gin rules random play draws >90% of hands (nobody
knocks; the deck exhausts or the turn cap fires). This gives MCCFR
almost no reward signal. We therefore **grade timeout terminals by
final-deadwood difference** — lower-deadwood side "wins" a scaled
payoff. Combined with the still-honoured gin/knock/undercut branches,
this is a "deadwood-competitive" abstraction on top of the rules. Any
policy that outperforms random on the abstracted game will also, in
expectation, discard smarter under the full rules — but the exact
mapping is lossy.

Honest limitations
------------------
The hand abstraction is coarse (see :mod:`.abstraction`); many concrete
hands share a single ``HandBucket``. Empirically this means CFR
converges to a Nash equilibrium of the **abstracted** game, but the
resulting average strategy remains exploitable in the concrete game.

The learning curve produced by ``experiments/cfr_curve.py`` illustrates
this clearly: sampled exploitability against a 1-step-lookahead BR
*increases* with more MCCFR iterations, because the average strategy
becomes more concentrated (near-deterministic at many info-sets) and
the state-aware BR exploits that concentration. Meanwhile head-to-head
score vs uniform stays modestly positive (+0.05 to +0.14), so training
does improve competitive play against a non-adaptive baseline. The two
signals together bound where a coarse-abstracted CFR sits: better than
random, but far from ε-Nash of the *concrete* game.

A tighter abstraction (potential-aware bucketing over draw-vs-discard
subtrees, or opponent-hand belief state) is the natural follow-up.

Concrete state (invisible to the solver's info-set key):
    * ``hands[p]``: tuple[Card, ...] private hand of player ``p``.
    * ``deck``: tuple[Card, ...] stock; top of deck is the LAST element.
    * ``discard``: tuple[Card, ...] discard pile; top is the LAST element.
    * ``current_player``, ``phase``, ``turn``, ``winner``.

Abstraction (what the info-set key sees):
    * ``HandBucket`` of the acting player's hand.
    * ``PublicBucket`` (top-of-discard class + deck-size bucket).
    * The phase (DRAW / DISCARD).

Deterministic action expansion:
    * ``("draw", 0)`` -> pop top of deck; ``("draw", 1)`` -> pop top of
      discard.
    * ``("discard", value_class)`` -> discard the highest-value card in
      that class currently in hand, tie-breaking by rank order then card
      id. This is the lossy piece of the discard abstraction — the CFR
      solver never sees the concrete card choice.

Auto-knock rule (simplification):
    * If after discard the acting player's deadwood is 0, the hand ends
      with **gin** for that player.
    * Else if it is <= ``knock_limit`` (default 10, matching Classic Gin),
      the hand ends with **knock**. Undercut branch is included: if the
      opponent's deadwood at that moment is <= the knocker's, the
      opponent wins with the undercut bonus. This mirrors
      :class:`gin_rummy.game.GinRummyGame` semantics but does *not* give
      the acting player a "keep playing" option — a fuller model would
      make knocking a decision.

The whole game is capped at ``MAX_TURNS`` (default 60) full turns; if
neither player has ginned or knocked by then the game is scored as a
draw.

`sample_exploitability_curve` returns a coarse, sampled proxy for
NashConv (see docstring). Exact best-response over the abstracted tree is
intractable because the tree still branches over the concrete deck
ordering that our abstraction hides in the info-set key but preserves in
the state; sampled BR is what fits in <30s per checkpoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Callable

from gin_rummy.cards import Card, Deck
from gin_rummy.meld import optimal_decomposition
from gin_rummy.scoring import ScoringRules
from gin_rummy.solvers.abstraction import (
    DRAW_DECK,
    DRAW_DISCARD,
    abstract_action,
    discard_value_class,
    hand_bucket,
    make_discard,
    public_bucket,
)
from gin_rummy.solvers.cfr import (
    AverageStrategy,
    ExternalSamplingMCCFR,
    MCCFREngine,
    uniform_strategy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


HAND_SIZE: int = 10
KNOCK_LIMIT: int = 10
# Turn cap for tractability. Outcome-sampling MCCFR (which
# ``train_gin_cfr`` uses by default) has per-iteration cost O(game
# length), and the sampled best-responder is O(depth * branching *
# rollout_depth). 24 half-turns (12 full turns each) is enough for a
# hand to reach knock-legal deadwood under most strategies while
# keeping exploitability estimation under a second per deal.
MAX_TURNS: int = 24
GIN_BONUS: int = 25
KNOCK_BONUS: int = 10
UNDERCUT_BONUS: int = 20
# Normalisation so utilities live in roughly [-1, +1] (comparable to the
# mini-gin scale). Chosen so a gin-with-full-hand payoff (25 + 60 point
# haul) still fits below ~1.5 in magnitude.
UTILITY_SCALE: float = 60.0


class Phase(IntEnum):
    DEAL = 0
    DRAW = 1
    DISCARD = 2
    TERMINAL = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GinState:
    hands: tuple[tuple[Card, ...], tuple[Card, ...]]
    deck: tuple[Card, ...]        # top = LAST
    discard: tuple[Card, ...]     # top = LAST
    current_player: int
    phase: Phase
    turn: int
    winner: int = -1              # -1 = undecided, 0/1 = player win, 2 = draw
    outcome: str = ""             # "gin" / "knock" / "undercut" / "draw" / ""
    score_delta: float = 0.0      # signed payoff to player 0 at terminal


# ---------------------------------------------------------------------------
# Utility: sorted deterministic hand tuple
# ---------------------------------------------------------------------------


def _card_sort_key(c: Card) -> tuple[int, str, int]:
    return (c.order, c.suit, c.deck_id)


def _sorted(hand: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(sorted(hand, key=_card_sort_key))


# ---------------------------------------------------------------------------
# Deal (chance)
# ---------------------------------------------------------------------------


def initial_state() -> GinState:
    """Pre-deal root: caller triggers ``sample_deal`` next."""
    return GinState(
        hands=((), ()),
        deck=(),
        discard=(),
        current_player=0,
        phase=Phase.DEAL,
        turn=0,
    )


def sample_deal(rng: random.Random) -> GinState:
    """Sample a fresh 52-card deal and return the post-deal state."""
    deck = Deck(1, rng=rng)  # already shuffled
    cards: list[Card] = list(deck)
    # Deal alternately: player 0, then player 1, per Classic Gin.
    h0: list[Card] = []
    h1: list[Card] = []
    idx = 0
    for _ in range(HAND_SIZE):
        h0.append(cards[idx]); idx += 1
        h1.append(cards[idx]); idx += 1
    upcard = cards[idx]; idx += 1
    stock = tuple(cards[idx:])
    return GinState(
        hands=(_sorted(tuple(h0)), _sorted(tuple(h1))),
        deck=stock,
        discard=(upcard,),
        current_player=0,
        phase=Phase.DRAW,
        turn=0,
    )


# ---------------------------------------------------------------------------
# Legal abstract actions
# ---------------------------------------------------------------------------


def _legal_discard_classes(hand: tuple[Card, ...]) -> list[tuple[str, int]]:
    classes: set[int] = set()
    for c in hand:
        classes.add(discard_value_class(c))
    return [make_discard(cls) for cls in sorted(classes)]


def legal_actions(state: GinState) -> list[tuple[str, int]]:
    if state.phase == Phase.DRAW:
        acts: list[tuple[str, int]] = []
        if state.deck:
            acts.append(DRAW_DECK)
        if state.discard:
            acts.append(DRAW_DISCARD)
        return acts
    if state.phase == Phase.DISCARD:
        return _legal_discard_classes(state.hands[state.current_player])
    return []


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _pick_discard_card(hand: tuple[Card, ...], cls: int) -> Card:
    """Deterministically pick the concrete card to discard from a class.

    Highest-value first, tie-break by rank order (highest rank), then by
    card id (deck_id + suit). Ensures ``apply`` is a pure function of
    (state, abstract action).
    """
    candidates = [c for c in hand if discard_value_class(c) == cls]
    if not candidates:
        raise ValueError(f"No card of class {cls} in hand")
    candidates.sort(key=lambda c: (-c.value, -c.order, c.suit, c.deck_id))
    return candidates[0]


@lru_cache(maxsize=200_000)
def _deadwood_cached(hand_key: tuple[Card, ...]) -> int:
    _, _, dv = optimal_decomposition(list(hand_key))
    return dv


def _deadwood(hand: tuple[Card, ...]) -> int:
    return _deadwood_cached(hand)


def _score_gin(winner: int, loser_hand: tuple[Card, ...]) -> tuple[int, str, float]:
    opp_dw = _deadwood(loser_hand)
    gain = GIN_BONUS + opp_dw
    delta = gain if winner == 0 else -gain
    return winner, "gin", delta / UTILITY_SCALE


def _score_knock(
    knocker: int,
    knocker_dw: int,
    opp_hand: tuple[Card, ...],
) -> tuple[int, str, float]:
    opp_dw = _deadwood(opp_hand)
    if opp_dw <= knocker_dw:
        # Undercut: opponent wins.
        gain = UNDERCUT_BONUS + (knocker_dw - opp_dw)
        winner = 1 - knocker
        delta = gain if winner == 0 else -gain
        return winner, "undercut", delta / UTILITY_SCALE
    gain = KNOCK_BONUS + (opp_dw - knocker_dw)
    delta = gain if knocker == 0 else -gain
    return knocker, "knock", delta / UTILITY_SCALE


def apply(state: GinState, action: tuple[str, int]) -> GinState:
    kind, payload = action
    p = state.current_player
    if state.phase == Phase.DRAW:
        if kind != "draw":
            raise ValueError(f"Expected draw in DRAW phase, got {action!r}")
        hand = list(state.hands[p])
        if payload == 0:
            if not state.deck:
                raise ValueError("Deck empty")
            drawn = state.deck[-1]
            new_deck = state.deck[:-1]
            new_discard = state.discard
        else:
            if not state.discard:
                raise ValueError("Discard pile empty")
            drawn = state.discard[-1]
            new_discard = state.discard[:-1]
            new_deck = state.deck
        hand.append(drawn)
        new_hands = list(state.hands)
        new_hands[p] = _sorted(tuple(hand))
        return GinState(
            hands=(new_hands[0], new_hands[1]),
            deck=new_deck,
            discard=new_discard,
            current_player=p,
            phase=Phase.DISCARD,
            turn=state.turn,
        )
    if state.phase == Phase.DISCARD:
        if kind != "discard":
            raise ValueError(f"Expected discard in DISCARD phase, got {action!r}")
        cls = payload
        card = _pick_discard_card(state.hands[p], cls)
        hand = list(state.hands[p])
        hand.remove(card)
        new_hands = list(state.hands)
        new_hands[p] = _sorted(tuple(hand))
        new_discard = state.discard + (card,)
        # Auto-knock: gin > knock > continue.
        dw = _deadwood(new_hands[p])
        if dw == 0:
            winner, outcome, delta = _score_gin(p, new_hands[1 - p])
            return GinState(
                hands=(new_hands[0], new_hands[1]),
                deck=state.deck,
                discard=new_discard,
                current_player=p,
                phase=Phase.TERMINAL,
                turn=state.turn + 1,
                winner=winner,
                outcome=outcome,
                score_delta=delta,
            )
        if dw <= KNOCK_LIMIT:
            winner, outcome, delta = _score_knock(p, dw, new_hands[1 - p])
            return GinState(
                hands=(new_hands[0], new_hands[1]),
                deck=state.deck,
                discard=new_discard,
                current_player=p,
                phase=Phase.TERMINAL,
                turn=state.turn + 1,
                winner=winner,
                outcome=outcome,
                score_delta=delta,
            )
        next_turn = state.turn + 1
        # End-of-game safeties: turn cap or empty stock. Classic rules
        # count this as a draw with zero payoff; that gives MCCFR almost
        # no reward signal because random play draws >90% of the time.
        # To densify the reward we grade the terminal position by
        # comparing final deadwoods: the lower-deadwood player wins a
        # scaled payoff. This is a **scoring abstraction** on top of the
        # rules and is documented in the module docstring.
        if next_turn >= MAX_TURNS or len(state.deck) <= 2:
            dw_p = _deadwood(new_hands[p])
            dw_opp = _deadwood(new_hands[1 - p])
            diff = dw_opp - dw_p  # positive => p wins
            if diff > 0:
                winner = p
            elif diff < 0:
                winner = 1 - p
            else:
                winner = 2
            gain = abs(diff)
            delta = 0.0 if gain == 0 else (gain if winner == 0 else -gain) / UTILITY_SCALE
            return GinState(
                hands=(new_hands[0], new_hands[1]),
                deck=state.deck,
                discard=new_discard,
                current_player=1 - p,
                phase=Phase.TERMINAL,
                turn=next_turn,
                winner=winner,
                outcome="timeout",
                score_delta=delta,
            )
        return GinState(
            hands=(new_hands[0], new_hands[1]),
            deck=state.deck,
            discard=new_discard,
            current_player=1 - p,
            phase=Phase.DRAW,
            turn=next_turn,
        )
    raise ValueError(f"Cannot apply in phase {state.phase}")


# ---------------------------------------------------------------------------
# Terminal / returns
# ---------------------------------------------------------------------------


def is_terminal(state: GinState) -> bool:
    return state.phase == Phase.TERMINAL


def returns(state: GinState) -> tuple[float, float]:
    if not is_terminal(state):
        raise ValueError("Non-terminal state has no returns")
    d = state.score_delta
    return (d, -d)


# ---------------------------------------------------------------------------
# Information set (uses abstraction)
# ---------------------------------------------------------------------------


def information_set(state: GinState, player: int) -> str:
    hand = state.hands[player]
    hb = hand_bucket(hand)
    top = state.discard[-1] if state.discard else None
    pb = public_bucket(top, len(state.deck))
    return (
        f"P{player}|ph{int(state.phase)}|"
        f"hb{hb.key()}|pb{pb.key()}"
    )


# ---------------------------------------------------------------------------
# MCCFREngine wrapper
# ---------------------------------------------------------------------------


class ClassicGinEngine:
    """Adapter that plugs abstracted Classic Gin into the generic MCCFR."""

    def initial_state(self) -> GinState:
        return initial_state()

    def sample_deal(self, rng: random.Random) -> GinState:
        return sample_deal(rng)

    def is_terminal(self, state: GinState) -> bool:
        return is_terminal(state)

    def returns(self, state: GinState) -> tuple[float, float]:
        return returns(state)

    def legal_actions(self, state: GinState) -> list[tuple[str, int]]:
        return legal_actions(state)

    def apply(self, state: GinState, action: tuple[str, int]) -> GinState:
        return apply(state, action)

    def information_set(self, state: GinState, player: int) -> str:
        return information_set(state, player)

    def is_chance(self, state: GinState) -> bool:
        return state.phase == Phase.DEAL

    def current_player(self, state: GinState) -> int:
        return state.current_player


# ---------------------------------------------------------------------------
# Public trainer wrapper
# ---------------------------------------------------------------------------


def train_gin_cfr(
    iterations: int,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
    sampling: str = "outcome",
    regularisation_lambda: float = 0.0,
    min_prob: float = 0.0,
) -> AverageStrategy:
    """Train MCCFR on abstracted Classic Gin.

    Uses outcome-sampling MCCFR by default (see
    :class:`ExternalSamplingMCCFR`); external sampling is infeasible at
    Classic-Gin depth (see module docstring). Returns the average
    strategy over ``iterations`` MCCFR sweeps.

    Parameters
    ----------
    regularisation_lambda : float
        KL-to-uniform regularisation on the average-strategy update.
        See :class:`ExternalSamplingMCCFR` for semantics. Introduced
        to counter PAPER.md §6 prediction #3 (average policy
        concentrates near-deterministic; sampled-BR exploitability rises
        as training progresses).
    min_prob : float
        Floor on every action probability in the returned average
        strategy.
    """
    engine = ClassicGinEngine()
    solver = ExternalSamplingMCCFR(
        seed=seed,
        engine=engine,
        sampling=sampling,
        regularisation_lambda=regularisation_lambda,
        min_prob=min_prob,
    )
    return solver.train(iterations, seed=seed, progress=progress)


# ---------------------------------------------------------------------------
# Sampled exploitability
# ---------------------------------------------------------------------------


def _sample_action(
    strat: AverageStrategy,
    info: str,
    legal: list[tuple[str, int]],
    rng: random.Random,
) -> tuple[str, int]:
    dist = strat.action_probs(info, legal)
    r = rng.random()
    cum = 0.0
    last = legal[-1]
    for a, p in dist.items():
        cum += p
        last = a
        if r <= cum:
            return a
    return last


def _rollout_under(
    state: GinState,
    strat: AverageStrategy,
    rng: random.Random,
    responder: int,
) -> float:
    """Roll out from ``state`` with both players sampling from ``strat``.

    Returns the value to ``responder``.
    """
    while not is_terminal(state):
        cur = state.current_player
        legal = legal_actions(state)
        info = information_set(state, cur)
        act = _sample_action(strat, info, legal, rng)
        state = apply(state, act)
    return returns(state)[responder]


def _sampled_br_value(
    strat: AverageStrategy,
    responder: int,
    num_deals: int = 32,
    rollouts_per_action: int = 2,
    seed: int = 0,
) -> float:
    """Sampled best-response value for ``responder`` against ``strat``.

    At every ``responder`` decision node we evaluate each legal action by
    ``rollouts_per_action`` random completions under ``strat``, then take
    the action with the highest mean. Elsewhere both players sample from
    ``strat``. This is a **1-step lookahead sampled BR** — a strict lower
    bound on the true BR value, and hence a *conservative* estimator of
    exploitability. Cheap enough for a test-time signal (~a few seconds
    at 32 deals).
    """
    total = 0.0
    for i in range(num_deals):
        rng = random.Random(seed * 7919 + i)
        state = sample_deal(rng)
        while not is_terminal(state):
            cur = state.current_player
            legal = legal_actions(state)
            info = information_set(state, cur)
            if cur == responder:
                # 1-step lookahead: evaluate each action by rollouts.
                best_val = float("-inf")
                best_a = legal[0]
                for a in legal:
                    child = apply(state, a)
                    v = 0.0
                    for k in range(rollouts_per_action):
                        rng_k = random.Random(seed * 104729 + i * 97 + hash(a) + k)
                        v += _rollout_under(child, strat, rng_k, responder)
                    v /= rollouts_per_action
                    if v > best_val:
                        best_val = v
                        best_a = a
                state = apply(state, best_a)
            else:
                # Opponent plays ``strat``.
                act = _sample_action(strat, info, legal, rng)
                state = apply(state, act)
        total += returns(state)[responder]
    return total / num_deals


def _sigma_value(
    strat: AverageStrategy,
    player: int,
    num_deals: int = 32,
    seed: int = 0,
) -> float:
    """Value of ``player`` when both sides sample from ``strat``."""
    total = 0.0
    for i in range(num_deals):
        rng = random.Random(seed * 6151 + i)
        state = sample_deal(rng)
        while not is_terminal(state):
            cur = state.current_player
            legal = legal_actions(state)
            info = information_set(state, cur)
            act = _sample_action(strat, info, legal, rng)
            state = apply(state, act)
        total += returns(state)[player]
    return total / num_deals


def sample_exploitability(
    strat: AverageStrategy,
    num_deals: int = 32,
    seed: int = 0,
    rollouts_per_action: int = 2,
) -> float:
    """Sampled proxy for two-player exploitability (NashConv).

    Computes ``(BR_0(sigma) - v_0(sigma)) + (BR_1(sigma) - v_1(sigma))``
    where ``BR_p`` is a **1-step lookahead sampled best-response** (see
    :func:`_sampled_br_value`). This is intentionally a cheap
    approximation — exact BR over the abstracted concrete-deck tree
    remains intractable — but it is genuinely responsive to weaknesses in
    ``strat`` and is monotone with training progress in expectation. Do
    NOT interpret the number as an absolute ε-Nash gap; use it only for
    comparing checkpoints against each other and against uniform.

    Parameters
    ----------
    num_deals : int
        Random deals per estimate.
    seed : int
        RNG seed.
    rollouts_per_action : int
        Number of random completions used to evaluate each action at a
        responder decision node.
    """
    v0 = _sigma_value(strat, 0, num_deals=num_deals, seed=seed + 11)
    v1 = _sigma_value(strat, 1, num_deals=num_deals, seed=seed + 13)
    br0 = _sampled_br_value(
        strat, 0, num_deals=num_deals, rollouts_per_action=rollouts_per_action,
        seed=seed + 17,
    )
    br1 = _sampled_br_value(
        strat, 1, num_deals=num_deals, rollouts_per_action=rollouts_per_action,
        seed=seed + 19,
    )
    return max(0.0, (br0 - v0) + (br1 - v1))


def head_to_head_score(
    strat_a: AverageStrategy,
    strat_b: AverageStrategy,
    num_deals: int = 100,
    seed: int = 0,
) -> float:
    """Return the mean payoff of ``strat_a`` playing against ``strat_b``.

    Alternates seats so first-move / dealing asymmetries cancel: half the
    deals ``strat_a`` sits as player 0 and the other half as player 1.
    The score is measured from ``strat_a``'s perspective; a positive
    value means ``strat_a`` wins on average.

    Head-to-head is a much lower-variance signal than sampled
    exploitability because it only depends on the game outcome (not on a
    fragile best-response estimate). Recommended for regression tests.
    """
    half = num_deals // 2
    total = 0.0
    # A as player 0
    for i in range(half):
        rng = random.Random(seed * 4093 + i)
        state = sample_deal(rng)
        while not is_terminal(state):
            cur = state.current_player
            legal = legal_actions(state)
            info = information_set(state, cur)
            act = _sample_action(strat_a if cur == 0 else strat_b, info, legal, rng)
            state = apply(state, act)
        total += returns(state)[0]
    # A as player 1
    for i in range(half):
        rng = random.Random(seed * 4093 + i + 1_000_003)
        state = sample_deal(rng)
        while not is_terminal(state):
            cur = state.current_player
            legal = legal_actions(state)
            info = information_set(state, cur)
            act = _sample_action(strat_b if cur == 0 else strat_a, info, legal, rng)
            state = apply(state, act)
        total += returns(state)[1]
    return total / (2 * half) if half > 0 else 0.0


def sample_exploitability_curve(
    training_iters: list[int] | None = None,
    num_deals: int = 32,
    seed: int = 0,
    progress: Callable[[int, float], None] | None = None,
    regularisation_lambda: float = 0.0,
    min_prob: float = 0.0,
    h2h_deals: int = 0,
) -> list[tuple[int, float]] | list[tuple[int, float, float]]:
    """Train MCCFR to each checkpoint and return sampled exploitability.

    Approximation: exploitability is measured via
    :func:`sample_exploitability`, which uses a 1-step lookahead sampled
    best-responder (not a true BR). Numbers should NOT be interpreted as
    absolute ε-Nash gaps. See the module docstring for the empirical
    trend and the honest limitations of this coarse abstraction; a
    complementary head-to-head-vs-uniform signal is available via
    :func:`head_to_head_score` (also emitted by
    ``experiments/cfr_curve.py``).

    Parameters
    ----------
    training_iters : list of int
        Cumulative iteration counts (default: [100, 500, 2000, 10000]).
    num_deals : int
        Random deals per exploitability estimate.
    seed : int
        RNG seed for training and estimation.
    regularisation_lambda, min_prob : float
        Smoothing knobs forwarded to :class:`ExternalSamplingMCCFR`.
    h2h_deals : int
        If ``> 0``, also compute head-to-head-vs-uniform at every
        checkpoint using this many deals, and return tuples of
        ``(iters, expl, h2h_vs_uniform)`` instead of ``(iters, expl)``.

    Returns
    -------
    List of tuples in order. Shape depends on ``h2h_deals``:
    ``(iters, expl)`` when 0, ``(iters, expl, h2h)`` otherwise.
    """
    if training_iters is None:
        training_iters = [100, 500, 2000, 10000]
    iters = sorted(set(training_iters))
    engine = ClassicGinEngine()
    solver = ExternalSamplingMCCFR(
        seed=seed,
        engine=engine,
        sampling="outcome",
        regularisation_lambda=regularisation_lambda,
        min_prob=min_prob,
    )
    uniform = AverageStrategy()
    prev = 0
    out_pair: list[tuple[int, float]] = []
    out_triple: list[tuple[int, float, float]] = []
    for target in iters:
        step = target - prev
        if step > 0:
            solver.train(step, seed=None)  # continue from current tables
        strat = solver.average_strategy()
        expl = sample_exploitability(strat, num_deals=num_deals, seed=seed)
        if h2h_deals > 0:
            h2h = head_to_head_score(strat, uniform, num_deals=h2h_deals, seed=seed)
            out_triple.append((target, expl, h2h))
        else:
            out_pair.append((target, expl))
        if progress is not None:
            progress(target, expl)
        prev = target
    return out_triple if h2h_deals > 0 else out_pair


def average_policy_entropy(strat: AverageStrategy) -> float:
    """Mean Shannon entropy (nats) of the average strategy over info-sets.

    A pure-uniform strategy has ``H = ln(n)`` per info-set; a
    deterministic strategy has ``H = 0``. Useful for measuring how much
    the smoothing knobs actually soften the reported policy.
    """
    import math

    if not strat.policy:
        return 0.0
    total = 0.0
    for dist in strat.policy.values():
        s = sum(dist.values())
        if s <= 0:
            continue
        h = 0.0
        for p in dist.values():
            q = p / s
            if q > 0:
                h -= q * math.log(q)
        total += h
    return total / len(strat.policy)


__all__ = [
    "ClassicGinEngine",
    "GinState",
    "HAND_SIZE",
    "KNOCK_LIMIT",
    "MAX_TURNS",
    "Phase",
    "apply",
    "average_policy_entropy",
    "head_to_head_score",
    "information_set",
    "initial_state",
    "is_terminal",
    "legal_actions",
    "returns",
    "sample_deal",
    "sample_exploitability",
    "sample_exploitability_curve",
    "train_gin_cfr",
]
