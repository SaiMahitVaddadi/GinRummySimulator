"""Worked example for the general Markov machinery.

Four self-contained demos, each printing a compact report to stdout:

1. Fit a :class:`MarkovChain` over four **hand-strength buckets**
   estimated from 200 seeded self-play games. Report the transition
   matrix and the stationary distribution.

2. Build an :class:`AbsorbingChain` with ``gin`` and ``knock`` as
   absorbing states. Report the expected number of turns to absorption
   from each transient (hand-strength) start bucket.

3. Run the :class:`DeckDepletionModel` and compare its
   independent-draws approximation against the empirical rank-class
   frequency at turn 30 across 500 seeded games. This is the
   promised sanity check on the approximation.

4. Run one three-deep cascade from a canonical mid-game state, print
   the top-5 cascade paths with their joint probabilities, and score
   the top path from the perspective of the acting player.

Nothing here writes to disk — the script is meant to be run
interactively. Use it as an executable spec of the API.
"""

from __future__ import annotations

import random
from collections import Counter

from gin_rummy.cards import Card
from gin_rummy.game import GinRummyGame
from gin_rummy.markov.absorbing import AbsorbingChain
from gin_rummy.markov.cascade import (
    CascadeState,
    cascade_value,
    greedy_response_model,
    probabilistic_cascade,
)
from gin_rummy.markov.chain import estimate_chain_from_sequences
from gin_rummy.markov.deck_process import (
    CLASS_NAMES,
    DeckDepletionModel,
    RANK_CLASSES,
)
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policies.heuristic import GreedyKnockPolicy


BUCKETS = ("hi", "mid", "lo", "melded")


def _bucket(hand: list[Card]) -> str:
    """Coarse gin-rummy hand-strength bucket from optimal deadwood value.

    * ``melded`` — deadwood == 0 (gin-ready).
    * ``lo`` — deadwood in [1, 10] (knock-legal in classic gin).
    * ``mid`` — deadwood in (10, 25].
    * ``hi`` — deadwood > 25.
    """
    _, _, dw = optimal_decomposition(hand)
    if dw == 0:
        return "melded"
    if dw <= 10:
        return "lo"
    if dw <= 25:
        return "mid"
    return "hi"


def _collect_bucket_sequences(num_games: int, seed: int) -> list[list[str]]:
    """Play ``num_games`` heuristic self-play games and return one
    per-turn bucket trajectory per (game, player). We reconstruct the
    per-turn hand from the recorded history so we can label each turn
    with the player's post-draw hand-strength bucket."""
    sequences: list[list[str]] = []
    for g in range(num_games):
        # Two greedy policies with independent RNGs but seed-derived.
        rng_a = random.Random(seed + 2 * g)
        rng_b = random.Random(seed + 2 * g + 1)
        game = GinRummyGame(
            num_players=2,
            policies=[GreedyKnockPolicy(rng_a), GreedyKnockPolicy(rng_b)],
            seed=seed + g,
        )
        result = game.play()
        # Group per player.
        by_player: dict[int, list[str]] = {0: [], 1: []}
        for rec in result.history:
            dw = rec.deadwood_value
            if rec.action == "gin":
                bucket = "melded"
            elif dw == 0:
                bucket = "melded"
            elif dw <= 10:
                bucket = "lo"
            elif dw <= 25:
                bucket = "mid"
            else:
                bucket = "hi"
            by_player.setdefault(rec.player_id, []).append(bucket)
        for pid in sorted(by_player):
            seq = by_player[pid]
            if len(seq) >= 2:
                sequences.append(seq)
    return sequences


# --------------------------------------------------------------------- demo 1


def demo_hand_strength_chain(num_games: int = 200, seed: int = 0xC0FFEE) -> None:
    print("=" * 72)
    print(f"[1] MarkovChain over hand-strength buckets ({num_games} games)")
    print("=" * 72)
    seqs = _collect_bucket_sequences(num_games, seed)
    chain = estimate_chain_from_sequences(seqs, BUCKETS, laplace_alpha=1.0)
    chain.validate()

    header = f"{'from \\ to':<10}" + "".join(f"{b:>10}" for b in BUCKETS)
    print(header)
    for s in BUCKETS:
        row = chain.transition[s]
        print(f"{s:<10}" + "".join(f"{row[t]:>10.4f}" for t in BUCKETS))
    print()

    pi = chain.stationary_distribution(tol=1e-10, max_iter=5000)
    print("Stationary distribution (power iteration):")
    for s in BUCKETS:
        print(f"  {s:<8} {pi[s]:.4f}")
    print()


# --------------------------------------------------------------------- demo 2


def demo_absorption_to_gin() -> None:
    print("=" * 72)
    print("[2] AbsorbingChain with gin/knock as terminal states")
    print("=" * 72)
    # We construct a small illustrative chain rather than fit from data
    # (fitting would require distinguishing "player ends the hand" from
    # "player continues" per turn, which the current TurnRecord doesn't
    # cleanly separate). The numbers below are deliberately schematic.
    states = ["hi", "mid", "lo", "melded", "knock", "gin"]
    transition = {
        "hi":     {"hi": 0.50, "mid": 0.45, "lo": 0.03, "melded": 0.00,
                   "knock": 0.01, "gin": 0.01},
        "mid":    {"hi": 0.05, "mid": 0.55, "lo": 0.34, "melded": 0.01,
                   "knock": 0.04, "gin": 0.01},
        "lo":     {"hi": 0.02, "mid": 0.13, "lo": 0.55, "melded": 0.05,
                   "knock": 0.22, "gin": 0.03},
        "melded": {"hi": 0.00, "mid": 0.00, "lo": 0.00, "melded": 0.60,
                   "knock": 0.30, "gin": 0.10},
        "knock":  {"knock": 1.0},
        "gin":    {"gin": 1.0},
    }
    chain = AbsorbingChain(
        states=states,
        transition=transition,
        absorbing_states=["knock", "gin"],
    )
    chain.validate()
    chain.validate_absorbing()

    print("Expected turns to absorption from each transient start bucket:")
    for start in ["hi", "mid", "lo", "melded"]:
        e = chain.expected_absorption(start)
        print(f"  start={start:<8}  E[turns] = {e:.2f}")

    print()
    print("Absorption probabilities (row=start, col=[knock, gin]):")
    B = chain.absorption_probabilities()
    transient = ["hi", "mid", "lo", "melded"]
    for i, s in enumerate(transient):
        print(f"  {s:<8}  P(knock)={B[i][0]:.3f}  P(gin)={B[i][1]:.3f}")
    print()


# --------------------------------------------------------------------- demo 3


def demo_deck_depletion_sanity(
    n_trials: int = 500, turn: int = 30, seed: int = 12345
) -> None:
    print("=" * 72)
    print(
        f"[3] Deck depletion: independence approximation vs. empirical "
        f"at turn {turn} ({n_trials} trials)"
    )
    print("=" * 72)
    model = DeckDepletionModel(num_decks=1)
    counter: Counter[str] = Counter()
    rng = random.Random(seed)
    for _ in range(n_trials):
        seq = model.simulate(turn, rng)
        # The "top card at turn t" is the t-th draw.
        counter[seq[-1]] += 1

    predicted = model.initial_distribution()
    print(f"{'class':<8}{'predicted':>12}{'empirical':>12}{'abs_error':>12}")
    max_err = 0.0
    for c in CLASS_NAMES:
        p = predicted[c]
        emp = counter[c] / n_trials
        err = abs(p - emp)
        max_err = max(max_err, err)
        print(f"{c:<8}{p:>12.4f}{emp:>12.4f}{err:>12.4f}")
    print(f"\n  max abs error: {max_err:.4f}")
    print(
        "  (Under the independence approximation both columns should agree "
        "up to sampling noise ~ 1/sqrt(n_trials).)\n"
    )


# --------------------------------------------------------------------- demo 4


def _canonical_midgame_state() -> tuple[CascadeState, list[Card]]:
    """A hand-picked mid-game state whose response distribution is
    interesting to watch cascade."""
    # Player 0: has a strong ace-set and a partial club run; high deadwood
    # from the K/Q/9/3 tail — canonical "cash in vs. hold" moment.
    hand0 = [
        Card("A", "♠"), Card("A", "♥"), Card("A", "♦"),
        Card("5", "♣"), Card("6", "♣"), Card("7", "♣"),
        Card("K", "♠"), Card("Q", "♥"), Card("9", "♦"), Card("3", "♣"),
    ]
    # Player 1: a decent hand with two pairs and mid-value stragglers.
    hand1 = [
        Card("2", "♠"), Card("2", "♥"), Card("2", "♣"),
        Card("8", "♥"), Card("9", "♥"), Card("10", "♥"),
        Card("4", "♠"), Card("J", "♣"), Card("6", "♦"), Card("K", "♦"),
    ]
    state = CascadeState(
        hands={0: hand0, 1: hand1},
        top_discard=Card("4", "♥"),  # mid-round discard
        turn=12,
    )
    return state, hand0


def demo_cascade() -> None:
    print("=" * 72)
    print("[4] Probabilistic cascade: top-5 paths from a canonical state")
    print("=" * 72)
    state, hand0 = _canonical_midgame_state()

    def model(s: CascadeState, actor_id: int) -> dict[Card, float]:
        return greedy_response_model(s, actor_id, temperature=1.0)

    paths = probabilistic_cascade(
        state, model, depth=3, top_k=5, actor_order=[0, 1]
    )
    for i, path in enumerate(paths, start=1):
        joint = 1.0
        pretty: list[str] = []
        for step in path:
            joint *= step.probability
            action_str = str(step.action)
            pretty.append(
                f"P{step.actor_id + 1}->{action_str}"
                f"(p={step.probability:.3f})"
            )
        print(f"  path {i}:  joint={joint:.4f}   " + "  |  ".join(pretty))

    if paths:
        v = cascade_value(
            paths[0], perspective_actor_id=0, initial_hand=hand0
        )
        print(
            f"\n  cascade_value(top path from P1's perspective) = {v:.4f} "
            f"(joint x deadwood delta)"
        )
    print()


# --------------------------------------------------------------------- main


def main() -> None:
    demo_hand_strength_chain(num_games=200, seed=0xC0FFEE)
    demo_absorption_to_gin()
    demo_deck_depletion_sanity(n_trials=500, turn=30, seed=12345)
    demo_cascade()


if __name__ == "__main__":
    main()
