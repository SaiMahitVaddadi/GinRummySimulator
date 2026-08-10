"""End-to-end demo of the graph_learning subpackage.

Runs, in order:

1. Build a bipartite ``HandGraph`` from a canonical 10-card hand.
2. Fit a :class:`SkipgramEmbeddings` on a DeepWalk corpus of the graph.
3. Play a small tournament with four seeded policies:
     - :class:`RandomPolicy`
     - :class:`GreedyKnockPolicy`
     - ``ComposableHeuristicPolicy(random discard, knock ASAP)``
     - ``ComposableHeuristicPolicy(highest-deadwood discard, never knock)``
4. Extract behavioural fingerprints per policy.
5. Cluster with :func:`discover_policy_families` and print assignments.

Run:

    uv run experiments/graph_learning_demo.py

Deterministic given the ``--seed`` flag; defaults are picked so the
demo runs in <5s on a laptop.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

# Make ``src/`` importable when invoked as a plain script.
import sys
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gin_rummy.cards import Deck  # noqa: E402
from gin_rummy.game import GameResult, GinRummyGame  # noqa: E402
from gin_rummy.graph_learning import (  # noqa: E402
    HashedCoocEmbeddings,
    PolicyFingerprintExtractor,
    deepwalk_corpus,
    discover_policy_families,
    graph_augmented_signature,
)
from gin_rummy.models.graph import build_hand_graph  # noqa: E402
from gin_rummy.policies import (  # noqa: E402
    ComposableHeuristicPolicy,
    GreedyKnockPolicy,
    discard_highest_deadwood,
    discard_random,
    draw_random,
    knock_asap,
    knock_never,
)
from gin_rummy.policy import Observation, RandomPolicy  # noqa: E402


def _canonical_hand(seed: int) -> Observation:
    """Deal a canonical 10-card observation to embed the graph on."""
    deck = Deck(1, rng=random.Random(seed))
    hand = tuple(deck.deal(10))
    top = deck.draw()
    return Observation(
        hand=hand,
        top_discard=top,
        discard_pile_size=1,
        deck_size=len(deck),
        turn_number=0,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10,),
    )


def _play_games(
    policy_name: str,
    policy_factory,
    opponent_factory,
    n_games: int,
    base_seed: int,
) -> list[GameResult]:
    """Play ``n_games`` head-to-heads, always seating the target policy at 0."""
    out: list[GameResult] = []
    for i in range(n_games):
        seed = base_seed + i
        p = policy_factory(seed)
        opp = opponent_factory(seed)
        game = GinRummyGame(2, policies=[p, opp], seed=seed)
        out.append(game.play())
    return out


def _policies():
    """Return ``[(name, factory), ...]`` for the four demo policies."""
    def random_factory(seed: int):
        return RandomPolicy(random.Random(seed + 101))

    def greedy_factory(seed: int):
        return GreedyKnockPolicy(random.Random(seed + 202))

    def rand_discard_knock_asap_factory(seed: int):
        return ComposableHeuristicPolicy(
            draw_rule=draw_random,
            discard_rule=discard_random,
            knock_rule=knock_asap,
            rng=random.Random(seed + 303),
        )

    def high_dw_never_knock_factory(seed: int):
        return ComposableHeuristicPolicy(
            draw_rule=draw_random,
            discard_rule=discard_highest_deadwood,
            knock_rule=knock_never,
            rng=random.Random(seed + 404),
        )

    return [
        ("random",              random_factory),
        ("greedy_knock",        greedy_factory),
        ("rand_discard_asap",   rand_discard_knock_asap_factory),
        ("highdw_never_knock",  high_dw_never_knock_factory),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Graph-learning policy-family demo.")
    ap.add_argument("--seed", type=int, default=1729, help="master seed")
    ap.add_argument("--games", type=int, default=20, help="games per policy")
    ap.add_argument("--walks-per-node", type=int, default=8)
    ap.add_argument("--walk-length", type=int, default=10)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args(argv)

    print(f"[demo] seed={args.seed}  games/policy={args.games}")

    # 1. Build a canonical hand graph and its adjacency.
    obs = _canonical_hand(args.seed)
    graph = build_hand_graph(obs)
    print(
        f"[demo] hand graph: {graph.num_card_nodes} card nodes, "
        f"{graph.num_meld_nodes} meld nodes, {graph.num_edges} edges"
    )

    # 2. DeepWalk corpus + hashed co-occurrence embedding (fast, deterministic).
    walk_rng = random.Random(args.seed + 1)
    corpus = deepwalk_corpus(
        graph,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        rng=walk_rng,
    )
    print(f"[demo] deepwalk corpus: {len(corpus)} walks")
    emb = HashedCoocEmbeddings(window=args.window)
    emb.fit(corpus)
    print(f"[demo] embedding dim (# unique nodes visited): {emb.dim}")

    # 3. Play games.
    policies = _policies()
    opponent_factory = policies[0][1]  # random opponent throughout for comparability
    all_results: dict[str, list[GameResult]] = {}
    for name, factory in policies:
        all_results[name] = _play_games(name, factory, opponent_factory,
                                         args.games, args.seed + 1000)
        wins = sum(1 for r in all_results[name] if r.winner_id == 0)
        print(f"[demo]   {name:20s}  wins as seat 0: {wins}/{args.games}")

    # 4. Extract fingerprints (behavioural + graph-augmented).
    extractor = PolicyFingerprintExtractor()
    behavioural: dict[str, list[float]] = {}
    augmented: dict[str, list[float]] = {}
    for name, results in all_results.items():
        fp = extractor.extract(name, seat_id=0, results=results)
        behavioural[name] = fp.features
        augmented[name] = graph_augmented_signature(name, fp, emb)

    # 5. Cluster both feature sets. With k_range=(2,3) and n=4 policies we're
    #    asking: do these separate into two families? three?
    print("\n[demo] --- clustering (behavioural only) ---")
    disc_b = discover_policy_families(behavioural, k_range=(2, 3),
                                      rng=random.Random(args.seed + 7))
    print(f"  best_k={disc_b.best_k}  silhouette={disc_b.silhouette:.3f}")
    for name, lb in disc_b.labels.items():
        print(f"    cluster {lb}  <-  {name}")

    print("\n[demo] --- clustering (behavioural + graph augmentation) ---")
    disc_g = discover_policy_families(augmented, k_range=(2, 3),
                                      rng=random.Random(args.seed + 11))
    print(f"  best_k={disc_g.best_k}  silhouette={disc_g.silhouette:.3f}")
    for name, lb in disc_g.labels.items():
        print(f"    cluster {lb}  <-  {name}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
