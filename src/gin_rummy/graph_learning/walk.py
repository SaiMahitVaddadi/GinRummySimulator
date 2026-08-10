"""Uniform random walks over the bipartite hand graph.

Implements the DeepWalk corpus construction of

    Perozzi, Al-Rfou & Skiena. "DeepWalk: Online Learning of Social
    Representations." KDD 2014. https://arxiv.org/abs/1403.6652

The bipartite ``HandGraph`` from :mod:`gin_rummy.models.graph` stores
edges as ``(card_node_id, meld_node_id)`` pairs. Card node ids live in
``[0, num_card_nodes)`` and meld node ids in
``[num_card_nodes, num_card_nodes + num_meld_nodes)`` after the offset
applied by :func:`build_adjacency` — this gives every node in the graph
a single canonical integer id that walks can traverse over.

Everything here is torch-free and depends only on stdlib.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from gin_rummy.cards import RANKS, SUITS, Card
from gin_rummy.models.graph import NUM_CARD_NODES, HandGraph


# Ordered enumeration matches the ``build_hand_graph`` card-feature loop:
# ``for suit in SUITS: for rank in RANKS``. Keep this in lockstep with
# ``_CARD_INDEX`` in models/graph.py.
_INDEX_TO_CARD: tuple[tuple[str, str], ...] = tuple(
    (suit, rank) for suit in SUITS for rank in RANKS
)


@dataclass(frozen=True)
class Adjacency:
    """Undirected adjacency list over the combined bipartite node id space.

    * Node ids ``0..num_card_nodes-1`` are card nodes.
    * Node ids ``num_card_nodes..num_card_nodes+num_meld_nodes-1`` are meld nodes.

    ``neighbours[i]`` is the sorted, deterministic list of node ids adjacent
    to node ``i``. Empty lists denote isolated nodes.
    """

    neighbours: list[list[int]]
    num_card_nodes: int
    num_meld_nodes: int

    @property
    def total_nodes(self) -> int:
        return self.num_card_nodes + self.num_meld_nodes

    def is_isolated(self, node: int) -> bool:
        return 0 <= node < len(self.neighbours) and not self.neighbours[node]


def build_adjacency(graph: HandGraph) -> Adjacency:
    """Materialise a bidirectional adjacency list from a :class:`HandGraph`.

    The underlying ``edge_index`` is stored as directed ``card -> meld``
    pairs; we symmetrise so a walk can bounce back and forth between
    partitions. Meld nodes are re-indexed with an offset of
    ``num_card_nodes`` so every node lives in one flat id space.
    """
    n_cards = graph.num_card_nodes
    n_melds = graph.num_meld_nodes
    n = n_cards + n_melds
    adj: list[set[int]] = [set() for _ in range(n)]
    for card_id, meld_id in graph.edge_index:
        m = n_cards + meld_id
        adj[card_id].add(m)
        adj[m].add(card_id)
    return Adjacency(
        neighbours=[sorted(s) for s in adj],
        num_card_nodes=n_cards,
        num_meld_nodes=n_melds,
    )


def _as_adjacency(graph: HandGraph | Adjacency) -> Adjacency:
    if isinstance(graph, Adjacency):
        return graph
    return build_adjacency(graph)


def random_walk(
    graph: HandGraph | Adjacency,
    start_node: int,
    length: int,
    rng: random.Random,
) -> list[int]:
    """Return a uniform random walk of at most ``length`` steps.

    Follows the classical DeepWalk walker: at each step, pick a uniformly
    random neighbour of the current node. If the walk hits a dead-end
    (an isolated node), the walk terminates early — matching how the
    reference implementation handles sinks.
    """
    if length <= 0:
        return []
    adj = _as_adjacency(graph)
    if start_node < 0 or start_node >= adj.total_nodes:
        raise IndexError(f"start_node {start_node} out of range [0, {adj.total_nodes})")
    walk = [start_node]
    while len(walk) < length:
        cur = walk[-1]
        nbrs = adj.neighbours[cur]
        if not nbrs:
            break
        walk.append(rng.choice(nbrs))
    return walk


def deepwalk_corpus(
    graph: HandGraph | Adjacency,
    walks_per_node: int,
    walk_length: int,
    rng: random.Random,
) -> list[list[int]]:
    """Generate a DeepWalk (Perozzi 2014) walk corpus.

    Perozzi et al. iterate over every non-isolated node and start
    ``walks_per_node`` uniform random walks of ``walk_length`` from it.
    The paper shuffles the node order between epochs so downstream
    skip-gram training sees a well-mixed stream — we do the same.
    """
    adj = _as_adjacency(graph)
    starts = [i for i in range(adj.total_nodes) if adj.neighbours[i]]
    corpus: list[list[int]] = []
    for _ in range(walks_per_node):
        order = starts[:]
        rng.shuffle(order)
        for s in order:
            corpus.append(random_walk(adj, s, walk_length, rng))
    return corpus


# ---------- interpretability helpers ----------


def node_id_to_card(node_id: int) -> Card | None:
    """Return the ``Card`` behind a card-partition node id, or ``None``.

    Meld-partition node ids (``>= NUM_CARD_NODES``) return ``None`` — use
    :func:`node_id_to_meld_cards` for those.
    """
    if not 0 <= node_id < NUM_CARD_NODES:
        return None
    suit, rank = _INDEX_TO_CARD[node_id]
    return Card(rank=rank, suit=suit)


def node_id_to_meld_cards(
    node_id: int, graph: HandGraph
) -> tuple[Card, ...] | None:
    """Return the cards making up a meld-partition node, or ``None``.

    ``node_id`` here is the *global* id used by walks: card ids come first,
    then meld ids offset by ``graph.num_card_nodes``.
    """
    if node_id < graph.num_card_nodes:
        return None
    meld_idx = node_id - graph.num_card_nodes
    if not 0 <= meld_idx < graph.num_meld_nodes:
        return None
    cards: list[Card] = []
    for card_id in graph.meld_cards[meld_idx]:
        c = node_id_to_card(card_id)
        if c is not None:
            cards.append(c)
    return tuple(cards)
