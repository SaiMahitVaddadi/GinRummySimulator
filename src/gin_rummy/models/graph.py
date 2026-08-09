"""Pure-Python bipartite graph builder for rummy hands.

Given an ``Observation``, ``build_hand_graph`` returns a ``HandGraph``
describing a bipartite graph whose left partition is the fixed set of
52 card nodes and whose right partition is a variable number of
*candidate meld* nodes. Card and meld nodes carry hand-crafted feature
vectors; edges connect each card to every meld it participates in.

This is the "unclaimed for card games" encoding the 2024 GNN-for-games
survey called out (see package docstring). Keeping it torch-free means
the encoding is unit-testable in the base install and doubles as a
featurizer for non-GNN downstream models (e.g. classical SFT baselines).

Feature layout
--------------
Card nodes (20 dims each, ordered 0..51 by ``(suit, rank)``):
    [0:13]  rank one-hot (A..K)
    [13:17] suit one-hot (spades, hearts, diamonds, clubs)
    [17]    deadwood-value scalar (raw pip value, 1..10)
    [18]    in-hand flag (1.0 if the card is in the current player's hand)
    [19]    in-discard flag (1.0 if the card is currently the top of discard)

Meld nodes (5 dims each):
    [0]     is_set (1.0 if the meld is a set)
    [1]     is_run (1.0 if the meld is a run)
    [2]     length
    [3]     total pip value
    [4]     average pip value

Edges
-----
Directed ``card -> meld`` tuples ``(card_node_id, meld_node_id)`` for
every (card, meld) pair such that ``card in meld``. The bipartite
structure means the caller can flip these to ``meld -> card`` cheaply
for a message-passing pass in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from gin_rummy.cards import RANK_VALUE, RANKS, SUITS, Card
from gin_rummy.policy import Observation

# 52 canonical card node ids, ordered as (suit index, rank index).
# NOTE: this ordering is fixed *for the model* and independent of the
# game's own card iteration order. It gives every card a stable index
# so learned parameters transfer across observations.
_CARD_INDEX: dict[tuple[str, str], int] = {
    (suit, rank): i
    for i, (suit, rank) in enumerate(
        (suit, rank) for suit in SUITS for rank in RANKS
    )
}
NUM_CARD_NODES: int = 52
CARD_FEATURE_DIM: int = 20
MELD_FEATURE_DIM: int = 5


@dataclass(frozen=True)
class HandGraph:
    """Bipartite ``cards <-> candidate melds`` graph in list-of-lists form.

    Fields are plain Python lists so this object serialises trivially and
    is safe to pickle / dump to JSONL. Convert to torch tensors at the
    boundary of a GNN model.
    """

    card_features: list[list[float]]
    meld_features: list[list[float]]
    edge_index: list[tuple[int, int]]  # (card_node_id, meld_node_id) pairs
    meld_cards: list[list[int]] = field(default_factory=list)
    """For each meld node, the sorted list of card node ids it contains.
    Useful for policy heads that need to look up meld membership without
    walking edges."""

    @property
    def num_card_nodes(self) -> int:
        return len(self.card_features)

    @property
    def num_meld_nodes(self) -> int:
        return len(self.meld_features)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index)


def card_node_id(card: Card) -> int:
    """Return the fixed 0..51 node id for a card.

    Multi-deck games collapse duplicate physical cards onto the same
    node id — this is intentional. The GNN sees "there is an A-of-spades
    in hand", not "there are two copies of A-of-spades." Games that need
    duplicate resolution should extend this module.
    """
    key = (card.suit, card.rank)
    idx = _CARD_INDEX.get(key)
    if idx is None:
        # Jokers or other exotics — fall back to hashing into the last slot.
        # This keeps the shape stable; downstream code should filter jokers
        # before invoking a GNN policy.
        raise KeyError(f"card {card!r} has no canonical node id")
    return idx


def _rank_onehot(rank: str) -> list[float]:
    out = [0.0] * len(RANKS)
    if rank in RANKS:  # jokers etc. get an all-zero one-hot
        out[RANKS.index(rank)] = 1.0
    return out


def _suit_onehot(suit: str) -> list[float]:
    out = [0.0] * len(SUITS)
    if suit in SUITS:
        out[SUITS.index(suit)] = 1.0
    return out


def _card_feature_vector(
    suit: str,
    rank: str,
    *,
    in_hand: bool,
    in_discard: bool,
) -> list[float]:
    return [
        *_rank_onehot(rank),
        *_suit_onehot(suit),
        float(RANK_VALUE.get(rank, 0)),
        1.0 if in_hand else 0.0,
        1.0 if in_discard else 0.0,
    ]


def _enumerate_candidate_melds(
    hand: list[Card],
) -> list[tuple[str, tuple[Card, ...]]]:
    """Enumerate every legal set/run drawn from the hand.

    Sets: 3-4 cards of the same rank. Runs: 3+ consecutive same-suit
    ranks. Matches the enumeration used by ``meld.optimal_decomposition``
    but returns cards rather than indices — cheaper for graph building.
    Jokers are ignored (as in the meld solver).
    """
    melds: list[tuple[str, tuple[Card, ...]]] = []

    # Sets — group by rank.
    by_rank: dict[str, list[Card]] = {}
    for c in hand:
        if c.rank == "JKR":
            continue
        by_rank.setdefault(c.rank, []).append(c)
    for group in by_rank.values():
        if len(group) < 3:
            continue
        for size in range(3, min(len(group), 4) + 1):
            for combo in combinations(group, size):
                melds.append(("set", combo))

    # Runs — group by suit, walk consecutive orders.
    by_suit: dict[str, dict[int, list[Card]]] = {}
    for c in hand:
        if c.rank == "JKR":
            continue
        by_suit.setdefault(c.suit, {}).setdefault(c.order, []).append(c)
    for by_order in by_suit.values():
        orders = sorted(by_order)
        i = 0
        while i < len(orders):
            j = i
            while j + 1 < len(orders) and orders[j + 1] == orders[j] + 1:
                j += 1
            span = orders[i : j + 1]
            for length in range(3, len(span) + 1):
                for start in range(len(span) - length + 1):
                    ranks = span[start : start + length]
                    frontier: list[list[Card]] = [[]]
                    for r in ranks:
                        frontier = [
                            path + [c] for path in frontier for c in by_order[r]
                        ]
                    for combo in frontier:
                        melds.append(("run", tuple(combo)))
            i = j + 1

    return melds


def build_hand_graph(obs: Observation) -> HandGraph:
    """Build the bipartite graph for an ``Observation``.

    Deterministic: the same observation always produces byte-identical
    ``card_features``, ``meld_features``, and ``edge_index`` (subject to
    the enumeration order in ``_enumerate_candidate_melds``, which is
    itself deterministic).
    """
    hand_by_id: set[int] = set()
    for c in obs.hand:
        try:
            hand_by_id.add(card_node_id(c))
        except KeyError:
            continue

    discard_id: int | None = None
    if obs.top_discard is not None:
        try:
            discard_id = card_node_id(obs.top_discard)
        except KeyError:
            discard_id = None

    card_features: list[list[float]] = []
    for suit in SUITS:
        for rank in RANKS:
            idx = _CARD_INDEX[(suit, rank)]
            card_features.append(
                _card_feature_vector(
                    suit,
                    rank,
                    in_hand=idx in hand_by_id,
                    in_discard=idx == discard_id,
                )
            )

    # Enumerate meld nodes over the current hand only. We could enumerate
    # over every card the player *could* end up holding, but that blows up
    # the graph without adding signal — a 2-layer GAT can hop from an
    # in-hand card to a meld and back inside its receptive field.
    hand_cards = list(obs.hand)
    meld_features: list[list[float]] = []
    edge_index: list[tuple[int, int]] = []
    meld_cards: list[list[int]] = []
    for meld_idx, (kind, cards) in enumerate(_enumerate_candidate_melds(hand_cards)):
        length = len(cards)
        total = sum(c.value for c in cards)
        meld_features.append(
            [
                1.0 if kind == "set" else 0.0,
                1.0 if kind == "run" else 0.0,
                float(length),
                float(total),
                float(total) / float(length) if length else 0.0,
            ]
        )
        node_ids: list[int] = []
        for c in cards:
            try:
                cid = card_node_id(c)
            except KeyError:
                continue
            node_ids.append(cid)
            edge_index.append((cid, meld_idx))
        meld_cards.append(sorted(set(node_ids)))

    return HandGraph(
        card_features=card_features,
        meld_features=meld_features,
        edge_index=edge_index,
        meld_cards=meld_cards,
    )
