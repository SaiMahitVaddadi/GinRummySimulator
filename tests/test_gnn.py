"""GNN policy scaffold — graph builder, optional-dep guard, determinism."""

from __future__ import annotations

import pytest

from gin_rummy.cards import Card
from gin_rummy.models.graph import (
    CARD_FEATURE_DIM,
    MELD_FEATURE_DIM,
    NUM_CARD_NODES,
    build_hand_graph,
    card_node_id,
)
from gin_rummy.models.gnn_policy import GNNPolicy, MissingGNNExtras
from gin_rummy.policy import Observation


def _make_obs(hand: tuple[Card, ...], top_discard: Card | None = None) -> Observation:
    return Observation(
        hand=hand,
        top_discard=top_discard,
        discard_pile_size=1 if top_discard else 0,
        deck_size=31,
        turn_number=1,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10,),
    )


def _sample_hand() -> tuple[Card, ...]:
    """A 10-card hand with a clean 4-card run and a 3-card set.

    Run: 2-5 of spades. Set: three 9s (♥ ♦ ♣). Plus three assorted cards.
    """
    return (
        Card("2", "♠"), Card("3", "♠"), Card("4", "♠"), Card("5", "♠"),
        Card("9", "♥"), Card("9", "♦"), Card("9", "♣"),
        Card("K", "♥"), Card("7", "♦"), Card("A", "♣"),
    )


def test_graph_shape_for_sample_hand():
    hand = _sample_hand()
    obs = _make_obs(hand, top_discard=Card("Q", "♥"))
    g = build_hand_graph(obs)

    # 52 card nodes always, regardless of hand size.
    assert g.num_card_nodes == NUM_CARD_NODES
    assert all(len(feat) == CARD_FEATURE_DIM for feat in g.card_features)
    # Every meld feature is the fixed 5 dims.
    assert all(len(feat) == MELD_FEATURE_DIM for feat in g.meld_features)


def test_graph_edge_structure_and_meld_enumeration():
    """For the sample hand the meld enumeration is exactly known.

    Sets: {9♥,9♦,9♣} — one 3-card set. (No 4-card set possible; only 3 nines.)
    Runs: 2♠-5♠ span of length 4 gives:
        - 2 length-3 runs (2-3-4, 3-4-5)
        - 1 length-4 run (2-3-4-5)
      Total 3 runs.
    So 1 + 3 = 4 candidate melds.
    """
    hand = _sample_hand()
    obs = _make_obs(hand)
    g = build_hand_graph(obs)
    assert g.num_meld_nodes == 4

    # Edge count = sum of meld lengths (each card->meld edge appears once).
    # 3 (set) + 3 + 3 + 4 (runs) = 13.
    assert g.num_edges == 13

    # Every edge's card_node_id lands in the in-hand card node set.
    hand_ids = {card_node_id(c) for c in hand}
    for card_id, meld_id in g.edge_index:
        assert card_id in hand_ids
        assert 0 <= meld_id < g.num_meld_nodes

    # in-hand flag (index 18) is set exactly for the 10 in-hand cards.
    in_hand_flags = [feat[18] for feat in g.card_features]
    assert sum(in_hand_flags) == 10.0
    for cid in hand_ids:
        assert g.card_features[cid][18] == 1.0


def test_graph_encodes_top_discard_flag():
    """The top-of-discard card is flagged in feature slot 19."""
    hand = _sample_hand()
    top = Card("Q", "♥")
    g = build_hand_graph(_make_obs(hand, top_discard=top))
    discard_id = card_node_id(top)
    assert g.card_features[discard_id][19] == 1.0
    # Exactly one card has the discard flag set.
    assert sum(feat[19] for feat in g.card_features) == 1.0


def test_graph_builder_is_deterministic():
    """Same observation -> byte-identical graph. Order-preserving enumeration."""
    hand = _sample_hand()
    obs = _make_obs(hand, top_discard=Card("Q", "♥"))
    g1 = build_hand_graph(obs)
    g2 = build_hand_graph(obs)
    assert g1.card_features == g2.card_features
    assert g1.meld_features == g2.meld_features
    assert g1.edge_index == g2.edge_index
    assert g1.meld_cards == g2.meld_cards


def test_graph_builder_handles_no_melds():
    """A hand with no possible sets or runs produces zero meld nodes."""
    hand = (
        Card("A", "♠"), Card("3", "♥"), Card("5", "♦"), Card("7", "♣"),
        Card("9", "♠"), Card("J", "♥"), Card("K", "♦"),
    )
    g = build_hand_graph(_make_obs(hand))
    assert g.num_card_nodes == NUM_CARD_NODES
    assert g.num_meld_nodes == 0
    assert g.num_edges == 0


def test_gnn_policy_raises_without_extras():
    """Without torch installed, ``GNNPolicy()`` gives a clean, actionable error."""
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(MissingGNNExtras) as exc:
            GNNPolicy()
        assert "uv sync --extra gnn" in str(exc.value)
        return

    # torch is present — the constructor should either succeed (if
    # torch-geometric is also present) or raise MissingGNNExtras for
    # the missing PyG dep. Either way, importing shouldn't blow up.
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        with pytest.raises(MissingGNNExtras) as exc:
            GNNPolicy()
        assert "uv sync --extra gnn" in str(exc.value)
        return

    # Full stack available: constructor succeeds.
    policy = GNNPolicy()
    assert policy is not None
