"""Tests for the ``gin_rummy.graph_learning`` subpackage."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from gin_rummy.cards import Deck
from gin_rummy.game import GinRummyGame
from gin_rummy.graph_learning import (
    HashedCoocEmbeddings,
    PolicyFingerprintExtractor,
    SkipgramEmbeddings,
    build_adjacency,
    deepwalk_corpus,
    discover_policy_families,
    graph_augmented_signature,
    kmeans,
    node2vec_corpus,
    node2vec_walk,
    node_id_to_card,
    random_walk,
    silhouette_score,
)
from gin_rummy.models.graph import build_hand_graph
from gin_rummy.policies import (
    ComposableHeuristicPolicy,
    GreedyKnockPolicy,
    discard_highest_deadwood,
    discard_random,
    draw_random,
    knock_asap,
    knock_never,
)
from gin_rummy.policy import Observation, RandomPolicy


# ---------- fixtures ----------


def _canonical_obs(seed: int = 42, n_cards: int = 10) -> Observation:
    deck = Deck(1, rng=random.Random(seed))
    hand = tuple(deck.deal(n_cards))
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
        other_hand_sizes=(n_cards,),
    )


@pytest.fixture
def graph_with_melds():
    """A hand with enough structure to produce >=1 meld."""
    # Loop over a few seeds to guarantee at least one candidate meld exists.
    for seed in range(200):
        obs = _canonical_obs(seed=seed, n_cards=10)
        g = build_hand_graph(obs)
        if g.num_meld_nodes > 0:
            return g
    pytest.fail("could not build a graph with any melds — model changed?")


@pytest.fixture
def graph_dense():
    """A dense graph — several melds sharing cards — for walk-quality tests."""
    # Construct a synthetic hand guaranteed to have multiple overlapping melds:
    # A♠ 2♠ 3♠ 4♠ 5♠ (a 5-run yielding 3 candidate 3-runs + 2 candidate 4-runs
    # + 1 candidate 5-run) plus 5♥ 5♦ (adds a set option touching 5♠).
    from gin_rummy.cards import Card
    hand = (
        Card("A", "♠"), Card("2", "♠"), Card("3", "♠"),
        Card("4", "♠"), Card("5", "♠"),
        Card("5", "♥"), Card("5", "♦"),
        Card("7", "♣"), Card("9", "♣"), Card("J", "♣"),
    )
    obs = Observation(
        hand=hand,
        top_discard=Card("6", "♠"),
        discard_pile_size=1,
        deck_size=42,
        turn_number=0,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10,),
    )
    return build_hand_graph(obs)


# ============================================================
# 1. random_walk stays inside the graph
# ============================================================


def test_random_walk_stays_in_graph(graph_with_melds):
    adj = build_adjacency(graph_with_melds)
    rng = random.Random(0)
    # Pick a non-isolated start node.
    start = next(i for i, ns in enumerate(adj.neighbours) if ns)
    walk = random_walk(adj, start, length=25, rng=rng)
    assert walk[0] == start
    for prev, nxt in zip(walk, walk[1:]):
        assert nxt in adj.neighbours[prev], (
            f"step {prev}->{nxt} is not an edge in the graph"
        )


# ============================================================
# 2. deepwalk_corpus shape
# ============================================================


def test_deepwalk_corpus_shape(graph_with_melds):
    rng = random.Random(1)
    adj = build_adjacency(graph_with_melds)
    n_active = sum(1 for ns in adj.neighbours if ns)
    walks_per_node = 3
    walk_length = 8
    corpus = deepwalk_corpus(
        graph_with_melds,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        rng=rng,
    )
    assert len(corpus) == walks_per_node * n_active
    # Every walk has at least 1 node (the start) and at most walk_length.
    for w in corpus:
        assert 1 <= len(w) <= walk_length


# ============================================================
# 3. node2vec with p=q=1 matches DeepWalk in distribution
# ============================================================


def test_node2vec_reduces_to_deepwalk_when_p_q_1(graph_dense):
    # For a fixed (start, second) pair, node2vec with p=q=1 gives the same
    # step distribution as uniform DeepWalk. We check on a node with a
    # neighbour that itself has multiple neighbours so the biased sampler
    # has real choices to make.
    adj = build_adjacency(graph_dense)
    start = None
    for i, ns in enumerate(adj.neighbours):
        if len(ns) >= 2 and any(len(adj.neighbours[j]) >= 2 for j in ns):
            start = i
            break
    if start is None:
        pytest.skip("graph too sparse to compare node2vec vs deepwalk")

    n_trials = 400
    dw_rng = random.Random(11)
    nv_rng = random.Random(11)

    dw_counts: Counter[int] = Counter()
    nv_counts: Counter[int] = Counter()
    for _ in range(n_trials):
        w1 = random_walk(adj, start, length=3, rng=dw_rng)
        w2 = node2vec_walk(adj, start, length=3, p=1.0, q=1.0, rng=nv_rng)
        if len(w1) >= 3:
            dw_counts[w1[2]] += 1
        if len(w2) >= 3:
            nv_counts[w2[2]] += 1

    # Both should have visited a similar set of third-step nodes with
    # similar frequencies. Compare via total variation distance.
    keys = set(dw_counts) | set(nv_counts)
    tvd = 0.5 * sum(
        abs(dw_counts.get(k, 0) / max(1, sum(dw_counts.values()))
            - nv_counts.get(k, 0) / max(1, sum(nv_counts.values())))
        for k in keys
    )
    assert tvd < 0.20, f"TVD between DeepWalk and node2vec(p=q=1) too large: {tvd:.3f}"


# ============================================================
# 4. k-means recovers planted clusters
# ============================================================


def test_kmeans_recovers_planted_clusters():
    rng = random.Random(2024)
    # Two well-separated 2D blobs.
    blob_a = [[rng.gauss(0.0, 0.15), rng.gauss(0.0, 0.15)] for _ in range(30)]
    blob_b = [[rng.gauss(5.0, 0.15), rng.gauss(5.0, 0.15)] for _ in range(30)]
    vectors = blob_a + blob_b
    truth = [0] * 30 + [1] * 30
    result = kmeans(vectors, k=2, n_init=5, max_iter=100,
                    rng=random.Random(7))
    # Labels are permutation-invariant — check via cluster purity.
    same = sum(1 for a, b in zip(result.labels, truth) if a == b)
    purity = max(same, len(truth) - same) / len(truth)
    assert purity >= 0.95, f"kmeans purity {purity:.3f} below 0.95"
    sil = silhouette_score(vectors, result.labels)
    assert sil > 0.7, f"silhouette too low: {sil:.3f}"


# ============================================================
# 5. policy fingerprints differ across policies
# ============================================================


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _play_n(policy_factory, n_games: int, base_seed: int):
    results = []
    for i in range(n_games):
        seed = base_seed + i
        p = policy_factory(seed)
        opp = RandomPolicy(random.Random(seed + 5000))
        game = GinRummyGame(2, policies=[p, opp], seed=seed)
        results.append(game.play())
    return results


def test_policy_fingerprints_differ():
    def rand_factory(seed):
        return RandomPolicy(random.Random(seed + 101))

    def greedy_factory(seed):
        return GreedyKnockPolicy(random.Random(seed + 202))

    rand_results = _play_n(rand_factory, 20, base_seed=999)
    greedy_results = _play_n(greedy_factory, 20, base_seed=999)
    extractor = PolicyFingerprintExtractor()
    fp_rand = extractor.extract("random", 0, rand_results)
    fp_greedy = extractor.extract("greedy", 0, greedy_results)
    # Vectors have the fixed extractor length.
    assert fp_rand.dim == extractor.feature_len
    assert fp_greedy.dim == extractor.feature_len
    # Cosine distance = 1 - cosine similarity.
    cos = _cosine(fp_rand.features, fp_greedy.features)
    cos_dist = 1.0 - cos
    assert cos_dist > 0.1, (
        f"random vs greedy fingerprints too similar: cos_dist={cos_dist:.3f}"
    )


# ============================================================
# 6. end-to-end discover_policy_families smoke test
# ============================================================


def test_discover_policy_families_smoke():
    def rand_factory(seed):
        return RandomPolicy(random.Random(seed + 101))

    def greedy_factory(seed):
        return GreedyKnockPolicy(random.Random(seed + 202))

    def rand_asap_factory(seed):
        return ComposableHeuristicPolicy(
            draw_rule=draw_random,
            discard_rule=discard_random,
            knock_rule=knock_asap,
            rng=random.Random(seed + 303),
        )

    def high_never_factory(seed):
        return ComposableHeuristicPolicy(
            draw_rule=draw_random,
            discard_rule=discard_highest_deadwood,
            knock_rule=knock_never,
            rng=random.Random(seed + 404),
        )

    policies = [
        ("random", rand_factory),
        ("greedy_knock", greedy_factory),
        ("rand_asap", rand_asap_factory),
        ("highdw_never", high_never_factory),
    ]

    extractor = PolicyFingerprintExtractor()
    fps: dict[str, list[float]] = {}
    for name, factory in policies:
        results = _play_n(factory, 20, base_seed=555)
        fps[name] = extractor.extract(name, 0, results).features

    # Build embeddings just to exercise the graph-augmented path too.
    obs = _canonical_obs(seed=7)
    graph = build_hand_graph(obs)
    corpus = deepwalk_corpus(graph, walks_per_node=4, walk_length=8,
                             rng=random.Random(0))
    emb = HashedCoocEmbeddings(window=3)
    emb.fit(corpus)
    # Sanity: augmented signature is at least as long as behavioural one.
    for name, factory in policies:
        results = _play_n(factory, 5, base_seed=222)
        fp = extractor.extract(name, 0, results)
        aug = graph_augmented_signature(name, fp, emb)
        assert len(aug) >= len(fp.features)

    disc = discover_policy_families(fps, k_range=(2, 3),
                                    rng=random.Random(33))
    # Sanity: every policy got assigned some cluster; at least 2 non-empty.
    assert set(disc.labels.keys()) == set(fps.keys())
    assert len(set(disc.labels.values())) >= 2
    assert disc.best_k in (2, 3)
    # Silhouette is finite, in [-1, 1].
    assert -1.0 <= disc.silhouette <= 1.0


# ============================================================
# bonus: node_id_to_card round-trips
# ============================================================


def test_node_id_to_card_roundtrip():
    from gin_rummy.models.graph import card_node_id

    obs = _canonical_obs(seed=3)
    for c in obs.hand:
        nid = card_node_id(c)
        back = node_id_to_card(nid)
        assert back is not None
        assert back.rank == c.rank
        assert back.suit == c.suit


# ============================================================
# bonus: embeddings smoke
# ============================================================


def test_skipgram_embeddings_smoke(graph_with_melds):
    corpus = deepwalk_corpus(graph_with_melds, walks_per_node=2, walk_length=6,
                             rng=random.Random(1))
    emb = SkipgramEmbeddings(dim=8, window=2, epochs=1,
                             rng=random.Random(0))
    emb.fit(corpus)
    # embed_all keys are exactly the observed nodes.
    all_emb = emb.embed_all()
    assert all(len(v) == 8 for v in all_emb.values())


def test_node2vec_corpus_shape(graph_with_melds):
    rng = random.Random(4)
    adj = build_adjacency(graph_with_melds)
    n_active = sum(1 for ns in adj.neighbours if ns)
    corpus = node2vec_corpus(graph_with_melds, walks_per_node=2,
                             walk_length=5, p=1.0, q=0.5, rng=rng)
    assert len(corpus) == 2 * n_active


def test_node2vec_walk_rejects_bad_params(graph_with_melds):
    with pytest.raises(ValueError):
        node2vec_walk(graph_with_melds, start=0, length=3, p=0.0, q=1.0,
                      rng=random.Random())
    with pytest.raises(ValueError):
        node2vec_walk(graph_with_melds, start=0, length=3, p=1.0, q=-1.0,
                      rng=random.Random())
