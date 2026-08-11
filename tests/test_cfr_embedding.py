"""Tests for embedding-based hand bucketing + MCCFR wiring.

Covers the four surfaces the U x C study touches:

* :func:`gin_rummy.solvers.embedding_abstraction.build_embedding_bucketing`
  — end-to-end pipeline (hands -> graphs -> walks -> embeddings -> kmeans).
* :class:`EmbeddingHandBucket` determinism and bucket-count bound.
* :func:`gin_rummy.solvers.gin_cfr.train_gin_cfr_embedding`
  — smoke test that a tiny bucketer actually drives CFR to produce
    non-empty tables.

Every test uses tiny knobs (k=4-8, corpus 100-200) so the full file
runs in a few seconds.
"""

from __future__ import annotations

import random
import time

import pytest

from gin_rummy.cards import Deck
from gin_rummy.solvers.embedding_abstraction import (
    DEFAULT_WALK_CONFIG,
    EmbeddingHandBucket,
    WalkConfig,
    build_embedding_bucketing,
    hand_to_embedding,
)
from gin_rummy.solvers.gin_cfr import (
    embedding_information_set,
    train_gin_cfr_embedding,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sample_hands(n: int, seed: int, hand_size: int = 10) -> list[list]:
    hands: list[list] = []
    rng = random.Random(seed)
    for _ in range(n):
        d = Deck(1, rng=random.Random(rng.randrange(2 ** 31 - 1)))
        hands.append(list(d.deal(hand_size)))
    return hands


@pytest.fixture
def tiny_bucketer() -> EmbeddingHandBucket:
    """A very small bucketer suitable for fast tests."""
    return build_embedding_bucketing(
        k=8,
        corpus_size=120,
        walk_config=WalkConfig(
            walks_per_node=2,
            walk_length=15,
            embedding_dim=8,
            window=3,
            epochs=1,
            negative=2,
        ),
        seed=0,
    )


# ---------------------------------------------------------------------------
# test_bucket_determinism
# ---------------------------------------------------------------------------


def test_bucket_determinism(tiny_bucketer: EmbeddingHandBucket) -> None:
    """The same hand should map to the same bucket on repeated calls."""
    hands = _sample_hands(30, seed=99)
    a = [tiny_bucketer.bucket_of(h) for h in hands]
    b = [tiny_bucketer.bucket_of(h) for h in hands]
    assert a == b, "bucketer is non-deterministic on repeated calls"


def test_bucket_determinism_across_reconstruction() -> None:
    """Two bucketers built with the same seed must agree on the same hand."""
    cfg = WalkConfig(
        walks_per_node=2, walk_length=12, embedding_dim=8,
        window=3, epochs=1, negative=2,
    )
    b1 = build_embedding_bucketing(k=8, corpus_size=100, walk_config=cfg, seed=7)
    b2 = build_embedding_bucketing(k=8, corpus_size=100, walk_config=cfg, seed=7)
    hands = _sample_hands(20, seed=13)
    # Centroid identity is not required (kmeans is a permutation up to
    # label ordering under equal seeds anyway — but we did seed the
    # inner RNG identically) so this really should match. If it ever
    # flakes, weaken to "assignments agree up to a permutation."
    a1 = [b1.bucket_of(h) for h in hands]
    a2 = [b2.bucket_of(h) for h in hands]
    assert a1 == a2, (
        "same-seed bucketers disagreed on identical hands:"
        f" {a1} vs {a2}"
    )


# ---------------------------------------------------------------------------
# test_bucket_count_bounded_by_k
# ---------------------------------------------------------------------------


def test_bucket_count_bounded_by_k(tiny_bucketer: EmbeddingHandBucket) -> None:
    """Distinct returned buckets must never exceed k (+1 sink)."""
    k = tiny_bucketer.k
    # 300 hands is plenty to hit every centroid with k=8.
    hands = _sample_hands(300, seed=1)
    ids = {tiny_bucketer.bucket_of(h) for h in hands}
    # Sink bucket may or may not appear; the theoretical ceiling is k + 1.
    assert len(ids) <= k + 1, (
        f"got {len(ids)} distinct buckets for k={k}"
    )
    # And every returned id must lie in [0, k]  (k itself = sink).
    for i in ids:
        assert 0 <= i <= k, f"bucket id {i} out of range for k={k}"


# ---------------------------------------------------------------------------
# test_embedding_bucketing_pipeline_end_to_end
# ---------------------------------------------------------------------------


def test_embedding_bucketing_pipeline_end_to_end() -> None:
    """corpus_size=200, k=8 must complete in <30s and produce a usable bucketer."""
    t0 = time.time()
    bucketer = build_embedding_bucketing(
        k=8,
        corpus_size=200,
        walk_config=DEFAULT_WALK_CONFIG,
        seed=0,
    )
    dt = time.time() - t0
    assert dt < 30.0, f"pipeline took {dt:.1f}s (>= 30s budget)"
    # Fresh hand must produce a bucket (int in valid range).
    fresh_hand = _sample_hands(1, seed=424242)[0]
    b = bucketer.bucket_of(fresh_hand)
    assert isinstance(b, int)
    assert 0 <= b <= bucketer.k
    # And the fitted state should have real centroids.
    assert bucketer.centroids, "no centroids fitted"
    assert bucketer.dim > 0
    # Round-trip through state_dict.
    other = EmbeddingHandBucket(k=8, seed=0)
    other.load_state_dict(bucketer.state_dict())
    other.embeddings = bucketer.embeddings  # weights don't serialise here
    assert other.bucket_of(fresh_hand) == b


# ---------------------------------------------------------------------------
# test_hand_to_embedding shape sanity
# ---------------------------------------------------------------------------


def test_hand_to_embedding_returns_fixed_dim(tiny_bucketer: EmbeddingHandBucket) -> None:
    assert tiny_bucketer.embeddings is not None
    hands = _sample_hands(15, seed=5)
    dims = set()
    for h in hands:
        v = hand_to_embedding(h, tiny_bucketer.embeddings)
        if v:
            dims.add(len(v))
    assert len(dims) == 1, f"embedding pooling produced ragged shapes: {dims}"


# ---------------------------------------------------------------------------
# test_train_gin_cfr_embedding_smoke
# ---------------------------------------------------------------------------


def test_train_gin_cfr_embedding_smoke() -> None:
    """200 CFR iters with a tiny (k=4, corpus=100) bucketer — must not crash
    and must populate at least a few info-sets."""
    bucketer = build_embedding_bucketing(
        k=4,
        corpus_size=100,
        walk_config=WalkConfig(
            walks_per_node=2, walk_length=12, embedding_dim=6,
            window=3, epochs=1, negative=2,
        ),
        seed=0,
    )
    strat = train_gin_cfr_embedding(bucketer, iterations=200, seed=0)
    n = strat.num_info_sets()
    assert n > 0, "CFR produced zero info-sets under the embedding abstraction"
    # And every stored distribution must sum to ~1 (or be at least
    # non-negative).
    for iset, dist in list(strat.policy.items())[:20]:
        s = sum(dist.values())
        assert s > 0.0, f"empty policy at {iset}"


# ---------------------------------------------------------------------------
# regression: embedding_information_set closure works on a live state
# ---------------------------------------------------------------------------


def test_embedding_information_set_produces_stable_key(
    tiny_bucketer: EmbeddingHandBucket,
) -> None:
    """The closure returned by :func:`embedding_information_set` must map
    the same (state, player) to the same string every time."""
    from gin_rummy.solvers.gin_cfr import sample_deal
    rng = random.Random(0)
    state = sample_deal(rng)
    info_fn = embedding_information_set(tiny_bucketer)
    k1 = info_fn(state, 0)
    k2 = info_fn(state, 0)
    assert k1 == k2
    # Key format should include the phase and both bucket components.
    assert k1.startswith("P0|ph")
    assert "hb_emb" in k1
    assert "pb(" in k1
