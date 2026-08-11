"""Learned, embedding-based hand-bucketing for MCCFR on Classic Gin.

Where :mod:`gin_rummy.solvers.abstraction` fingerprints a 10-card hand
with four *hand-crafted* integers (deadwood bucket, longest run, number
of sets, top-card class), this module replaces the fingerprint with a
**k-means cluster label over a learned vector representation** of the
hand.

Pipeline (see :func:`build_embedding_bucketing`):

    1. Sample a *corpus* of random 10-card hands from a fresh
       :class:`gin_rummy.cards.Deck`.
    2. Build a :class:`gin_rummy.models.graph.HandGraph` per hand.
    3. Run a DeepWalk corpus over the union of those graphs.
    4. Fit :class:`gin_rummy.graph_learning.embeddings.SkipgramEmbeddings`
       on the walk corpus.
    5. Turn each corpus hand into a fixed-dim vector (sum- or mean-pooled
       node embeddings over card / meld nodes).
    6. Fit :class:`EmbeddingHandBucket` = pure-Python k-means over those
       vectors (:func:`gin_rummy.graph_learning.clustering.kmeans`).

At CFR time the bucketer takes a hand, embeds it, and returns the
closest cluster index — this integer stands in for
:class:`gin_rummy.solvers.abstraction.HandBucket` inside the info-set
key. Compare against the hand-crafted baseline via
``experiments/cfr_embedding.py`` (the flagship U x C interaction study).

Design notes
------------
* Pooling: default **mean**. Sum-pooling is available and biases the
  vector toward hands with more embeddable nodes (many candidate
  melds ⇒ bigger vector norm ⇒ tends to cluster large-meld hands
  together). Mean-pooling is invariant to node count and empirically
  yields more balanced clusters.
* We embed **both** card nodes and meld nodes. Card-only pooling
  discards the very structure the bipartite graph encodes; meld-only
  pooling is unstable when a hand has zero candidate melds. Combining
  them is the honest default.
* Determinism: given a fixed ``seed``, ``build_embedding_bucketing`` is
  reproducible end-to-end.
* Serialisation: :meth:`EmbeddingHandBucket.state_dict` /
  :meth:`load_state_dict` support round-tripping through pickle so a
  fitted bucketer can travel alongside a CFR checkpoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from gin_rummy.cards import Card, Deck
from gin_rummy.graph_learning.clustering import KMeansResult, kmeans
from gin_rummy.graph_learning.embeddings import (
    HashedCoocEmbeddings,
    SkipgramEmbeddings,
)
from gin_rummy.graph_learning.walk import deepwalk_corpus
from gin_rummy.models.graph import build_hand_graph
from gin_rummy.policy import Observation


Embeddings = HashedCoocEmbeddings | SkipgramEmbeddings
Pooling = Literal["mean", "sum"]

HAND_SIZE_DEFAULT: int = 10


# ---------------------------------------------------------------------------
# Walk configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WalkConfig:
    """DeepWalk + skip-gram hyper-parameters.

    Deliberately tiny defaults: the graphs are small (52 card nodes +
    a few dozen candidate melds per hand) and we care about *shape*
    rather than distributional quality — the downstream consumer is a
    k-means with tens of centres, not a similarity search.
    """

    walks_per_node: int = 3
    walk_length: int = 20
    embedding_dim: int = 16
    window: int = 5
    epochs: int = 1
    negative: int = 3
    lr: float = 0.025


DEFAULT_WALK_CONFIG: WalkConfig = WalkConfig()


# ---------------------------------------------------------------------------
# Hand -> vector
# ---------------------------------------------------------------------------


def _hand_to_observation(hand: Sequence[Card]) -> Observation:
    return Observation(
        hand=tuple(hand),
        top_discard=None,
        discard_pile_size=0,
        deck_size=0,
        turn_number=0,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(len(hand),),
    )


def _pool(vectors: list[list[float]], mode: Pooling) -> tuple[float, ...]:
    """Sum- or mean-pool a list of vectors into a single tuple.

    Returns an empty tuple for an empty input; the caller decides how to
    handle unrepresentable hands (typically: treat as a "dead" bucket).
    """
    if not vectors:
        return ()
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            acc[i] += x
    if mode == "sum":
        return tuple(acc)
    if mode == "mean":
        n = len(vectors)
        return tuple(x / n for x in acc)
    raise ValueError(f"Unknown pooling mode: {mode!r}")


def hand_to_embedding(
    hand: Sequence[Card],
    embeddings: Embeddings,
    *,
    pooling: Pooling = "mean",
) -> tuple[float, ...]:
    """Build the hand graph and pool its node embeddings into a vector.

    We look up the embedding for **each card node in hand and each
    candidate-meld node** the graph produced, then pool. Nodes the
    embedding model has never seen contribute the model's zero vector
    (see :meth:`SkipgramEmbeddings.embed`).

    Returns an empty tuple only when the graph has zero embeddable nodes
    — this happens for the trivial empty-hand case only. Under a
    realistic 10-card hand the vector has ``embedding_dim`` entries.
    """
    obs = _hand_to_observation(hand)
    graph = build_hand_graph(obs)
    node_vecs: list[list[float]] = []
    # Card nodes: the graph reserves ids 0..num_card_nodes-1, but only
    # cards in the hand carry real signal — the rest have in_hand=0 and
    # never appear in a walk (they are isolated). Nevertheless we look
    # them up under the same global id space DeepWalk uses.
    hand_ids: set[int] = set()
    for c in hand:
        try:
            from gin_rummy.models.graph import card_node_id  # local: avoid cycle
            hand_ids.add(card_node_id(c))
        except Exception:
            continue
    for cid in sorted(hand_ids):
        v = embeddings.embed(cid)
        if v:
            node_vecs.append(v)
    # Meld nodes are offset by graph.num_card_nodes in the DeepWalk id
    # space (see graph_learning/walk.py).
    for meld_local_idx in range(graph.num_meld_nodes):
        global_id = graph.num_card_nodes + meld_local_idx
        v = embeddings.embed(global_id)
        if v:
            node_vecs.append(v)
    return _pool(node_vecs, pooling)


# ---------------------------------------------------------------------------
# EmbeddingHandBucket
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingHandBucket:
    """A fitted k-means bucketer over hand-embedding vectors.

    Usage::

        b = EmbeddingHandBucket(k=32, seed=0)
        b.fit(training_hands, embeddings=my_skipgram)
        idx = b.bucket_of(some_hand)   # -> int in [0, k)

    The bucketer holds a reference to the fitted embedding model — the
    same model used at fit time must be used at query time, otherwise
    the vector geometry no longer matches the centroids.
    """

    k: int
    seed: int = 0
    pooling: Pooling = "mean"
    embeddings: Embeddings | None = None
    centroids: list[list[float]] = field(default_factory=list)
    dim: int = 0
    _fitted: bool = False
    inertia: float = 0.0
    n_train: int = 0

    # ---- fit ----

    def fit(
        self,
        hands: Iterable[Sequence[Card]],
        *,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Fit k-means over the embeddings of ``hands``.

        ``embeddings`` may be passed explicitly; otherwise the one
        already stored on the bucketer is used. It is an error to call
        ``fit`` without an embedding model available.
        """
        if embeddings is not None:
            self.embeddings = embeddings
        if self.embeddings is None:
            raise ValueError(
                "EmbeddingHandBucket.fit requires an embeddings model"
                " (either passed in or set on the instance)."
            )
        vectors: list[list[float]] = []
        for hand in hands:
            v = hand_to_embedding(hand, self.embeddings, pooling=self.pooling)
            if not v:
                continue
            vectors.append(list(v))
        if not vectors:
            raise ValueError(
                "EmbeddingHandBucket.fit received a corpus that produced no"
                " non-empty embeddings. Are the hands empty or the embedding"
                " unfitted?"
            )
        # Every non-empty vector should share a dimension — take the
        # majority just in case the embedding shape wobbled.
        dim = max({len(v) for v in vectors}, key=lambda d: sum(1 for v in vectors if len(v) == d))
        vectors = [v for v in vectors if len(v) == dim]
        self.dim = dim
        k = min(self.k, len(vectors))
        if k < 1:
            raise ValueError("Not enough hands to fit even k=1.")
        result: KMeansResult = kmeans(
            vectors, k, n_init=3, max_iter=50, rng=random.Random(self.seed)
        )
        self.centroids = [list(c) for c in result.centroids]
        self.inertia = result.inertia
        self.n_train = len(vectors)
        self._fitted = True

    # ---- query ----

    def bucket_of(self, hand: Sequence[Card]) -> int:
        """Return the nearest cluster index in ``[0, k_effective)``.

        Hands whose embedding is empty (embedding model has never seen
        any of the hand's nodes) fall through to a dedicated *sink*
        bucket ``k``; this keeps the returned integer stable even for
        out-of-distribution hands.
        """
        if not self._fitted:
            raise RuntimeError("call .fit(hands) before .bucket_of(...)")
        if self.embeddings is None:
            raise RuntimeError("no embeddings model attached")
        v = hand_to_embedding(hand, self.embeddings, pooling=self.pooling)
        if not v or len(v) != self.dim:
            # Unrepresentable hand: dump to the sink bucket. Keeps CFR
            # info-set count bounded by ``k + 1``.
            return len(self.centroids)
        best_j = 0
        best_d = float("inf")
        for j, c in enumerate(self.centroids):
            d = 0.0
            for a, b in zip(v, c):
                diff = a - b
                d += diff * diff
                if d >= best_d:
                    break
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    @property
    def num_buckets(self) -> int:
        """Effective number of buckets (including the sink)."""
        return len(self.centroids) + 1

    # ---- serialise ----

    def state_dict(self) -> dict:
        return {
            "k": self.k,
            "seed": self.seed,
            "pooling": self.pooling,
            "centroids": [list(c) for c in self.centroids],
            "dim": self.dim,
            "inertia": self.inertia,
            "n_train": self.n_train,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        self.k = int(state["k"])
        self.seed = int(state["seed"])
        self.pooling = state.get("pooling", "mean")
        self.centroids = [list(c) for c in state.get("centroids", [])]
        self.dim = int(state.get("dim", 0))
        self.inertia = float(state.get("inertia", 0.0))
        self.n_train = int(state.get("n_train", 0))
        self._fitted = bool(state.get("fitted", False))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _sample_random_hands(
    corpus_size: int,
    hand_size: int,
    seed: int,
) -> list[list[Card]]:
    """Sample ``corpus_size`` random 10-card hands from a fresh deck each.

    Uses a **fresh** shuffled deck per hand — cheaper than a Fisher-Yates
    over a big card pool, and lets each hand see the full 52-card suit
    universe. Reproducible under ``seed``.
    """
    hands: list[list[Card]] = []
    rng = random.Random(seed)
    for i in range(corpus_size):
        # Per-hand RNG derived from the outer seed for reproducibility.
        d = Deck(1, rng=random.Random(rng.randrange(2 ** 31 - 1)))
        hand = list(d.deal(hand_size))
        hands.append(hand)
    return hands


def _train_embeddings_on_hands(
    hands: list[list[Card]],
    walk_config: WalkConfig,
    seed: int,
) -> SkipgramEmbeddings:
    """Build a combined walk corpus over ``hands`` and fit skip-gram.

    Each hand contributes its own DeepWalk sub-corpus; concatenating them
    is the simplest honest way to give the skip-gram model a mix of
    per-hand structural patterns without collapsing to a single graph.
    """
    corpus: list[list[int]] = []
    walk_rng = random.Random(seed + 1)
    for hand in hands:
        obs = _hand_to_observation(hand)
        graph = build_hand_graph(obs)
        # Skip degenerate graphs with no melds — no walk can leave a
        # card node without at least one edge.
        if graph.num_meld_nodes == 0:
            continue
        sub = deepwalk_corpus(
            graph,
            walks_per_node=walk_config.walks_per_node,
            walk_length=walk_config.walk_length,
            rng=walk_rng,
        )
        corpus.extend(sub)
    emb = SkipgramEmbeddings(
        dim=walk_config.embedding_dim,
        window=walk_config.window,
        epochs=walk_config.epochs,
        negative=walk_config.negative,
        lr=walk_config.lr,
        rng=random.Random(seed + 2),
    )
    emb.fit(corpus)
    return emb


def build_embedding_bucketing(
    k: int = 32,
    corpus_size: int = 2000,
    walk_config: WalkConfig | None = None,
    seed: int = 0,
    hand_size: int = HAND_SIZE_DEFAULT,
    pooling: Pooling = "mean",
) -> EmbeddingHandBucket:
    """End-to-end pipeline: hands -> graphs -> walks -> embeddings -> k-means.

    Returns a fitted :class:`EmbeddingHandBucket` ready to be passed to
    :func:`gin_rummy.solvers.gin_cfr.train_gin_cfr_embedding`.

    ``corpus_size=2000`` and ``k=32`` are the default operating point of
    ``experiments/cfr_embedding.py``. For smoke tests, ``corpus_size=100``
    and ``k=8`` is fast (<5s).
    """
    wc = walk_config if walk_config is not None else DEFAULT_WALK_CONFIG
    hands = _sample_random_hands(corpus_size, hand_size, seed=seed)
    embeddings = _train_embeddings_on_hands(hands, wc, seed=seed)
    bucketer = EmbeddingHandBucket(k=k, seed=seed, pooling=pooling)
    bucketer.fit(hands, embeddings=embeddings)
    return bucketer


__all__ = [
    "DEFAULT_WALK_CONFIG",
    "EmbeddingHandBucket",
    "Embeddings",
    "Pooling",
    "WalkConfig",
    "build_embedding_bucketing",
    "hand_to_embedding",
]
