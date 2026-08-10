"""Node embeddings from walk corpora — pure Python (numpy optional).

Two embedding models ship here:

* :class:`HashedCoocEmbeddings` — for each node, count co-occurrence with
  every other node inside a symmetric context window ``w`` over the walk
  corpus, then L2-normalise. No gradients, no epochs, no hyperparameters
  beyond the window. Cheap, deterministic, and interpretable. Vector
  dimension equals the number of observed nodes (sparse dict form).
* :class:`SkipgramEmbeddings` — a minimal Word2Vec skip-gram-with-
  negative-sampling loop (Mikolov et al. 2013) that produces a dense
  ``dim``-vector per node. Uses stdlib ``random`` and ``math``; if numpy
  is importable we transparently upgrade the inner loops for a large
  speedup on graphs with more than a handful of nodes.

Both models expose the same read API::

    emb.fit(corpus)
    emb.embed(node_id)      -> list[float]
    emb.embed_all()         -> dict[int, list[float]]

Empty corpora produce empty embeddings. Nodes that never appear in the
corpus return zero vectors from :meth:`embed`.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Sequence

try:  # numpy is optional — used strictly for a speed win.
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NUMPY = False


Corpus = Sequence[Sequence[int]]


# ============================================================
# Hashed co-occurrence embedding
# ============================================================


class HashedCoocEmbeddings:
    """Sparse co-occurrence counts, L2-normalised. No training loop.

    For each walk ``w1, w2, ..., wL`` and each centre position ``i``, we
    increment ``cooc[wi][wj]`` for every ``j`` with ``0 < |i - j| <= window``.
    After all walks are consumed we L2-normalise each row so cosine
    similarity is a plain dot product.

    The resulting embedding is a dict-of-dicts; :meth:`embed` densifies a
    single vector on demand, using the same node-id ordering as
    :meth:`embed_all` so the same ``dim`` positions mean the same nodes
    across calls.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._cooc: dict[int, Counter[int]] = defaultdict(Counter)
        self._nodes: list[int] = []
        self._node_index: dict[int, int] = {}
        self._fitted = False

    def fit(self, corpus: Corpus) -> None:
        cooc: dict[int, Counter[int]] = defaultdict(Counter)
        seen: set[int] = set()
        w = self.window
        for walk in corpus:
            L = len(walk)
            for i in range(L):
                a = walk[i]
                seen.add(a)
                lo = max(0, i - w)
                hi = min(L, i + w + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    cooc[a][walk[j]] += 1
        # L2-normalise rows.
        for node, row in cooc.items():
            norm = math.sqrt(sum(v * v for v in row.values()))
            if norm > 0:
                for k in list(row):
                    row[k] = row[k] / norm  # type: ignore[assignment]
        self._cooc = cooc
        self._nodes = sorted(seen)
        self._node_index = {n: i for i, n in enumerate(self._nodes)}
        self._fitted = True

    @property
    def dim(self) -> int:
        return len(self._nodes)

    def embed(self, node_id: int) -> list[float]:
        if not self._fitted:
            raise RuntimeError("call .fit(corpus) before .embed(...)")
        vec = [0.0] * len(self._nodes)
        row = self._cooc.get(node_id)
        if row is None:
            return vec
        for k, v in row.items():
            idx = self._node_index.get(k)
            if idx is not None:
                vec[idx] = float(v)
        return vec

    def embed_all(self) -> dict[int, list[float]]:
        return {n: self.embed(n) for n in self._nodes}


# ============================================================
# Skip-gram with negative sampling (pure python, numpy optional)
# ============================================================


def _sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class SkipgramEmbeddings:
    """A tiny Word2Vec-style skip-gram with negative sampling.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    window : int
        Symmetric context window.
    epochs : int
        Full passes over the corpus.
    negative : int
        Number of negative samples per positive pair.
    lr : float
        SGD learning rate (linearly decayed to ``0.0001 * lr`` across
        training).
    rng : random.Random, optional
        RNG for weight init and negative sampling.

    Notes
    -----
    This is a *reference implementation*, not a speed contender — it is
    tolerable at graph sizes up to a few hundred nodes and a few
    thousand walks. If numpy is importable we use vectorised weight
    updates; otherwise we fall back to nested Python loops.
    """

    def __init__(
        self,
        dim: int = 32,
        window: int = 5,
        epochs: int = 1,
        negative: int = 5,
        lr: float = 0.025,
        rng: random.Random | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if negative < 0:
            raise ValueError("negative must be >= 0")
        self.dim = dim
        self.window = window
        self.epochs = epochs
        self.negative = negative
        self.lr = lr
        self._rng = rng if rng is not None else random.Random()
        self._nodes: list[int] = []
        self._node_index: dict[int, int] = {}
        # Two matrices — input (target) and output (context) — as in the
        # original word2vec.c implementation.
        self._W_in: list[list[float]] | object = []
        self._W_out: list[list[float]] | object = []
        self._neg_table: list[int] = []
        self._fitted = False

    # ---- fit ----

    def _build_neg_table(self, freqs: Counter[int]) -> None:
        # Unigram^0.75 negative sampling table, cf. Mikolov 2013 §2.2.
        table_size = 1000
        weights = {n: (c ** 0.75) for n, c in freqs.items()}
        total = sum(weights.values()) or 1.0
        table: list[int] = []
        for n, w in weights.items():
            k = max(1, int(round(table_size * w / total)))
            table.extend([n] * k)
        self._rng.shuffle(table)
        self._neg_table = table

    def _init_weights(self) -> None:
        n = len(self._nodes)
        r = self._rng
        # Xavier-ish init in [-0.5/dim, 0.5/dim], matching the original C.
        scale = 0.5 / self.dim
        if _HAS_NUMPY:
            arr_in = [[(r.random() - 0.5) / self.dim for _ in range(self.dim)] for _ in range(n)]
            arr_out = [[0.0 for _ in range(self.dim)] for _ in range(n)]
            self._W_in = _np.array(arr_in, dtype=_np.float64)
            self._W_out = _np.array(arr_out, dtype=_np.float64)
        else:
            self._W_in = [
                [(r.random() - 0.5) * (2 * scale) for _ in range(self.dim)]
                for _ in range(n)
            ]
            self._W_out = [
                [0.0 for _ in range(self.dim)] for _ in range(n)
            ]

    def _sample_negative(self, exclude: int) -> int:
        if not self._neg_table:
            # Every node is exclude — fallback.
            return exclude
        for _ in range(10):
            cand = self._rng.choice(self._neg_table)
            if cand != exclude:
                return cand
        return self._neg_table[0]

    def fit(self, corpus: Corpus) -> None:
        freqs: Counter[int] = Counter()
        for walk in corpus:
            for n in walk:
                freqs[n] += 1
        self._nodes = sorted(freqs)
        self._node_index = {n: i for i, n in enumerate(self._nodes)}
        if not self._nodes:
            self._fitted = True
            return
        self._build_neg_table(freqs)
        self._init_weights()

        pair_count = sum(max(0, len(w) - 1) * self.window for w in corpus) * self.epochs
        pair_count = max(1, pair_count)
        processed = 0
        min_lr = self.lr * 0.0001

        for _ in range(self.epochs):
            for walk in corpus:
                L = len(walk)
                for i in range(L):
                    centre = self._node_index.get(walk[i])
                    if centre is None:
                        continue
                    lo = max(0, i - self.window)
                    hi = min(L, i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        ctx = self._node_index.get(walk[j])
                        if ctx is None:
                            continue
                        lr = max(min_lr, self.lr * (1.0 - processed / pair_count))
                        self._sgns_update(centre, ctx, lr)
                        processed += 1
        self._fitted = True

    def _sgns_update(self, centre_i: int, context_i: int, lr: float) -> None:
        neg_i = [self._node_index.get(self._sample_negative(self._nodes[context_i]))
                 for _ in range(self.negative)]
        neg_i = [i for i in neg_i if i is not None]
        if _HAS_NUMPY:
            W_in = self._W_in  # type: ignore[assignment]
            W_out = self._W_out  # type: ignore[assignment]
            v = W_in[centre_i]  # type: ignore[index]
            grad = _np.zeros(self.dim, dtype=_np.float64)  # type: ignore[union-attr]
            # Positive sample.
            u_pos = W_out[context_i]  # type: ignore[index]
            score = float(_np.dot(v, u_pos))  # type: ignore[union-attr]
            g = (_sigmoid(score) - 1.0) * lr
            grad += g * u_pos
            W_out[context_i] = u_pos - g * v  # type: ignore[index]
            # Negatives.
            for ni in neg_i:
                u_neg = W_out[ni]  # type: ignore[index]
                score = float(_np.dot(v, u_neg))  # type: ignore[union-attr]
                g = _sigmoid(score) * lr
                grad += g * u_neg
                W_out[ni] = u_neg - g * v  # type: ignore[index]
            W_in[centre_i] = v - grad  # type: ignore[index]
        else:
            W_in = self._W_in  # type: ignore[assignment]
            W_out = self._W_out  # type: ignore[assignment]
            v = W_in[centre_i]  # type: ignore[index]
            grad = [0.0] * self.dim
            # Positive.
            u_pos = W_out[context_i]  # type: ignore[index]
            score = sum(v[k] * u_pos[k] for k in range(self.dim))
            g = (_sigmoid(score) - 1.0) * lr
            for k in range(self.dim):
                grad[k] += g * u_pos[k]
                u_pos[k] -= g * v[k]
            # Negatives.
            for ni in neg_i:
                u_neg = W_out[ni]  # type: ignore[index]
                score = sum(v[k] * u_neg[k] for k in range(self.dim))
                g = _sigmoid(score) * lr
                for k in range(self.dim):
                    grad[k] += g * u_neg[k]
                    u_neg[k] -= g * v[k]
            for k in range(self.dim):
                v[k] -= grad[k]

    # ---- read ----

    def embed(self, node_id: int) -> list[float]:
        if not self._fitted:
            raise RuntimeError("call .fit(corpus) before .embed(...)")
        idx = self._node_index.get(node_id)
        if idx is None:
            return [0.0] * self.dim
        if _HAS_NUMPY:
            return list(map(float, self._W_in[idx]))  # type: ignore[index]
        return list(self._W_in[idx])  # type: ignore[index]

    def embed_all(self) -> dict[int, list[float]]:
        return {n: self.embed(n) for n in self._nodes}


def average_embedding(
    node_ids: Iterable[int],
    embeddings: HashedCoocEmbeddings | SkipgramEmbeddings,
) -> list[float]:
    """Element-wise mean of the embeddings for ``node_ids``.

    Zero-embedding contributions (from unseen nodes) count toward the
    denominator only if the id is present at all — this keeps the mean
    meaningful even when a policy discards cards that never showed up in
    the walk corpus.
    """
    acc: list[float] = []
    count = 0
    for nid in node_ids:
        v = embeddings.embed(nid)
        if not v:
            continue
        if not acc:
            acc = [0.0] * len(v)
        elif len(v) != len(acc):
            # Dim mismatch shouldn't happen; guard anyway.
            continue
        for i, x in enumerate(v):
            acc[i] += x
        count += 1
    if count == 0:
        return acc
    return [a / count for a in acc]
