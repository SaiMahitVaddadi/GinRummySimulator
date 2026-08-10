"""From-scratch k-means + silhouette + auto-``k`` clustering.

Everything here is stdlib-only — designed to cluster the dozens of
policy fingerprints a research sweep produces, not millions of points.

Reference: Lloyd, "Least squares quantization in PCM." IEEE TIT 1982.
Silhouette score follows Rousseeuw, JCAM 1987.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence


Vector = Sequence[float]


def _squared_distance(a: Vector, b: Vector) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _euclidean(a: Vector, b: Vector) -> float:
    return math.sqrt(_squared_distance(a, b))


def _mean(vectors: Sequence[Vector]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return [x / len(vectors) for x in acc]


@dataclass
class KMeansResult:
    labels: list[int]
    centroids: list[list[float]]
    inertia: float
    n_iter: int
    k: int
    seed_used: int | None = None
    trials: int = 1


def _kmeans_pp_init(
    vectors: Sequence[Vector], k: int, rng: random.Random
) -> list[list[float]]:
    """k-means++ (Arthur & Vassilvitskii 2007) seeding."""
    n = len(vectors)
    first = rng.randrange(n)
    centroids: list[list[float]] = [list(vectors[first])]
    while len(centroids) < k:
        d2 = [
            min(_squared_distance(v, c) for c in centroids) for v in vectors
        ]
        total = sum(d2)
        if total <= 0:
            centroids.append(list(vectors[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i, d in enumerate(d2):
            acc += d
            if acc >= r:
                chosen = i
                break
        centroids.append(list(vectors[chosen]))
    return centroids


def _kmeans_once(
    vectors: Sequence[Vector],
    k: int,
    max_iter: int,
    rng: random.Random,
) -> KMeansResult:
    n = len(vectors)
    centroids = _kmeans_pp_init(vectors, k, rng)
    labels = [0] * n
    for it in range(max_iter):
        changed = False
        for i, v in enumerate(vectors):
            best_j, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                d = _squared_distance(v, c)
                if d < best_d:
                    best_d = d
                    best_j = j
            if labels[i] != best_j:
                labels[i] = best_j
                changed = True
        # Recompute centroids.
        new_centroids: list[list[float]] = []
        for j in range(k):
            members = [vectors[i] for i in range(n) if labels[i] == j]
            if members:
                new_centroids.append(_mean(members))
            else:
                # Empty cluster — re-seed at the point farthest from any centroid.
                far_i, far_d = 0, -1.0
                for i, v in enumerate(vectors):
                    d = min(_squared_distance(v, c) for c in centroids)
                    if d > far_d:
                        far_d = d
                        far_i = i
                new_centroids.append(list(vectors[far_i]))
        centroids = new_centroids
        if not changed:
            break
    inertia = sum(_squared_distance(v, centroids[labels[i]]) for i, v in enumerate(vectors))
    return KMeansResult(
        labels=labels,
        centroids=centroids,
        inertia=inertia,
        n_iter=it + 1,
        k=k,
    )


def kmeans(
    vectors: Sequence[Vector],
    k: int,
    *,
    n_init: int = 5,
    max_iter: int = 100,
    rng: random.Random | None = None,
) -> KMeansResult:
    """Lloyd's k-means with k-means++ seeding and ``n_init`` restarts.

    The restart with the lowest inertia is returned. Empty clusters are
    re-seeded at the point farthest from any current centroid.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if not vectors:
        raise ValueError("vectors must be non-empty")
    if k > len(vectors):
        raise ValueError(f"k ({k}) exceeds number of vectors ({len(vectors)})")
    rng = rng if rng is not None else random.Random()
    best: KMeansResult | None = None
    for trial in range(n_init):
        # Fresh per-trial RNG derived from the outer RNG so each init is
        # independent yet reproducible.
        seed = rng.randrange(2 ** 31 - 1)
        trial_rng = random.Random(seed)
        result = _kmeans_once(vectors, k, max_iter, trial_rng)
        if best is None or result.inertia < best.inertia:
            result.seed_used = seed
            result.trials = trial + 1
            best = result
    assert best is not None
    best.trials = n_init
    return best


def silhouette_score(
    vectors: Sequence[Vector], labels: Sequence[int]
) -> float:
    """Mean silhouette coefficient (Rousseeuw 1987), in ``[-1, 1]``.

    Returns ``0.0`` if there is only one cluster (silhouette undefined).
    """
    n = len(vectors)
    if n != len(labels):
        raise ValueError("vectors and labels must be the same length")
    clusters: dict[int, list[int]] = {}
    for i, lb in enumerate(labels):
        clusters.setdefault(lb, []).append(i)
    if len(clusters) <= 1:
        return 0.0
    total = 0.0
    counted = 0
    for i, v in enumerate(vectors):
        own = clusters[labels[i]]
        if len(own) <= 1:
            # Silhouette undefined for singletons — count as 0.
            counted += 1
            continue
        a = sum(_euclidean(v, vectors[j]) for j in own if j != i) / (len(own) - 1)
        b = float("inf")
        for lb, members in clusters.items():
            if lb == labels[i]:
                continue
            d = sum(_euclidean(v, vectors[j]) for j in members) / len(members)
            if d < b:
                b = d
        denom = max(a, b)
        if denom > 0:
            total += (b - a) / denom
        counted += 1
    return total / counted if counted else 0.0


@dataclass
class ClusterDiscovery:
    best_k: int
    silhouette: float
    labels: dict[str, int]
    centroids: list[list[float]]
    inertia: float
    per_k: dict[int, float] = field(default_factory=dict)


def discover_policy_families(
    fingerprints: dict[str, Vector],
    k_range: tuple[int, int] = (2, 5),
    *,
    n_init: int = 5,
    max_iter: int = 100,
    rng: random.Random | None = None,
) -> ClusterDiscovery:
    """Sweep ``k`` over ``k_range`` and pick the assignment with best silhouette.

    Returns a :class:`ClusterDiscovery` with:

    * ``best_k``       — the ``k`` chosen.
    * ``silhouette``   — its silhouette score.
    * ``labels``       — policy-name → cluster id mapping.
    * ``centroids``    — cluster centroids in the input feature space.
    * ``per_k``        — silhouette per candidate ``k``, for reporting.
    """
    if not fingerprints:
        raise ValueError("fingerprints must be non-empty")
    lo, hi = k_range
    if lo < 2:
        raise ValueError("k_range lower bound must be >= 2")
    if hi < lo:
        raise ValueError("k_range upper bound must be >= lower bound")
    hi = min(hi, len(fingerprints))
    if hi < lo:
        # Not enough policies to sweep — degrade gracefully to k=lo capped at n.
        hi = max(lo, min(lo, len(fingerprints)))
    names = list(fingerprints.keys())
    vectors = [list(fingerprints[n]) for n in names]
    rng = rng if rng is not None else random.Random()

    best: tuple[int, float, "KMeansResult"] | None = None
    per_k: dict[int, float] = {}
    for k in range(lo, hi + 1):
        if k > len(vectors):
            continue
        result = kmeans(vectors, k, n_init=n_init, max_iter=max_iter, rng=rng)
        sil = silhouette_score(vectors, result.labels)
        per_k[k] = sil
        if best is None or sil > best[1]:
            best = (k, sil, result)
    assert best is not None
    k, sil, result = best
    return ClusterDiscovery(
        best_k=k,
        silhouette=sil,
        labels={names[i]: result.labels[i] for i in range(len(names))},
        centroids=result.centroids,
        inertia=result.inertia,
        per_k=per_k,
    )
