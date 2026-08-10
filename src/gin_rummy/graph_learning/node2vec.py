"""Biased 2nd-order random walks (node2vec).

Implements the return-parameter ``p`` / in-out-parameter ``q`` walker from

    Grover & Leskovec. "node2vec: Scalable Feature Learning for Networks."
    KDD 2016. https://arxiv.org/abs/1607.00653

Per-step sampling logic (given previous node ``t`` and current node ``v``):

    * For each candidate next node ``x`` in ``N(v)``:
        - ``x == t``           →  unnormalised weight ``1/p``  (return)
        - ``x`` is neighbour of ``t``  →  weight ``1``        (BFS-like, DFS-neutral)
        - otherwise            →  weight ``1/q``              (outward exploration)

Setting ``p == q == 1`` collapses to a uniform 1st-order random walk,
which matches DeepWalk's walker in distribution — a sanity check the
tests exercise directly.
"""

from __future__ import annotations

import random

from gin_rummy.graph_learning.walk import Adjacency, _as_adjacency
from gin_rummy.models.graph import HandGraph


def _weighted_choice(
    items: list[int],
    weights: list[float],
    rng: random.Random,
) -> int:
    """Reservoir-free weighted sample. ``rng.choices`` avoids importing numpy."""
    return rng.choices(items, weights=weights, k=1)[0]


def node2vec_walk(
    graph: HandGraph | Adjacency,
    start: int,
    length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[int]:
    """One biased 2nd-order random walk of at most ``length`` steps.

    See module docstring for the transition weights. When the walk hits an
    isolated node the walk terminates early, mirroring the DeepWalk walker.
    ``p`` and ``q`` must be strictly positive so their reciprocals are
    finite.
    """
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be strictly positive")
    if length <= 0:
        return []
    adj = _as_adjacency(graph)
    if start < 0 or start >= adj.total_nodes:
        raise IndexError(f"start {start} out of range [0, {adj.total_nodes})")
    walk = [start]
    if length == 1:
        return walk
    nbrs = adj.neighbours[start]
    if not nbrs:
        return walk
    # First step: no previous node — uniform choice matches DeepWalk.
    walk.append(rng.choice(nbrs))
    while len(walk) < length:
        cur = walk[-1]
        prev = walk[-2]
        cur_nbrs = adj.neighbours[cur]
        if not cur_nbrs:
            break
        prev_nbr_set = set(adj.neighbours[prev])
        weights: list[float] = []
        for x in cur_nbrs:
            if x == prev:
                weights.append(1.0 / p)
            elif x in prev_nbr_set:
                weights.append(1.0)
            else:
                weights.append(1.0 / q)
        walk.append(_weighted_choice(cur_nbrs, weights, rng))
    return walk


def node2vec_corpus(
    graph: HandGraph | Adjacency,
    walks_per_node: int,
    walk_length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[list[int]]:
    """DeepWalk-style corpus, but each walk is biased by ``p`` / ``q``.

    Matches the training-set construction in Grover & Leskovec 2016: for
    every non-isolated node, launch ``walks_per_node`` walks of the given
    length; shuffle node order per epoch so the downstream skip-gram model
    sees a well-mixed sequence.
    """
    adj = _as_adjacency(graph)
    starts = [i for i in range(adj.total_nodes) if adj.neighbours[i]]
    corpus: list[list[int]] = []
    for _ in range(walks_per_node):
        order = starts[:]
        rng.shuffle(order)
        for s in order:
            corpus.append(node2vec_walk(adj, s, walk_length, p, q, rng))
    return corpus
