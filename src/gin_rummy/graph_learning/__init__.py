"""Unsupervised graph-learning primitives on top of the bipartite hand graph.

This subpackage layers *label-free* representation learning and policy
introspection onto ``gin_rummy.models.graph.HandGraph``:

* :mod:`walk`               — DeepWalk-style uniform random walks (Perozzi 2014).
* :mod:`node2vec`           — biased 2nd-order walks (Grover & Leskovec 2016).
* :mod:`embeddings`         — pure-python skip-gram / hashed co-occurrence.
* :mod:`policy_signatures`  — behavioural fingerprints + graph-augmented vectors.
* :mod:`clustering`         — from-scratch k-means + silhouette + auto-``k``.

Everything is torch-free — pure Python plus stdlib. NumPy is imported
opportunistically for speed only and never required.
"""

from gin_rummy.graph_learning.clustering import (
    discover_policy_families,
    kmeans,
    silhouette_score,
)
from gin_rummy.graph_learning.embeddings import (
    HashedCoocEmbeddings,
    SkipgramEmbeddings,
)
from gin_rummy.graph_learning.node2vec import node2vec_corpus, node2vec_walk
from gin_rummy.graph_learning.policy_signatures import (
    PolicyFingerprintExtractor,
    graph_augmented_signature,
)
from gin_rummy.graph_learning.walk import (
    build_adjacency,
    deepwalk_corpus,
    node_id_to_card,
    node_id_to_meld_cards,
    random_walk,
)

__all__ = [
    "HashedCoocEmbeddings",
    "PolicyFingerprintExtractor",
    "SkipgramEmbeddings",
    "build_adjacency",
    "deepwalk_corpus",
    "discover_policy_families",
    "graph_augmented_signature",
    "kmeans",
    "node2vec_corpus",
    "node2vec_walk",
    "node_id_to_card",
    "node_id_to_meld_cards",
    "random_walk",
    "silhouette_score",
]
