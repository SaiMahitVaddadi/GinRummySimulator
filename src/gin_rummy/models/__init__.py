"""Graph neural network policy scaffold for rummy.

This package houses a bipartite ``cards <-> candidate-melds`` graph
representation of a rummy hand and a GAT-based policy trained on top of
it. The graph builder is pure Python; the neural pieces are guarded
behind an optional ``gnn`` extra so importing the package never fails
when torch / torch-geometric are absent.

Public API
----------
* ``HandGraph`` / ``build_hand_graph`` — torch-free graph construction.
* ``GNNPolicy`` — implements the ``Policy`` protocol; raises
  ``MissingGNNExtras`` at construction time if torch isn't installed.
* ``MissingGNNExtras`` — clean, actionable error raised when the extra
  isn't installed.

Novel research opening: essentially zero verified GNN work exists on
rummy, poker, bridge, mahjong, or Euchre (survey, 2024). The strongest
adjacent precedents are Rigaux et al. NeurIPS 2024 (chess-GNN) and the
Hex-GNN line (2023). The bipartite ``card -> meld`` edge type used here
is the encoding the survey names as "unclaimed for card games."
"""

from __future__ import annotations

from gin_rummy.models.graph import HandGraph, build_hand_graph
from gin_rummy.models.gnn_policy import GNNPolicy, MissingGNNExtras

__all__ = [
    "HandGraph",
    "build_hand_graph",
    "GNNPolicy",
    "MissingGNNExtras",
]
