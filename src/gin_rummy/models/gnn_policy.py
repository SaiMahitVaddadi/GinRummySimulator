"""GNN policy scaffold.

Wraps a small ``CardMeldGAT`` model over the bipartite hand graph and
adapts it to the ``Policy`` protocol so it drops straight into
``GinRummyGame(policies=...)``.

Design mirrors ``finetune.trainers`` for the optional-dep pattern:
importing this module never fails, but instantiating ``GNNPolicy``
raises ``MissingGNNExtras`` with an actionable install hint when
``torch`` / ``torch_geometric`` are missing.

Architecture
------------
Two GAT layers over the bipartite ``card <-> meld`` graph (undirected —
we symmetrise the edge index at forward time). After message passing we
have:

* A per-card embedding — a score head projects each to a scalar. The
  *in-hand* mask selects candidates for discard; softmax over those
  scores yields the discard distribution.
* A pooled global embedding — two scalar heads decide (a) draw source
  (deck vs. discard) and (b) whether to knock.

The heads are intentionally tiny; the meld graph carries almost all of
the structural signal in the classical rummy sense (Bayer et al. 2005,
Rigaux et al. 2024). This is a scaffold — the exact head shapes are the
first thing you'll want to sweep.
"""

from __future__ import annotations

import random
from typing import Any

from gin_rummy.cards import Card
from gin_rummy.models.graph import (
    CARD_FEATURE_DIM,
    MELD_FEATURE_DIM,
    NUM_CARD_NODES,
    build_hand_graph,
    card_node_id,
)
from gin_rummy.policy import DrawSource, Observation

_INSTALL_HINT = (
    "The GNN policy requires torch and torch-geometric. Install the extra:\n"
    "    uv sync --extra gnn"
)


class MissingGNNExtras(RuntimeError):
    """Raised when the GNN extra isn't installed and the user tries to use it."""


def _require(name: str, hint: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise MissingGNNExtras(
            f"'{name}' is required for GNNPolicy ({hint}).\n{_INSTALL_HINT}"
        ) from exc


def _torch_bits() -> tuple[Any, Any, Any, Any]:
    """Import torch stack lazily. Raises ``MissingGNNExtras`` on failure."""
    torch = _require("torch", "core tensor ops")
    _require("torch_geometric", "GAT layers")
    from torch import nn  # noqa: WPS433 — deliberate late import
    from torch_geometric.nn import GATConv  # noqa: WPS433

    return torch, nn, GATConv, torch.nn.functional


def _build_gat_module(hidden_dim: int, heads: int) -> Any:
    """Construct ``CardMeldGAT`` on demand — kept out of module scope so
    the ``nn.Module`` subclass is only defined when torch is available."""
    torch, nn, GATConv, F = _torch_bits()

    class CardMeldGAT(nn.Module):
        """Two-layer GAT over the bipartite ``card <-> meld`` graph.

        Cards and melds have different feature dims so we project each
        into a shared hidden space before message passing. We use a
        single homogeneous GAT (with the union of node types) rather
        than PyG's ``HeteroConv`` — simpler, and the type distinction is
        already encoded in the input projection.
        """

        def __init__(self) -> None:
            super().__init__()
            self.card_proj = nn.Linear(CARD_FEATURE_DIM, hidden_dim)
            self.meld_proj = nn.Linear(MELD_FEATURE_DIM, hidden_dim)
            self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            self.card_score = nn.Linear(hidden_dim, 1)
            self.draw_head = nn.Linear(hidden_dim, 2)  # deck vs discard
            self.knock_head = nn.Linear(hidden_dim, 1)  # knock logit

        def forward(
            self,
            card_x: Any,
            meld_x: Any,
            edge_index: Any,
        ) -> tuple[Any, Any, Any]:
            c = F.elu(self.card_proj(card_x))
            m = F.elu(self.meld_proj(meld_x)) if meld_x.numel() else meld_x.new_zeros((0, c.size(-1)))
            x = torch.cat([c, m], dim=0)
            x = F.elu(self.gat1(x, edge_index))
            x = F.elu(self.gat2(x, edge_index))
            card_x_out = x[: NUM_CARD_NODES]
            pooled = x.mean(dim=0, keepdim=True)
            card_scores = self.card_score(card_x_out).squeeze(-1)
            draw_logits = self.draw_head(pooled).squeeze(0)
            knock_logit = self.knock_head(pooled).squeeze()
            return card_scores, draw_logits, knock_logit

    return CardMeldGAT()


class GNNPolicy:
    """Bipartite-GAT policy. Implements the ``Policy`` protocol.

    Parameters
    ----------
    model_path : str | None
        Path to a torch ``state_dict``. When ``None`` the model uses
        random init — useful for smoke tests and self-play warmup.
    hidden_dim : int, default 64
        Hidden width of the GAT layers.
    heads : int, default 4
        Number of attention heads per GAT layer.
    device : str, default "cpu"
        Torch device string.
    rng : random.Random | None
        Fallback RNG for tie-breaking / no-op edge cases.

    Instantiating this class without torch installed raises
    ``MissingGNNExtras`` with a copy-pasteable install command.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        device: str = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        torch, _nn, _GATConv, _F = _torch_bits()
        self._torch = torch
        self._device = torch.device(device)
        self._model = _build_gat_module(hidden_dim, heads).to(self._device)
        if model_path is not None:
            state = torch.load(model_path, map_location=self._device)
            self._model.load_state_dict(state)
        self._model.eval()
        self._rng = rng if rng is not None else random.Random()

    # ------- Policy protocol -------

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        if obs.top_discard is None:
            return "deck"
        _, draw_logits, _ = self._forward(obs)
        # index 0 = deck, index 1 = discard (per draw_head ordering above)
        return "discard" if draw_logits[1].item() > draw_logits[0].item() else "deck"

    def choose_discard(self, obs: Observation) -> Card:
        card_scores, _, _ = self._forward(obs)
        # Mask to in-hand cards only; pick argmax.
        hand_ids: list[int] = []
        card_lookup: dict[int, Card] = {}
        for c in obs.hand:
            try:
                cid = card_node_id(c)
            except KeyError:
                continue
            hand_ids.append(cid)
            card_lookup[cid] = c
        if not hand_ids:
            return self._rng.choice(obs.hand)
        # Higher score = better to *keep*, so discard the lowest-scored card.
        scores = [(cid, card_scores[cid].item()) for cid in hand_ids]
        worst_cid = min(scores, key=lambda kv: kv[1])[0]
        return card_lookup[worst_cid]

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        _, _, knock_logit = self._forward(obs)
        # Illegal knocks are filtered by the engine; we only advise here.
        return bool(knock_logit.item() > 0)

    # ------- internals -------

    def _forward(self, obs: Observation) -> tuple[Any, Any, Any]:
        torch = self._torch
        graph = build_hand_graph(obs)
        card_x = torch.tensor(
            graph.card_features, dtype=torch.float32, device=self._device
        )
        if graph.num_meld_nodes:
            meld_x = torch.tensor(
                graph.meld_features, dtype=torch.float32, device=self._device
            )
        else:
            meld_x = torch.zeros((0, MELD_FEATURE_DIM), dtype=torch.float32, device=self._device)

        # Offset meld node ids by NUM_CARD_NODES and symmetrise.
        edges: list[list[int]] = [[], []]
        for card_id, meld_id in graph.edge_index:
            m = meld_id + NUM_CARD_NODES
            edges[0].extend([card_id, m])
            edges[1].extend([m, card_id])
        if not edges[0]:
            # Add a single self-loop on card 0 so GATConv doesn't choke on
            # an empty edge index (bipartite hand with no legal melds).
            edges = [[0], [0]]
        edge_index = torch.tensor(edges, dtype=torch.long, device=self._device)

        with torch.no_grad():
            return self._model(card_x, meld_x, edge_index)
