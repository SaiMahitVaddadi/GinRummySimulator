"""Training-loop scaffold for the GNN policy.

Consumes ``DataCollector.to_sft_rows`` output — the same corpus format
used by the LLM SFT scaffold — and turns each row into a supervised
``(graph, target_card_id)`` example for the discard head. The data
loading / graph-encoding path is fully runnable without torch installed
so you can dry-run the pipeline; the training loop itself raises
``NotImplementedError`` once torch is required.

Row format
----------
Each collector row is ``{"prompt": <serialised observation>, "completion":
<card string>}``. We currently rely on the caller to supply a
``prompt_to_observation`` decoder — the collector prompt format is
project-specific, and hard-coding a parser here would bake in the
wrong assumption.

To wire this up end-to-end:

1. Implement an ``Observation`` (de)serialiser somewhere central.
2. Pass it in as ``prompt_decoder``.
3. Pick a ``completion`` decoder (``str -> Card``) that mirrors the
   game's rendering.

Then ``prepare_dataset`` will produce a list of
``TrainingExample(graph, target_card_node_id)`` you can batch and shove
through the GAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from gin_rummy.cards import Card
from gin_rummy.models.graph import HandGraph, build_hand_graph, card_node_id
from gin_rummy.models.gnn_policy import MissingGNNExtras, _require
from gin_rummy.policy import Observation


@dataclass(frozen=True)
class TrainingExample:
    """One supervised discard example.

    Attributes
    ----------
    graph : HandGraph
        Bipartite ``card <-> meld`` encoding of the observation.
    target_card_node_id : int
        The 0..51 canonical id of the card that was actually discarded.
        Cross-entropy target for the per-card score head, masked to the
        in-hand cards.
    """

    graph: HandGraph
    target_card_node_id: int


def prepare_dataset(
    rows: Sequence[dict[str, str]],
    *,
    prompt_decoder: Callable[[str], Observation],
    completion_decoder: Callable[[str], Card],
) -> list[TrainingExample]:
    """Turn collector SFT rows into ``TrainingExample`` instances.

    Any row that fails to decode (bad prompt, joker discards, etc.) is
    silently dropped. This is the "please still run when the corpus is
    dirty" behaviour we want in a scaffold — swap in a strict decoder
    once you're ready to trust the input.
    """
    out: list[TrainingExample] = []
    for row in rows:
        try:
            obs = prompt_decoder(row["prompt"])
            card = completion_decoder(row["completion"])
            target = card_node_id(card)
        except (KeyError, ValueError):
            continue
        graph = build_hand_graph(obs)
        out.append(TrainingExample(graph=graph, target_card_node_id=target))
    return out


def iter_batches(
    examples: Sequence[TrainingExample],
    batch_size: int,
) -> Iterable[Sequence[TrainingExample]]:
    """Yield fixed-size mini-batches. No shuffling — the caller can wrap.

    Kept as a bare generator so it's exercised by dry-run tests without
    torch. When you plug in torch-geometric, replace this with
    ``torch_geometric.loader.DataLoader`` that batches by disjoint union
    of the per-example graphs (standard PyG idiom).
    """
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def train_sft(
    examples: Sequence[TrainingExample],
    *,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> str:
    """SFT the discard head on ``examples``. Returns a checkpoint path.

    This is the intended entry point once the corpus + decoders are in
    place. Right now it verifies torch is present and then punts to the
    caller — wire up a standard PyG training loop here.
    """
    _ = _require("torch", "training loop")
    _ = _require("torch_geometric", "DataLoader / GAT layers")
    raise NotImplementedError(
        "GNN training loop is intentionally scaffolded — wire up a "
        f"torch_geometric.loader.DataLoader over the {len(examples)} "
        f"examples (batch_size={batch_size}, epochs={epochs}, lr={lr}, "
        f"device={device}) and cross-entropy against target_card_node_id "
        "masked to in-hand cards."
    )


__all__ = [
    "TrainingExample",
    "prepare_dataset",
    "iter_batches",
    "train_sft",
    "MissingGNNExtras",
]
