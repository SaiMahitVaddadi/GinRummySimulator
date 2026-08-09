r"""Training loop for the GNN policy scaffold.

This module now implements a *small-scale, CPU-feasible* SFT training
loop on top of ``CardMeldGAT``. The public surface is:

* :class:`TrainingExample` — one supervised triple ``(graph, discard
  target, draw target, knock target)`` derived from a trajectory row.
* :func:`prepare_dataset` — turn ``DataCollector.to_sft_rows``-style
  data (with an ``Observation`` decoder) into ``TrainingExample``\ s.
  Retained for the LLM-prompt input path.
* :func:`collect_trajectories` / :func:`load_trajectories` /
  :func:`save_trajectories` — a **self-contained** logger and codec
  that dumps trajectories to JSONL as plain (observation, action)
  dictionaries — no LLM prompt encoding involved, so the training loop
  never has to re-run games or parse LLM prompt strings.
* :func:`examples_from_trajectories` — convert those trajectory rows
  into ``TrainingExample``\ s.
* :func:`train_sft` — Adam + weighted CE/BCE, verbose per-epoch
  reporting, held-out validation accuracy, checkpoint save. Full run
  must finish in ~3 min on a laptop CPU.

The optional-dep guard is preserved: importing this module never fails,
and any torch-requiring entry point raises :class:`MissingGNNExtras`
with an actionable install hint when torch is absent.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from gin_rummy.cards import RANKS, SUITS, Card
from gin_rummy.models.graph import (
    CARD_FEATURE_DIM,
    MELD_FEATURE_DIM,
    NUM_CARD_NODES,
    HandGraph,
    build_hand_graph,
    card_node_id,
)
from gin_rummy.models.gnn_policy import MissingGNNExtras, _require, _torch_bits
from gin_rummy.policy import DrawSource, Observation


# ---------------------------------------------------------------------------
# Trajectory serialisation (torch-free)
# ---------------------------------------------------------------------------

def _card_to_str(card: Card) -> str:
    return f"{card.rank}{card.suit}"


def _str_to_card(s: str) -> Card:
    # Rank is 1-2 chars ("10" or "A"-"K"), suit is 1 char.
    suit = s[-1]
    rank = s[:-1]
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"invalid card string {s!r}")
    return Card(rank, suit)


def _serialise_obs(obs: Observation) -> dict[str, Any]:
    return {
        "hand": [_card_to_str(c) for c in obs.hand],
        "top_discard": _card_to_str(obs.top_discard) if obs.top_discard else None,
        "discard_pile_size": obs.discard_pile_size,
        "deck_size": obs.deck_size,
        "turn_number": obs.turn_number,
        "knock_limit": obs.knock_limit,
        "player_id": obs.player_id,
        "num_players": obs.num_players,
        "other_hand_sizes": list(obs.other_hand_sizes),
    }


def _deserialise_obs(d: dict[str, Any]) -> Observation:
    return Observation(
        hand=tuple(_str_to_card(s) for s in d["hand"]),
        top_discard=_str_to_card(d["top_discard"]) if d.get("top_discard") else None,
        discard_pile_size=int(d["discard_pile_size"]),
        deck_size=int(d["deck_size"]),
        turn_number=int(d["turn_number"]),
        knock_limit=int(d["knock_limit"]),
        player_id=int(d["player_id"]),
        num_players=int(d["num_players"]),
        other_hand_sizes=tuple(d["other_hand_sizes"]),
    )


# ---------------------------------------------------------------------------
# Trajectory collection (uses the real game engine, but writes plain dicts)
# ---------------------------------------------------------------------------

@dataclass
class _RecordingPolicy:
    """Wraps an inner Policy and logs (obs, action) tuples to an in-memory list."""

    inner: Any  # a Policy
    rows: list[dict[str, Any]]
    player_id: int

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        src = self.inner.choose_draw_source(obs)
        self.rows.append(
            {
                "kind": "draw",
                "player_id": self.player_id,
                "obs": _serialise_obs(obs),
                "action": {"source": src},
            }
        )
        return src

    def choose_discard(self, obs: Observation) -> Card:
        card = self.inner.choose_discard(obs)
        self.rows.append(
            {
                "kind": "discard",
                "player_id": self.player_id,
                "obs": _serialise_obs(obs),
                "action": {"card": _card_to_str(card)},
            }
        )
        return card

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        knock = self.inner.choose_to_knock(obs, deadwood_value)
        self.rows.append(
            {
                "kind": "knock",
                "player_id": self.player_id,
                "obs": _serialise_obs(obs),
                "action": {"knock": bool(knock)},
                "deadwood_value": int(deadwood_value),
            }
        )
        return knock


def collect_trajectories(
    *,
    num_games: int,
    learner_factory: Callable[[random.Random], Any],
    opponent_factory: Callable[[random.Random], Any],
    seed: int = 0,
    game_cls: type | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Play ``num_games`` matches and return trajectory rows for the *learner* seat.

    Only the learner's decisions are logged; the opponent seat plays its
    inner policy directly. This keeps the corpus focused and prevents the
    learner from learning to imitate the (weaker) opponent.

    Rows are plain dicts serialisable as JSONL — the training pipeline
    never has to re-run the game engine.
    """
    # Local import to avoid a hard dep at module load time — the game
    # engine only needs stdlib, but keeping this local mirrors the rest
    # of the file's discipline of "no import surprises".
    from gin_rummy.variants.classic import ClassicGin

    cls = game_cls or ClassicGin
    root_rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for gi in range(num_games):
        game_seed = root_rng.randrange(2**31)
        game_rng = random.Random(game_seed)
        learner = learner_factory(game_rng)
        opponent = opponent_factory(game_rng)
        rec = _RecordingPolicy(inner=learner, rows=rows, player_id=0)
        game = cls(num_players=2, seed=game_seed, policies=[rec, opponent])
        game.play()
        if max_rows is not None and len(rows) >= max_rows:
            break
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def save_trajectories(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Training example dataclass (torch-free — plain Python lists)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingExample:
    """A supervised triple for the multi-head GNN.

    Any single row targets *one* head — the others are marked absent
    (``-1``) so the loss code can skip them. This lets us pack draw,
    discard, and knock decisions into a single dataset without inflating
    the feature space.

    Attributes
    ----------
    graph : HandGraph
        Bipartite card <-> meld encoding of the observation.
    hand_card_ids : tuple[int, ...]
        The 0..51 card ids currently in hand. Discard head is masked to
        this set.
    discard_target : int
        Card node id (0..51) that was discarded, or ``-1`` if this is
        not a discard example.
    draw_target : int
        0 = deck, 1 = discard, ``-1`` if not a draw example.
    knock_target : int
        0 = pass, 1 = knock, ``-1`` if not a knock example.
    """

    graph: HandGraph
    hand_card_ids: tuple[int, ...]
    discard_target: int = -1
    draw_target: int = -1
    knock_target: int = -1


def examples_from_trajectories(
    rows: Sequence[dict[str, Any]],
) -> list[TrainingExample]:
    """Convert trajectory rows into ``TrainingExample`` instances."""
    out: list[TrainingExample] = []
    for r in rows:
        try:
            obs = _deserialise_obs(r["obs"])
        except (KeyError, ValueError):
            continue
        graph = build_hand_graph(obs)
        hand_ids: list[int] = []
        for c in obs.hand:
            try:
                hand_ids.append(card_node_id(c))
            except KeyError:
                continue
        kind = r.get("kind")
        action = r.get("action", {})
        if kind == "discard":
            card_str = action.get("card")
            if not card_str:
                continue
            try:
                target = card_node_id(_str_to_card(card_str))
            except (ValueError, KeyError):
                continue
            out.append(
                TrainingExample(
                    graph=graph,
                    hand_card_ids=tuple(hand_ids),
                    discard_target=target,
                )
            )
        elif kind == "draw":
            src = action.get("source")
            if src not in ("deck", "discard"):
                continue
            out.append(
                TrainingExample(
                    graph=graph,
                    hand_card_ids=tuple(hand_ids),
                    draw_target=0 if src == "deck" else 1,
                )
            )
        elif kind == "knock":
            knock = action.get("knock")
            if knock is None:
                continue
            out.append(
                TrainingExample(
                    graph=graph,
                    hand_card_ids=tuple(hand_ids),
                    knock_target=1 if knock else 0,
                )
            )
    return out


# Retained for backwards compatibility with the previous scaffold. Consumes
# ``DataCollector.to_sft_rows`` output — the LLM-prompt-based corpus.
def prepare_dataset(
    rows: Sequence[dict[str, str]],
    *,
    prompt_decoder: Callable[[str], Observation],
    completion_decoder: Callable[[str], Card],
) -> list[TrainingExample]:
    """Turn LLM-style ``(prompt, completion)`` rows into examples (discard only)."""
    out: list[TrainingExample] = []
    for row in rows:
        try:
            obs = prompt_decoder(row["prompt"])
            card = completion_decoder(row["completion"])
            target = card_node_id(card)
        except (KeyError, ValueError):
            continue
        graph = build_hand_graph(obs)
        hand_ids: list[int] = []
        for c in obs.hand:
            try:
                hand_ids.append(card_node_id(c))
            except KeyError:
                continue
        out.append(
            TrainingExample(
                graph=graph,
                hand_card_ids=tuple(hand_ids),
                discard_target=target,
            )
        )
    return out


def iter_batches(
    examples: Sequence[TrainingExample],
    batch_size: int,
) -> Iterable[Sequence[TrainingExample]]:
    """Yield fixed-size mini-batches. No shuffling — the caller can wrap."""
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


# ---------------------------------------------------------------------------
# Training loop (torch-only)
# ---------------------------------------------------------------------------

def _example_to_tensors(ex: TrainingExample, torch: Any, device: Any) -> dict[str, Any]:
    card_x = torch.tensor(ex.graph.card_features, dtype=torch.float32, device=device)
    if ex.graph.num_meld_nodes:
        meld_x = torch.tensor(
            ex.graph.meld_features, dtype=torch.float32, device=device
        )
    else:
        meld_x = torch.zeros((0, MELD_FEATURE_DIM), dtype=torch.float32, device=device)

    edges: list[list[int]] = [[], []]
    for card_id, meld_id in ex.graph.edge_index:
        m = meld_id + NUM_CARD_NODES
        edges[0].extend([card_id, m])
        edges[1].extend([m, card_id])
    if not edges[0]:
        edges = [[0], [0]]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    return {"card_x": card_x, "meld_x": meld_x, "edge_index": edge_index}


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_discard_acc: float
    val_draw_acc: float
    val_knock_acc: float


@dataclass
class TrainingReport:
    checkpoint_path: str
    epoch_stats: list[EpochStats] = field(default_factory=list)
    num_train: int = 0
    num_val: int = 0


def train_sft(
    examples: Sequence[TrainingExample],
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    val_frac: float = 0.1,
    hidden_dim: int = 64,
    heads: int = 4,
    weights: tuple[float, float, float] = (3.0, 1.0, 1.0),
    checkpoint_path: str | Path = "checkpoints/gnn_sft.pt",
    seed: int = 0,
    verbose: bool = True,
) -> TrainingReport:
    """SFT ``CardMeldGAT`` on ``examples``. Returns a :class:`TrainingReport`.

    Loss: weighted CE (discard, per-hand mask) + BCE-with-logits (draw,
    knock). ``weights`` is a ``(discard, draw, knock)`` triple, default
    3:1:1 as specified.

    Notes
    -----
    We forward per-example and accumulate loss across the mini-batch
    rather than building a single batched PyG graph. The model is small
    and the batch is small — this is faster to write and stays under the
    3-minute laptop-CPU budget for ~2000 examples × 5 epochs.
    """
    torch, nn, _GATConv, F = _torch_bits()
    from gin_rummy.models.gnn_policy import _build_gat_module

    if not examples:
        raise ValueError("no training examples")

    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_val = max(1, int(math.floor(len(order) * val_frac))) if len(order) > 1 else 0
    val_idx = set(order[:n_val])
    train_examples = [examples[i] for i in order if i not in val_idx]
    val_examples = [examples[i] for i in order if i in val_idx]

    dev = torch.device(device)
    torch.manual_seed(seed)
    model = _build_gat_module(hidden_dim, heads).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    w_discard, w_draw, w_knock = weights

    report = TrainingReport(
        checkpoint_path=str(checkpoint_path),
        num_train=len(train_examples),
        num_val=len(val_examples),
    )

    def _forward_loss(ex: TrainingExample) -> tuple[Any, dict[str, Any]]:
        """Compute per-example loss + per-head correctness flags."""
        tensors = _example_to_tensors(ex, torch, dev)
        card_scores, draw_logits, knock_logit = model(
            tensors["card_x"], tensors["meld_x"], tensors["edge_index"]
        )
        total = card_scores.new_zeros(())
        correct: dict[str, Any] = {}
        if ex.discard_target >= 0 and ex.hand_card_ids:
            hand_ids = torch.tensor(
                list(ex.hand_card_ids), dtype=torch.long, device=dev
            )
            masked = card_scores.index_select(0, hand_ids)
            # target index within the masked vector
            try:
                tgt_pos = ex.hand_card_ids.index(ex.discard_target)
            except ValueError:
                tgt_pos = -1
            if tgt_pos >= 0:
                # NOTE: higher score = "keep" (per GNNPolicy.choose_discard),
                # so the label for CE is the *negation* — the discarded card
                # is the argmin. Equivalently, train on -scores.
                logits = -masked.unsqueeze(0)
                target = torch.tensor([tgt_pos], dtype=torch.long, device=dev)
                total = total + w_discard * F.cross_entropy(logits, target)
                pred = int(torch.argmin(masked).item())
                correct["discard"] = 1.0 if pred == tgt_pos else 0.0
        if ex.draw_target >= 0:
            target = torch.tensor([ex.draw_target], dtype=torch.long, device=dev)
            total = total + w_draw * F.cross_entropy(draw_logits.unsqueeze(0), target)
            pred = int(torch.argmax(draw_logits).item())
            correct["draw"] = 1.0 if pred == ex.draw_target else 0.0
        if ex.knock_target >= 0:
            target = torch.tensor(
                [float(ex.knock_target)], dtype=torch.float32, device=dev
            )
            total = total + w_knock * F.binary_cross_entropy_with_logits(
                knock_logit.reshape(1), target
            )
            pred = 1 if knock_logit.item() > 0 else 0
            correct["knock"] = 1.0 if pred == ex.knock_target else 0.0
        return total, correct

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_examples)
        running_loss = 0.0
        n_seen = 0
        for batch in iter_batches(train_examples, batch_size):
            optim.zero_grad()
            batch_loss = None
            for ex in batch:
                l, _ = _forward_loss(ex)
                if batch_loss is None:
                    batch_loss = l
                else:
                    batch_loss = batch_loss + l
            if batch_loss is None:
                continue
            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            optim.step()
            running_loss += batch_loss.item() * len(batch)
            n_seen += len(batch)
        train_loss = running_loss / max(1, n_seen)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val_seen = 0
        head_correct = {"discard": (0, 0), "draw": (0, 0), "knock": (0, 0)}
        with torch.no_grad():
            for ex in val_examples:
                l, correct = _forward_loss(ex)
                val_loss_sum += l.item()
                n_val_seen += 1
                for head, ok in correct.items():
                    c, t = head_correct[head]
                    head_correct[head] = (c + int(ok), t + 1)
        val_loss = val_loss_sum / max(1, n_val_seen)

        def _acc(head: str) -> float:
            c, t = head_correct[head]
            return c / t if t else 0.0

        stats = EpochStats(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_discard_acc=_acc("discard"),
            val_draw_acc=_acc("draw"),
            val_knock_acc=_acc("knock"),
        )
        report.epoch_stats.append(stats)
        if verbose:
            print(
                f"[gnn-sft] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc[disc/draw/knock]="
                f"{stats.val_discard_acc:.3f}/"
                f"{stats.val_draw_acc:.3f}/"
                f"{stats.val_knock_acc:.3f}"
            )

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f"[gnn-sft] saved checkpoint to {ckpt_path}")
    return report


__all__ = [
    "TrainingExample",
    "TrainingReport",
    "EpochStats",
    "prepare_dataset",
    "iter_batches",
    "train_sft",
    "collect_trajectories",
    "save_trajectories",
    "load_trajectories",
    "examples_from_trajectories",
    "MissingGNNExtras",
]
