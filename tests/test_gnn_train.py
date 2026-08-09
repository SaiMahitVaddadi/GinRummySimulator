"""Tests for the GNN SFT training loop.

Two paths:

* **With torch installed** (``gnn`` extra): run a tiny 30-sample training
  loop end-to-end. Verifies loss decreases and a checkpoint is written.
* **Without torch**: :func:`train_sft` must raise
  :class:`MissingGNNExtras` with an actionable install hint (never a
  cryptic ImportError).

The torch-free portions (trajectory codec, example construction) are
always exercised.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from gin_rummy.cards import Card
from gin_rummy.models.graph import build_hand_graph, card_node_id
from gin_rummy.models.gnn_policy import MissingGNNExtras
from gin_rummy.models.train_gnn import (
    TrainingExample,
    collect_trajectories,
    examples_from_trajectories,
    load_trajectories,
    save_trajectories,
    train_sft,
)
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import Observation, RandomPolicy


try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _sample_hand() -> tuple[Card, ...]:
    return (
        Card("2", "♠"), Card("3", "♠"), Card("4", "♠"), Card("5", "♠"),
        Card("9", "♥"), Card("9", "♦"), Card("9", "♣"),
        Card("K", "♥"), Card("7", "♦"), Card("A", "♣"),
    )


def _make_obs(hand: tuple[Card, ...], top: Card | None = None) -> Observation:
    return Observation(
        hand=hand,
        top_discard=top,
        discard_pile_size=1 if top else 0,
        deck_size=31,
        turn_number=1,
        knock_limit=10,
        player_id=0,
        num_players=2,
        other_hand_sizes=(10,),
    )


# ---------------------------------------------------------------------------
# Torch-free tests — always run
# ---------------------------------------------------------------------------

def test_trajectory_roundtrip_jsonl(tmp_path: Path) -> None:
    """collect -> save -> load returns the same structural rows."""
    rows = collect_trajectories(
        num_games=2,
        learner_factory=lambda rng: GreedyKnockPolicy(rng),
        opponent_factory=lambda rng: RandomPolicy(rng),
        seed=7,
        max_rows=40,
    )
    assert rows, "collector produced no rows"
    kinds = {r["kind"] for r in rows}
    assert kinds.issubset({"draw", "discard", "knock"})
    p = tmp_path / "traj.jsonl"
    save_trajectories(rows, p)
    loaded = load_trajectories(p)
    assert loaded == rows


def test_examples_from_trajectories_covers_all_heads() -> None:
    """Trajectory rows convert into TrainingExamples with the right target field set."""
    rows = collect_trajectories(
        num_games=4,
        learner_factory=lambda rng: GreedyKnockPolicy(rng),
        opponent_factory=lambda rng: RandomPolicy(rng),
        seed=1,
        max_rows=80,
    )
    examples = examples_from_trajectories(rows)
    assert examples
    has_discard = any(e.discard_target >= 0 for e in examples)
    has_draw = any(e.draw_target >= 0 for e in examples)
    # Knock rows are only emitted when a knock is legal — allow zero.
    assert has_discard, "no discard examples produced"
    assert has_draw, "no draw examples produced"
    # Every example has exactly one head labelled.
    for e in examples:
        labelled = [
            e.discard_target >= 0,
            e.draw_target >= 0,
            e.knock_target >= 0,
        ]
        assert sum(labelled) == 1


def test_train_sft_without_torch_raises_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch isn't available, train_sft raises MissingGNNExtras with the hint."""
    if _HAS_TORCH:
        pytest.skip("torch is installed; extras path not exercised here")
    with pytest.raises(MissingGNNExtras) as exc:
        train_sft([
            TrainingExample(
                graph=build_hand_graph(_make_obs(_sample_hand())),
                hand_card_ids=tuple(card_node_id(c) for c in _sample_hand()),
                discard_target=card_node_id(Card("K", "♥")),
            )
        ])
    assert "uv sync --extra gnn" in str(exc.value)


# ---------------------------------------------------------------------------
# Torch-required tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="requires the gnn extra")
def test_tiny_training_run_reduces_loss(tmp_path: Path) -> None:
    """30-sample end-to-end training run: completes, saves checkpoint, loss drops."""
    rows = collect_trajectories(
        num_games=6,
        learner_factory=lambda rng: GreedyKnockPolicy(rng),
        opponent_factory=lambda rng: RandomPolicy(rng),
        seed=42,
        max_rows=30,
    )
    examples = examples_from_trajectories(rows)
    assert len(examples) >= 20
    ckpt = tmp_path / "gnn_tiny.pt"
    report = train_sft(
        examples,
        epochs=3,
        batch_size=8,
        lr=1e-2,
        hidden_dim=16,
        heads=2,
        val_frac=0.2,
        checkpoint_path=ckpt,
        seed=0,
        verbose=False,
    )
    assert ckpt.exists()
    assert len(report.epoch_stats) == 3
    first = report.epoch_stats[0].train_loss
    last = report.epoch_stats[-1].train_loss
    # Loss should decrease (allowing tiny wiggle for a 30-sample run).
    assert last < first + 1e-4, (
        f"train loss did not decrease: first={first:.4f} last={last:.4f}"
    )
