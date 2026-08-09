"""DPO trainer + self-play pipeline tests.

These exercise the two research contributions the ``finetune`` package
adds:

* Novel opening #5 — **DPO strictly from game outcomes on a card game**
  (fine-tuning survey flagged this as an open niche: PokerGPT uses RLHF,
  LLM4CardGame uses SFT only).
* SPIN-style self-play iteration (Chen et al. ICML 2024,
  arXiv:2401.01335).

None of the tests require ``trl`` / ``transformers`` / ``peft`` — the
scaffolds either raise :class:`MissingFineTuneExtras` cleanly, or the
collector-only path exercises game play without heavy deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gin_rummy.finetune import (
    FineTuneRun,
    SelfPlayConfig,
    outcome_dpo_pairs,
    run_selfplay,
)
from gin_rummy.finetune.trainers import (
    MissingFineTuneExtras,
    run_dpo,
    run_grpo,
    run_kto,
    run_sft,
)


# ---------------------------------------------------------------------------
# All four trainers must raise MissingFineTuneExtras cleanly without trl.
# ---------------------------------------------------------------------------

def _skip_if_trl_installed() -> None:
    try:
        import trl  # noqa: F401
    except ImportError:
        return
    pytest.skip("trl is installed; extras-missing path not exercised")


def _run() -> FineTuneRun:
    return FineTuneRun(name="t", base_model="Qwen/Qwen2.5-0.5B", method="dpo")


def test_run_sft_missing_extras():
    _skip_if_trl_installed()
    with pytest.raises(MissingFineTuneExtras) as exc:
        run_sft(_run(), [])
    assert "uv sync --extra finetune" in str(exc.value)


def test_run_dpo_missing_extras():
    _skip_if_trl_installed()
    with pytest.raises(MissingFineTuneExtras) as exc:
        run_dpo(_run(), [])
    assert "uv sync --extra finetune" in str(exc.value)


def test_run_kto_missing_extras():
    _skip_if_trl_installed()
    with pytest.raises(MissingFineTuneExtras) as exc:
        run_kto(_run(), [])
    assert "uv sync --extra finetune" in str(exc.value)


def test_run_grpo_missing_extras():
    _skip_if_trl_installed()
    with pytest.raises(MissingFineTuneExtras) as exc:
        run_grpo(_run(), ["prompt"], lambda comps: [0.0] * len(comps))
    assert "uv sync --extra finetune" in str(exc.value)


# ---------------------------------------------------------------------------
# Self-play pipeline runs end-to-end without any heavy deps.
# ---------------------------------------------------------------------------

def test_run_selfplay_populates_collector():
    cfg = SelfPlayConfig(num_games=2, opponent_policy_spec="greedy")
    collector = run_selfplay(cfg, seed=42)
    assert len(collector.samples) > 0, "collector should be non-empty"
    # Every sample should have a back-filled reward (game ended one way or another).
    assert all(s.reward is not None for s in collector.samples)
    # Both seats should show up (recording wraps learner + opponent).
    player_ids = {s.player_id for s in collector.samples}
    assert player_ids == {0, 1}
    # All three decision kinds should appear across a couple of games.
    kinds = {s.kind for s in collector.samples}
    assert "draw" in kinds
    assert "discard" in kinds


def test_run_selfplay_dumps_jsonl(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    cfg = SelfPlayConfig(
        num_games=1, opponent_policy_spec="greedy", output_path=out
    )
    collector = run_selfplay(cfg, seed=7)
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == len(collector.samples) > 0


def test_selfplay_dpo_pairs_are_wellformed():
    """The pairs mined from a self-play run must have the DPO schema:
    ``{prompt, chosen, rejected}`` — the exact shape ``trl.DPOTrainer`` wants."""
    cfg = SelfPlayConfig(num_games=4, opponent_policy_spec="greedy")
    collector = run_selfplay(cfg, seed=1)
    # ``outcome_dpo_pairs`` is the outcome-only pair miner — the actual
    # research contribution (fine-tuning survey opening #5). The generic
    # ``collector.to_dpo_pairs`` requires same-turn alignment that never
    # happens in serialised turn-taking games.
    pairs = outcome_dpo_pairs(collector)
    # With mixed outcomes across 4 games we should get *some* pairs.
    assert len(pairs) > 0, "expected at least one DPO pair across 4 games"
    for pair in pairs:
        assert set(pair.keys()) == {"prompt", "chosen", "rejected"}
        assert isinstance(pair["prompt"], str) and pair["prompt"]
        assert isinstance(pair["chosen"], str) and pair["chosen"]
        assert isinstance(pair["rejected"], str) and pair["rejected"]
        # Chosen != rejected — otherwise the preference signal is degenerate.
        assert pair["chosen"] != pair["rejected"]


def test_outcome_dpo_pairs_rejects_unknown_match_on():
    cfg = SelfPlayConfig(num_games=1, opponent_policy_spec="greedy")
    collector = run_selfplay(cfg, seed=0)
    with pytest.raises(ValueError):
        outcome_dpo_pairs(collector, match_on="turn")


def test_selfplay_random_opponent_still_works():
    cfg = SelfPlayConfig(num_games=2, opponent_policy_spec="random")
    collector = run_selfplay(cfg, seed=0)
    assert len(collector.samples) > 0


def test_selfplay_unknown_spec_raises():
    cfg = SelfPlayConfig(num_games=1, opponent_policy_spec="nonexistent")
    with pytest.raises(ValueError):
        run_selfplay(cfg, seed=0)
