"""Fine-tune scaffold — config dataclasses, collector, trainer error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from gin_rummy.finetune.collector import DataCollector, TrajectorySample
from gin_rummy.finetune.config import (
    FineTuneRun,
    OptimizerConfig,
    PEFTConfig,
    SchedulerConfig,
)
from gin_rummy.finetune.trainers import MissingFineTuneExtras, run_sft


def test_peft_config_qlora_flips_load_in_4bit():
    cfg = PEFTConfig(method="qlora")
    assert cfg.load_in_4bit is True


def test_finetunerun_serialises_round_trip():
    run = FineTuneRun(
        name="test", base_model="Qwen/Qwen2.5-0.5B", method="sft",
    )
    d = run.to_dict()
    assert d["name"] == "test"
    assert d["peft"]["method"] == "lora"


def test_collector_backfills_rewards_and_dumps(tmp_path: Path):
    c = DataCollector()
    c.record(TrajectorySample(
        game_id="g1", turn=1, player_id=0, kind="discard",
        prompt="hand: A♠ ...", action="7♥",
    ))
    c.record(TrajectorySample(
        game_id="g1", turn=1, player_id=1, kind="discard",
        prompt="hand: ...", action="K♣",
    ))
    c.close_game("g1", winner_id=0)
    assert c.samples[0].reward == 1.0
    assert c.samples[1].reward == -1.0
    out = tmp_path / "traj.jsonl"
    c.dump(out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2


def test_collector_sft_and_dpo_conversions():
    c = DataCollector()
    c.record(TrajectorySample(
        game_id="g1", turn=1, player_id=0, kind="discard",
        prompt="P", action="A",
    ))
    c.record(TrajectorySample(
        game_id="g1", turn=1, player_id=1, kind="discard",
        prompt="P", action="B",
    ))
    c.close_game("g1", winner_id=0)
    sft = c.to_sft_rows()
    dpo = c.to_dpo_pairs()
    kto = c.to_kto_rows()
    assert len(sft) == 1
    assert sft[0]["completion"] == "A"
    assert len(dpo) == 1
    assert dpo[0]["chosen"] == "A"
    assert dpo[0]["rejected"] == "B"
    assert len(kto) == 2


def test_trainer_raises_helpful_error_without_extras():
    """The scaffold only works when the finetune extra is installed. Without
    it, callers should get a clean, actionable error — not a stack trace."""
    run = FineTuneRun(name="t", base_model="Qwen/Qwen2.5-0.5B", method="sft")
    try:
        import trl  # noqa: F401
        # If trl is somehow installed in the dev env, this test is a no-op.
        pytest.skip("trl is installed; extras path not exercised")
    except ImportError:
        pass
    with pytest.raises(MissingFineTuneExtras) as exc:
        run_sft(run, [])
    assert "uv sync --extra finetune" in str(exc.value)


def test_optimizer_and_scheduler_defaults():
    o = OptimizerConfig()
    s = SchedulerConfig()
    assert o.name == "paged_adamw_8bit"
    assert s.name == "cosine"
