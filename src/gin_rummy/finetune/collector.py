"""Trajectory collector — hooks the game loop to build training corpora.

Every match run through the collector produces a stream of
``TrajectorySample`` rows. From these rows you can build:

* **SFT dataset**: ``(prompt, chosen_action)`` for winning trajectories.
* **DPO/IPO/SimPO pairs**: ``(prompt, winner_action, loser_action)`` from
  paired matches on the same seed.
* **KTO labels**: ``(prompt, action, binary_reward)`` — natural fit for
  binary game outcomes (arXiv preprints on KTO argue this is the least
  data-shaping preference method).
* **GRPO groups**: sample G actions per state, roll out each, then use
  the group-standardised reward (DeepSeekMath, arXiv:2402.03300).

No heavy deps needed to run the collector; you get JSONL rows and can
feed them to trl / axolotl / unsloth from a separate process.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectorySample:
    """One decision, plus the eventual outcome the collector can back-fill."""

    game_id: str
    turn: int
    player_id: int
    kind: str  # "draw" | "discard" | "knock"
    prompt: str  # rendered observation text
    action: str  # rendered chosen action
    metadata: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None  # filled at end of game


@dataclass
class DataCollector:
    """Append-only buffer of trajectory samples. ``dump`` serialises to JSONL."""

    samples: list[TrajectorySample] = field(default_factory=list)

    def record(self, sample: TrajectorySample) -> None:
        self.samples.append(sample)

    def close_game(self, game_id: str, *, winner_id: int | None) -> None:
        """Back-fill the outcome reward for every sample in this game."""
        for s in self.samples:
            if s.game_id != game_id:
                continue
            if s.reward is not None:
                continue
            if winner_id is None:
                s.reward = 0.0
            else:
                s.reward = 1.0 if s.player_id == winner_id else -1.0

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for s in self.samples:
                fh.write(json.dumps(asdict(s)) + "\n")

    # ---------- format converters ----------

    def to_sft_rows(self, *, winning_only: bool = True) -> list[dict[str, str]]:
        """Return one ``{"prompt": ..., "completion": ...}`` row per sample."""
        out: list[dict[str, str]] = []
        for s in self.samples:
            if winning_only and (s.reward is None or s.reward <= 0):
                continue
            out.append({"prompt": s.prompt, "completion": s.action})
        return out

    def to_kto_rows(self) -> list[dict[str, Any]]:
        """Return one row per sample with a binary label (KTO expects unpaired)."""
        out: list[dict[str, Any]] = []
        for s in self.samples:
            if s.reward is None:
                continue
            out.append(
                {"prompt": s.prompt, "completion": s.action, "label": s.reward > 0}
            )
        return out

    def to_dpo_pairs(self) -> list[dict[str, str]]:
        """Pair same-turn actions across (game_id, turn, kind, player_id≠) tuples
        where one player won and the other lost. Cheap; imperfect; useful."""
        by_key: dict[tuple[str, int, str], list[TrajectorySample]] = {}
        for s in self.samples:
            by_key.setdefault((s.game_id, s.turn, s.kind), []).append(s)
        pairs: list[dict[str, str]] = []
        for group in by_key.values():
            if len(group) < 2:
                continue
            winners = [g for g in group if g.reward and g.reward > 0]
            losers = [g for g in group if g.reward and g.reward < 0]
            for w in winners:
                for lo in losers:
                    pairs.append(
                        {"prompt": w.prompt, "chosen": w.action, "rejected": lo.action}
                    )
        return pairs
