"""K-fold analogues for RL/agent evaluation.

Standard k-fold doesn't apply directly (no fixed dataset), but the RL
literature has settled on a handful of analogues — this module ships the
two that matter for a card-game testbed:

* ``opponent_kfold`` — partition an opponent pool into k folds; for each
  fold, evaluate the focal policy against the held-out opponents. Reports
  per-fold IQM and across-fold variance. (Cross-play with disjoint pools;
  see Nekoei et al. CoLLAs 2023, arXiv:2308.10284.)
* ``deal_kfold`` — partition pre-generated deal seeds into k folds so that
  every focal-vs-opponent matchup within a fold plays the *same* hands
  (paired-comparison variance reduction, the same idea as duplicate
  poker / ProcGen level splits).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from gin_rummy.eval.stats import bootstrap_ci, iqm
from gin_rummy.eval.tournament import PolicyEntry, Tournament, TournamentResult
from gin_rummy.game import GinRummyGame


@dataclass
class FoldReport:
    fold: int
    win_rate: float
    n: int
    iqm_turns: float
    result: TournamentResult


@dataclass
class KFoldReport:
    focal: str
    folds: list[FoldReport] = field(default_factory=list)

    @property
    def per_fold_win_rates(self) -> list[float]:
        return [f.win_rate for f in self.folds]

    @property
    def iqm_win_rate(self) -> float:
        return iqm(self.per_fold_win_rates)

    def summary(self) -> str:
        lines = [f"K-fold report for '{self.focal}' across {len(self.folds)} folds"]
        for f in self.folds:
            lines.append(
                f"  fold {f.fold}: {f.win_rate * 100:5.2f}% over {f.n} games "
                f"(IQM turns={f.iqm_turns:5.1f})"
            )
        rates = self.per_fold_win_rates
        boot = bootstrap_ci(rates, n_resamples=2000, rng=random.Random(0))
        lines.append(
            f"  aggregate: IQM={self.iqm_win_rate * 100:5.2f}%  "
            f"bootstrap 95% CI [{boot.lower * 100:5.2f}, {boot.upper * 100:5.2f}]"
        )
        return "\n".join(lines)


def opponent_kfold(
    *,
    game_cls: type[GinRummyGame],
    focal: PolicyEntry,
    opponents: Sequence[PolicyEntry],
    games_per_pair: int,
    k: int = 5,
    seed: int | None = 0,
    paired: bool = True,
    **game_kwargs,
) -> KFoldReport:
    """Partition opponents into k folds. For each fold, run a 2-agent
    tournament of the focal policy against the *held-out* opponents; report
    win rate per fold plus aggregated IQM + bootstrap CI.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(opponents) < k:
        raise ValueError(
            f"Need at least k={k} opponents to run k-fold; got {len(opponents)}"
        )

    # Shuffle deterministically so fold assignment is reproducible.
    rng = random.Random(seed)
    shuffled = list(opponents)
    rng.shuffle(shuffled)

    folds: list[list[PolicyEntry]] = [[] for _ in range(k)]
    for i, opp in enumerate(shuffled):
        folds[i % k].append(opp)

    report = KFoldReport(focal=focal.name)
    for f_idx, holdout in enumerate(folds):
        wins = 0
        losses = 0
        turns: list[float] = []
        sub_result: TournamentResult | None = None
        for opp in holdout:
            t = Tournament(
                game_cls=game_cls,
                entries=[focal, opp],
                games_per_pair=games_per_pair,
                seed=(None if seed is None else seed + f_idx * 100 + hash(opp.name) % 100),
                paired=paired,
                **game_kwargs,
            )
            r = t.run()
            sub_result = r  # keep the last one for inspection
            f_wins, o_wins, _ = r.wins_between(focal.name, opp.name)
            wins += f_wins
            losses += o_wins
            turns.extend(float(m.turns) for m in r.matches)
        total = wins + losses
        win_rate = wins / total if total else 0.0
        report.folds.append(
            FoldReport(
                fold=f_idx,
                win_rate=win_rate,
                n=total,
                iqm_turns=iqm(turns),
                result=sub_result,  # type: ignore[arg-type]
            )
        )
    return report


def deal_kfold(seeds: Sequence[int], k: int = 5) -> list[list[int]]:
    """Split a pool of pre-chosen deal seeds into k contiguous folds. Use the
    returned lists as ``seed`` arguments so every agent within a fold plays
    the identical set of hands."""
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(seeds) < k:
        raise ValueError("Need at least k seeds")
    fold_size = len(seeds) // k
    return [list(seeds[i * fold_size : (i + 1) * fold_size]) for i in range(k)]
