"""Bradley–Terry ratings with bootstrap confidence intervals.

Fits the Bradley–Terry MLE via the classical **Minorization–Maximization
(MM) algorithm** (Hunter 2004) — no external deps. Bootstrap CIs come from
resampling matches with replacement and refitting. Ratings are converted
to Elo-like units so they read the way you'd expect from a chess
leaderboard.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RatingCI:
    name: str
    elo: float
    lower: float
    upper: float
    level: float


def bradley_terry_ratings(
    matches: Sequence[tuple[str, str, str]],
    *,
    tol: float = 1e-7,
    max_iter: int = 500,
    prior_strength: float = 1.0,
) -> dict[str, float]:
    """Fit Bradley–Terry strengths ``w_i`` from ordered win/loss pairs.

    ``matches`` is a sequence of ``(winner, loser, "any_string")`` triples;
    the third element is retained for interface symmetry with match logs
    (typically the seed) and is ignored here. Ties should be pre-filtered.

    Uses MM iteration. Adds a symmetric Dirichlet-style prior of
    ``prior_strength`` pseudo-wins on every ordered pair so the fit stays
    well-defined for undefeated / winless players.
    """
    if not matches:
        return {}
    names: list[str] = sorted({p for m in matches for p in (m[0], m[1])})
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)

    wins = [0.0] * N
    games = [[0.0] * N for _ in range(N)]
    for winner, loser, _ in matches:
        i, j = idx[winner], idx[loser]
        wins[i] += 1.0
        games[i][j] += 1.0
        games[j][i] += 1.0

    # Symmetric prior — one pseudo-win each direction between every pair.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            games[i][j] += prior_strength
            wins[i] += prior_strength * 0.5

    w = [1.0] * N
    for _ in range(max_iter):
        w_new = [0.0] * N
        for i in range(N):
            denom = 0.0
            for j in range(N):
                if i == j:
                    continue
                if games[i][j]:
                    denom += games[i][j] / (w[i] + w[j])
            w_new[i] = wins[i] / denom if denom > 0 else w[i]
        # Normalize to preserve identifiability (geometric mean = 1).
        log_mean = sum(math.log(x) for x in w_new) / N
        gm = math.exp(log_mean)
        w_new = [x / gm for x in w_new]
        # Convergence
        if max(abs(a - b) for a, b in zip(w, w_new)) < tol:
            w = w_new
            break
        w = w_new
    return dict(zip(names, w))


def _to_elo(strength: float, anchor: float = 1500.0) -> float:
    """400·log10 scale — same units as chess Elo, anchored so mean = anchor."""
    return anchor + 400.0 * math.log10(max(strength, 1e-12))


def rating_ci_table(
    matches: Sequence[tuple[str, str, str]],
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int | None = 0,
    prior_strength: float = 1.0,
    anchor: float = 1500.0,
) -> list[RatingCI]:
    """Bootstrap the match list; refit BT each resample; return Elo CIs."""
    rng = random.Random(seed)
    strengths = bradley_terry_ratings(matches, prior_strength=prior_strength)
    if not strengths:
        return []
    names = sorted(strengths)
    point_elo = {n: _to_elo(strengths[n], anchor=anchor) for n in names}

    boots: dict[str, list[float]] = defaultdict(list)
    n = len(matches)
    for _ in range(n_resamples):
        resample = [matches[rng.randrange(n)] for _ in range(n)]
        s = bradley_terry_ratings(resample, prior_strength=prior_strength)
        for name in names:
            if name in s:
                boots[name].append(_to_elo(s[name], anchor=anchor))
    alpha = (1 - level) / 2
    table: list[RatingCI] = []
    for name in names:
        samples = sorted(boots[name])
        if not samples:
            lo = up = point_elo[name]
        else:
            lo = samples[int(alpha * len(samples))]
            up = samples[min(int((1 - alpha) * len(samples)), len(samples) - 1)]
        table.append(RatingCI(name=name, elo=point_elo[name], lower=lo, upper=up, level=level))
    table.sort(key=lambda r: r.elo, reverse=True)
    return table
