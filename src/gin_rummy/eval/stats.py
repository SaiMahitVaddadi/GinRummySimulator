"""Statistical helpers for the eval framework.

Dependency-free (no numpy/scipy). Wilson CI for binary proportions,
non-parametric percentile bootstrap for continuous statistics,
Interquartile Mean (rliable), Holm–Bonferroni step-down, and a paired
sign test for head-to-head comparisons.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence


# ---------- Wilson score CI for proportions ----------

@dataclass(frozen=True)
class ProportionCI:
    successes: int
    trials: int
    lower: float
    upper: float
    level: float

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


def wilson_ci(successes: int, trials: int, level: float = 0.95) -> ProportionCI:
    """Wilson score CI for a binomial proportion (Agresti–Coull 1998)."""
    if trials <= 0:
        return ProportionCI(successes, trials, 0.0, 0.0, level)
    z = _z_for(level)
    n = trials
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    lower = centre - half
    upper = centre + half
    if lower < 1e-12 or successes == 0:
        lower = 0.0
    if upper > 1 - 1e-12 or successes == trials:
        upper = 1.0
    return ProportionCI(successes, trials, lower, upper, level)


def _z_for(level: float) -> float:
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if level in table:
        return table[level]
    raise ValueError(f"Supported levels: {sorted(table)}. Got {level}.")


# ---------- Non-parametric bootstrap for continuous statistics ----------

@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    level: float
    n_resamples: int


def bootstrap_ci(
    samples: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    n_resamples: int = 10_000,
    level: float = 0.95,
    rng: random.Random | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI. Useful for non-Gaussian metrics."""
    if not samples:
        return BootstrapCI(0.0, 0.0, 0.0, level, n_resamples)
    rng = rng if rng is not None else random.Random(0)
    n = len(samples)
    point = statistic(samples)
    if n == 1:
        return BootstrapCI(point, point, point, level, n_resamples)
    stats: list[float] = []
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(resample))
    stats.sort()
    alpha = (1 - level) / 2
    lower = stats[int(alpha * n_resamples)]
    upper = stats[min(int((1 - alpha) * n_resamples), n_resamples - 1)]
    return BootstrapCI(point, lower, upper, level, n_resamples)


# ---------- Interquartile mean (rliable) ----------

def iqm(samples: Sequence[float]) -> float:
    """Mean of the middle 50% — robust to outliers, keeps more signal than
    the median. Agarwal et al. NeurIPS 2021 (rliable)."""
    if not samples:
        return 0.0
    sorted_s = sorted(samples)
    n = len(sorted_s)
    lo = n // 4
    hi = n - lo
    trimmed = sorted_s[lo:hi] if hi > lo else sorted_s
    return statistics.fmean(trimmed)


# ---------- Multiple-comparisons correction ----------

def holm_bonferroni(pvalues: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down. Returns which nulls to reject at FWER=alpha.

    Uniformly more powerful than Bonferroni while still controlling FWER.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank, idx in enumerate(order, start=1):
        threshold = alpha / (m - rank + 1)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


# ---------- Paired sign test ----------

def paired_sign_test(deltas: Sequence[float]) -> float:
    """Two-sided p-value for H0: median(delta) == 0.

    ``deltas`` is per-pair outcome (e.g. +1 if A won the paired game, -1 if
    B won, 0 for a tie or paired-cancelled game). Uses the exact binomial.
    """
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = 2 * tail / (2 ** n)
    return min(1.0, p)
