"""Small statistical helpers used by the benchmark harness.

Kept dependency-free (no numpy/scipy) so the package stays stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionCI:
    """A binomial-proportion estimate with a two-sided confidence interval."""

    successes: int
    trials: int
    lower: float
    upper: float
    level: float  # e.g. 0.95

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


def wilson_ci(successes: int, trials: int, level: float = 0.95) -> ProportionCI:
    """Wilson score confidence interval for a binomial proportion.

    Wilson beats the naive Wald interval on small samples and near 0/1
    (Agresti & Coull 1998). Two-sided. No external deps.
    """
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
    # Clamp to [0, 1] and snap floating-point noise at the edges.
    if lower < 1e-12 or successes == 0:
        lower = 0.0
    if upper > 1 - 1e-12 or successes == trials:
        upper = 1.0
    return ProportionCI(
        successes=successes,
        trials=trials,
        lower=lower,
        upper=upper,
        level=level,
    )


def _z_for(level: float) -> float:
    """Inverse-normal quantile for the given two-sided confidence level.

    Hard-coded for the levels a benchmark actually uses so we don't take a
    scipy dependency.
    """
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if level in table:
        return table[level]
    raise ValueError(
        f"Only these confidence levels are supported: {sorted(table)}. Got {level}."
    )
