"""Pure-Python numerics helpers for the HMM subpackage.

Everything downstream runs in log-space, so ``logsumexp`` is the workhorse
that keeps forward / backward / Baum-Welch numerically stable without any
NumPy dependency.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

NEG_INF = float("-inf")


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable ``log(sum(exp(v) for v in values))``.

    Handles the all -inf case (returns -inf) and the singleton case. The
    trick: factor out the max so the largest term contributes exp(0)=1
    and everything else contributes something in (0, 1].
    """
    vals = [v for v in values]
    if not vals:
        return NEG_INF
    m = max(vals)
    if m == NEG_INF:
        return NEG_INF
    total = 0.0
    for v in vals:
        if v == NEG_INF:
            continue
        total += math.exp(v - m)
    if total <= 0.0:
        return NEG_INF
    return m + math.log(total)


def normalise(vec: Sequence[float]) -> list[float]:
    """L1-normalise a non-negative vector; returns a uniform vector if the
    input sums to zero (so downstream code never divides by zero)."""
    total = sum(vec)
    n = len(vec)
    if n == 0:
        return []
    if total <= 0.0:
        return [1.0 / n] * n
    return [v / total for v in vec]


def log_normalise(vec: Sequence[float]) -> list[float]:
    """Normalise a probability vector already in log-space so ``exp`` sums
    to 1. Used inside Baum-Welch when re-estimating rows of ``trans`` or
    ``emit`` from log-space accumulators."""
    if not vec:
        return []
    lse = logsumexp(vec)
    if lse == NEG_INF:
        n = len(vec)
        return [-math.log(n)] * n
    return [v - lse for v in vec]


def safe_log(x: float) -> float:
    """``log(x)`` that maps 0 and tiny negatives to ``-inf`` instead of
    raising. Keeps the HMM robust to zero rows from user-supplied initial
    parameters."""
    if x <= 0.0:
        return NEG_INF
    return math.log(x)


__all__ = ["NEG_INF", "log_normalise", "logsumexp", "normalise", "safe_log"]
