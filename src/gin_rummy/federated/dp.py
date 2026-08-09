"""Differential-privacy primitives for federated telemetry.

Implements the canonical Laplace mechanism of Dwork & Roth,
"The Algorithmic Foundations of Differential Privacy" (Foundations and
Trends in Theoretical Computer Science, 2014), Definition 3.3 /
Theorem 3.6: for a numeric query ``f`` with L1 sensitivity ``Δf``, the
mechanism ``M(x) = f(x) + Lap(Δf / ε)`` satisfies ε-differential
privacy.

Scope caveat: the threat model for card-game telemetry is thin — a
DecisionRecord holds latency and success flags, not player identity or
raw hands. This module exists as an infrastructure primitive so future
work can plug it into aggregates that actually need DP (e.g. cross-org
LLM cost telemetry), not because we believe latency histograms warrant
formal privacy protection today.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    rng: random.Random | None = None,
) -> float:
    """Return ``value + Lap(sensitivity / epsilon)``.

    Parameters
    ----------
    value:
        The true query answer to be privatised.
    sensitivity:
        L1 sensitivity ``Δf`` of the query — the maximum change in
        ``f(x)`` when a single record is added / removed / changed.
        Must be non-negative.
    epsilon:
        Privacy budget (ε > 0). Smaller ε → more noise, stronger
        privacy. ``epsilon == 0`` is undefined (infinite noise scale)
        and raises ``ValueError``.
    rng:
        Optional ``random.Random`` for reproducibility. Uses module
        default RNG when None.

    Notes
    -----
    See Dwork & Roth (Foundations and Trends in TCS, 2014), Def. 3.3
    and Thm. 3.6.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0 for the Laplace mechanism, got {epsilon}")
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
    if sensitivity == 0:
        # Degenerate query: no dependence on data → no noise needed.
        return value

    scale = sensitivity / epsilon
    r = rng if rng is not None else random
    # Inverse-CDF sample of Laplace(0, scale) via a uniform on (-0.5, 0.5).
    u = r.random() - 0.5
    noise = -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))
    return value + noise


def dp_mean(
    values: Sequence[float],
    sensitivity: float,
    epsilon: float,
    rng: random.Random | None = None,
) -> float:
    """DP mean via per-element clipping + Laplace noise on the average.

    Each ``v`` is clipped to ``[-sensitivity, sensitivity]`` so the L1
    sensitivity of the mean over ``n`` records is ``2 * sensitivity / n``
    (Dwork & Roth 2014, §3.3 discussion of bounded queries). Laplace
    noise with that scale / ε is then added.

    Returns ``0.0`` for an empty input (nothing to noise).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
    if not values:
        return 0.0

    n = len(values)
    clipped = [max(-sensitivity, min(sensitivity, v)) for v in values]
    true_mean = sum(clipped) / n
    mean_sensitivity = (2.0 * sensitivity) / n
    return laplace_mechanism(true_mean, mean_sensitivity, epsilon, rng=rng)


__all__ = ["dp_mean", "laplace_mechanism"]
