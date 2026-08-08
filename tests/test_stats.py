"""Statistical helpers — Wilson CI properties."""

from __future__ import annotations

import pytest

from gin_rummy.stats import wilson_ci


def test_wilson_ci_zero_trials():
    ci = wilson_ci(0, 0)
    assert ci.rate == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_wilson_ci_rate_matches_ratio():
    ci = wilson_ci(30, 100)
    assert ci.rate == pytest.approx(0.30)
    # Wilson width for 30/100 at 95% is roughly ±9pp.
    assert 0.20 < ci.lower < 0.25
    assert 0.35 < ci.upper < 0.42


def test_wilson_ci_edges_are_bounded():
    """0-success and full-success intervals must not overflow [0, 1]."""
    ci_lo = wilson_ci(0, 100)
    ci_hi = wilson_ci(100, 100)
    assert ci_lo.lower == 0.0
    assert ci_lo.upper < 0.05
    assert ci_hi.upper == 1.0
    assert ci_hi.lower > 0.95


def test_wilson_ci_narrows_with_more_data():
    """More trials at the same rate → tighter interval."""
    small = wilson_ci(30, 100)
    big = wilson_ci(3000, 10000)
    assert (big.upper - big.lower) < (small.upper - small.lower)


def test_wilson_ci_rejects_unsupported_level():
    with pytest.raises(ValueError):
        wilson_ci(30, 100, level=0.97)
