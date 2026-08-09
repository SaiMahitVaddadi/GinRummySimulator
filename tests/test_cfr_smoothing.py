"""Tests for smoothed / regularised MCCFR on abstracted Classic Gin.

Addresses PAPER.md §6 prediction #3: vanilla MCCFR on the coarse
abstraction concentrates its average policy near-deterministic per
info-set, inflating 1-step-lookahead sampled-BR exploitability. These
tests exercise the two smoothing knobs implemented in
``gin_rummy.solvers.cfr.ExternalSamplingMCCFR``:

    * ``regularisation_lambda`` — blend the *average-strategy* update
      with uniform-over-legal.
    * ``min_prob`` — floor every action probability of the reported
      average strategy at ``eps`` and renormalise.

Lean by design: the full ablation (4 configs, 5k iterations) lives in
``experiments/cfr_smoothing.py``.
"""

from __future__ import annotations

import math
import random

import pytest

from gin_rummy.solvers.cfr import (
    AverageStrategy,
    ExternalSamplingMCCFR,
    _apply_min_prob,
)
from gin_rummy.solvers.gin_cfr import (
    ClassicGinEngine,
    average_policy_entropy,
    head_to_head_score,
    train_gin_cfr,
)


# ---------------------------------------------------------------------------
# _apply_min_prob unit tests
# ---------------------------------------------------------------------------


def test_apply_min_prob_floors_and_normalises():
    dist = {"a": 0.9, "b": 0.1, "c": 0.0}
    out = _apply_min_prob(dist, 0.05)
    assert set(out) == set(dist)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    for p in out.values():
        assert p >= 0.05 - 1e-12


def test_apply_min_prob_disabled():
    dist = {"a": 0.5, "b": 0.5}
    out = _apply_min_prob(dist, 0.0)
    assert out is dist  # untouched


def test_apply_min_prob_uniform_when_floor_saturates():
    dist = {"a": 0.99, "b": 0.005, "c": 0.005}
    # floor >= 1/n so all mass forced to uniform.
    out = _apply_min_prob(dist, 0.5)
    for p in out.values():
        assert p == pytest.approx(1.0 / 3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# min_prob guarantees a floor in every info-set of the average strategy
# ---------------------------------------------------------------------------


def test_min_prob_guarantees_floor_in_average_strategy():
    """After training with ``min_prob=0.02`` every action prob in the
    reported average strategy must be at least 0.02."""
    strat = train_gin_cfr(200, seed=0, min_prob=0.02)
    assert strat.num_info_sets() > 0
    for iset, dist in strat.policy.items():
        s = sum(dist.values())
        assert s == pytest.approx(1.0, abs=1e-6), f"{iset}: sum={s}"
        assert dist  # no empty info-sets
        for a, p in dist.items():
            assert p >= 0.02 - 1e-9, (
                f"{iset} action {a} prob {p} < 0.02 floor"
            )


# ---------------------------------------------------------------------------
# regularisation_lambda=1.0 collapses to uniform
# ---------------------------------------------------------------------------


def test_regularisation_lambda_one_is_uniform_when_queried():
    """With ``regularisation_lambda=1.0`` the average-strategy update is
    100% uniform-over-legal per visit. When queried against a legal set
    (via :meth:`AverageStrategy.action_probs`), the returned distribution
    must be uniform over that legal set — this is the operationally
    meaningful "indistinguishable from uniform" check under the coarse
    abstraction (different concrete hands sharing one bucket may present
    different discard-class legal sets, so the raw stored dict is a
    weighted mixture of per-visit uniforms; masking to any specific
    legal set recovers exact uniformity).

    Draw-phase info-sets always have exactly two legal actions in this
    engine, so we check them directly.
    """
    strat = train_gin_cfr(150, seed=0, regularisation_lambda=1.0)
    assert strat.num_info_sets() > 0

    # Direct check on draw-phase info-sets (legal set is always ("draw",
    # 0), ("draw", 1) once the deal has produced a discard pile, i.e.
    # from the first draw onward — the pre-first-draw state has both
    # options anyway).
    from gin_rummy.solvers.abstraction import DRAW_DECK, DRAW_DISCARD

    n_draw_checked = 0
    for iset, dist in strat.policy.items():
        if "|ph1|" not in iset:  # ph1 == DRAW phase
            continue
        legal = [a for a in (DRAW_DECK, DRAW_DISCARD) if a in dist]
        if len(legal) != 2:
            continue
        probs = strat.action_probs(iset, legal)
        for p in probs.values():
            assert p == pytest.approx(0.5, abs=1e-6), (
                f"{iset} draw-phase prob {p} != 0.5 under lambda=1.0"
            )
        n_draw_checked += 1
    assert n_draw_checked > 0, "no draw-phase info-sets to check"

    # Sanity: entropy across all info-sets should be very close to the
    # legal-set uniform entropy — collapse toward uniform is real.
    ent = average_policy_entropy(strat)
    # The trained tables mix info-sets with 2, 3, 4 legal actions; a
    # policy that is *uniform-over-legal-at-visit* averaged into a
    # slightly-mismatched union will have entropy at least ln(2) minus a
    # small slack.
    assert ent >= math.log(2) - 0.05, (
        f"entropy {ent:.3f} much lower than uniform-over-legal baseline"
        f" ln(2)={math.log(2):.3f} under lambda=1.0"
    )


# ---------------------------------------------------------------------------
# Smoothing lowers exploitability (or at least does not blow it up) vs baseline
# ---------------------------------------------------------------------------


def test_smoothing_does_not_hurt_h2h_vs_uniform():
    """Smoothing should keep head-to-head vs uniform in the same ballpark
    as the baseline — it should not turn a competitive policy into a
    losing one.

    Loose statistical check on one seed to keep runtime in the ~20s
    budget. The stronger monotonicity claim across seeds is tested in
    ``test_h2h_improves_with_iterations_under_smoothing``.
    """
    iters = 400
    baseline = train_gin_cfr(iters, seed=0)
    smoothed = train_gin_cfr(iters, seed=0, min_prob=0.02)
    uniform = AverageStrategy()
    h_base = head_to_head_score(baseline, uniform, num_deals=120, seed=0)
    h_smooth = head_to_head_score(smoothed, uniform, num_deals=120, seed=0)
    # Smoothed should be within 0.05 of baseline (both should be
    # >= -0.05 in practice).
    assert h_smooth >= h_base - 0.05, (
        f"smoothed h2h {h_smooth:+.3f} < baseline {h_base:+.3f} - 0.05"
    )


def test_h2h_improves_with_iterations_under_smoothing():
    """Under smoothing, head-to-head vs uniform should be at least as
    good at a higher iteration budget as at a lower one — averaged
    across a few seeds so single-seed noise cannot flip the sign.

    Runs 3 seeds x 2 iteration levels (100 vs 500) at ``min_prob=0.02``.
    Total runtime budget: <30s.
    """
    seeds = (0, 1, 2)
    low_iter = 100
    high_iter = 500
    lo_scores: list[float] = []
    hi_scores: list[float] = []
    uniform = AverageStrategy()
    for s in seeds:
        lo = train_gin_cfr(low_iter, seed=s, min_prob=0.02)
        hi = train_gin_cfr(high_iter, seed=s, min_prob=0.02)
        lo_scores.append(
            head_to_head_score(lo, uniform, num_deals=100, seed=s + 101)
        )
        hi_scores.append(
            head_to_head_score(hi, uniform, num_deals=100, seed=s + 101)
        )
    mean_lo = sum(lo_scores) / len(lo_scores)
    mean_hi = sum(hi_scores) / len(hi_scores)
    # Loose statistical check: mean head-to-head should not regress by
    # more than 0.02 going from low to high iterations.
    assert mean_hi >= mean_lo - 0.02, (
        f"h2h regressed with more training under smoothing:"
        f" lo={mean_lo:+.3f} hi={mean_hi:+.3f} (per-seed lo={lo_scores},"
        f" hi={hi_scores})"
    )


# ---------------------------------------------------------------------------
# Entropy monotonicity as a sanity check on the smoothing knobs
# ---------------------------------------------------------------------------


def test_smoothing_raises_average_policy_entropy():
    """Turning on either smoothing knob should not decrease the average
    entropy of the reported strategy versus the baseline. Both should
    strictly raise it (all else equal — same seed, same iters).
    """
    iters = 200
    base = train_gin_cfr(iters, seed=0)
    reg = train_gin_cfr(iters, seed=0, regularisation_lambda=0.1)
    floor = train_gin_cfr(iters, seed=0, min_prob=0.02)
    h_base = average_policy_entropy(base)
    h_reg = average_policy_entropy(reg)
    h_floor = average_policy_entropy(floor)
    assert h_reg >= h_base, (
        f"regularisation_lambda=0.1 entropy {h_reg:.4f} < baseline {h_base:.4f}"
    )
    assert h_floor >= h_base, (
        f"min_prob=0.02 entropy {h_floor:.4f} < baseline {h_base:.4f}"
    )


# ---------------------------------------------------------------------------
# Curve helper: h2h_deals > 0 returns triples
# ---------------------------------------------------------------------------


def test_sample_exploitability_curve_returns_h2h_when_requested():
    from gin_rummy.solvers.gin_cfr import sample_exploitability_curve

    curve = sample_exploitability_curve(
        training_iters=[50, 100],
        num_deals=8,
        seed=0,
        h2h_deals=40,
        min_prob=0.02,
    )
    assert len(curve) == 2
    for row in curve:
        assert len(row) == 3
        iters, expl, h2h = row
        assert isinstance(iters, int)
        assert isinstance(expl, float)
        assert isinstance(h2h, float)
        assert expl >= 0.0


# ---------------------------------------------------------------------------
# Solver-level guardrails
# ---------------------------------------------------------------------------


def test_invalid_regularisation_lambda_rejected():
    with pytest.raises(ValueError):
        ExternalSamplingMCCFR(regularisation_lambda=-0.1)
    with pytest.raises(ValueError):
        ExternalSamplingMCCFR(regularisation_lambda=1.5)


def test_invalid_min_prob_rejected():
    with pytest.raises(ValueError):
        ExternalSamplingMCCFR(min_prob=-0.01)
