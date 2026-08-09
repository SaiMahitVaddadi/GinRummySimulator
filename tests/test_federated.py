"""Tests for the federated telemetry aggregator + DP primitives.

Scope note: these are correctness checks on the plumbing (merge counts,
origin bookkeeping, DP mechanism sanity), not calibration checks on the
DP noise. Card-game telemetry has no real adversary; see
``PAPER.md`` §10 and the docstrings in ``gin_rummy.federated`` for the
threat-model caveat.
"""

from __future__ import annotations

import random
import statistics

import pytest

from gin_rummy.eval.telemetry import DecisionRecord, PolicyTelemetry
from gin_rummy.federated import (
    FederatedNode,
    aggregate_scalar,
    aggregate_telemetry,
    build_private_report,
    dp_mean,
    laplace_mechanism,
)
from gin_rummy.federated.aggregator import MergedPolicyTelemetry


def _make_telemetry(name: str, n: int, base_latency: float = 1.0) -> PolicyTelemetry:
    t = PolicyTelemetry(name=name)
    for i in range(n):
        t.decisions.append(
            DecisionRecord(kind="draw", elapsed_ms=base_latency + i)
        )
    return t


def test_aggregator_merges_telemetry() -> None:
    """3 nodes × 5 decisions each → 15 merged decisions with correct origins."""
    nodes = [
        FederatedNode(name=f"node-{i}", telemetry=_make_telemetry(f"pol-{i}", 5))
        for i in range(3)
    ]
    merged = aggregate_telemetry(nodes)

    assert isinstance(merged, MergedPolicyTelemetry)
    assert merged.n == 15
    assert len(merged.record_origins) == 15
    # Origin bookkeeping is 1:1 with decisions and preserves node order.
    assert merged.record_origins[:5] == ["node-0"] * 5
    assert merged.record_origins[5:10] == ["node-1"] * 5
    assert merged.record_origins[10:] == ["node-2"] * 5
    # Sanity: every merged record is a DecisionRecord.
    assert all(isinstance(r, DecisionRecord) for r in merged.decisions)


def test_aggregate_scalar_weighted_mean() -> None:
    """aggregate_scalar should return an n-weighted mean by default."""
    nodes = [
        FederatedNode(name="a", telemetry=_make_telemetry("pol-a", 10, base_latency=1.0)),
        FederatedNode(name="b", telemetry=_make_telemetry("pol-b", 2, base_latency=100.0)),
    ]
    # Node a mean latency = 1..10 → 5.5; node b = 100,101 → 100.5.
    # Weighted by n: (10*5.5 + 2*100.5) / 12 = (55 + 201) / 12 = 21.333...
    got = aggregate_scalar(nodes, lambda t: t.mean_latency_ms)
    assert got == pytest.approx(256 / 12)


def test_laplace_mechanism_zero_epsilon_raises() -> None:
    with pytest.raises(ValueError):
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.0)
    with pytest.raises(ValueError):
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=-0.5)


def test_laplace_mechanism_zero_sensitivity_is_noise_free() -> None:
    """A query with zero sensitivity carries no information → no noise."""
    out = laplace_mechanism(3.14, sensitivity=0.0, epsilon=0.1)
    assert out == 3.14


def test_dp_mean_noisy_but_bounded() -> None:
    """dp_mean(sensitivity=1, ε=1) stays within a loose bound of the truth.

    This is a *correctness of the noise wiring* test, not calibration.
    Values in [-1, 1] have per-element clipping sensitivity 1, so mean
    sensitivity is 2/n and noise scale is 2/n over 100 draws → tiny; a
    loose bound of 20 is well beyond anything the mechanism should
    produce here.
    """
    rng = random.Random(42)
    values = [rng.uniform(-1, 1) for _ in range(100)]
    true_mean = statistics.fmean(values)

    noisy = [dp_mean(values, sensitivity=1.0, epsilon=1.0, rng=rng) for _ in range(100)]
    for v in noisy:
        assert abs(v - true_mean) < 20  # loose sanity bound


def test_dp_mean_empty_returns_zero() -> None:
    assert dp_mean([], sensitivity=1.0, epsilon=1.0) == 0.0


def test_build_private_report_records_budget() -> None:
    """PrivateReport should carry node id, epsilon spent, sensitivities, and n."""
    node = FederatedNode(
        name="runner-7",
        telemetry=_make_telemetry("pol", 20),
        local_secret_seed=123,
    )
    report = build_private_report(
        node,
        epsilon=0.5,
        sensitivities={"mean_latency_ms": 5.0, "fallback_rate": 1.0},
    )

    assert report.node_id == "runner-7"
    assert report.epsilon_spent == 0.5
    assert set(report.metrics.keys()) == {"mean_latency_ms", "fallback_rate"}
    assert report.sensitivities == {"mean_latency_ms": 5.0, "fallback_rate": 1.0}
    assert report.n_records == 20
    # Values should be finite floats even after noising.
    for v in report.metrics.values():
        assert isinstance(v, float)
        assert v == v  # NaN check


def test_build_private_report_requires_matching_sensitivities() -> None:
    node = FederatedNode(name="n", telemetry=_make_telemetry("p", 3))
    with pytest.raises(ValueError):
        build_private_report(node, epsilon=0.1, sensitivities={})


def test_build_private_report_rejects_zero_epsilon() -> None:
    node = FederatedNode(name="n", telemetry=_make_telemetry("p", 3))
    with pytest.raises(ValueError):
        build_private_report(
            node, epsilon=0.0, sensitivities={"mean_latency_ms": 1.0}
        )


def test_build_private_report_reproducible_with_seed() -> None:
    """Same local_secret_seed → same DP-noised output (given same inputs)."""
    tele = _make_telemetry("p", 20)
    node1 = FederatedNode(name="n", telemetry=tele, local_secret_seed=99)
    node2 = FederatedNode(name="n", telemetry=tele, local_secret_seed=99)
    r1 = build_private_report(node1, epsilon=0.7, sensitivities={"mean_latency_ms": 5.0})
    r2 = build_private_report(node2, epsilon=0.7, sensitivities={"mean_latency_ms": 5.0})
    assert r1.metrics == r2.metrics
