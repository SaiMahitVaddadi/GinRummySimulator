"""Wire shape for a node's upstream report.

A ``PrivateReport`` is the sanitised, DP-noised bundle a
``FederatedNode`` publishes to the aggregator. It carries only:

* the node id (auditable origin),
* a dict of DP-noised scalar aggregates,
* the ε (privacy budget) consumed producing them,
* the sensitivities used per metric (for reproducibility).

Design intent — matching the "sanitised metrics only" note in the comms
survey — is that this dataclass be the ONLY thing that leaves a node in
a real deployment. Raw ``DecisionRecord``s stay local.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Mapping

from gin_rummy.eval.telemetry import PolicyTelemetry
from gin_rummy.federated.aggregator import FederatedNode
from gin_rummy.federated.dp import laplace_mechanism


# A metric extractor: given a telemetry, produce a scalar.
MetricFn = Callable[[PolicyTelemetry], float]

# Sensible built-in extractors: name -> (fn, default sensitivity if caller
# doesn't override). Sensitivity defaults are approximate upper bounds on
# how much a single decision could shift the aggregate.
DEFAULT_METRICS: Mapping[str, MetricFn] = {
    "mean_latency_ms": lambda t: t.mean_latency_ms,
    "fallback_rate": lambda t: t.fallback_rate,
    "parse_fail_rate": lambda t: t.parse_fail_rate,
}


@dataclass
class PrivateReport:
    """A node's outbound DP report.

    Attributes
    ----------
    node_id:
        Origin label (matches ``FederatedNode.name``).
    metrics:
        Metric name → DP-noised value.
    epsilon_spent:
        Sum of ε budget consumed across all metrics in this report.
        Under sequential composition (Dwork & Roth 2014, Thm. 3.16) the
        total privacy loss upper-bounds this quantity.
    sensitivities:
        Metric name → sensitivity used. Kept for auditability so the
        aggregator can confirm no over-noised / under-noised claims.
    n_records:
        How many raw records were aggregated (also released; assumed
        non-sensitive count).
    """

    node_id: str
    metrics: dict[str, float]
    epsilon_spent: float
    sensitivities: dict[str, float] = field(default_factory=dict)
    n_records: int = 0


def build_private_report(
    node: FederatedNode,
    epsilon: float,
    sensitivities: Mapping[str, float],
    *,
    metrics: Mapping[str, MetricFn] | None = None,
    rng: random.Random | None = None,
) -> PrivateReport:
    """Produce a ``PrivateReport`` by DP-noising each requested metric.

    ε is split uniformly across the requested metrics (basic sequential
    composition; caller is free to pre-allocate a per-metric ε by
    calling this function multiple times with disjoint metric sets).

    Only metrics that appear in *both* ``metrics`` (or ``DEFAULT_METRICS``)
    and ``sensitivities`` are released — this makes it a KeyError to
    forget a sensitivity, rather than silently leaking an un-noised value.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    metric_fns = dict(metrics) if metrics is not None else dict(DEFAULT_METRICS)
    keys = [k for k in metric_fns if k in sensitivities]
    if not keys:
        raise ValueError(
            "No overlap between requested metrics and provided sensitivities."
        )

    # If the node carries a local_secret_seed and no rng was passed, use
    # it so nodes can independently reproduce their own noise without
    # sharing RNG state upstream.
    if rng is None and node.local_secret_seed is not None:
        rng = random.Random(node.local_secret_seed)

    per_metric_eps = epsilon / len(keys)
    telemetries = node.iter_telemetries()
    # Aggregate across a node's telemetries by mean of the extracted scalar,
    # weighted by decision count.
    total_n = sum(t.n for t in telemetries)

    out: dict[str, float] = {}
    used_sens: dict[str, float] = {}
    for k in keys:
        fn = metric_fns[k]
        if total_n == 0:
            value = 0.0
        else:
            value = (
                sum(fn(t) * t.n for t in telemetries if t.n > 0) / total_n
            )
        s = float(sensitivities[k])
        out[k] = laplace_mechanism(value, s, per_metric_eps, rng=rng)
        used_sens[k] = s

    return PrivateReport(
        node_id=node.name,
        metrics=out,
        epsilon_spent=epsilon,
        sensitivities=used_sens,
        n_records=total_n,
    )


__all__ = [
    "DEFAULT_METRICS",
    "MetricFn",
    "PrivateReport",
    "build_private_report",
]
