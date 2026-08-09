"""Federated telemetry aggregation with optional differential privacy.

See ``PAPER.md`` §10 for motivation (a follow-up item; scaffolding, not
a production federated system). Public surface:

* ``FederatedNode`` / ``FederatedRun`` / ``MergedPolicyTelemetry`` — nodes
  and aggregation output shape.
* ``aggregate_telemetry`` / ``aggregate_scalar`` — the two aggregation
  entry points.
* ``laplace_mechanism`` / ``dp_mean`` — Laplace-mechanism DP primitives
  (Dwork & Roth, 2014).
* ``PrivateReport`` / ``build_private_report`` — the sanitised, DP-noised
  wire shape a node ships upstream.
"""

from gin_rummy.federated.aggregator import (
    FederatedNode,
    FederatedRun,
    MergedPolicyTelemetry,
    aggregate_scalar,
    aggregate_telemetry,
)
from gin_rummy.federated.dp import dp_mean, laplace_mechanism
from gin_rummy.federated.protocol import (
    DEFAULT_METRICS,
    PrivateReport,
    build_private_report,
)

__all__ = [
    "DEFAULT_METRICS",
    "FederatedNode",
    "FederatedRun",
    "MergedPolicyTelemetry",
    "PrivateReport",
    "aggregate_scalar",
    "aggregate_telemetry",
    "build_private_report",
    "dp_mean",
    "laplace_mechanism",
]
