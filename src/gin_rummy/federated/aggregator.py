"""Federated telemetry aggregation.

Minimal, dependency-free primitives for combining per-agent
``PolicyTelemetry`` collected by independent runners. This is the
scaffolding motivated (weakly, per ``PAPER.md`` §10 and the comms
survey) by "federated + differential-privacy telemetry aggregator" as
follow-up work; the aim here is to have a merge/aggregate surface that
downstream work can drop real transport and real threat modelling into,
not to ship a production federated system.

Sanitised-metrics design: nodes ship ``PolicyTelemetry`` objects made of
``DecisionRecord``s (latency + fallback / parse flags + optional token
counts). No hands, cards, prompts, or player identities leave the node —
only per-decision timing and success flags. The one item added on the
central side is a per-record *node id* (an auditable origin label) so
downstream analyses can slice by runner without re-fetching raw traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from gin_rummy.eval.telemetry import DecisionRecord, PolicyTelemetry


@dataclass
class FederatedNode:
    """A single local runner contributing telemetry.

    Parameters
    ----------
    name:
        Human-readable node identifier used as the origin label attached
        to every merged decision. Should be unique across a run.
    telemetry:
        One ``PolicyTelemetry`` or a list of them (e.g. per-policy).
        All decisions from all telemetries under this node inherit the
        same node id in the merged output.
    local_secret_seed:
        Optional per-node seed. Reserved for DP noise draws — kept here
        so a node can produce reproducible-yet-independent noise without
        the aggregator having to distribute RNG state.
    """

    name: str
    telemetry: PolicyTelemetry | list[PolicyTelemetry]
    local_secret_seed: int | None = None

    def iter_telemetries(self) -> list[PolicyTelemetry]:
        if isinstance(self.telemetry, PolicyTelemetry):
            return [self.telemetry]
        return list(self.telemetry)


class MergedPolicyTelemetry(PolicyTelemetry):
    """A ``PolicyTelemetry`` that also tracks the origin node of each
    decision (parallel to ``decisions``). Subclassed rather than
    monkey-patched so ``isinstance`` checks against ``PolicyTelemetry``
    downstream keep working.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.record_origins: list[str] = []

    def add(self, record: DecisionRecord, origin: str) -> None:
        self.decisions.append(record)
        self.record_origins.append(origin)


@dataclass
class FederatedRun:
    """A group of nodes plus an optional custom combiner.

    The default combiner is :func:`aggregate_telemetry`; users can
    supply any callable mapping ``Sequence[FederatedNode]`` to whatever
    aggregate shape they want (e.g. per-policy dict, per-metric report).
    """

    nodes: Sequence[FederatedNode]
    combiner: Callable[[Sequence["FederatedNode"]], object] = field(
        default=lambda ns: aggregate_telemetry(ns)
    )

    def run(self) -> object:
        return self.combiner(self.nodes)


def aggregate_telemetry(
    nodes: Sequence[FederatedNode],
    *,
    name: str = "federated",
) -> MergedPolicyTelemetry:
    """Merge every DecisionRecord across nodes into one ``PolicyTelemetry``.

    Each merged record retains an auditable origin label (the source
    node's ``name``) in a parallel ``record_origins`` list. Ordering is
    node-major then telemetry-major then original decision order — this
    is deterministic so downstream percentile / mean computations are
    reproducible.
    """
    merged = MergedPolicyTelemetry(name=name)
    for node in nodes:
        for tel in node.iter_telemetries():
            for rec in tel.decisions:
                merged.add(rec, origin=node.name)
    return merged


def aggregate_scalar(
    nodes: Sequence[FederatedNode],
    extract_fn: Callable[[PolicyTelemetry], float],
    *,
    weight_fn: Callable[[PolicyTelemetry], float] | None = None,
) -> float:
    """Weighted mean of a scalar extracted per telemetry across all nodes.

    ``extract_fn`` receives each ``PolicyTelemetry`` (a node may hold
    several) and returns a scalar (e.g. mean latency, win rate). The
    default weight is ``PolicyTelemetry.n`` so under-sampled runners do
    not dominate; pass ``weight_fn=lambda t: 1.0`` for unweighted mean.
    Returns 0.0 if the total weight is zero (no data).
    """
    if weight_fn is None:
        weight_fn = lambda t: float(t.n)  # noqa: E731 — trivial default

    total_w = 0.0
    total_v = 0.0
    for node in nodes:
        for tel in node.iter_telemetries():
            w = weight_fn(tel)
            if w <= 0:
                continue
            total_v += w * extract_fn(tel)
            total_w += w
    if total_w == 0.0:
        return 0.0
    return total_v / total_w


__all__ = [
    "FederatedNode",
    "FederatedRun",
    "MergedPolicyTelemetry",
    "aggregate_scalar",
    "aggregate_telemetry",
]
