"""Per-policy telemetry: latency, tokens, parse-fail, fallback rate.

Every decision made by a policy is timed by the ``TelemetryPolicy`` wrapper.
For LLM policies we also surface tokens-in / tokens-out (if the underlying
provider reports them) and the parse-fail / fallback flags already captured
by ``LLMPolicy.trace``.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field

from gin_rummy.cards import Card
from gin_rummy.policies.llm import LLMPolicy
from gin_rummy.policy import DrawSource, Observation, Policy


@dataclass
class DecisionRecord:
    kind: str  # "draw" | "discard" | "knock"
    elapsed_ms: float
    fell_back: bool = False
    parse_error: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class PolicyTelemetry:
    name: str
    decisions: list[DecisionRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.decisions)

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(d.elapsed_ms for d in self.decisions) if self.decisions else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.decisions:
            return 0.0
        s = sorted(d.elapsed_ms for d in self.decisions)
        return s[min(int(0.95 * len(s)), len(s) - 1)]

    @property
    def fallback_rate(self) -> float:
        return sum(d.fell_back for d in self.decisions) / self.n if self.n else 0.0

    @property
    def parse_fail_rate(self) -> float:
        return sum(d.parse_error for d in self.decisions) / self.n if self.n else 0.0

    @property
    def total_tokens_in(self) -> int:
        return sum(d.tokens_in or 0 for d in self.decisions)

    @property
    def total_tokens_out(self) -> int:
        return sum(d.tokens_out or 0 for d in self.decisions)

    def merge(self, other: "PolicyTelemetry") -> None:
        self.decisions.extend(other.decisions)


class TelemetryPolicy:
    """Wraps a Policy and times every decision. LLM-specific fields are
    populated when the wrapped policy is an ``LLMPolicy``."""

    def __init__(self, inner: Policy, telemetry: PolicyTelemetry) -> None:
        self._inner = inner
        self.telemetry = telemetry

    def _record(self, kind: str, elapsed_ms: float) -> None:
        rec = DecisionRecord(kind=kind, elapsed_ms=elapsed_ms)
        if isinstance(self._inner, LLMPolicy) and self._inner.trace:
            last = self._inner.trace[-1]
            if last.kind == kind:
                rec.fell_back = last.fell_back
                rec.parse_error = (last.reason == "parse_error") or (last.reason == "illegal_card")
        self.telemetry.decisions.append(rec)

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        t0 = time.perf_counter()
        out = self._inner.choose_draw_source(obs)
        self._record("draw", (time.perf_counter() - t0) * 1000)
        return out

    def choose_discard(self, obs: Observation) -> Card:
        t0 = time.perf_counter()
        out = self._inner.choose_discard(obs)
        self._record("discard", (time.perf_counter() - t0) * 1000)
        return out

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        t0 = time.perf_counter()
        out = self._inner.choose_to_knock(obs, deadwood_value)
        self._record("knock", (time.perf_counter() - t0) * 1000)
        return out
