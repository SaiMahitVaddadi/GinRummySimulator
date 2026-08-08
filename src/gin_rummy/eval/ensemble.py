"""Ensemble and Mixture-of-Experts policies.

Implements four composition patterns, each of which is itself a ``Policy``
and drops into the standard game engine:

* ``VotingEnsemble``           — plurality vote across expert policies.
* ``ConfidenceWeightedEnsemble`` — expert confidences drive a soft vote.
* ``PhaseGatedMoE``            — deterministic router by turn / stock size
  (the pattern behind M2CTS in chess, arXiv:2401.16852).
* ``HandStrengthGatedMoE``     — route by current deadwood, so an expensive
  expert (e.g. LLM) is only invoked when it can plausibly matter.

Also exposes ``seed_ensemble`` — a convenience for building N independent
copies of a policy with distinct RNGs, useful for bootstrap-DQN-style
uncertainty in policies that carry internal randomness.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from gin_rummy.cards import Card
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation, Policy


@dataclass
class VotingEnsemble:
    """Plurality vote across experts. Ties broken by first-listed expert.

    For continuous actions like ``choose_discard`` (many candidate cards), the
    voted card is the one with the most first-place votes, then the first
    expert's choice among tied cards.
    """

    experts: Sequence[Policy]
    name: str = "vote"

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        votes = Counter(e.choose_draw_source(obs) for e in self.experts)
        return votes.most_common(1)[0][0]

    def choose_discard(self, obs: Observation) -> Card:
        votes = Counter(e.choose_discard(obs) for e in self.experts)
        winners = votes.most_common()
        top_count = winners[0][1]
        tied = {c for c, n in winners if n == top_count}
        for e in self.experts:
            choice = e.choose_discard(obs)
            if choice in tied:
                return choice
        return winners[0][0]

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        votes = sum(1 for e in self.experts if e.choose_to_knock(obs, deadwood_value))
        return votes * 2 > len(self.experts)


@dataclass
class ConfidenceWeightedEnsemble:
    """Weighted vote where each expert declares a confidence in (0, 1].

    Confidences come from ``confidence_fn(expert, obs) -> float``. Defaults
    to uniform weights (equivalent to ``VotingEnsemble``).
    """

    experts: Sequence[Policy]
    confidence_fn: Callable[[Policy, Observation], float] = field(
        default=lambda _e, _o: 1.0
    )
    name: str = "confvote"

    def _weighted(self, choices: Sequence[tuple[object, float]]) -> object:
        totals: dict[object, float] = {}
        for choice, w in choices:
            totals[choice] = totals.get(choice, 0.0) + w
        return max(totals, key=totals.get)  # type: ignore[arg-type]

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        choices = [
            (e.choose_draw_source(obs), self.confidence_fn(e, obs)) for e in self.experts
        ]
        return self._weighted(choices)  # type: ignore[return-value]

    def choose_discard(self, obs: Observation) -> Card:
        choices = [
            (e.choose_discard(obs), self.confidence_fn(e, obs)) for e in self.experts
        ]
        return self._weighted(choices)  # type: ignore[return-value]

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        weighted_yes = 0.0
        total = 0.0
        for e in self.experts:
            w = self.confidence_fn(e, obs)
            total += w
            if e.choose_to_knock(obs, deadwood_value):
                weighted_yes += w
        return weighted_yes * 2 >= total


@dataclass
class PhaseGatedMoE:
    """Router by game phase: early = expand-melds, mid = optimise, late = knock.

    ``phase_thresholds`` is a ``(early_end, mid_end)`` turn-number pair. Below
    ``early_end`` the ``early`` expert runs; between ``early_end`` and
    ``mid_end`` the ``mid`` expert runs; above ``mid_end`` the ``late``
    expert runs.

    Precedent: Helfenstein et al. M2CTS (arXiv:2401.16852) uses phase-gated
    MoE for chess and reports +122 Elo over a single-net baseline.
    """

    early: Policy
    mid: Policy
    late: Policy
    phase_thresholds: tuple[int, int] = (8, 20)
    name: str = "phasemoe"

    def _route(self, obs: Observation) -> Policy:
        early_end, mid_end = self.phase_thresholds
        if obs.turn_number < early_end:
            return self.early
        if obs.turn_number < mid_end:
            return self.mid
        return self.late

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        return self._route(obs).choose_draw_source(obs)

    def choose_discard(self, obs: Observation) -> Card:
        return self._route(obs).choose_discard(obs)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._route(obs).choose_to_knock(obs, deadwood_value)


@dataclass
class HandStrengthGatedMoE:
    """Route by current deadwood: weak hand → cheap expert, strong hand → careful expert.

    Rationale: expensive experts (LLM, deep RL) can only matter when the
    decision is *decisive* — a hand with 40 deadwood is largely locked in.
    The threshold is the deadwood value below which we route to ``careful``.
    """

    cheap: Policy
    careful: Policy
    deadwood_threshold: int = 25
    name: str = "handmoe"

    def _route(self, obs: Observation) -> Policy:
        _, _, dv = optimal_decomposition(list(obs.hand))
        return self.careful if dv <= self.deadwood_threshold else self.cheap

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        return self._route(obs).choose_draw_source(obs)

    def choose_discard(self, obs: Observation) -> Card:
        return self._route(obs).choose_discard(obs)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._route(obs).choose_to_knock(obs, deadwood_value)


def seed_ensemble(
    factory: Callable[[random.Random], Policy], n: int, base_seed: int
) -> list[Policy]:
    """N independent copies of ``factory``, one per seed. Analogous to
    Bootstrapped DQN's K-heads pattern (Osband et al. NeurIPS 2016)."""
    return [factory(random.Random(base_seed + i)) for i in range(n)]
