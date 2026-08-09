"""Tests for the ablation framework in ``gin_rummy.eval.ablation``."""

from __future__ import annotations

import random

import pytest

from gin_rummy.eval.ablation import (
    AblationResult,
    Factor,
    FactorAssignment,
    FactorialDesign,
    FractionalFactorialDesign,
    analysis,
    run_ablation,
)
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import Observation, Policy, RandomPolicy
from gin_rummy.variants.classic import ClassicGin


# ---------- design enumeration ----------


def _dummy_build(levels, rng):
    return RandomPolicy(rng)


def test_factor_rejects_duplicate_levels():
    with pytest.raises(ValueError):
        Factor("f", ["a", "a"])


def test_factor_rejects_single_level():
    with pytest.raises(ValueError):
        Factor("f", ["only"])


def test_full_factorial_2x2_produces_four_assignments():
    design = FactorialDesign(
        [Factor("draw", ["deck", "discard"]), Factor("knock", ["always", "never"])],
        build_fn=_dummy_build,
    )
    assignments = design.assignments()
    assert len(assignments) == 4
    keys = {a.key for a in assignments}
    assert keys == {
        (("draw", "deck"), ("knock", "always")),
        (("draw", "deck"), ("knock", "never")),
        (("draw", "discard"), ("knock", "always")),
        (("draw", "discard"), ("knock", "never")),
    }


def test_full_factorial_2x3x2_produces_twelve_assignments():
    design = FactorialDesign(
        [
            Factor("a", ["x", "y"]),
            Factor("b", ["1", "2", "3"]),
            Factor("c", ["p", "q"]),
        ],
        build_fn=_dummy_build,
    )
    assert len(design.assignments()) == 12


def test_fractional_factorial_rejects_non_two_level():
    with pytest.raises(ValueError):
        FractionalFactorialDesign(
            [Factor("f", ["a", "b", "c"])],
            generators=None,
            build_fn=_dummy_build,
        )


def test_fractional_ofat_has_k_plus_one_runs():
    factors = [
        Factor("a", ["a0", "a1"]),
        Factor("b", ["b0", "b1"]),
        Factor("c", ["c0", "c1"]),
        Factor("d", ["d0", "d1"]),
    ]
    design = FractionalFactorialDesign(factors, generators="OFAT", build_fn=_dummy_build)
    assignments = design.assignments()
    assert len(assignments) == len(factors) + 1
    # The first is the all-baseline run
    assert assignments[0].levels == {f.name: f.levels[0] for f in factors}
    # Each subsequent run flips exactly one factor
    baseline = assignments[0].levels
    for extra in assignments[1:]:
        differences = [k for k in extra.levels if extra.levels[k] != baseline[k]]
        assert len(differences) == 1


def test_fractional_half_fraction_gives_half_of_full():
    factors = [Factor(f"f{i}", ["0", "1"]) for i in range(4)]
    full = FactorialDesign(factors, build_fn=_dummy_build)
    half = FractionalFactorialDesign(factors, generators=None, build_fn=_dummy_build)
    assert len(half.assignments()) == len(full.assignments()) // 2
    assert "III" in half.resolution() or "IV" in half.resolution() or "V" in half.resolution()


def test_fractional_resolution_v_for_five_factors():
    factors = [Factor(f"f{i}", ["0", "1"]) for i in range(5)]
    half = FractionalFactorialDesign(factors, generators=None, build_fn=_dummy_build)
    assert len(half.assignments()) == 16  # 2^{5-1}
    assert half.resolution() == "V"


# ---------- runner end-to-end ----------


def _random_opponent(rng):
    return RandomPolicy(rng)


def _build_mock(levels, rng):
    """Two-factor mock with a known dominant factor.

    Factor "strength" ∈ {"weak", "strong"}: strong = greedy (dominates
    random), weak = random. "cosmetic" ∈ {"x", "y"} does nothing.
    """
    if levels["strength"] == "strong":
        return GreedyKnockPolicy(rng)
    return RandomPolicy(rng)


def test_run_ablation_end_to_end_small():
    factors = [
        Factor("strength", ["weak", "strong"]),
        Factor("cosmetic", ["x", "y"]),
    ]
    design = FactorialDesign(factors, build_fn=_build_mock)
    result = run_ablation(
        design,
        opponent_factory=_random_opponent,
        opponent_name="random",
        game_cls=ClassicGin,
        games_per_pair=20,
        seed=0,
        paired=True,
    )
    # 4 assignments -> 4 tournament runs
    assert len(result.runs) == 4
    for run in result.runs.values():
        assert run.wins + run.losses + run.draws == 20
    assert set(result.main_effects.keys()) == {"strength", "cosmetic"}
    # Both levels populated
    assert set(result.main_effects["strength"].keys()) == {"weak", "strong"}


def test_main_effects_recovers_dominant_factor():
    """When one factor deterministically produces a strong policy (greedy)
    versus a weak one (random), the main effect on ``strength`` should be
    close to the full head-to-head win margin of greedy vs random."""
    factors = [
        Factor("strength", ["weak", "strong"]),
        Factor("cosmetic", ["x", "y"]),
    ]
    design = FactorialDesign(factors, build_fn=_build_mock)
    result = run_ablation(
        design,
        opponent_factory=_random_opponent,
        opponent_name="random",
        game_cls=ClassicGin,
        games_per_pair=30,
        seed=0,
        paired=True,
    )
    strong_rate = result.main_effects["strength"]["strong"].marginal_win_rate
    weak_rate = result.main_effects["strength"]["weak"].marginal_win_rate
    # Greedy dominates random by a wide margin.
    assert strong_rate > 0.80
    assert weak_rate < 0.55  # random-vs-random is ~50%
    # Cosmetic factor should be a tiny effect (both levels similar).
    cosmetic_rates = [me.marginal_win_rate for me in result.main_effects["cosmetic"].values()]
    assert abs(cosmetic_rates[0] - cosmetic_rates[1]) < 0.20
    # Ranking should put "strength" first.
    assert result.ranking[0][0] == "strength"


# ---------- interaction detection ----------


class _ConstantOutcomePolicy:
    """Testing helper: makes the game engine irrelevant by wrapping
    a deterministic seat-outcome via a policy. Not used directly here
    because outcomes depend on the game engine; instead we use a
    synthetic-run harness below.
    """

    def choose_draw_source(self, obs: Observation):
        return "deck"

    def choose_discard(self, obs: Observation):
        return obs.hand[0]

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return True


def _synth_run(win_rate_at: dict[tuple[str, str], float], n: int = 50) -> AblationResult:
    """Build an ``AblationResult`` by direct injection so we can test
    the interaction detector on a known ground-truth surface (bypassing
    the game engine)."""
    from gin_rummy.eval.ablation import AssignmentRun
    from gin_rummy.eval.stats import wilson_ci
    from gin_rummy.eval.tournament import TournamentResult

    factors = [
        Factor("a", ["a0", "a1"]),
        Factor("b", ["b0", "b1"]),
    ]
    result = AblationResult(
        design_kind="full",
        factors=factors,
        opponent_name="opp",
    )
    for (a_lvl, b_lvl), rate in win_rate_at.items():
        wins = int(round(rate * n))
        losses = n - wins
        deltas = [1] * wins + [-1] * losses
        levels = {"a": a_lvl, "b": b_lvl}
        key = tuple(sorted(levels.items()))
        asn = FactorAssignment(levels=levels, build=lambda rng: _ConstantOutcomePolicy())
        result.runs[key] = AssignmentRun(
            assignment=asn,
            tournament=TournamentResult(
                variant="synthetic",
                policy_names=["asn", "opp"],
                games_per_pair=n,
                paired=True,
            ),
            wins=wins,
            losses=losses,
            draws=0,
            per_seed_deltas=deltas,
            win_rate_ci=wilson_ci(wins, wins + losses),
            win_rate_including_draws_ci=wilson_ci(wins, n),
        )
    # Recompute derived stats using the module's internals.
    from gin_rummy.eval.ablation import (
        _compute_interactions,
        _compute_main_effects,
        _compute_ranking,
    )
    _compute_main_effects(result)
    _compute_interactions(result, alpha=0.05)
    _compute_ranking(result)
    return result


def test_no_interaction_detected_when_effects_are_additive():
    # Additive surface: a1 adds +0.2, b1 adds +0.1, no interaction.
    surface = {
        ("a0", "b0"): 0.50,
        ("a1", "b0"): 0.70,
        ("a0", "b1"): 0.60,
        ("a1", "b1"): 0.80,
    }
    result = _synth_run(surface, n=200)
    eff = result.interaction_effects[("a", "b")]
    # Additive surface -> |(0.80-0.60) - (0.70-0.50)| = 0.
    assert eff.magnitude < 1e-9


def test_constructive_interaction_is_detected():
    # Only (a1, b1) combo lifts win rate; individual flips do little.
    surface = {
        ("a0", "b0"): 0.50,
        ("a1", "b0"): 0.52,
        ("a0", "b1"): 0.48,
        ("a1", "b1"): 0.95,
    }
    result = _synth_run(surface, n=200)
    eff = result.interaction_effects[("a", "b")]
    # (0.95-0.48) - (0.52-0.50) = 0.45
    assert eff.magnitude > 0.30
    # The paired sign test should also flag this: with n=200 and a
    # sign-consistent contrast, p should be tiny.
    assert eff.p_value < 0.01
    assert eff.rejected is True


# ---------- printers ----------


def test_main_effects_table_renders():
    factors = [Factor("strength", ["weak", "strong"]), Factor("cosmetic", ["x", "y"])]
    design = FactorialDesign(factors, build_fn=_build_mock)
    result = run_ablation(
        design,
        opponent_factory=_random_opponent,
        opponent_name="random",
        game_cls=ClassicGin,
        games_per_pair=10,
        seed=0,
        paired=True,
    )
    text = analysis.main_effects_table(result)
    assert "strength" in text
    assert "cosmetic" in text
    assert "win%" in text
    interact_text = analysis.interactions_table(result)
    assert "strength x cosmetic" in interact_text or "strength" in interact_text
