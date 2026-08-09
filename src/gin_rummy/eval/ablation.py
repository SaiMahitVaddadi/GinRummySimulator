"""Ablation framework: full and fractional factorial designs over policy factors.

The matched-arm pattern in ``experiments/matched_arm.py`` treats each
policy as a black box and runs a round-robin. This module upgrades that
into a proper **ablation study**: every architectural component
(heuristic sub-modules, MoE, ensemble, tools, signalling, fine-tuning)
becomes a *factor* with named *levels*, and we enumerate combinations of
those factors -- exactly or via a fractional design (Box/Hunter DoE).

The output is a set of estimated **main effects** (marginal win rate per
level, holding all other factors uniform) and **pairwise interactions**
(difference-in-differences with paired-sign p-values, Holm-corrected).

Every assignment plays a 2-way paired tournament against a single fixed
``opponent_factory``. This is deliberately narrower than a full N-way
round-robin: the win rate vs. the opponent is a *contrast* estimator,
not an absolute skill rating. If you want cross-play across assignments,
just feed the same ``FactorialDesign`` builder into ``Tournament``
directly.

References
----------
* Box, Hunter & Hunter, *Statistics for Experimenters* (2005), Ch. 5-8.
  Two-level fractional factorials, resolution III/IV/V.
* Agarwal et al., NeurIPS 2021 (rliable) -- motivates the paired /
  bootstrap-CI presentation.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from gin_rummy.eval.stats import (
    ProportionCI,
    holm_bonferroni,
    paired_sign_test,
    wilson_ci,
)
from gin_rummy.eval.tournament import (
    PolicyEntry,
    Tournament,
    TournamentResult,
)
from gin_rummy.game import GinRummyGame
from gin_rummy.policy import Policy


# ---------- factors ----------


@dataclass(frozen=True)
class Factor:
    """One experimental factor with a fixed set of named levels.

    Example: ``Factor("discard", ["random", "highest_deadwood", "safe_from_opp"])``.
    """

    name: str
    levels: tuple[str, ...]

    def __init__(self, name: str, levels: Sequence[str]) -> None:
        if len(levels) < 2:
            raise ValueError(f"Factor {name!r} needs at least 2 levels; got {list(levels)}")
        if len(set(levels)) != len(levels):
            raise ValueError(f"Factor {name!r} has duplicate levels: {list(levels)}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "levels", tuple(levels))


AssignmentDict = dict[str, str]
BuildFn = Callable[[AssignmentDict, random.Random], Policy]


@dataclass(frozen=True)
class FactorAssignment:
    """One concrete combination of levels across factors, plus a build callable.

    ``levels`` maps factor.name -> chosen level. ``build(rng)`` yields the
    ``Policy`` for that assignment (the design's ``build_fn`` is baked in).
    """

    levels: AssignmentDict
    build: Callable[[random.Random], Policy]

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.levels.items()))

    def label(self, sep: str = "|") -> str:
        return sep.join(f"{k}={v}" for k, v in sorted(self.levels.items()))


# ---------- designs ----------


class FactorialDesign:
    """Full factorial: enumerate every combination of factor levels.

    For k factors with L_i levels each, that is prod(L_i) assignments.
    Use ``FractionalFactorialDesign`` when this is too many runs.
    """

    def __init__(self, factors: Sequence[Factor], build_fn: BuildFn) -> None:
        if not factors:
            raise ValueError("FactorialDesign needs at least one factor")
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate factor names: {names}")
        self.factors: tuple[Factor, ...] = tuple(factors)
        self.build_fn = build_fn

    def assignments(self) -> list[FactorAssignment]:
        out: list[FactorAssignment] = []
        level_lists = [f.levels for f in self.factors]
        for combo in itertools.product(*level_lists):
            levels = {f.name: lvl for f, lvl in zip(self.factors, combo)}
            build_fn = self.build_fn
            frozen_levels = dict(levels)
            out.append(
                FactorAssignment(
                    levels=frozen_levels,
                    build=lambda rng, _lv=frozen_levels: build_fn(_lv, rng),
                )
            )
        return out

    def __len__(self) -> int:
        n = 1
        for f in self.factors:
            n *= len(f.levels)
        return n


class FractionalFactorialDesign:
    """Two-level fractional factorial design (Box/Hunter DoE, Ch. 6-8).

    Requires every factor to have exactly two levels. Supports:

    * ``"OFAT"`` -- one-factor-at-a-time (resolution III-like): a single
      baseline run plus one run per factor with only that factor flipped.
      Run count: k + 1.
    * ``"half"`` -- classical half fraction 2^{k-1} of resolution V (for
      k >= 5) constructed from the defining relation
      ``I = X1*X2*...*Xk``. For 3 <= k <= 4 the half-fraction reduces to
      resolution III and IV respectively; we still emit 2^{k-1} runs and
      warn callers via ``resolution``.

    Callers who want a specific custom fraction can pass ``generators`` as
    a list of factor-index tuples that define the confounding relations
    (e.g. ``[(0, 1, 2)]`` for I = ABC in a 2^{3-1} design). If ``None``,
    the default rule above is used.
    """

    def __init__(
        self,
        factors: Sequence[Factor],
        generators: str | Sequence[Sequence[int]] | None,
        build_fn: BuildFn,
    ) -> None:
        if not factors:
            raise ValueError("FractionalFactorialDesign needs at least one factor")
        for f in factors:
            if len(f.levels) != 2:
                raise ValueError(
                    f"FractionalFactorialDesign requires 2-level factors; "
                    f"factor {f.name!r} has {len(f.levels)} levels"
                )
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate factor names: {names}")
        self.factors: tuple[Factor, ...] = tuple(factors)
        self.build_fn = build_fn
        self.generators = generators

    def resolution(self) -> str:
        """Nominal resolution of the design (informational)."""
        if self.generators == "OFAT":
            return "III (main effects only; no interactions estimable)"
        k = len(self.factors)
        if k <= 2:
            return "full (fractional degenerates for k<=2)"
        if k == 3:
            return "III"
        if k == 4:
            return "IV"
        return "V"

    def assignments(self) -> list[FactorAssignment]:
        k = len(self.factors)
        gens = self.generators
        if gens == "OFAT":
            return self._ofat_assignments()
        # Half-fraction: 2^{k-1} runs picked by a defining word.
        # Default word: I = product of all factors, i.e. include a run iff
        # the parity of "level 2 selected" across all factors is even.
        # Custom generators: list of index-tuples; a run is kept iff the
        # product of its indicator values on each generator is +1.
        if gens is None:
            generator_sets = [tuple(range(k))]
        else:
            generator_sets = [tuple(g) for g in gens]  # type: ignore[assignment]
        out: list[FactorAssignment] = []
        for combo in itertools.product(*(f.levels for f in self.factors)):
            # +1 for level index 0, -1 for level index 1.
            signs = [1 if lvl == self.factors[i].levels[0] else -1 for i, lvl in enumerate(combo)]
            keep = True
            for gset in generator_sets:
                prod = 1
                for idx in gset:
                    prod *= signs[idx]
                if prod != 1:
                    keep = False
                    break
            if not keep:
                continue
            levels = {f.name: lvl for f, lvl in zip(self.factors, combo)}
            build_fn = self.build_fn
            frozen_levels = dict(levels)
            out.append(
                FactorAssignment(
                    levels=frozen_levels,
                    build=lambda rng, _lv=frozen_levels: build_fn(_lv, rng),
                )
            )
        if not out:
            # Fall back to full factorial if the defining relation is empty
            # (defensive; shouldn't happen for valid generators).
            return FactorialDesign(self.factors, self.build_fn).assignments()
        return out

    def _ofat_assignments(self) -> list[FactorAssignment]:
        baseline_levels: AssignmentDict = {f.name: f.levels[0] for f in self.factors}
        out: list[FactorAssignment] = []
        build_fn = self.build_fn
        out.append(
            FactorAssignment(
                levels=dict(baseline_levels),
                build=lambda rng, _lv=dict(baseline_levels): build_fn(_lv, rng),
            )
        )
        for f in self.factors:
            perturbed = dict(baseline_levels)
            perturbed[f.name] = f.levels[1]
            out.append(
                FactorAssignment(
                    levels=perturbed,
                    build=lambda rng, _lv=dict(perturbed): build_fn(_lv, rng),
                )
            )
        return out

    def __len__(self) -> int:
        return len(self.assignments())


Design = FactorialDesign | FractionalFactorialDesign


# ---------- runner ----------


PolicyFactory = Callable[[random.Random], Policy]


@dataclass
class AssignmentRun:
    """One assignment's outcome vs. the opponent.

    Two win-rate flavours:

    * ``win_rate_ci`` -- Wilson CI over ``wins / (wins + losses)``.
      Standard head-to-head rate that ignores draws. Sensitive when the
      opponent is weak enough that draws dominate.
    * ``win_rate_including_draws_ci`` -- Wilson CI over
      ``wins / total_games``, counting draws as non-wins. When the two
      policies dead-end in undecidable games (e.g. random-vs-heuristic
      often runs out the deck), this is the metric that actually
      differentiates architectures.
    """

    assignment: FactorAssignment
    tournament: TournamentResult
    wins: int
    losses: int
    draws: int
    per_seed_deltas: list[int]  # +1 assignment win, -1 opp win, 0 draw (paired)
    win_rate_ci: ProportionCI
    win_rate_including_draws_ci: ProportionCI


@dataclass
class MainEffect:
    factor: str
    level: str
    n_assignments: int
    marginal_win_rate: float
    ci_lower: float
    ci_upper: float
    n_games: int
    # Draws-as-losses view; useful when the opponent forces many
    # inconclusive games and the head-to-head rate saturates at 100%.
    marginal_win_rate_incl_draws: float = 0.0
    ci_lower_incl_draws: float = 0.0
    ci_upper_incl_draws: float = 0.0


@dataclass
class InteractionEffect:
    """Pairwise interaction: how the effect of factor A shifts across levels of B.

    For each level combination (a_lvl, b_lvl) we compute the marginal win
    rate across all assignments matching both. The interaction magnitude
    is the max spread of ``main-effect(A|B=b)`` minus ``main-effect(A)``,
    following the classical difference-in-differences framing.

    ``p_value`` is a two-sided paired sign test on per-seed deltas of
    (a1 vs a0) at level b1 minus (a1 vs a0) at level b0, i.e. the
    interaction contrast, restricted to factors that share a seed
    schedule (all runs here do, since ``run_ablation`` uses the same base
    seed for every assignment). The p-values across all pairs are
    Holm-Bonferroni corrected; ``rejected`` records whether the null of
    no interaction is rejected at FWER=alpha.
    """

    factor_a: str
    factor_b: str
    magnitude: float
    p_value: float
    p_value_holm: float
    rejected: bool
    per_cell_rate: dict[tuple[str, str], float]


@dataclass
class AblationResult:
    design_kind: str  # "full" or "fractional-<resolution>"
    factors: list[Factor]
    opponent_name: str
    runs: dict[tuple[tuple[str, str], ...], AssignmentRun] = field(default_factory=dict)
    main_effects: dict[str, dict[str, MainEffect]] = field(default_factory=dict)
    interaction_effects: dict[tuple[str, str], InteractionEffect] = field(default_factory=dict)
    ranking: list[tuple[str, float]] = field(default_factory=list)  # (factor, |effect|)

    def factor_by_name(self, name: str) -> Factor:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)


def run_ablation(
    design: Design,
    *,
    opponent_factory: PolicyFactory,
    opponent_name: str = "opponent",
    game_cls: type[GinRummyGame],
    games_per_pair: int = 50,
    seed: int = 0,
    paired: bool = True,
    interaction_alpha: float = 0.05,
    **game_kwargs,
) -> AblationResult:
    """Run every assignment in ``design`` vs. ``opponent_factory``.

    Each assignment plays a 2-policy paired tournament using the shared
    ``games_per_pair`` and ``seed``. Because the seed schedule is
    identical across assignments, per-seed outcomes are paired -- which
    unlocks the paired-sign-test-based interaction contrasts below.

    Parameters
    ----------
    design : FactorialDesign | FractionalFactorialDesign
    opponent_factory : (rng) -> Policy
        Held fixed across every assignment. Typically ``RandomPolicy`` or
        the strongest heuristic you want to contrast against.
    game_cls : subclass of GinRummyGame
    games_per_pair : int
        Games each assignment plays vs. the opponent. Must be even if
        ``paired``.
    seed : int
        Base seed. Reproducible.
    paired : bool
        If True, uses the paired-seat protocol from ``Tournament``.
    interaction_alpha : float
        FWER threshold for Holm-Bonferroni across interactions.
    """
    if paired and games_per_pair % 2:
        raise ValueError("Paired mode requires an even games_per_pair")

    if isinstance(design, FractionalFactorialDesign):
        design_kind = f"fractional-{design.resolution()}"
    else:
        design_kind = "full"

    assignments = design.assignments()
    if not assignments:
        raise RuntimeError("Design produced zero assignments")

    result = AblationResult(
        design_kind=design_kind,
        factors=list(design.factors),
        opponent_name=opponent_name,
    )

    for asn in assignments:
        assignment_name = asn.label()
        entries = [
            PolicyEntry(name=assignment_name, build=asn.build),
            PolicyEntry(name=opponent_name, build=opponent_factory),
        ]
        # Guard: if user picked an opponent name that collides with the
        # generated assignment label, disambiguate.
        if entries[0].name == entries[1].name:
            entries[1] = PolicyEntry(
                name=f"{opponent_name}__opp",
                build=opponent_factory,
            )
        tourney = Tournament(
            game_cls=game_cls,
            entries=entries,
            games_per_pair=games_per_pair,
            seed=seed,
            paired=paired,
            **game_kwargs,
        )
        t_result = tourney.run()
        wins, losses, draws = t_result.wins_between(entries[0].name, entries[1].name)
        trials = wins + losses
        ci = wilson_ci(wins, trials)
        total = wins + losses + draws
        ci_incl = wilson_ci(wins, total)
        deltas = _per_seed_deltas(t_result, entries[0].name, entries[1].name)
        result.runs[asn.key] = AssignmentRun(
            assignment=asn,
            tournament=t_result,
            wins=wins,
            losses=losses,
            draws=draws,
            per_seed_deltas=deltas,
            win_rate_ci=ci,
            win_rate_including_draws_ci=ci_incl,
        )

    _compute_main_effects(result)
    _compute_interactions(result, alpha=interaction_alpha)
    _compute_ranking(result)
    return result


def _per_seed_deltas(t: TournamentResult, a: str, b: str) -> list[int]:
    """Return +1 / -1 / 0 for each match where ``a`` played vs ``b``, ordered by seed."""
    deltas: list[tuple[int | None, int]] = []
    for m in t.matches:
        if a not in m.seat_names or b not in m.seat_names:
            continue
        if m.winner_seat is None:
            d = 0
        else:
            winner = m.seat_names[m.winner_seat]
            d = 1 if winner == a else -1
        deltas.append((m.seed, d))
    deltas.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
    return [d for _, d in deltas]


def _compute_main_effects(result: AblationResult) -> None:
    for factor in result.factors:
        bucket: dict[str, list[AssignmentRun]] = {lvl: [] for lvl in factor.levels}
        for run in result.runs.values():
            lvl = run.assignment.levels.get(factor.name)
            if lvl is None:
                continue
            bucket[lvl].append(run)
        result.main_effects[factor.name] = {}
        for lvl, runs in bucket.items():
            if not runs:
                continue
            wins = sum(r.wins for r in runs)
            trials = sum(r.wins + r.losses for r in runs)
            ci = wilson_ci(wins, trials)
            games = sum(r.wins + r.losses + r.draws for r in runs)
            ci_incl = wilson_ci(wins, games)
            result.main_effects[factor.name][lvl] = MainEffect(
                factor=factor.name,
                level=lvl,
                n_assignments=len(runs),
                marginal_win_rate=ci.rate,
                ci_lower=ci.lower,
                ci_upper=ci.upper,
                n_games=games,
                marginal_win_rate_incl_draws=ci_incl.rate,
                ci_lower_incl_draws=ci_incl.lower,
                ci_upper_incl_draws=ci_incl.upper,
            )


def _compute_interactions(result: AblationResult, *, alpha: float) -> None:
    factor_names = [f.name for f in result.factors]
    if len(factor_names) < 2:
        return

    interactions: list[tuple[str, str, InteractionEffect, float]] = []
    for a_name, b_name in itertools.combinations(factor_names, 2):
        fa = result.factor_by_name(a_name)
        fb = result.factor_by_name(b_name)
        per_cell_rate: dict[tuple[str, str], float] = {}
        per_cell_rate_incl: dict[tuple[str, str], float] = {}
        cell_deltas: dict[tuple[str, str], list[int]] = {}
        cell_totals: dict[tuple[str, str], tuple[int, int]] = {}
        for a_lvl in fa.levels:
            for b_lvl in fb.levels:
                wins = losses = draws = 0
                merged_deltas: list[int] = []
                for run in result.runs.values():
                    if run.assignment.levels.get(a_name) != a_lvl:
                        continue
                    if run.assignment.levels.get(b_name) != b_lvl:
                        continue
                    wins += run.wins
                    losses += run.losses
                    draws += run.draws
                    merged_deltas.extend(run.per_seed_deltas)
                cell_totals[(a_lvl, b_lvl)] = (wins, losses)
                cell_deltas[(a_lvl, b_lvl)] = merged_deltas
                trials = wins + losses
                per_cell_rate[(a_lvl, b_lvl)] = wins / trials if trials else 0.0
                total = wins + losses + draws
                per_cell_rate_incl[(a_lvl, b_lvl)] = wins / total if total else 0.0

        # Interaction magnitude: max over (a1, a0) of |Delta_b1 - Delta_b0|
        # where Delta_bj = rate(a1, bj) - rate(a0, bj). We report the max
        # across all pairs of a-levels crossed with all pairs of b-levels
        # (=1 pair each for 2-level factors, so simple).
        magnitudes: list[float] = []
        contrast_deltas: list[int] = []
        a_levels = fa.levels
        b_levels = fb.levels
        for a0, a1 in itertools.combinations(a_levels, 2):
            for b0, b1 in itertools.combinations(b_levels, 2):
                delta_at_b1 = per_cell_rate[(a1, b1)] - per_cell_rate[(a0, b1)]
                delta_at_b0 = per_cell_rate[(a1, b0)] - per_cell_rate[(a0, b0)]
                magnitudes.append(abs(delta_at_b1 - delta_at_b0))

                # Build per-seed interaction contrast: sum(a1-a0 at b1) -
                # sum(a1-a0 at b0). We take the same-length paired vector
                # by truncating to the minimum length -- assignments share
                # a seed schedule so lengths align in practice.
                d_a1b1 = cell_deltas[(a1, b1)]
                d_a0b1 = cell_deltas[(a0, b1)]
                d_a1b0 = cell_deltas[(a1, b0)]
                d_a0b0 = cell_deltas[(a0, b0)]
                m = min(len(d_a1b1), len(d_a0b1), len(d_a1b0), len(d_a0b0))
                for i in range(m):
                    contrast_deltas.append(
                        (d_a1b1[i] - d_a0b1[i]) - (d_a1b0[i] - d_a0b0[i])
                    )
        magnitude = max(magnitudes) if magnitudes else 0.0
        # If head-to-head saturates (every cell 100%), fall back to
        # draws-inclusive rates so the interaction can still be measured.
        if magnitude < 1e-9:
            incl_mags: list[float] = []
            for a0, a1 in itertools.combinations(fa.levels, 2):
                for b0, b1 in itertools.combinations(fb.levels, 2):
                    d1 = per_cell_rate_incl[(a1, b1)] - per_cell_rate_incl[(a0, b1)]
                    d0 = per_cell_rate_incl[(a1, b0)] - per_cell_rate_incl[(a0, b0)]
                    incl_mags.append(abs(d1 - d0))
            if incl_mags:
                magnitude = max(incl_mags)
        p_raw = paired_sign_test(contrast_deltas) if contrast_deltas else 1.0
        interactions.append(
            (
                a_name,
                b_name,
                InteractionEffect(
                    factor_a=a_name,
                    factor_b=b_name,
                    magnitude=magnitude,
                    p_value=p_raw,
                    p_value_holm=p_raw,  # populated below
                    rejected=False,
                    per_cell_rate=per_cell_rate,
                ),
                p_raw,
            )
        )

    pvals = [entry[3] for entry in interactions]
    rejects = holm_bonferroni(pvals, alpha=alpha)
    # Holm-corrected p: for each i, sorted rank r, p_corr = min(1, p * (m-r+1)),
    # then enforce monotonicity along the sorted order.
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    corrected = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min(1.0, pvals[idx] * (m - rank + 1))
        running = max(running, adj)
        corrected[idx] = running
    for i, (a, b, eff, _) in enumerate(interactions):
        eff.p_value_holm = corrected[i]
        eff.rejected = rejects[i]
        result.interaction_effects[(a, b)] = eff


def _compute_ranking(result: AblationResult) -> None:
    """Rank factors by |max - min| marginal win rate.

    Uses the head-to-head rate as the primary key. When that rate
    saturates (e.g. every level beats the opponent 100% ex-draws), we
    fall back to the draws-inclusive rate so the ranking still reflects
    which factor actually shifts outcomes.
    """
    ranks: list[tuple[str, float]] = []
    for factor_name, levels in result.main_effects.items():
        rates = [me.marginal_win_rate for me in levels.values()]
        if not rates:
            continue
        span = max(rates) - min(rates)
        if span < 1e-9:
            incl = [me.marginal_win_rate_incl_draws for me in levels.values()]
            span = max(incl) - min(incl)
        ranks.append((factor_name, span))
    ranks.sort(key=lambda kv: kv[1], reverse=True)
    result.ranking = ranks


# ---------- analysis / printers ----------


class analysis:
    """Namespace for plain-text printers. Static methods only; no state."""

    @staticmethod
    def main_effects_table(result: AblationResult) -> str:
        lines: list[str] = []
        lines.append(
            f"Main effects  (design={result.design_kind}, opponent={result.opponent_name})"
        )
        lines.append(
            f"{'factor':<20} {'level':<24} {'assigns':>7} {'games':>7} "
            f"{'win% (H2H)':>11} {'95% CI':>16} {'win% (incl draws)':>18} {'95% CI':>16}"
        )
        lines.append("-" * 121)
        for factor in result.factors:
            entries = result.main_effects.get(factor.name, {})
            for lvl in factor.levels:
                me = entries.get(lvl)
                if me is None:
                    lines.append(
                        f"{factor.name:<20} {lvl:<24} {0:>7} {0:>7} "
                        f"{'-':>11} {'-':>16} {'-':>18} {'-':>16}"
                    )
                    continue
                ci = f"[{me.ci_lower * 100:5.1f}, {me.ci_upper * 100:5.1f}]"
                ci_i = f"[{me.ci_lower_incl_draws * 100:5.1f}, {me.ci_upper_incl_draws * 100:5.1f}]"
                lines.append(
                    f"{factor.name:<20} {lvl:<24} {me.n_assignments:>7} "
                    f"{me.n_games:>7} {me.marginal_win_rate * 100:>10.1f}% {ci:>16} "
                    f"{me.marginal_win_rate_incl_draws * 100:>17.1f}% {ci_i:>16}"
                )
        if result.ranking:
            lines.append("")
            lines.append("Factor ranking by |max - min| marginal win rate:")
            for name, span in result.ranking:
                lines.append(f"  {name:<20} span={span * 100:5.1f}pp")
        return "\n".join(lines)

    @staticmethod
    def interactions_table(result: AblationResult) -> str:
        if not result.interaction_effects:
            return "(no interaction effects computed: need at least 2 factors)"
        lines: list[str] = []
        lines.append("Pairwise interactions  (Holm-Bonferroni across all pairs)")
        lines.append(f"{'A x B':<32} {'magnitude':>10} {'p-raw':>8} {'p-holm':>8} "
                     f"{'reject':>7}")
        lines.append("-" * 72)
        items = sorted(
            result.interaction_effects.items(),
            key=lambda kv: kv[1].magnitude,
            reverse=True,
        )
        for (a, b), eff in items:
            label = f"{a} x {b}"
            reject = "yes" if eff.rejected else "no"
            lines.append(
                f"{label:<32} {eff.magnitude * 100:>9.1f}pp {eff.p_value:>8.3f} "
                f"{eff.p_value_holm:>8.3f} {reject:>7}"
            )
        return "\n".join(lines)


__all__ = [
    "AblationResult",
    "AssignmentRun",
    "Factor",
    "FactorAssignment",
    "FactorialDesign",
    "FractionalFactorialDesign",
    "InteractionEffect",
    "MainEffect",
    "analysis",
    "run_ablation",
]
