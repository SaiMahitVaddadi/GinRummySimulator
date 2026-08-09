"""Composite multi-factor ablation: heuristic sub-rules x wrappers x MoE.

This is the paper's *flagship* ablation. It crosses every architectural
add the codebase currently supports into a single 2^4 = 16-cell full
factorial and reports main effects + all pairwise interactions.

Factors (all 2-level, so the design collapses cleanly to a fractional
half-fraction when a resolution-IV sketch is desired):

* ``discard``  in {``random``, ``highest_deadwood``}
    Which card to shed. ``random`` is the null baseline; the greedy
    ``highest_deadwood`` rule from :mod:`gin_rummy.policies.heuristic_components`
    is the current best hand-crafted default.

* ``knock``    in {``wait_for_gin``, ``always``}
    Knock timing. ``wait_for_gin`` = knock only at deadwood 0 (hold out
    for the gin bonus); ``always`` = knock the moment it's legal.

* ``wrapper``  in {``bare``, ``voting3``}
    Whether to wrap the base heuristic in a 3-seed :class:`VotingEnsemble`.
    Since our base policies are deterministic given a fixed rule triple,
    voting3 is a pass-through *unless* combined with a randomised base
    (i.e. ``discard=random``), which turns voting3 into a genuine
    variance-reduction ablation.

* ``moe``      in {``flat``, ``phase_gated``}
    Whether to wrap the base heuristic in a :class:`PhaseGatedMoE` router.
    We route ``early``/``mid``/``late`` to the *same* base expert for now
    (single-expert-per-phase). This isolates the routing overhead; the
    slot is ready for genuinely-different phase experts in follow-on work.

Design (per the task brief):

    base = ComposableHeuristicPolicy(draw_if_reduces_deadwood, discard, knock, rng)
    if moe == "phase_gated":
        base = PhaseGatedMoE(early=base, mid=base, late=base)
    if wrapper == "voting3":
        base = VotingEnsemble(experts=[base, base, base])

Opponent: :class:`RandomPolicy`. Paired hands, ``games_per_pair=80`` -> 1,280
games total for the full factorial (~90 s on a laptop).

Additionally emits a resolution-IV **half-fraction** (2^{4-1} = 8 cells)
using the framework's default ``I = ABCD`` generator, to demonstrate the
fractional design and to sanity-check that main-effect signs agree with
the full run.

Outputs
-------
* ``experiments/results/ablation_composite_full.jsonl`` + ``.txt``
* ``experiments/results/ablation_composite_frac.jsonl`` + ``.txt``
* stdout mirrors both tables.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from gin_rummy.eval.ablation import (
    AblationResult,
    Factor,
    FactorialDesign,
    FractionalFactorialDesign,
    analysis,
    run_ablation,
)
from gin_rummy.eval.ensemble import PhaseGatedMoE, VotingEnsemble
from gin_rummy.policies.heuristic import ComposableHeuristicPolicy
from gin_rummy.policies.heuristic_components import (
    DiscardRule,
    KnockRule,
    discard_highest_deadwood,
    discard_random,
    draw_if_reduces_deadwood,
    knock_asap,
)
from gin_rummy.policy import Observation, Policy, RandomPolicy
from gin_rummy.variants.classic import ClassicGin


# ---------- knock rules ---------------------------------------------------


def _knock_wait_for_gin(obs: Observation, deadwood_value: int, rng: random.Random) -> bool:
    """Only knock at deadwood 0 (i.e. hold out for gin)."""
    return deadwood_value == 0


DISCARD_TABLE: dict[str, DiscardRule] = {
    "random": discard_random,
    "highest_deadwood": discard_highest_deadwood,
}
KNOCK_TABLE: dict[str, KnockRule] = {
    "wait_for_gin": _knock_wait_for_gin,
    "always": knock_asap,
}


# ---------- policy builder ------------------------------------------------


def _build_composite(levels: dict[str, str], rng: random.Random) -> Policy:
    """Build the (draw, discard, knock) heuristic and wrap by (moe, wrapper).

    The wrapping order (MoE inside, ensemble outside) is fixed by the task
    brief and mirrors the flagship recipe in the paper: MoE routes per
    turn, then a top-level voting ensemble aggregates multiple such
    routers.
    """
    discard_rule = DISCARD_TABLE[levels["discard"]]
    knock_rule = KNOCK_TABLE[levels["knock"]]

    def _make_base(sub_rng: random.Random) -> Policy:
        return ComposableHeuristicPolicy(
            draw_rule=draw_if_reduces_deadwood,
            discard_rule=discard_rule,
            knock_rule=knock_rule,
            rng=sub_rng,
        )

    def _wrap_moe(base_factory: Callable[[random.Random], Policy],
                  sub_rng: random.Random) -> Policy:
        if levels["moe"] == "phase_gated":
            # Single-expert-per-phase: independent RNG streams for each
            # phase's expert so that any internal randomness (e.g.
            # discard_random) stays decorrelated across phases.
            return PhaseGatedMoE(
                early=base_factory(random.Random(sub_rng.randrange(1 << 30))),
                mid=base_factory(random.Random(sub_rng.randrange(1 << 30))),
                late=base_factory(random.Random(sub_rng.randrange(1 << 30))),
            )
        return base_factory(sub_rng)

    if levels["wrapper"] == "voting3":
        experts = [
            _wrap_moe(_make_base, random.Random(rng.randrange(1 << 30)))
            for _ in range(3)
        ]
        return VotingEnsemble(experts=experts)
    return _wrap_moe(_make_base, rng)


# ---------- factors --------------------------------------------------------


FACTORS: list[Factor] = [
    Factor("discard", ["random", "highest_deadwood"]),
    Factor("knock", ["wait_for_gin", "always"]),
    Factor("wrapper", ["bare", "voting3"]),
    Factor("moe", ["flat", "phase_gated"]),
]


# ---------- opponent -------------------------------------------------------


def _random_opponent(rng: random.Random) -> Policy:
    return RandomPolicy(rng)


# ---------- IO -------------------------------------------------------------


def _dump_jsonl(result: AblationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "summary",
                    "design_kind": result.design_kind,
                    "opponent": result.opponent_name,
                    "factors": [
                        {"name": f.name, "levels": list(f.levels)}
                        for f in result.factors
                    ],
                    "ranking": [
                        {"factor": n, "span": s} for n, s in result.ranking
                    ],
                }
            )
            + "\n"
        )
        for f_name, levels in result.main_effects.items():
            for lvl, me in levels.items():
                fh.write(json.dumps({"kind": "main_effect", **asdict(me)}) + "\n")
        for (a, b), eff in result.interaction_effects.items():
            payload = {
                "kind": "interaction",
                "factor_a": eff.factor_a,
                "factor_b": eff.factor_b,
                "magnitude": eff.magnitude,
                "p_value": eff.p_value,
                "p_value_holm": eff.p_value_holm,
                "rejected": eff.rejected,
                "per_cell_rate": [
                    {"a_level": k[0], "b_level": k[1], "rate": v}
                    for k, v in eff.per_cell_rate.items()
                ],
            }
            fh.write(json.dumps(payload) + "\n")
        for key, run in result.runs.items():
            fh.write(
                json.dumps(
                    {
                        "kind": "assignment",
                        "levels": dict(key),
                        "wins": run.wins,
                        "losses": run.losses,
                        "draws": run.draws,
                        "win_rate": run.win_rate_ci.rate,
                        "ci_lower": run.win_rate_ci.lower,
                        "ci_upper": run.win_rate_ci.upper,
                        "win_rate_incl_draws": run.win_rate_including_draws_ci.rate,
                    }
                )
                + "\n"
            )


def _dump_text(result: AblationResult, path: Path, header: str) -> str:
    text_parts = [
        header,
        "",
        analysis.main_effects_table(result),
        "",
        analysis.interactions_table(result),
        "",
    ]
    text = "\n".join(text_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ---------- library entry --------------------------------------------------


def run_composite_ablation(
    games_per_pair: int = 80,
    seed: int = 0,
    out_dir: str | Path = "experiments/results",
    also_fractional: bool = True,
) -> dict[str, AblationResult]:
    """Run the full 2^4 factorial (and optionally its half-fraction).

    Parameters
    ----------
    games_per_pair : int
        Games each cell plays vs. the random opponent. Must be even
        (paired mode). Default 80 -> 1,280 games for the full design.
    seed : int
        Reproducible base seed shared across all cells.
    out_dir : str | Path
        Destination directory for ``.jsonl`` + ``.txt`` outputs.
    also_fractional : bool
        If True, also run a resolution-IV half-fraction (2^{4-1}=8 cells)
        for demonstration and cross-check the main-effect signs.

    Returns
    -------
    dict[str, AblationResult]
        Keys: ``"full"`` and (if enabled) ``"frac"``.
    """
    out_dir = Path(out_dir)
    results: dict[str, AblationResult] = {}

    # ---- full 2^4 factorial (16 cells) ----
    full_design = FactorialDesign(FACTORS, build_fn=_build_composite)
    t0 = time.time()
    full_result = run_ablation(
        full_design,
        opponent_factory=_random_opponent,
        opponent_name="random",
        game_cls=ClassicGin,
        games_per_pair=games_per_pair,
        seed=seed,
        paired=True,
    )
    dt_full = time.time() - t0
    results["full"] = full_result

    full_jsonl = out_dir / "ablation_composite_full.jsonl"
    full_txt = out_dir / "ablation_composite_full.txt"
    _dump_jsonl(full_result, full_jsonl)
    header = (
        f"Composite ablation (FULL 2^4 = 16 cells, {games_per_pair} games/cell, "
        f"{len(FACTORS)} factors, opponent=random, elapsed={dt_full:.1f}s)"
    )
    text_full = _dump_text(full_result, full_txt, header)
    print(text_full)
    print(f"[ablation_composite] wrote {full_jsonl}")
    print(f"[ablation_composite] wrote {full_txt}")

    # ---- resolution-IV half-fraction (8 cells) ----
    if also_fractional:
        frac_design = FractionalFactorialDesign(
            FACTORS,
            generators=None,  # default I = ABCD -> resolution IV
            build_fn=_build_composite,
        )
        t0 = time.time()
        frac_result = run_ablation(
            frac_design,
            opponent_factory=_random_opponent,
            opponent_name="random",
            game_cls=ClassicGin,
            games_per_pair=games_per_pair,
            seed=seed,
            paired=True,
        )
        dt_frac = time.time() - t0
        results["frac"] = frac_result

        frac_jsonl = out_dir / "ablation_composite_frac.jsonl"
        frac_txt = out_dir / "ablation_composite_frac.txt"
        _dump_jsonl(frac_result, frac_jsonl)
        header = (
            f"Composite ablation (FRACTIONAL half-fraction, 8 cells, "
            f"{games_per_pair} games/cell, resolution={frac_design.resolution()}, "
            f"opponent=random, elapsed={dt_frac:.1f}s)"
        )
        text_frac = _dump_text(frac_result, frac_txt, header)
        print(text_frac)
        print(f"[ablation_composite] wrote {frac_jsonl}")
        print(f"[ablation_composite] wrote {frac_txt}")

        # Cross-check main-effect signs between full and half-fraction.
        cross_lines = ["", "Main-effect sign agreement (FULL vs FRACTIONAL):"]
        for factor in FACTORS:
            full_rates = full_result.main_effects[factor.name]
            frac_rates = frac_result.main_effects.get(factor.name, {})
            lvl0, lvl1 = factor.levels
            full_delta = (
                full_rates[lvl1].marginal_win_rate_incl_draws
                - full_rates[lvl0].marginal_win_rate_incl_draws
            )
            if lvl0 not in frac_rates or lvl1 not in frac_rates:
                cross_lines.append(f"  {factor.name:<10}  (missing in fractional)")
                continue
            frac_delta = (
                frac_rates[lvl1].marginal_win_rate_incl_draws
                - frac_rates[lvl0].marginal_win_rate_incl_draws
            )
            agree = "yes" if (full_delta * frac_delta >= 0) else "NO"
            cross_lines.append(
                f"  {factor.name:<10}  full_delta={full_delta * 100:+5.1f}pp  "
                f"frac_delta={frac_delta * 100:+5.1f}pp  agree={agree}"
            )
        cross_text = "\n".join(cross_lines)
        print(cross_text)
        with frac_txt.open("a") as fh:
            fh.write(cross_text + "\n")

    return results


# ---------- CLI ------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.ablation_composite",
        description=(
            "Multi-factor ablation composing every architectural add the "
            "codebase supports (heuristic sub-rules x MoE x voting ensemble)."
        ),
    )
    p.add_argument("--games-per-pair", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    p.add_argument(
        "--no-fractional",
        action="store_true",
        help="Skip the resolution-IV half-fraction demo run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    run_composite_ablation(
        games_per_pair=args.games_per_pair,
        seed=args.seed,
        out_dir=args.out_dir,
        also_fractional=not args.no_fractional,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
