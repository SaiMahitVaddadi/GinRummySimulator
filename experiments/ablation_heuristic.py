"""Worked ablation example: decompose the ``GreedyKnockPolicy`` recipe.

Treats the three sub-decisions of a hand-crafted heuristic as independent
factors and runs a full 2 x 3 x 2 = 12-cell factorial ablation vs.
``RandomPolicy``. Because the current codebase ships only one production
draw/discard/knock heuristic, we synthesise 2-3 minimal alternatives per
factor to give the ablation something to measure. When the parallel
decomposition agent adds richer sub-modules, they can be swapped in
without changing this script.

Factors (paper-friendly names):

* ``draw`` -- draw source policy
   - ``deck``      : always draw from the deck
   - ``smart``     : take discard iff it reduces deadwood (greedy default)

* ``discard`` -- which card to discard
   - ``random``            : uniform over non-meld cards (fallback to any card)
   - ``highest_deadwood``  : max-value deadwood card (greedy default)
   - ``safe_from_opp``     : STUB. Same as highest_deadwood but tie-breaks
                              on the card *least* likely to help the opponent
                              (approximated as: prefer cards whose rank does
                              not match the top discard). Marked STUB in the
                              docstring so the decomposition agent knows to
                              slot in a real belief-model version.

* ``knock`` -- when to knock
   - ``always``       : knock whenever legal (greedy default)
   - ``wait_for_gin`` : only knock at 0 deadwood

Full factorial: 2 x 3 x 2 = 12 assignments. Each vs. ``RandomPolicy``,
``games_per_pair=50``, ``paired=True`` -> ~600 games total, ~10 s.

Outputs
-------
* ``experiments/results/ablation_heuristic.jsonl`` -- per-assignment
  win rates + main effects + interactions.
* stdout -- the main-effects table (paper-ready).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from gin_rummy.cards import Card
from gin_rummy.eval.ablation import (
    AblationResult,
    Factor,
    FactorialDesign,
    analysis,
    run_ablation,
)
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import DrawSource, Observation, Policy, RandomPolicy
from gin_rummy.variants.classic import ClassicGin


# ---------- pluggable sub-modules ----------


DrawFn = Callable[[Observation], DrawSource]
DiscardFn = Callable[[Observation, random.Random], Card]
KnockFn = Callable[[Observation, int], bool]


def draw_deck(obs: Observation) -> DrawSource:
    return "deck"


def draw_smart(obs: Observation) -> DrawSource:
    if obs.top_discard is None:
        return "deck"
    _, _, before = optimal_decomposition(list(obs.hand))
    _, _, after = optimal_decomposition(list(obs.hand) + [obs.top_discard])
    return "discard" if after < before else "deck"


def _deadwood_cards(obs: Observation) -> list[Card]:
    _, deadwood, _ = optimal_decomposition(list(obs.hand))
    return deadwood


def discard_random(obs: Observation, rng: random.Random) -> Card:
    dw = _deadwood_cards(obs)
    if not dw:
        return rng.choice(obs.hand)
    return rng.choice(dw)


def discard_highest_deadwood(obs: Observation, rng: random.Random) -> Card:
    dw = _deadwood_cards(obs)
    if not dw:
        return rng.choice(obs.hand)
    return max(dw, key=lambda c: c.value)


def discard_safe_from_opp(obs: Observation, rng: random.Random) -> Card:
    """STUB. Placeholder for a real opponent-belief-informed policy.

    Currently: pick the highest-value deadwood card, but among ties in
    value prefer the card whose rank does NOT match the top discard's
    rank (a crude 'don't feed the opponent the set they just built' proxy).
    Replace with a proper belief-tracker once the decomposition agent
    adds one.
    """
    dw = _deadwood_cards(obs)
    if not dw:
        return rng.choice(obs.hand)
    top = obs.top_discard
    max_v = max(c.value for c in dw)
    top_tier = [c for c in dw if c.value == max_v]
    if top is not None:
        safer = [c for c in top_tier if c.rank != top.rank]
        if safer:
            return safer[0]
    return top_tier[0]


def knock_always(obs: Observation, dw_val: int) -> bool:
    return True


def knock_wait_for_gin(obs: Observation, dw_val: int) -> bool:
    return dw_val == 0


DRAW_TABLE: dict[str, DrawFn] = {"deck": draw_deck, "smart": draw_smart}
DISCARD_TABLE: dict[str, DiscardFn] = {
    "random": discard_random,
    "highest_deadwood": discard_highest_deadwood,
    "safe_from_opp": discard_safe_from_opp,  # STUB
}
KNOCK_TABLE: dict[str, KnockFn] = {"always": knock_always, "wait_for_gin": knock_wait_for_gin}


# ---------- composed policy from an assignment ----------


class _ComposedHeuristic:
    def __init__(self, draw: DrawFn, discard: DiscardFn, knock: KnockFn, rng: random.Random):
        self._draw = draw
        self._discard = discard
        self._knock = knock
        self._rng = rng

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        return self._draw(obs)

    def choose_discard(self, obs: Observation) -> Card:
        return self._discard(obs, self._rng)

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        return self._knock(obs, deadwood_value)


def _build_from_assignment(levels: dict[str, str], rng: random.Random) -> Policy:
    return _ComposedHeuristic(
        draw=DRAW_TABLE[levels["draw"]],
        discard=DISCARD_TABLE[levels["discard"]],
        knock=KNOCK_TABLE[levels["knock"]],
        rng=rng,
    )


# ---------- driver ----------


FACTORS = [
    Factor("draw", ["deck", "smart"]),
    Factor("discard", ["random", "highest_deadwood", "safe_from_opp"]),
    Factor("knock", ["always", "wait_for_gin"]),
]


def _random_opponent(rng: random.Random) -> Policy:
    return RandomPolicy(rng)


def _dump_jsonl(result: AblationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "summary",
                    "design_kind": result.design_kind,
                    "opponent": result.opponent_name,
                    "factors": [{"name": f.name, "levels": list(f.levels)} for f in result.factors],
                    "ranking": [{"factor": n, "span": s} for n, s in result.ranking],
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
                    }
                )
                + "\n"
            )


def run(
    games_per_pair: int = 50,
    seed: int = 0,
    out_dir: str | Path = "experiments/results",
) -> AblationResult:
    design = FactorialDesign(FACTORS, build_fn=_build_from_assignment)
    result = run_ablation(
        design,
        opponent_factory=_random_opponent,
        opponent_name="random",
        game_cls=ClassicGin,
        games_per_pair=games_per_pair,
        seed=seed,
        paired=True,
    )
    out_path = Path(out_dir) / "ablation_heuristic.jsonl"
    _dump_jsonl(result, out_path)
    print(analysis.main_effects_table(result))
    print()
    print(analysis.interactions_table(result))
    print()
    print(f"[ablation_heuristic] wrote {out_path}")
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.ablation_heuristic",
        description="Full-factorial ablation of the heuristic draw/discard/knock sub-modules.",
    )
    p.add_argument("--games-per-pair", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    run(games_per_pair=args.games_per_pair, seed=args.seed, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
