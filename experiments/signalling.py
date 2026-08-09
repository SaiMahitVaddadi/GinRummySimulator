"""§6 predictions #4 and #5 — structured signalling in Euchre.

Runs four treatments over the same ``N`` seeds (0..N-1):

* **silent**       — all four seats play :class:`RandomEuchrePolicy`.
* **cooperative**  — seats 0 & 2 (team 0) wrap their random policy in
                     :class:`EuchreTalkingPolicy` (emit-only); team 1
                     stays silent.
* **byzantine**    — seat 0 uses :class:`EuchreTalkingPolicy`; seat 2
                     (its partner) uses :class:`EuchreByzantinePolicy`
                     wrapping a talker; team 1 stays silent. Still
                     emit-only.
* **closed_loop**  — seats 0 & 2 use talk-and-listen wiring via
                     :func:`ClosedLoopTeam`: each plays a listener
                     wrapping a talker over the shared channel. Team 1
                     stays silent. This is the treatment that
                     actually closes the coordination loop; the
                     three above are emit-only baselines kept for
                     comparison.

For each treatment we report:

* Team-0 win rate + Wilson CI (games where team 0 outscores team 1).
* Team-0 partner-MI + block-bootstrap CI. The MI is
  I(action; partner_hand | own_hand, public) restricted to team-0
  seats' plays, computed with the same coarse buckets as
  ``comms.analyze`` so numbers are comparable across treatments.
* Average number of signals broadcast per game.
* Listener fidelity: total listener actions, and the fraction that
  were actually signal-conditioned (i.e. the listener rule fired and
  chose a different card than the inner policy). Zero for the three
  emit-only treatments — they carry no listener.

The point is to observe what our concrete implementation does. In the
current build the closed_loop arm is the only one wired to *react* to
signals; the silent / cooperative / byzantine arms are byte-identical
in play behaviour and exist only to prove that signals alone don't
move partner-MI or win rate.

Usage
-----
CLI:      ``uv run python -m experiments.signalling --games 200 --seed 0``
Library:  ``from experiments.signalling import run_signalling``
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

from gin_rummy.comms.analyze import (
    PlaySample,
    _bootstrap_mi,
    _InstrumentedEuchreGame,
    _mi_over_samples,
)
from gin_rummy.comms.channel import MessageChannel
from gin_rummy.comms.euchre_signals import (
    ClosedLoopTeam,
    EuchreByzantinePolicy,
    EuchreListeningPolicy,
    EuchreTalkingPolicy,
)
from gin_rummy.comms.euchre_search_listener import (
    ClosedLoopSearchTeam,
    SearchListeningPolicy,
)
from gin_rummy.eval.stats import BootstrapCI, ProportionCI, wilson_ci
from gin_rummy.variants.euchre import EuchrePolicy, RandomEuchrePolicy


Treatment = Literal[
    "silent",
    "cooperative",
    "byzantine",
    "closed_loop",
    "closed_loop_search",
]

TEAM0_SEATS: tuple[int, ...] = (0, 2)


# --------------------------------------------------------------- builders ---


PolicyBuilder = Callable[[random.Random, MessageChannel], list[EuchrePolicy]]


def _silent_builder(rng: random.Random, channel: MessageChannel) -> list[EuchrePolicy]:
    del channel
    return [RandomEuchrePolicy(rng) for _ in range(4)]


def _cooperative_builder(
    rng: random.Random, channel: MessageChannel
) -> list[EuchrePolicy]:
    return [
        EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=0),
        RandomEuchrePolicy(rng),
        EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=2),
        RandomEuchrePolicy(rng),
    ]


def _byzantine_builder(
    rng: random.Random, channel: MessageChannel
) -> list[EuchrePolicy]:
    honest = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=0)
    liar_inner = EuchreTalkingPolicy(RandomEuchrePolicy(rng), channel, sender_id=2)
    liar = EuchreByzantinePolicy(liar_inner)
    return [
        honest,
        RandomEuchrePolicy(rng),
        liar,
        RandomEuchrePolicy(rng),
    ]


def _closed_loop_builder(
    rng: random.Random, channel: MessageChannel
) -> list[EuchrePolicy]:
    """Talk-and-listen on both team-0 seats; team-1 silent random.

    ``ClosedLoopTeam`` wires an :class:`EuchreListeningPolicy` at each
    of seats 0 and 2, with an :class:`EuchreTalkingPolicy` as its
    inner, all sharing ``channel``. That combines emission (via the
    talker) with reaction (via the listener) so partner signals can
    actually change plays.
    """
    listener0, listener2 = ClosedLoopTeam(
        seat0=RandomEuchrePolicy(rng),
        seat2=RandomEuchrePolicy(rng),
        channel=channel,
    )
    return [
        listener0,
        RandomEuchrePolicy(rng),
        listener2,
        RandomEuchrePolicy(rng),
    ]


def _closed_loop_search_builder(
    rng: random.Random, channel: MessageChannel
) -> list[EuchrePolicy]:
    """Talk-and-search-listen on both team-0 seats; team-1 silent random.

    Same wiring as :func:`_closed_loop_builder` but with a
    :class:`SearchListeningPolicy` in the listener slot instead of the
    hand-crafted :class:`EuchreListeningPolicy`. See
    ``gin_rummy.comms.euchre_search_listener`` for the rollout details.
    """
    # Draw a deterministic sub-seed from the shared policy RNG so the
    # search's own reject-sampler is reproducible per game while still
    # varying with the top-level ``seed``.
    seed = rng.randrange(1 << 31)
    listener0, listener2 = ClosedLoopSearchTeam(
        seat0=RandomEuchrePolicy(rng),
        seat2=RandomEuchrePolicy(rng),
        channel=channel,
        k_samples=32,
        rng_seed=seed,
    )
    return [
        listener0,
        RandomEuchrePolicy(rng),
        listener2,
        RandomEuchrePolicy(rng),
    ]


BUILDERS: dict[Treatment, PolicyBuilder] = {
    "silent": _silent_builder,
    "cooperative": _cooperative_builder,
    "byzantine": _byzantine_builder,
    "closed_loop": _closed_loop_builder,
    "closed_loop_search": _closed_loop_search_builder,
}


# ------------------------------------------------------------- data types ---


@dataclass
class TreatmentResult:
    treatment: Treatment
    n_games: int
    n_plays_team0: int
    team0_wins: int
    team0_ties: int
    team0_losses: int
    team0_passouts: int
    team0_win_rate: float
    team0_wilson_lower: float
    team0_wilson_upper: float
    partner_mi_point: float
    partner_mi_lower: float
    partner_mi_upper: float
    own_mi_point: float
    signals_per_game: float
    total_signals: int
    # Listener fidelity — 0 for emit-only treatments.
    listener_plays_total: int = 0
    listener_plays_signal_conditioned: int = 0
    listener_signal_conditioned_fraction: float = 0.0


# ------------------------------------------------------------- experiment ---


def _play_one(
    *,
    treatment: Treatment,
    seed: int,
    game_id: int,
) -> tuple[list[PlaySample], int | None, int, int, int]:
    """Play one Euchre hand under ``treatment``.

    Returns
    -------
    samples
        Per-play snapshots from the instrumented game (used for MI).
    winner_team
        0, 1, or ``None`` for pass-out.
    signal_count
        Number of messages broadcast by the treatment's talkers.
    listener_plays
        Total ``choose_play`` invocations across any
        :class:`EuchreListeningPolicy` seats in this game (0 for
        emit-only treatments).
    listener_signal_conditioned
        Subset of the above where the listener actually diverged from
        its inner policy because of a partner signal.
    """
    channel = MessageChannel()
    # RNG for the policies is kept separate from the game's dealing RNG
    # so that a stochastic policy consuming an entropy bit doesn't shift
    # the shuffle across treatments — crucial for a like-for-like MI
    # comparison across the silent / cooperative / byzantine arms.
    policy_rng = random.Random(seed ^ 0xA5A5A5)
    policies = BUILDERS[treatment](policy_rng, channel)
    game = _InstrumentedEuchreGame(
        policies=policies, seed=seed, game_id=game_id
    )
    result = game.play()

    listener_plays = 0
    listener_conditioned = 0
    for p in policies:
        if isinstance(p, (EuchreListeningPolicy, SearchListeningPolicy)):
            listener_plays += p.plays_total
            listener_conditioned += p.plays_signal_conditioned

    return (
        game.samples,
        result.winner_team,
        len(channel),
        listener_plays,
        listener_conditioned,
    )


def _mi_ci(
    per_game_team0_samples: Sequence[Sequence[PlaySample]],
    *,
    hand_field: str,
    condition_on_own: bool,
    n_resamples: int,
    seed: int,
    level: float,
) -> BootstrapCI:
    return _bootstrap_mi(
        per_game_team0_samples,
        hand_field,
        condition_on_own=condition_on_own,
        n_resamples=n_resamples,
        level=level,
        rng=random.Random(seed),
    )


def run_treatment(
    treatment: Treatment,
    *,
    n_games: int,
    seed: int = 0,
    n_resamples: int = 500,
    level: float = 0.95,
) -> TreatmentResult:
    """Run ``n_games`` deals under ``treatment`` and return summary stats."""
    per_game_team0: list[list[PlaySample]] = []
    wins = ties = losses = passouts = 0
    total_signals = 0
    listener_plays_total = 0
    listener_conditioned_total = 0
    for gid in range(n_games):
        (
            samples,
            winner_team,
            sig_count,
            l_plays,
            l_cond,
        ) = _play_one(treatment=treatment, seed=seed + gid, game_id=gid)
        total_signals += sig_count
        listener_plays_total += l_plays
        listener_conditioned_total += l_cond
        team0_samples = [s for s in samples if s.seat in TEAM0_SEATS]
        per_game_team0.append(team0_samples)
        if winner_team is None:
            passouts += 1
        elif winner_team == 0:
            wins += 1
        elif winner_team == 1:
            losses += 1
        else:  # pragma: no cover
            ties += 1

    decisive = wins + losses  # exclude pass-outs from the win-rate denominator
    wilson: ProportionCI = wilson_ci(wins, decisive, level=level)

    n_plays_team0 = sum(len(g) for g in per_game_team0)
    if n_plays_team0:
        flat = [s for g in per_game_team0 for s in g]
        own_mi = _mi_over_samples(
            flat, "own_hand", condition_on_own=False
        )
        partner_ci = _mi_ci(
            per_game_team0,
            hand_field="partner_hand",
            condition_on_own=True,
            n_resamples=n_resamples,
            seed=seed ^ 0xBEEF,
            level=level,
        )
    else:
        own_mi = 0.0
        partner_ci = BootstrapCI(0.0, 0.0, 0.0, level, n_resamples)

    conditioned_frac = (
        listener_conditioned_total / listener_plays_total
        if listener_plays_total
        else 0.0
    )
    return TreatmentResult(
        treatment=treatment,
        n_games=n_games,
        n_plays_team0=n_plays_team0,
        team0_wins=wins,
        team0_ties=ties,
        team0_losses=losses,
        team0_passouts=passouts,
        team0_win_rate=wilson.rate,
        team0_wilson_lower=wilson.lower,
        team0_wilson_upper=wilson.upper,
        partner_mi_point=partner_ci.point,
        partner_mi_lower=partner_ci.lower,
        partner_mi_upper=partner_ci.upper,
        own_mi_point=own_mi,
        signals_per_game=total_signals / n_games if n_games else 0.0,
        total_signals=total_signals,
        listener_plays_total=listener_plays_total,
        listener_plays_signal_conditioned=listener_conditioned_total,
        listener_signal_conditioned_fraction=conditioned_frac,
    )


def run_signalling(
    *,
    n_games: int = 200,
    seed: int = 0,
    n_resamples: int = 500,
    treatments: Sequence[Treatment] = (
        "silent",
        "cooperative",
        "byzantine",
        "closed_loop",
        "closed_loop_search",
    ),
    output_path: Path | None = None,
) -> list[TreatmentResult]:
    """Run all four treatments and (optionally) write JSONL results."""
    results = [
        run_treatment(
            t, n_games=n_games, seed=seed, n_resamples=n_resamples
        )
        for t in treatments
    ]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "config",
                        "n_games": n_games,
                        "seed": seed,
                        "n_resamples": n_resamples,
                    }
                )
                + "\n"
            )
            for r in results:
                f.write(
                    json.dumps({"kind": "treatment", **asdict(r)}) + "\n"
                )
    return results


# ------------------------------------------------------------------ CLI ----


def _format_table(results: Sequence[TreatmentResult]) -> str:
    header = (
        f"{'treatment':<20}{'games':>6}{'wins':>6}{'losses':>7}"
        f"{'win rate (Wilson 95%)':>26}"
        f"{'partner MI [95% CI]':>30}{'sig/game':>10}"
        f"{'listener fired':>22}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        wilson = (
            f"{r.team0_win_rate:.3f} [{r.team0_wilson_lower:.3f},"
            f"{r.team0_wilson_upper:.3f}]"
        )
        mi = (
            f"{r.partner_mi_point:.4f} "
            f"[{r.partner_mi_lower:.4f},{r.partner_mi_upper:.4f}]"
        )
        listener_cell = (
            f"{r.listener_plays_signal_conditioned}/{r.listener_plays_total}"
            f" ({r.listener_signal_conditioned_fraction * 100:.1f}%)"
            if r.listener_plays_total
            else "n/a"
        )
        lines.append(
            f"{r.treatment:<20}{r.n_games:>6}{r.team0_wins:>6}"
            f"{r.team0_losses:>7}{wilson:>26}{mi:>30}"
            f"{r.signals_per_game:>10.2f}"
            f"{listener_cell:>22}"
        )
    return "\n".join(lines)


def _default_output() -> Path:
    return Path(__file__).resolve().parent / "results" / "signalling.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test §6 predictions #4 and #5 (structured signalling)."
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="JSONL output path (default: experiments/results/signalling.jsonl)",
    )
    args = parser.parse_args(argv)

    results = run_signalling(
        n_games=args.games,
        seed=args.seed,
        n_resamples=args.n_resamples,
        output_path=args.output,
    )
    print(_format_table(results))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
