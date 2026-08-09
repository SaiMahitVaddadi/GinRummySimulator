"""Matched-arm §5 experiment: paired round-robin across policy families.

This script populates the Classic Gin row of the ``PAPER.md`` §5 table
with actual numbers. It runs a paired round-robin tournament between:

* ``random``     — the ``RandomPolicy`` baseline.
* ``greedy``     — the ``GreedyKnockPolicy`` gold-standard heuristic.
* ``vote3``      — plurality-vote ``VotingEnsemble`` of three ``GreedyKnockPolicy``
                   experts seeded independently.
* ``phase_moe``  — ``PhaseGatedMoE`` routing early / mid / late-turn decisions
                   to three independently-seeded ``GreedyKnockPolicy`` experts.
                   Runs the router even though every arm is greedy; this
                   isolates routing cost from expert diversity.
* ``hand_moe``   — ``HandStrengthGatedMoE`` (careful=greedy, cheap=random) so
                   the router only invokes greedy when the hand can plausibly
                   be shaped by careful play.
* ``llm``        — placeholder ``LLMPolicy`` entry. Skipped cleanly when no
                   ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``OPENROUTER_API_KEY``
                   is set. Never issues a real API call from tests.

Outputs land in ``experiments/results/`` by default:

* ``matched_arm.jsonl`` — one summary line + one line per match.
* ``matched_arm.txt``   — human-readable ``print_tournament`` capture.

Usage
-----
CLI:      ``python -m experiments.matched_arm --games-per-pair 200 --seed 0``
Library:  ``from experiments.matched_arm import run_matched_arm``
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gin_rummy.eval.ensemble import (
    HandStrengthGatedMoE,
    PhaseGatedMoE,
    VotingEnsemble,
)
from gin_rummy.eval.reporters import export_jsonl, print_tournament
from gin_rummy.eval.tournament import PolicyEntry, Tournament, TournamentResult
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policy import RandomPolicy
from gin_rummy.variants.classic import ClassicGin


class SkipError(RuntimeError):
    """Raised by a policy factory when a required credential is missing.

    The tournament driver treats a ``SkipError`` at build-time as an
    instruction to omit the entry from the round-robin rather than
    crash. See ``_filter_entries``.
    """


# ---------- policy factories ----------


def _random_builder(rng: random.Random) -> RandomPolicy:
    return RandomPolicy(rng)


def _greedy_builder(rng: random.Random) -> GreedyKnockPolicy:
    return GreedyKnockPolicy(rng)


def _vote3_builder(rng: random.Random) -> VotingEnsemble:
    # Three independently seeded greedy experts. Because greedy is
    # near-deterministic, most votes will agree — but the ``VotingEnsemble``
    # path is still exercised so its cost telemetry is realistic.
    experts = [
        GreedyKnockPolicy(random.Random(rng.randrange(1 << 30))) for _ in range(3)
    ]
    return VotingEnsemble(experts=experts, name="vote3")


def _phase_moe_builder(rng: random.Random) -> PhaseGatedMoE:
    early = GreedyKnockPolicy(random.Random(rng.randrange(1 << 30)))
    mid = GreedyKnockPolicy(random.Random(rng.randrange(1 << 30)))
    late = GreedyKnockPolicy(random.Random(rng.randrange(1 << 30)))
    return PhaseGatedMoE(early=early, mid=mid, late=late, name="phase_moe")


def _hand_moe_builder(rng: random.Random) -> HandStrengthGatedMoE:
    return HandStrengthGatedMoE(
        cheap=RandomPolicy(random.Random(rng.randrange(1 << 30))),
        careful=GreedyKnockPolicy(random.Random(rng.randrange(1 << 30))),
        deadwood_threshold=25,
        name="hand_moe",
    )


def _llm_builder(rng: random.Random):
    """Placeholder LLM builder — raises ``SkipError`` when no API key is set.

    We deliberately do NOT construct a live ``LLMPolicy`` here even when a
    key is present, to keep the experiment offline-friendly. Users who want
    to enable this arm should replace this factory with one that supplies
    a ``completion_fn`` or a live LiteLLM model string.
    """
    if not _has_llm_credentials():
        raise SkipError(
            "no LLM credentials in environment "
            "(set OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY "
            "and edit experiments/matched_arm.py:_llm_builder to enable)"
        )
    raise SkipError(
        "LLM arm is credentialed but disabled by default to avoid API spend; "
        "edit experiments/matched_arm.py:_llm_builder to enable"
    )


def _has_llm_credentials() -> bool:
    return any(
        os.environ.get(k)
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
    )


# ---------- entry catalogue ----------


@dataclass(frozen=True)
class CandidateEntry:
    name: str
    builder: Callable[[random.Random], object]


CANDIDATES: tuple[CandidateEntry, ...] = (
    CandidateEntry("random", _random_builder),
    CandidateEntry("greedy", _greedy_builder),
    CandidateEntry("vote3", _vote3_builder),
    CandidateEntry("phase_moe", _phase_moe_builder),
    CandidateEntry("hand_moe", _hand_moe_builder),
    CandidateEntry("llm", _llm_builder),
)


def _filter_entries(
    candidates: tuple[CandidateEntry, ...],
) -> tuple[list[PolicyEntry], list[tuple[str, str]]]:
    """Instantiate each candidate once with a probe RNG to detect ``SkipError``.

    Returns
    -------
    entries : list[PolicyEntry]
        Only the candidates whose probe-build succeeded.
    skipped : list[(name, reason)]
        The candidates that raised ``SkipError`` during the probe.
    """
    entries: list[PolicyEntry] = []
    skipped: list[tuple[str, str]] = []
    probe_rng = random.Random(0xC0FFEE)
    for c in candidates:
        try:
            c.builder(probe_rng)
        except SkipError as exc:
            skipped.append((c.name, str(exc)))
            continue
        entries.append(PolicyEntry(name=c.name, build=c.builder))
    return entries, skipped


# ---------- driver ----------


def run_matched_arm(
    games_per_pair: int = 500,
    seed: int = 0,
    out_dir: str | Path = "experiments/results",
) -> TournamentResult:
    """Run the matched-arm round-robin and write result artefacts.

    Parameters
    ----------
    games_per_pair : int
        Number of games per pairing. Paired mode requires this be even; if an
        odd value is supplied it is rounded up by 1 with a warning.
    seed : int
        Base seed for the tournament driver. Deterministic given ``paired`` +
        the same policy-build closures.
    out_dir : str | Path
        Where to write ``matched_arm.jsonl`` and ``matched_arm.txt``.

    Returns
    -------
    TournamentResult
    """
    if games_per_pair % 2:
        print(
            f"[matched_arm] warning: games_per_pair={games_per_pair} is odd; "
            "bumping to make paired mode happy.",
            file=sys.stderr,
        )
        games_per_pair += 1

    entries, skipped = _filter_entries(CANDIDATES)
    if len(entries) < 2:
        raise RuntimeError(
            f"Need at least 2 runnable policies; only {len(entries)} survived filtering."
        )

    if skipped:
        print("[matched_arm] skipped entries:", file=sys.stderr)
        for name, reason in skipped:
            print(f"  - {name}: {reason}", file=sys.stderr)

    tournament = Tournament(
        game_cls=ClassicGin,
        entries=entries,
        games_per_pair=games_per_pair,
        seed=seed,
        paired=True,
    )
    result = tournament.run()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_path / "matched_arm.jsonl"
    export_jsonl(result, jsonl_path)

    # Capture print_tournament so we can both echo to stdout AND persist a
    # plain-text summary suitable for pasting into the paper draft.
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tournament(result)
    summary_text = buf.getvalue()

    txt_path = out_path / "matched_arm.txt"
    header_lines = [
        f"# Matched-arm experiment (games_per_pair={games_per_pair}, seed={seed})",
        f"# Policies included: {', '.join(e.name for e in entries)}",
    ]
    if skipped:
        header_lines.append("# Skipped: " + "; ".join(f"{n} ({r})" for n, r in skipped))
    txt_path.write_text("\n".join(header_lines) + "\n" + summary_text)

    # Echo to real stdout for the operator.
    print(summary_text)
    print(f"[matched_arm] wrote {jsonl_path}")
    print(f"[matched_arm] wrote {txt_path}")

    return result


# ---------- CLI ----------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.matched_arm",
        description="Populate the §5 matched-arm table for Classic Gin.",
    )
    p.add_argument(
        "--games-per-pair",
        type=int,
        default=500,
        help="Games per policy pairing (paired mode; must be even). Default 500.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Base seed for reproducibility."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments/results",
        help="Directory to write matched_arm.{jsonl,txt}.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    run_matched_arm(
        games_per_pair=args.games_per_pair, seed=args.seed, out_dir=args.out_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
