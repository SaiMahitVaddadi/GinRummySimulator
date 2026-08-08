"""Human-readable printers and JSONL export for tournament results."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import TextIO

from gin_rummy.eval.rating import RatingCI, rating_ci_table
from gin_rummy.eval.tournament import TournamentResult


def _matches_for_rating(result: TournamentResult) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for m in result.matches:
        if m.winner_seat is None:
            continue
        winner = m.seat_names[m.winner_seat]
        loser = m.seat_names[1 - m.winner_seat]  # 2-seat only
        triples.append((winner, loser, str(m.seed)))
    return triples


def print_tournament(result: TournamentResult, *, level: float = 0.95) -> None:
    print(f"\n=== Tournament: {result.variant} · {len(result.matches)} matches ===")
    print(f"Policies: {', '.join(result.policy_names)}")
    print(f"Games per pair: {result.games_per_pair}  Paired: {result.paired}")
    print()

    # Cross-play matrix
    print(f"Cross-play win rates ({int(level * 100)}% Wilson CI):")
    names = result.policy_names
    if len(names) == 2:
        a, b = names
        a_wins, b_wins, draws = result.wins_between(a, b)
        matrix = result.crossplay_matrix(level=level)
        print(f"  {a} vs {b}: {matrix[(a, b)].rate * 100:6.2f}%  "
              f"[{matrix[(a, b)].lower * 100:5.2f}, {matrix[(a, b)].upper * 100:5.2f}]  "
              f"({a_wins}-{b_wins}, {draws} draws)")
    else:
        matrix = result.crossplay_matrix(level=level)
        width = max(len(n) for n in names)
        header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in names)
        print(header)
        for a in names:
            row = [f"{a:>{width}}: "]
            for b in names:
                if a == b:
                    row.append(f"{'—':>{width}}")
                else:
                    row.append(f"{matrix[(a, b)].rate * 100:>{width - 1}.1f}%")
            print("  ".join(row))
    print()

    # Bradley–Terry ratings
    triples = _matches_for_rating(result)
    if triples:
        ratings = rating_ci_table(triples, level=level)
        print(f"Bradley–Terry Elo ({int(level * 100)}% bootstrap CI, anchor 1500):")
        for r in ratings:
            print(f"  {r.name:<20}  {r.elo:7.1f}  [{r.lower:7.1f}, {r.upper:7.1f}]")
        print()

    # Outcome distribution
    dist = result.outcome_distribution()
    total = sum(dist.values())
    if total:
        print("Outcome distribution:")
        for outcome, count in dist.most_common():
            print(f"  {outcome:<10}  {count / total * 100:5.2f}%  ({count})")
        print()

    # Turn / deadwood stats
    if result.matches:
        turns_ci = result.turns_ci(level=level)
        print(f"Turns:            mean={turns_ci.point:5.1f}  "
              f"[{turns_ci.lower:5.1f}, {turns_ci.upper:5.1f}]")
        dw_ci = result.deadwood_ci(level=level)
        if dw_ci.point:
            print(f"Loser deadwood:   mean={dw_ci.point:5.1f}  "
                  f"[{dw_ci.lower:5.1f}, {dw_ci.upper:5.1f}]")
    print()

    # Telemetry per policy
    if any(t.n for t in result.telemetry.values()):
        print("Per-policy telemetry:")
        print(f"  {'name':<20}  {'decisions':>9}  {'mean ms':>9}  {'p95 ms':>9}  "
              f"{'parse%':>7}  {'fallback%':>10}  {'tok in':>7}  {'tok out':>8}")
        for name in result.policy_names:
            t = result.telemetry[name]
            print(
                f"  {name:<20}  {t.n:>9}  {t.mean_latency_ms:>9.3f}  {t.p95_latency_ms:>9.3f}  "
                f"{t.parse_fail_rate * 100:>6.2f}%  {t.fallback_rate * 100:>9.2f}%  "
                f"{t.total_tokens_in:>7}  {t.total_tokens_out:>8}"
            )
        print()


def export_jsonl(result: TournamentResult, path: str | Path, *, level: float = 0.95) -> None:
    """One line per match + summary lines. Enables post-hoc re-analysis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        _write_summary(result, fh, level=level)
        for m in result.matches:
            fh.write(json.dumps({"kind": "match", **asdict(m)}) + "\n")


def _write_summary(result: TournamentResult, fh: TextIO, *, level: float) -> None:
    matrix = result.crossplay_matrix(level=level)
    summary = {
        "kind": "summary",
        "variant": result.variant,
        "policies": result.policy_names,
        "games_per_pair": result.games_per_pair,
        "paired": result.paired,
        "num_matches": len(result.matches),
        "outcome_distribution": dict(result.outcome_distribution()),
        "crossplay": [
            {"row": a, "col": b, "rate": ci.rate, "lower": ci.lower, "upper": ci.upper}
            for a, b in matrix
            for ci in [matrix[(a, b)]]
        ],
        "telemetry": {
            name: {
                "n": t.n,
                "mean_latency_ms": t.mean_latency_ms,
                "p95_latency_ms": t.p95_latency_ms,
                "parse_fail_rate": t.parse_fail_rate,
                "fallback_rate": t.fallback_rate,
                "total_tokens_in": t.total_tokens_in,
                "total_tokens_out": t.total_tokens_out,
            }
            for name, t in result.telemetry.items()
        },
    }
    triples = _matches_for_rating(result)
    if triples:
        ratings = rating_ci_table(triples, level=level)
        summary["bradley_terry"] = [
            {"name": r.name, "elo": r.elo, "lower": r.lower, "upper": r.upper}
            for r in ratings
        ]
    fh.write(json.dumps(summary) + "\n")
