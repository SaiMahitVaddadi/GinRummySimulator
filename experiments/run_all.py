"""Thin dispatcher that runs every experiment in the ``experiments/`` folder.

Currently wired:

* ``matched_arm`` — §5 Classic Gin table.

Placeholders for peer experiments (add these when they land):

* # TODO: ``euchre_coordination`` — §5 Euchre row + §6 prediction 4.
* # TODO: ``cfr_minigin_exploitability`` — §6 prediction 3.
* # TODO: ``llm_tool_ablation`` — §6 prediction 2.
* # TODO: ``cot_fingerprint_audit`` — §6 prediction 6.

Usage
-----
CLI:  ``python -m experiments.run_all --games-per-pair 500 --seed 0``
"""

from __future__ import annotations

import argparse
import sys

from experiments.matched_arm import run_matched_arm


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.run_all",
        description="Run every registered experiment end-to-end.",
    )
    p.add_argument("--games-per-pair", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated experiment names to run (default: all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    registry = {
        "matched_arm": lambda: run_matched_arm(
            games_per_pair=args.games_per_pair,
            seed=args.seed,
            out_dir=args.out_dir,
        ),
        # TODO: "euchre_coordination": lambda: ...
        # TODO: "cfr_minigin_exploitability": lambda: ...
        # TODO: "llm_tool_ablation": lambda: ...
        # TODO: "cot_fingerprint_audit": lambda: ...
    }

    failures: list[tuple[str, BaseException]] = []
    for name, fn in registry.items():
        if only is not None and name not in only:
            continue
        print(f"\n===== running experiment: {name} =====", file=sys.stderr)
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — surface everything
            failures.append((name, exc))
            print(f"[run_all] {name} raised {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n[run_all] {len(failures)} experiment(s) failed.", file=sys.stderr)
        return 1
    print("\n[run_all] all experiments completed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
