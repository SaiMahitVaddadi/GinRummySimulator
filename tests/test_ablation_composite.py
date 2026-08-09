"""Smoke tests for the composite (2^4) ablation experiment.

Kept intentionally tiny (``games_per_pair=4``) so the test runs in a
couple of seconds even on CI. Larger-N verification happens by running
``python -m experiments.ablation_composite`` directly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.ablation_composite import (
    FACTORS,
    _build_composite,
    run_composite_ablation,
)
from gin_rummy.eval.ensemble import PhaseGatedMoE, VotingEnsemble
from gin_rummy.policies.heuristic import ComposableHeuristicPolicy


# ---------- builder unit tests ---------------------------------------------


def test_factor_names_match_task_spec():
    names = [f.name for f in FACTORS]
    assert names == ["discard", "knock", "wrapper", "moe"]
    for f in FACTORS:
        assert len(f.levels) == 2, f"factor {f.name} must be 2-level"


def test_build_bare_flat_is_plain_heuristic():
    p = _build_composite(
        {"discard": "highest_deadwood", "knock": "always",
         "wrapper": "bare", "moe": "flat"},
        random.Random(0),
    )
    assert isinstance(p, ComposableHeuristicPolicy)


def test_build_voting3_wraps_ensemble():
    p = _build_composite(
        {"discard": "highest_deadwood", "knock": "always",
         "wrapper": "voting3", "moe": "flat"},
        random.Random(0),
    )
    assert isinstance(p, VotingEnsemble)
    assert len(list(p.experts)) == 3


def test_build_phase_gated_wraps_moe():
    p = _build_composite(
        {"discard": "highest_deadwood", "knock": "always",
         "wrapper": "bare", "moe": "phase_gated"},
        random.Random(0),
    )
    assert isinstance(p, PhaseGatedMoE)


def test_build_voting3_of_phase_gated_stacks_correctly():
    p = _build_composite(
        {"discard": "highest_deadwood", "knock": "always",
         "wrapper": "voting3", "moe": "phase_gated"},
        random.Random(0),
    )
    assert isinstance(p, VotingEnsemble)
    for expert in p.experts:
        assert isinstance(expert, PhaseGatedMoE)


# ---------- end-to-end smoke ------------------------------------------------


def test_run_composite_ablation_smoke(tmp_path: Path):
    results = run_composite_ablation(
        games_per_pair=4,
        seed=0,
        out_dir=tmp_path,
        also_fractional=True,
    )
    assert set(results) == {"full", "frac"}

    full = results["full"]
    frac = results["frac"]

    # 2^4 = 16 cells for full, 2^{4-1} = 8 for the half-fraction.
    assert len(full.runs) == 16
    assert len(frac.runs) == 8

    # All four factors have both levels populated in the main-effects table.
    for factor in FACTORS:
        assert factor.name in full.main_effects
        assert set(full.main_effects[factor.name].keys()) == set(factor.levels)

    # C(4, 2) = 6 pairwise interactions.
    assert len(full.interaction_effects) == 6

    # JSONL and TXT outputs exist and are non-trivial.
    for stem in ("ablation_composite_full", "ablation_composite_frac"):
        jsonl = tmp_path / f"{stem}.jsonl"
        txt = tmp_path / f"{stem}.txt"
        assert jsonl.exists()
        assert txt.exists()
        # First line is a valid summary record.
        first = jsonl.read_text().splitlines()[0]
        summary = json.loads(first)
        assert summary["kind"] == "summary"
        assert "factors" in summary
        # Text output includes both tables.
        text = txt.read_text()
        assert "Main effects" in text
        assert "Pairwise interactions" in text or "no interaction" in text


def test_full_and_fractional_agree_on_dominant_factor_sign_smoke(tmp_path: Path):
    """At tiny N the *dominant* factor's sign should still agree between
    the full and fractional runs. This is a weak but non-vacuous check
    that the fractional design isn't mis-aliased."""
    results = run_composite_ablation(
        games_per_pair=4,
        seed=1,
        out_dir=tmp_path,
        also_fractional=True,
    )
    full = results["full"]
    frac = results["frac"]
    # Take the top-ranked factor from the full run and confirm its
    # direction (which level wins) matches in the fractional run.
    if not full.ranking:
        pytest.skip("empty ranking (should not happen)")
    top_factor = full.ranking[0][0]
    lvl0, lvl1 = [f for f in FACTORS if f.name == top_factor][0].levels
    full_delta = (
        full.main_effects[top_factor][lvl1].marginal_win_rate_incl_draws
        - full.main_effects[top_factor][lvl0].marginal_win_rate_incl_draws
    )
    frac_delta = (
        frac.main_effects[top_factor][lvl1].marginal_win_rate_incl_draws
        - frac.main_effects[top_factor][lvl0].marginal_win_rate_incl_draws
    )
    # Signs should agree (product >= 0). We allow exact-zero deltas since
    # at N=4 ties are possible.
    assert full_delta * frac_delta >= 0.0
