"""Smoke tests for the ``experiments/`` scripts.

These deliberately use a tiny ``games_per_pair`` so the whole file runs
in well under 30 s. Full-resolution runs live under
``experiments/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.matched_arm import (
    CANDIDATES,
    SkipError,
    _filter_entries,
    run_matched_arm,
)


def test_llm_builder_skips_when_no_credentials(monkeypatch):
    """The LLM arm must skip cleanly rather than crash the tournament when
    no API key is present. We clear the keys explicitly so the test is
    stable regardless of the developer's shell environment."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    entries, skipped = _filter_entries(CANDIDATES)
    skipped_names = {name for name, _reason in skipped}
    assert "llm" in skipped_names
    assert all(e.name != "llm" for e in entries)
    # We still expect the offline arms to survive.
    assert {"random", "greedy", "vote3", "phase_moe", "hand_moe"} <= {
        e.name for e in entries
    }


def test_llm_builder_raises_skip_error_directly():
    """Direct invocation should raise ``SkipError`` — not ``KeyError`` or
    ``RuntimeError`` — so the filter can catch it precisely."""
    import random

    from experiments.matched_arm import _llm_builder

    try:
        _llm_builder(random.Random(0))
    except SkipError:
        return
    # If we got here without SkipError the contract is broken. We do not
    # assert False directly because the placeholder may be swapped for a
    # real LLM once someone wires up credentials; in that case the test
    # should still be updated deliberately.
    raise AssertionError("expected SkipError from the disabled-by-default LLM stub")


def test_run_matched_arm_smoke(tmp_path: Path, monkeypatch):
    """End-to-end smoke: run 6 games per pair and assert both artefacts
    materialise with the shape ``print_tournament`` / ``export_jsonl``
    guarantee."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = run_matched_arm(games_per_pair=6, seed=0, out_dir=tmp_path)

    jsonl_path = tmp_path / "matched_arm.jsonl"
    txt_path = tmp_path / "matched_arm.txt"
    assert jsonl_path.exists(), "matched_arm.jsonl was not written"
    assert txt_path.exists(), "matched_arm.txt was not written"

    lines = jsonl_path.read_text().splitlines()
    assert len(lines) >= 2, "expected at least a summary line + one match"
    summary = json.loads(lines[0])
    assert summary["kind"] == "summary"
    assert summary["variant"] == "ClassicGin"
    # 5 offline policies (LLM skipped) → C(5, 2) = 10 pairings × 6 games = 60 matches.
    assert summary["num_matches"] == len(result.matches) == 60

    txt = txt_path.read_text()
    assert "Tournament: ClassicGin" in txt
    assert "Cross-play win rates" in txt
    assert "greedy" in txt and "random" in txt
