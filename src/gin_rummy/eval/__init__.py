"""Evaluation and benchmarking framework.

Layered modules:

* ``stats``       — Wilson / bootstrap CIs, IQM, Holm–Bonferroni, paired sign test.
* ``telemetry``   — per-decision latency / tokens / parse-fail / fallback rate.
* ``tournament``  — N-way round-robin with cross-play matrix and telemetry roll-up.
* ``rating``      — Bradley–Terry MLE + bootstrap-CI Elo ratings.
* ``kfold``       — opponent-pool cross-validation and deal-seed splits.
* ``ensemble``    — voting, confidence-weighted, phase-gated, hand-strength-gated, seed-averaged.
* ``xplain``      — hand introspection, discard regret, reward channels, behavioral fingerprints, rollout regret.
* ``reporters``   — text summary and JSONL export for reproducibility.
"""

from gin_rummy.eval.ablation import (
    AblationResult,
    AssignmentRun,
    Factor,
    FactorAssignment,
    FactorialDesign,
    FractionalFactorialDesign,
    InteractionEffect,
    MainEffect,
    analysis,
    run_ablation,
)
from gin_rummy.eval.ensemble import (
    ConfidenceWeightedEnsemble,
    HandStrengthGatedMoE,
    PhaseGatedMoE,
    VotingEnsemble,
    seed_ensemble,
)
from gin_rummy.eval.kfold import KFoldReport, deal_kfold, opponent_kfold
from gin_rummy.eval.rating import RatingCI, bradley_terry_ratings, rating_ci_table
from gin_rummy.eval.reporters import export_jsonl, print_tournament
from gin_rummy.eval.stats import (
    BootstrapCI,
    ProportionCI,
    bootstrap_ci,
    holm_bonferroni,
    iqm,
    paired_sign_test,
    wilson_ci,
)
from gin_rummy.eval.telemetry import DecisionRecord, PolicyTelemetry, TelemetryPolicy
from gin_rummy.eval.tournament import (
    MatchOutcome,
    PolicyEntry,
    Tournament,
    TournamentResult,
)
from gin_rummy.eval.xplain import (
    BehavioralFingerprint,
    ChannelContribution,
    HandIntrospection,
    discard_regret_table,
    fingerprint_from_history,
    introspect_hand,
    reward_channels,
    rollout_regret,
)

__all__ = [
    "AblationResult",
    "AssignmentRun",
    "BehavioralFingerprint",
    "BootstrapCI",
    "ChannelContribution",
    "ConfidenceWeightedEnsemble",
    "DecisionRecord",
    "Factor",
    "FactorAssignment",
    "FactorialDesign",
    "FractionalFactorialDesign",
    "HandIntrospection",
    "HandStrengthGatedMoE",
    "InteractionEffect",
    "KFoldReport",
    "MainEffect",
    "MatchOutcome",
    "PhaseGatedMoE",
    "PolicyEntry",
    "PolicyTelemetry",
    "ProportionCI",
    "RatingCI",
    "TelemetryPolicy",
    "Tournament",
    "TournamentResult",
    "VotingEnsemble",
    "analysis",
    "bootstrap_ci",
    "bradley_terry_ratings",
    "deal_kfold",
    "discard_regret_table",
    "export_jsonl",
    "fingerprint_from_history",
    "holm_bonferroni",
    "introspect_hand",
    "iqm",
    "opponent_kfold",
    "paired_sign_test",
    "print_tournament",
    "rating_ci_table",
    "reward_channels",
    "rollout_regret",
    "run_ablation",
    "seed_ensemble",
    "wilson_ci",
]
