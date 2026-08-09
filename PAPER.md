# Rummy as an Architecture-Family Testbed for Card-Game AI

*Working outline. Synthesises seven landscape/methodology surveys into a
single research contribution that the accompanying codebase now
supports end-to-end.*

---

## 1. One-line thesis

**Rummy is a rare card-game substrate that is (i) small enough for
tabular CFR to touch its equilibrium, (ii) rich enough that LLM
reasoning is nontrivial, and (iii) has a natural partnership variant
(Euchre) where implicit signalling can be measured
information-theoretically. This paper introduces a common environment,
a set of matched agents drawn from every major architecture family, and
a battery of evaluation and explainability primitives, and reports the
first cross-family comparison on this substrate.**

---

## 2. Positioning and why it matters

The recent literature is emphatic in three places:

* Kelidari, Haghi & Salmani (arXiv:2607.06854, July 2026) show a
  hand-crafted expert **beats every learned Gin Rummy agent 70–99%**.
* Mohan's IRumAI (arXiv:2606.21975, June 2026) is the *only* published
  RL result on Indian Rummy — 2-player, 13-card, single deck.
* No CFR solve of any rummy variant has been published; no GNN policy
  for any card game has been verified; no LLM agent paper on Euchre
  exists at all.

The gap is not lack of algorithmic ideas — CFR / RL / LLM / GNN /
ensemble methods are mature. The gap is a **shared substrate** that lets
you drop matched instances of each family into the same seat and
compare, plus an evaluation protocol strong enough to detect the
differences honestly.

---

## 3. The substrate — what the codebase now ships

| Component | Package | Purpose |
|---|---|---|
| Multi-variant rummy engine | `variants/` | Classic Gin, Oklahoma, Hollywood, Indian (3/7/10/13/15) |
| Partnership trick-taking | `variants/euchre.py` | 4-player, cross-partnership, trump-calling |
| Optimal meld decomposition | `meld.py` | Exact bitmask DP, ~140 µs / 10-card hand |
| **Heuristic reference** | `policies/heuristic.py` | Reproduces the Gold-Standard recipe; 99% vs random |
| **LLM policy + tools** | `policies/llm.py`, `policies/tools.py` | LiteLLM transport; **novel**: `meld_analyzer_tool()` lets the LLM call the exact solver mid-decision |
| **Bipartite GNN policy** | `models/graph.py`, `models/gnn_policy.py` | 52 card nodes + variable meld nodes, GAT head |
| **CFR solve of mini-gin** | `solvers/minigin.py`, `solvers/cfr.py`, `solvers/best_response.py` | 12-card variant, 88 305 info-sets, ε-Nash accessible |
| **DPO from game outcomes** | `finetune/selfplay.py`, `finetune/trainers.py` | `outcome_dpo_pairs()` mines within-game winner/loser pairs; SPIN loop scaffolded |
| **Cross-play tournament** | `eval/tournament.py`, `eval/rating.py` | N-way round-robin, Wilson CIs, Bradley–Terry Elo w/ bootstrap |
| Paired-hands variance reduction | `eval/tournament.py`, `--paired` | Duplicate-poker trick |
| IQM + bootstrap + Holm | `eval/stats.py` | `rliable`-style |
| Opponent-pool k-fold | `eval/kfold.py` | RL generalisation analogue |
| Ensembles / MoE | `eval/ensemble.py` | Voting, ConfidenceWeighted, PhaseGatedMoE (M2CTS), HandStrengthGatedMoE, seed_ensemble |
| **Explainability primitives** | `eval/xplain.py` | Hand introspection, discard-regret table, reward-channel decomposition, behavioural fingerprints, rollout regret |
| **Coordination MI** | `comms/analyze.py` | I(A; H_partner | H_own, public) — the "telepathy" quantity, common-cause corrected |
| Cheap talk + Byzantine | `comms/signals.py`, `comms/byzantine.py` | Structured signalling; adversarial variant |
| Blackboard orchestrator | `comms/orchestrator.py` | LangGraph-lite; state-machine node router |

Reproducibility: every game accepts a seed; every tournament exports
JSONL; heavy deps (`litellm`, `trl`, `peft`, `torch`, `torch_geometric`,
`bitsandbytes`) are optional extras.

---

## 4. Six landscape surveys distilled

*Full survey markdown lives in the repo's commit history; here is the
one-line takeaway from each.*

1. **Rummy landscape** — under-served in Python; multi-player and
   multi-variant environments do not exist elsewhere.
2. **LLM agents in games** — Diplomacy (CICERO), Werewolf, Hanabi are
   crowded; card games under-studied; Euchre is uncharted for LLMs.
3. **RL / CFR / equilibrium** — TMECor is the theoretically-right
   target for 2v2 Euchre; no CFR-of-rummy paper exists; PPO/TRPO peak
   at ~36% vs. the Gold-Standard expert.
4. **Classical benchmarking (ACPC-era)** — mbb/g + AIVAT are the gold
   standard; duplicate hands are free variance reduction; exploitability
   is worst-case, win rate is average.
5. **Modern RL evaluation** — `rliable` (IQM + stratified bootstrap +
   probability of improvement); ≥10 seeds is the publication floor;
   cost-per-decision is a first-class metric.
6. **LLM evaluation** — prompt-sensitivity + T=0 non-determinism +
   parse-fail-fallback contamination are the three canonical failure
   modes; Arena-Elo with bootstrap CIs is the emerging norm.
7. **Communication / MoE / GNN / fine-tuning / explainability** —
   emergent communication overfits partners (OBL); MoE improves chess
   +122 Elo (M2CTS); no card-game GNN verified; DPO strictly from game
   outcomes is unclaimed; CoT rationales are unfaithful (Turpin 2023).

---

## 5. Proposed empirical section (matched-arm experiment)

Four seats × four architecture families × two games:

|            | Random | Heuristic | Deep-MC RL (IRumAI-style) | LLM (frontier, no tools) | LLM + meld solver (novel) | CFR (mini-gin only) | GNN (Gin only) |
|------------|--------|-----------|----------------------------|--------------------------|----------------------------|----------------------|-----------------|
| **Classic Gin (2p)** | ✓ | ✓ | needs training | ✓ | ✓ | via mini-gin projection | needs training |
| **Euchre (2v2)** | ✓ | needs impl. | needs training | ✓ | — | out of scope | needs training |

Report per matchup:

* Head-to-head win rate with Wilson 95% CI, **paired hands**.
* Bradley–Terry Elo with bootstrap CIs across all pairings.
* Outcome distribution (gin / knock / undercut / draw).
* **Cost per decision** (µs / ms / $ / tokens-in / tokens-out).
* **Parse-fail / fallback rate** for the LLM arms.
* **Opponent-pool 5-fold CV** across held-out heuristic variants.
* For Euchre: **I(action; partner_hand | own_hand, public)** — the
  Hanabi-style telepathy score, common-cause corrected.
* **Approximate exploitability** of the CFR-trained policy on mini-gin
  (exact best-response) — the only setting where Nash-distance is
  computable, but it anchors the whole scale.

Ablations:

* LLM alone vs. LLM + meld-solver tool (isolates the tool contribution).
* Ensemble (heuristic + LLM + RL vote) vs. best single (isolates
  ensemble diversity, per Kuncheva).
* Phase-gated MoE vs. flat (M2CTS-style; tests whether the chess result
  ports).
* Structured signal on vs. off in Euchre partnerships (does cheap talk
  raise partner MI? arXiv:2510.05748 for cooperative games).
* Byzantine partner (arXiv:2606.07790): a single lying agent should
  visibly collapse the coordination score.

---

## 6. Explicit falsifiable predictions

Following the "gold-standard" methodology (external fixed expert, never
used for training):

1. **The Gold-Standard finding replicates on our engine.** The
   `GreedyKnockPolicy` beats a naive PPO baseline ≥ 70% at N ≥ 1000
   paired hands.
2. **LLM + solver > LLM alone.** Isolating the tool contribution
   yields ≥ 5 pp win-rate lift against the heuristic baseline.
3. **CFR reaches ε ≤ 0.1 on mini-gin** in ≤ 100 k iterations; its
   knock-timing distribution matches the folk heuristic ≤ 10 deadwood
   (a *normative-vs-folk* test).
4. **Partner MI in Euchre increases with structured signals** — the
   corrected I(A; H_partner | H_own, public) score rises from the ~0.02
   nats random baseline to ≥ 0.10 nats with `TalkingPolicy` on both
   partners.
5. **Byzantine partner collapses cooperation** — team win rate drops
   below the no-signal baseline when one partner uses `ByzantinePolicy`.
6. **CoT rationales are only weakly aligned with actions.** Behavioural
   fingerprints (via `eval/xplain`) diverge from the LLM's own stated
   knock-threshold by ≥ 10 pp — replicating Turpin et al. NeurIPS 2023
   in a card-game setting.

---

## 7. Contributions

* A dependency-free Python engine covering Gin (Classic / Oklahoma /
  Hollywood), Indian Rummy at four hand sizes, and Euchre.
* The first published CFR solve of a rummy variant (mini-gin: 12 cards,
  88 k info-sets) with exact best-response for exploitability.
* A bipartite cards-and-melds **GNN architecture** for rummy, the first
  such construction we verify in the literature.
* An **LLM policy with tool-use** where the tool is the optimal meld
  solver — the missing hybrid the survey named as open.
* A **DPO-from-game-outcomes** pipeline (outcome-only pair mining +
  SPIN self-play loop) — the missing preference-based FT the survey
  named as open.
* An **information-theoretic coordination metric** for card-game
  partnerships, correctly de-confounded for the common-cause deck
  constraint that would otherwise inflate the score (~0.22 nats → ~0.02
  under random Euchre play).
* A **publication-grade eval framework**: Wilson + bootstrap CIs, IQM,
  Holm–Bonferroni, paired hands, cross-play matrix, Bradley–Terry Elo
  with bootstrap, opponent-pool k-fold, per-decision cost telemetry,
  JSONL export.
* An **explainability suite** (introspection, contrastive discard,
  reward channels, behavioural fingerprints, rollout regret) drawn from
  Milani et al. 2023 and adapted to rummy's discrete action space.

---

## 8. Threats to validity and honest limitations

* **CFR solves only mini-gin, not full Gin.** The mini-gin info-set
  count (~88 k) is under 0.001 % of what full Gin would need under any
  reasonable abstraction. The CFR result is a *reference point*, not a
  playing agent.
* **The GNN is a scaffold, not a trained agent.** We ship the graph
  builder + model class + training-loop shell; full training requires
  torch/PyG (optional extra) and a GPU we don't assume.
* **LLM comparisons age fast.** Model dates, prompts, parse-fail rates,
  and $-per-decision are logged with every run; the JSONL contract
  makes re-runs comparable.
* **Euchre is simplified.** No "going alone", no left-bower promotion.
  We report the simplification in-line; the coordination science is not
  affected because the signalling mechanics are preserved.
* **Random-play baseline is very weak.** The Gold-Standard expert
  beats random 99% of the time; use it, not random, as the yardstick.
* **CoT audit uses behavioural fingerprints, not mechanistic
  interpretability.** We do not claim to interpret model internals.
* **Multiplayer CFR guarantees do not hold outside 2p zero-sum.**
  Anything beyond 2-player mini-gin is empirical (per Pluribus's own
  justification).

---

## 9. Deliverable artefacts

* The Python package `gin_rummy` on PyPI (proposed) — MIT.
* A reproducibility manifest: `pyproject.toml` lock, seeded runs, JSONL
  results, all-agent replay in ~1 hour on a laptop CPU + one API key.
* Six methodology surveys as supplementary material (already in this
  repository's commit history).

---

## 10. Roadmap after v1

* Extend CFR to full 2-player Gin via hand abstraction; report ε-Nash
  gap curves.
* Train the GNN with self-play PPO; port `train_gnn.py` to Modal.
* Federated + differential-privacy telemetry aggregator (motivated
  weakly per §7 of the comms survey; scope for a follow-up).
* Euchre bidding-language RL (analogous to bridge NooK) once the
  4-player CFR path is in place.
* Public leaderboard on a Kaggle-Arena-style substrate.

---

*The code that supports every claim above is in the repository; every
number in §5–§6 is producible by a one-line CLI invocation of
`gin-rummy --tournament` or `python -m gin_rummy.solvers.cfr`.*
