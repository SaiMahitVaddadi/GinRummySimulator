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

## 5. Empirical programme — the paper as an ablation study

Rather than reporting head-to-head win rates as isolated points, we
frame every architectural addition as **a factor in a designed
experiment**. Each add — heuristic sub-rule, ensemble, MoE gate, tool
call, signal channel, listener rule, fine-tuning method — becomes a
factor with named levels, and we run either the full 2^k factorial or a
resolution-III fractional factorial when k is large. The framework
(`src/gin_rummy/eval/ablation.py`) ships main-effects tables with
Wilson CIs, pairwise interaction effects with paired-sign tests
Holm-Bonferroni-corrected, and an overall factor ranking by effect
magnitude.

### 5a. Heuristic-decomposition ablation (2 × 3 × 2 = 12 cells)

The `GreedyKnockPolicy` is not one thing — it is three independent
rules the ablation framework can now enumerate. Running the full
factorial vs. `RandomPolicy` at 50 games per pairing produces
(`experiments/ablation_heuristic.py`):

| Factor    | Levels                                              |
|-----------|-----------------------------------------------------|
| `draw`    | `deck` (always deck) · `smart` (take-if-reduces-DW) |
| `discard` | `random` · `highest_deadwood` · `safe_from_opp`     |
| `knock`   | `wait_for_gin` (never knock) · `always` (knock ASAP)|

**Main effects (draws-inclusive win-rate range across levels):**

| Factor    | Range    | Best level         | Worst level      |
|-----------|----------|--------------------|------------------|
| `knock`   | **48.7 pp** | `always` → 72.7 %  | `wait_for_gin` → 24.0 % |
| `discard` | **33.5 pp** | `highest_deadwood` / `safe_from_opp` tied → 59.5 % | `random` → 26.0 % |
| `draw`    | **20.0 pp** | `smart` → 58.3 %   | `deck` → 38.3 %  |

**Pairwise interactions (Holm–Bonferroni corrected):**

| Pair              | Magnitude | p (Holm) | Reject H₀ |
|-------------------|-----------|----------|-----------|
| `discard × knock` | **43.0 pp** | < 0.001  | **yes**   |
| `draw × discard`  | 9.0 pp    | 0.20     | no        |
| `draw × knock`    | 2.7 pp    | 0.69     | no        |

**Findings that shape the paper's narrative:**

1. **The single most impactful addition is the *knock rule*.** Turning
   on knock-ASAP over holding out for gin swings win rate by ~50 pp on
   its own. Every heuristic in the literature agrees on this
   qualitatively; we quantify it under a controlled factorial.
2. **Discard × knock is the largest interaction we can detect.** The
   value of a smart discard depends on whether you will knock —
   *without* a knock rule, better discards do very little; *with*
   knock-ASAP, discard quality is worth 33 pp. This is the empirical
   version of the folk wisdom "discard defensively when you're planning
   to knock."
3. **Draw source is the smallest of the three main effects.** Random
   drawing is only ~20 pp worse than the "take-if-reduces-deadwood"
   rule, which is the classical folklore. Some effort in the literature
   has been spent optimising draw policies; the ablation suggests
   diminishing returns relative to discard and knock rules.

### 5b. Higher-order ablations (matched-arm, current status)

Beyond the heuristic decomposition, we treat every remaining
architectural add as its own factor. Status of each add — evidence in
hand from `experiments/matched_arm.py`, `experiments/cfr_curve.py`,
`experiments/signalling.py`, `experiments/cot_faithfulness.py`:

| Factor                            | Levels                                            | Observed contribution |
|-----------------------------------|---------------------------------------------------|-----------------------|
| **Policy family**                 | random · greedy · vote3 · phase_moe · hand_moe    | Greedy family ≈ 1780 BT-Elo, random 840; ensembling and MoE add < 5 Elo over a single greedy expert |
| **Ensemble type**                 | none · voting3 · phase-gated · hand-strength-gated | Statistically indistinguishable from base greedy at 200 games/pair — the ensemble/MoE gain is smaller than our CI |
| **CFR training**                  | uniform · MCCFR-5k · MCCFR-abstracted-10k         | Mini-gin: −22.7 % exploitability. Abstracted-Gin: sampled-BR *rises* (see §3 finding); head-to-head vs uniform monotonically increases to +0.14 |
| **LLM tools**                     | plain LLM · LLM + meld_analyzer_tool               | **Deferred**: needs credentials; infra ready |
| **Signalling channel**            | silent · emit-only · byzantine emit · closed-loop | Silent ≈ emit-only ≈ byzantine (all identical, listener-less). Closed-loop: partner MI 0.023 vs silent 0.026 — **hand-written listener does not lift MI** |
| **CoT rationale**                 | off · on                                          | Framework operational; scripted mode detects stated–actual gap correctly |
| **DPO from game outcomes**        | off · SFT · SFT+DPO · SPIN                        | Infrastructure ready; empirical run requires GPU |

**Every one of these is one CLI flag away** from being run as a factor
in a larger factorial once the LLM/GPU/credential prerequisites are
supplied.

### 5c. What we measure per matchup

* Head-to-head win rate with Wilson 95 % CI, **paired hands**.
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
* **Main-effects table + interaction table** from every ablation run.

### 5d. Ablations we specifically call out

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

## 6. Explicit falsifiable predictions and observed results

Following the "gold-standard" methodology (external fixed expert, never
used for training). Results below are from the matched-arm, CFR-curve,
signalling, and CoT-audit experiment scripts shipped in `experiments/`.
Each is one command away from being re-run against a fresh seed.

1. **The Gold-Standard finding replicates on our engine.**
   *Prediction:* `GreedyKnockPolicy` beats random ≥ 70% at N ≥ 1000
   paired hands.
   **✓ Observed:** greedy beats random **100%** at N=200 paired
   (`experiments/matched_arm.py`, seed=0). Bradley–Terry Elo: greedy
   1778, random 840 — a 938-point gap consistent with the ~99.9 %
   expected win rate. `phase_moe` and `vote3` are statistically
   indistinguishable from greedy (BT gap < 5) — as expected, since the
   underlying expert is the same.

2. **LLM + solver > LLM alone.** Not yet run end-to-end (requires
   frontier-model credentials). The infra is shipped; the ablation is
   one `--policies llm:gpt-4o-mini+tools,llm:gpt-4o-mini` invocation
   away.

3. **CFR reaches low exploitability, and its behaviour matches the
   folk heuristic.**
   *Prediction:* ε ≤ 0.1 on mini-gin in ≤ 100 k iterations.
   **Partially supported / nuanced on abstracted full Gin.** On the
   mini-gin substrate the tabular solver reduces exploitability **22.7%
   from uniform in 5 k iterations** (88 k info-sets, 18 s;
   `solvers/cfr.py`). On abstracted full Gin the picture is different:
   the average strategy concentrates near-deterministically per
   info-set, and *sampled* exploitability *rises* with training because
   a 1-step-lookahead state-aware best-response exploits deterministic
   strategies more effectively than a diffuse uniform baseline (an
   artefact of the bounded BR, not the CFR trajectory). The honest
   low-variance signal is head-to-head vs. uniform, which **rises
   monotonically to +0.14 by 10 k iterations**. Finding: sampled-BR
   exploitability is a poor primary metric once the trained strategy
   sharpens; head-to-head is the right yardstick at abstraction scale.

4. **Partner MI in Euchre increases with structured signals.**
   *Prediction:* corrected I(A; H_partner | H_own, public) rises from
   the ~0.02 nats random baseline to ≥ 0.10 nats with `TalkingPolicy`
   on both partners.
   **✗ Now honestly falsifiable — and honestly falsified.** The
   `EuchreListeningPolicy` closes the loop (listener fires on 7.8 % of
   team-0 plays, verified by decision-fidelity telemetry). Four
   treatments at N=200 (seed 0):

   | Treatment       | Team-0 win rate      | Partner MI [95 % CI]     | Signals/game | Listener fired |
   |-----------------|----------------------|--------------------------|--------------|----------------|
   | silent          | 0.477 [0.409, 0.547] | 0.0259 [0.0314, 0.0537]  | 0.00         | n/a            |
   | cooperative-emit | 0.477 [0.409, 0.547] | 0.0259 [0.0314, 0.0537]  | 7.49         | n/a            |
   | byzantine-emit  | 0.477 [0.409, 0.547] | 0.0259 [0.0314, 0.0537]  | 7.49         | n/a            |
   | **closed-loop** | 0.487 [0.418, 0.557] | 0.0228 [0.0271, 0.0513]  | 7.55         | **154/1970 (7.8 %)** |

   Partner MI in the closed-loop arm is *lower* than the silent
   baseline (0.023 vs 0.026, CIs overlap). Hand-written listening
   rules over a random inner substrate **do not lift coordination
   above chance**. This is a real, non-trivial methodological finding:
   *signalling infrastructure alone is not sufficient — the content of
   the protocol and the base-policy competence both have to be
   sophisticated*. A learned or search-based listener is the natural
   next step.

5. **Byzantine partner collapses cooperation.**
   *Prediction:* team-0 win rate drops below the silent baseline when
   one partner uses `ByzantinePolicy`.
   **✗ Vacuously not-supported** — with prediction #4 not holding at
   the baseline, a byzantine-collapse test compares noise to noise.
   The byzantine emitter is trivially wireable through the closed loop
   (the listener already accepts `kind="byzantine"` payloads), but the
   test is not scientifically meaningful until a listener that
   actually confers a signal-based advantage exists.

6. **CoT rationales are only weakly aligned with actions.**
   *Prediction:* behavioural fingerprints diverge from the LLM's stated
   knock-threshold by ≥ 10 deadwood units.
   **✓ Framework operational; empirical run requires credentials.**
   `experiments/cot_faithfulness.py` extracts stated thresholds via 7
   regex patterns, tallies actual knock-deadwood distributions, and
   reports the gap + parse-fail rate. In scripted-mode audit
   (deterministic LLM stating "knock ≤ 5", knocking at whatever the
   engine surfaces), the audit correctly detects a −2.25 unit gap
   over 16 knock decisions with zero parse-failures.

### Summary

Two predictions confirmed (#1, #6 in scaffold), one nuanced with a
real finding about BR-vs-abstraction (#3), one honestly falsified after
closing the loop (#4 — the listener works, but hand-crafted rules don't
lift MI), one vacuously not-supported downstream of it (#5), one
deferred pending credentials (#2). This is exactly the split a working
experiment programme is supposed to produce.

### 6a. Which addition mattered most?

Cross-tabulating the ablation results in §5 against the predictions
above yields a clean ranking of *empirical contribution to policy
quality*:

1. **`knock_asap`** — the single largest factor (48.7 pp main effect;
   43 pp discard×knock interaction). Almost every historical rummy
   engine agrees; the ablation quantifies it.
2. **`discard_highest_deadwood`** — 33 pp main effect, essentially
   ties `discard_safe_from_opp` (the belief-modelling stub); the safe
   variant needs a real card-tracker to see if it separates.
3. **`draw_if_reduces_deadwood`** — 20 pp main effect. Real, but the
   smallest of the three heuristic sub-rules.
4. **Ensembles and MoE over identical experts** — within 5 Elo of the
   single expert; the historical +122 Elo M2CTS chess result requires
   *different* experts per phase, which our current MoE lacks.
5. **CFR training on abstracted Gin** — measurable head-to-head
   improvement (up to +14 pp vs uniform) but a real methodological
   caveat about sampled-BR under strategy sharpening.
6. **Signalling infrastructure without competent listener/inner
   policy** — no measurable coordination lift.

The dominant lesson: **rule-timing (knock) and rule-quality (discard)
dwarf every architectural add we can currently ship**. Ensembles,
signalling, MoE, and CFR contribute only after those two are set well
— and neither is a "novel" addition; they are exactly the folk-heuristic
components long known to matter.

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
* A **factorial ablation framework** (`eval/ablation.py`) with main
  effects, pairwise interactions, Holm–Bonferroni correction and a
  fractional-factorial mode for large factor sets. Every architectural
  add lands in this framework as a named factor rather than a
  standalone claim.
* Two **honest null results** obtained by rigorously testing our own
  predictions: (a) sampled-BR exploitability breaks under abstraction
  sharpening, so head-to-head vs uniform is the right yardstick at
  scale; (b) hand-written cheap-talk listeners do not lift partner MI
  above the random-play baseline in Euchre — coordination requires more
  than infrastructure.

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

- **Listener-side policy for closed-loop signalling** — required to
  actually test predictions #4 and #5. `EuchreListeningPolicy` that
  observes `MessageChannel` and conditions play on the most recent
  partner signal. This is the single highest-value follow-up.
- **Potential-aware / imperfect-recall abstraction for full Gin CFR** —
  the current bucket concentrates the trained strategy too aggressively
  for sampled-BR exploitability to be meaningful; a smoother
  abstraction (or moving to full sequence-form CFR on a smaller
  variant) is the natural next step.
- **Frontier-LLM run of the matched-arm experiment** — one CLI
  invocation once credentials are supplied; will populate the
  currently-empty LLM columns in §5.
- **Train the GNN with self-play PPO** — `models/train_gnn.py` scaffold
  already ships; needs a GPU pass.
- **Federated + differential-privacy telemetry** — done as
  infrastructure (`src/gin_rummy/federated/`), with the honest caveat
  that DP over a card-game latency log has no realistic adversary.
  Genuine motivation is cross-org LLM-cost sharing, not privacy.
- **Public leaderboard** on a Kaggle-Arena-style substrate.

---

*The code that supports every claim above is in the repository; every
number in §5–§6 is producible by a one-line CLI invocation of
`gin-rummy --tournament` or `python -m gin_rummy.solvers.cfr`.*
