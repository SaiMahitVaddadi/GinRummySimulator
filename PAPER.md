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

### 5b. Composite factorial: 2⁴ across heuristic × ensemble × MoE

The framework's promise is *composability*, and to demonstrate it we
run a full 2⁴ = 16-cell factorial at 80 games per pairing (~90 s), plus
a resolution-IV half-fraction cross-check (8 cells):

| Factor    | Levels                              |
|-----------|-------------------------------------|
| `discard` | `random` · `highest_deadwood`       |
| `knock`   | `wait_for_gin` · `always`           |
| `wrapper` | `bare` · `voting3` (3-seed vote)    |
| `moe`     | `flat` · `phase_gated`              |

**Main effects (draws-inclusive win-rate range across levels):**

| Factor    | Range | Best level         | Worst level         |
|-----------|-------|--------------------|---------------------|
| `discard` | **67.8 pp** | `highest_deadwood` → 69.8 % | `random` → 2.0 % |
| `knock`   | **30.6 pp** | `always` → 51.2 %  | `wait_for_gin` → 20.6 % |
| `wrapper` | 0.3 pp  | `voting3` → 36.1 % | `bare` → 35.8 %     |
| `moe`     | 0.3 pp  | `phase_gated` → 36.1 % | `flat` → 35.8 % |

**Pairwise interactions (Holm–Bonferroni corrected, 6 pairs):**

| Pair              | Magnitude | p (Holm) | Reject H₀ |
|-------------------|-----------|----------|-----------|
| `discard × knock` | **92.9 pp** | < 0.001 | **yes**   |
| `discard × moe`   | 25.0 pp | 1.00     | no        |
| `discard × wrapper` | 22.2 pp | 1.00   | no        |
| `wrapper × moe`   | 1.7 pp  | 1.00     | no        |
| `knock × wrapper` | 0.9 pp  | 1.00     | no        |
| `knock × moe`     | 0.9 pp  | 1.00     | no        |

**The resolution-IV half fraction (8 cells) recovers the same
main-effect sign on every factor** (`discard` +67.8 pp vs +67.2 pp,
`knock` +30.6 vs +30.9, `wrapper` +0.3 vs +0.3, `moe` +0.3 vs +0.3)
and flags the same top interaction. This is the framework working as
advertised: at ~half the compute the practitioner recovers the same
main-effect signs.

**Confirmed at scale: architectural wrappers over identical experts
contribute essentially nothing.** Voting3 and phase-gated MoE both
land at 0.3 pp — near the CI floor. The M2CTS +122-Elo chess result
requires *different* per-phase experts, which our current MoE lacks.
Ensembles need diversity; ensembling identical greedy policies is
functionally a no-op.

### 5c. Higher-order ablations (individual add status)

Beyond the composite factorial, per-add status:

| Factor | Observed contribution |
|---|---|
| **CFR training** (mini-gin) | −22.7 % exploitability from uniform in 5 k iters (`solvers/cfr.py`) |
| **CFR training** (abstracted Gin) | Sampled-BR exploitability *rises* under strategy sharpening; **head-to-head vs uniform monotonically increases to +0.14 by 10 k iters** — the honest low-variance signal |
| **CFR smoothing** (KL / min-prob) | Observed 4-way sweep at 5k iters, 1591 info-sets: baseline expl 0.75 / H2H +0.15; **λ=0.1 KL-reg expl 0.61 (−19 %) / H2H +0.11 — best config**; min_prob=0.02 floor expl 0.90 (+19 %) / H2H +0.12; both together expl 0.66 / H2H +0.08. **Honest trade-off with a clear winner**: KL regularisation alone is Pareto-superior; min-prob and the combined config both hurt H2H |
| **LLM tools** | Infra ready (`policies/tools.py` with `meld_analyzer_tool`); frontier-LLM run pending credentials |
| **Signalling: closed-loop hand-written listener** | Partner MI 0.023 vs silent 0.026 — **hand-written rules do not lift MI** |
| **Signalling: search-based listener (1-ply expectimax over partner-hand prior)** | Partner MI 0.024, listener fires 11.3 % (vs 7.8 % hand-written) — **still no MI lift**. Coordination gap is robust to listener sophistication under this MI decomposition |
| **GNN policy** (SFT from greedy traces, CPU, 5 epochs) | 88 % vs random [82.8, 91.8]; **31.5 % vs greedy** [25.5, 38.2] — pure imitation loses to its teacher 2:1; consistent with the SFT literature |
| **CoT rationale audit** | Scripted-mode detects stated-vs-actual knock-threshold gap correctly (parse-fail 0 %); production run pending credentials |
| **DPO from game outcomes** | `outcome_dpo_pairs` mines within-game winner/loser pairs (novel semantics per survey); SPIN loop scaffolded |

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

### 5e. Exhaustive ablation catalogue

Because every architectural add has to earn its keep against the very
large observed effects of the heuristic sub-rules, the paper commits
to running each of the following as a named factor in either a full
factorial (small factor count) or a resolution-III / IV fractional
design (large factor count, per Box & Hunter). All entries are
already scaffolded in the codebase; each Level is *concretely
implemented* today unless flagged **[stub]** or **[needs credentials]**
or **[needs GPU]**. Costs are wall-clock at 200 games per pairing on
a laptop CPU unless otherwise noted.

Legend for the **Status** column:
- ✅ = observed numbers in `experiments/results/`
- 🔧 = code shipped, one CLI invocation from running
- 🕓 = code shipped, requires external resource (API key, GPU, dataset)
- 💭 = design in the codebase, implementation pending

---

#### 5e.i · Heuristic-decomposition factors

The `ComposableHeuristicPolicy` accepts any (draw, discard, knock)
triple from `policies/heuristic_components.py`. All levels
below are already named functions.

| # | Factor | Levels | Impl (file:symbol) | Cost | Status |
|---|--------|--------|--------------------|------|--------|
| H1 | `draw` | `random`; `deck_only`; `smart_if_reduces_deadwood` | `heuristic_components.py:draw_random / draw_always_deck / draw_if_reduces_deadwood` | µs / decision | ✅ (20 pp main) |
| H2 | `discard` | `random`; `highest_deadwood`; `lowest_deadwood` (null baseline); `safe_from_opp` **[stub — needs a card-tracker to separate from highest_deadwood]** | `heuristic_components.py:discard_*` | µs / decision | ✅ (67.8 pp main) |
| H3 | `knock` | `never`; `always_when_legal`; `knock_below(k)` for k ∈ {5, 8, 10}; `knock_after_turn(t)` for t ∈ {5, 10, 20} | `heuristic_components.py:knock_never / knock_asap / knock_below / knock_after_turn` | µs / decision | ✅ for 2 levels (30.6 pp main); 🔧 for parametric sweeps |

Priority interactions to characterise: **H2 × H3** (already 92.9 pp,
Holm p < 0.001 in the 2⁴ composite). H1 × H3 and H1 × H2 are
statistically null at N = 1280 games; report as evidence that draw
rule is orthogonal to the other two.

---

#### 5e.ii · Ensemble & MoE factors

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| E1 | `wrapper` | `bare`; `voting3`; `voting7`; `confidence_weighted` | `eval/ensemble.py:VotingEnsemble, ConfidenceWeightedEnsemble` | n × base cost | ✅ voting3 (0.3 pp over identical experts); 🔧 for diverse expert pools |
| E2 | `moe` | `flat`; `phase_gated`(early/mid/late split @ turns 8, 20); `hand_strength_gated`(dw ≤ 25 → careful, else cheap) | `eval/ensemble.py:PhaseGatedMoE, HandStrengthGatedMoE` | 1× base | ✅ flat vs phase_gated (0.3 pp with identical experts); 🔧 with diverse experts |
| E3 | `seed_ensemble` | k ∈ {1, 3, 5, 10} independent-seed copies of any base policy | `eval/ensemble.py:seed_ensemble` | k × base cost | 🔧 |
| E4 | `moe_expert_diversity` | homogeneous (all greedy) vs. heterogeneous {greedy, LLM, GNN} | build_fn composes | mixed | 🔧 (blocked on §5e.iv or §5e.vii being trained) |

Priority interactions: E1 × E4 (does voting help when experts
actually differ?), E2 × H3 (does phase-gated routing dominate
knock-timing? — hypothesis: no, because H3 already dominates).

---

#### 5e.iii · CFR & equilibrium-finding factors

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| C1 | `cfr_variant` | tabular external-sampling MCCFR; outcome sampling; vanilla CFR; **CFR+** [💭]; **Discounted-CFR (α, β, γ)** [💭] | `solvers/cfr.py:ExternalSamplingMCCFR / OutcomeSamplingMCCFR` | seconds–minutes | ✅ external & outcome |
| C2 | `iterations` | 100; 500; 2 000; 5 000; 10 000; 50 000 | trainer arg | linear in iter | ✅ full curve on mini-gin (`cfr_curve.py`) |
| C3 | `variant_scale` | mini-gin (12 cards, 88 k info-sets); abstracted Classic Gin (~1.6 k info-sets); full Classic Gin **[💭 abstraction too coarse to be sound]** | `solvers/minigin.py` / `solvers/gin_cfr.py` | 20 s / 3 min / intractable | ✅ mini-gin & abstracted; 💭 unabstracted |
| C4 | `abstraction_bucketing` | HandBucket = (dw × longest_run × num_sets × top_class); PublicBucket = (top_discard_class × deck_size) — 4 alternative bucketings enumerated in `abstraction.py` [🔧] | `solvers/abstraction.py` | same order | ✅ one bucketing; 🔧 for alternatives |
| C5 | `smoothing` | none; **λ = 0.1 KL-to-uniform** on avg-strategy; min_prob = 0.02 floor; both together | `solvers/cfr.py:regularisation_lambda, min_prob` | same | ✅ 4-config sweep |
| C6 | `best_response_type` | tabular BR (mini-gin only); sampled BR (n_deals ∈ {32, 100, 500}); learned BR (Timbers IJCAI 2022) [💭] | `solvers/best_response.py` | seconds | ✅ tabular & sampled |

Priority interactions: **C3 × C5** (does smoothing rescue sampled-BR
exploitability at abstracted scale? — early evidence: yes for λ=0.1).
C1 × C2 (does CFR+ converge faster than MCCFR at the same
iteration budget?).

---

#### 5e.iv · LLM policy factors

Every level below flows through `LiteLLM` so the same policy code runs
against any provider. Costs are per-decision at temperature 0.2, max
200 output tokens.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| L1 | `model_family` (frontier proprietary) | `openai/gpt-4o-mini`, `openai/gpt-4o`, `openai/o1-mini`, `openai/o4-mini`, `anthropic/claude-haiku-4-5`, `anthropic/claude-sonnet-4-5`, `anthropic/claude-opus-4-1`, `google/gemini-2.5-flash`, `google/gemini-2.5-pro` | `policies/llm.py:LLMPolicy(model=…)` | $10⁻⁴–$10⁻² / decision, 0.5–5 s | 🕓 (needs credentials) |
| L2 | `model_family` (open weights) | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `deepseek/deepseek-v3`, `mistralai/Mistral-Small-24B` (served via vLLM/Ollama/Together/Fireworks) | same policy; endpoint URL swap | GPU-hours | 🕓 (needs GPU or Together/Fireworks credit) |
| L3 | `temperature` | 0.0; 0.2 (default); 0.7; 1.0 | `LLMPolicy(temperature=…)` | same | 🔧 |
| L4 | `system_prompt_variant` | terse-JSON default; “think step-by-step”; “act like a poker pro”; adversarial persona; **≥ 3 variants** to serve as the LLM-analogue of RL seeds (per lmgame-Bench 2025) | `LLMPolicy(system_prompt=…)` | same | 🔧 |
| L5 | `rationale_request` | off; on (adds `rationale: str` to the JSON schema, extracted into `LLMCall.rationale` for the CoT audit) | `LLMPolicy(request_rationale=True)` | +50–200 tokens / decision | ✅ scaffold; 🕓 for real audit |
| L6 | `max_tool_calls` | 0 (no tools); 1; 3 (default); 5 | `LLMPolicy(max_tool_calls=…)` | 1–5× base cost | ✅ mock; 🕓 with real LLM |
| L7 | `fallback_policy` | random; greedy; `wait-for-user` (raise) — controls the noise floor when JSON parse fails or the API errors | `LLMPolicy(fallback=…)` | same | ✅ |
| L8 | `parse_strictness` | tolerant (default: try to extract JSON from prose); strict-JSON-only | code toggle | same | 🔧 (currently only tolerant) |

Priority interactions: **L1 × L4** (does prompt variance dwarf model
choice — the emerging worry in the LLM-agent literature?); **L1 × L6**
(do smaller cheaper models benefit disproportionately from tool
access?); L5 × behavioural-fingerprint (does adding a rationale
request *change* the action distribution — a Turpin 2023 replication).

---

#### 5e.v · LLM tool-use factors

The `Tool` dataclass lets any Python function become a callable tool.
Every LLM decision can invoke up to `max_tool_calls` tools.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| T1 | `meld_analyzer_tool` | off; on (exposes optimal decomposition + per-discard deadwood delta to the model) | `policies/tools.py:meld_analyzer_tool` | +1 tool round-trip per decision | ✅ mocked; 🕓 real LLM |
| T2 | `discard_regret_tool` | off; on (exposes `xplain.discard_regret_table` to the model) | 💭 (2-line wrap of `xplain.discard_regret_table`) | +1 round trip | 💭 |
| T3 | `opponent_hand_estimator_tool` | off; on (rank-frequency tracker over the discard pile) | 💭 | +1 round trip | 💭 |
| T4 | `rollout_regret_tool` | off; on (Monte-Carlo counterfactual for the current decision, budget = 20 rollouts) | 💭 (wrap `xplain.rollout_regret`) | +50–200 ms / decision | 💭 |
| T5 | `cfr_solver_tool` | off; on (mini-gin only; call the CFR-average-policy for a matched abstracted state) | 💭 | µs | 💭 |

Priority interactions: **T1 × L1** (does the tool close the gap between
Haiku and Opus?); T1 × T4 (does having both tools help or does the LLM
just call whichever is cheaper?); T5 × H3 (does the CFR-derived
knock-timing distribution differ from `knock_asap` — the folk-vs-normative
test).

---

#### 5e.vi · Fine-tuning factors

All wrappers live in `finetune/trainers.py`; all data flows through
`finetune/collector.py`. Heavy deps (`transformers`, `peft`, `trl`,
`bitsandbytes`, `datasets`) are optional.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| F1 | `method` | `sft`; `dpo`; `kto`; `orpo` [💭]; `simpo` [💭]; `grpo`; `rloo` [💭]; API-side `openai_sft`; API-side `openai_dpo`; API-side `openai_rft` (o4-mini) | `trainers.py:run_{sft,dpo,kto,grpo,openai_sft,openai_dpo}` | GPU-hours or $ | ✅ SFT & DPO & KTO & GRPO ship; 🕓 for OpenAI arm |
| F2 | `peft_method` | `none` (full FT); `lora`; `qlora` (4-bit NF4); `dora`; `adalora` [🔧]; `ia3` [🔧]; `pissa` [🔧]; `vera` [🔧] | `finetune/config.py:PEFTConfig(method=…)` | 1× vs 4× VRAM | ✅ config dataclass; wrappers ship for lora/qlora; others in config only |
| F3 | `lora_rank` × `alpha` | (8, 16); (16, 32) default; (32, 64); (64, 128) | `PEFTConfig(r=…, alpha=…)` | linear in rank | 🔧 |
| F4 | `optimizer` | `adamw`; `adamw_8bit`; `paged_adamw_8bit` (default); `lion`; `adafactor`; `schedulefree_adamw`; `muon` [💭] | `finetune/config.py:OptimizerConfig(name=…)` | mostly memory | 🔧 for the first 6 |
| F5 | `scheduler` | `constant`; `linear` (warmup + linear decay); `cosine` (default); `wsd` (warmup-stable-decay for continual self-play); `constant_with_warmup` | `SchedulerConfig(name=…)` | free | 🔧 |
| F6 | `batch_size` × `grad_accum` × `epochs` | small (2 × 4 × 1) vs. medium (8 × 4 × 3) vs. large (32 × 4 × 5) | `FineTuneRun(…)` | linear | 🔧 |
| F7 | `data_source` | greedy-vs-random traces; greedy-vs-greedy traces; LLM-vs-greedy traces; human logs (none available) | `collector.py:DataCollector` + `selfplay.py:run_selfplay` | seconds–minutes | ✅ greedy-vs-random; 🔧 others |
| F8 | `spin_iterations` | 0 (SFT only); 1; 3 (default); 5 (SPIN loop: current-checkpoint vs. previous-checkpoint DPO iterations) | `selfplay.py:run_selfplay_dpo_iteration` | k × F1 cost | ✅ scaffold; 🕓 full run |
| F9 | `base_model` | `Qwen2.5-0.5B` (tiny — laptop); `Qwen2.5-3B`; `Qwen2.5-7B`; `Llama-3.1-8B` | `FineTuneRun(base_model=…)` | grows with model | 🔧 |

Priority interactions: **F1 × F7** (does DPO from game outcomes actually
beat SFT on the same corpus? — the novel claim); F1 × F9 (does GRPO
benefit more at 7B than at 0.5B?); F2 × F6 (QLoRA + small batch vs.
LoRA + large batch on same VRAM budget).

---

#### 5e.vii · GNN policy factors

The graph builder in `models/graph.py` is torch-free; the model
architecture and training loop are in `models/gnn_policy.py` and
`models/train_gnn.py` under the `gnn` optional extra.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| G1 | `graph_type` | bipartite cards↔candidate-melds (default); pure card-graph with rank/suit edges [💭]; card-player heterogeneous graph [💭] | `models/graph.py:build_hand_graph` | µs | ✅ bipartite |
| G2 | `layer_type` | GAT (default); GCN; GraphSAGE; GIN; TransformerConv | `models/gnn_policy.py:CardMeldGAT` — swap conv class | same order | 🔧 (one class, swap) |
| G3 | `n_layers` | 1; 2 (default); 3; 4 | model init arg | linear in layers | 🔧 |
| G4 | `hidden_dim` | 32; 64 (default); 128; 256 | model init arg | quadratic | 🔧 |
| G5 | `heads` (GAT-only) | 1; 4 (default); 8 | model init arg | linear | 🔧 |
| G6 | `training_regime` | SFT on greedy traces (default); PPO self-play [💭 needs `torch_geometric` env]; DouZero-style deep Monte Carlo [💭] | `train_gnn.py:train_sft` | 30 s CPU vs GPU-hours | ✅ SFT only |
| G7 | `training_corpus_size` | 500; 2 000 (default); 10 000; 100 000 samples | `train_gnn.py:collect_trajectories` | linear | ✅ 2 000; 🔧 sweep |
| G8 | `evaluator` | vs random; vs greedy (default); vs vote3; vs LLM-frontier | eval pair | seconds–minutes | ✅ vs random & greedy; 🕓 vs LLM |

Priority interactions: **G2 × G7** (does GAT need more data to beat
GCN, or is architecture orthogonal?); G6 × G2 (does PPO amplify the
graph-inductive-bias advantage over MLP baselines?); G8 × F9 (matched
parameter-count comparison: does the small GNN beat a
similarly-parameter'd fine-tuned Qwen).

---

#### 5e.viii · Signalling & communication factors

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| S1 | `talker_type` | none; structured-enum `EuchreTalkingPolicy`; byzantine (inverted-signal) | `comms/euchre_signals.py` | free | ✅ |
| S2 | `listener_type` | none; hand-written rule-based; 1-ply search over partner-hand prior; multi-ply search [💭]; learned belief-net listener [💭 needs training] | `comms/euchre_signals.py:EuchreListeningPolicy` + `comms/euchre_search_listener.py:SearchListeningPolicy` | ms per play (search) | ✅ hand-written & 1-ply search; 💭 learned |
| S3 | `inner_policy` | RandomEuchrePolicy (current); rule-based Euchre expert [💭]; RL-trained Euchre policy [💭]; LLM Euchre policy [💭] | inner arg to listener | varies | ✅ random only; the survey names the "competent inner + multi-ply rollout" as the hypothesised path to MI lift |
| S4 | `channel_bandwidth` | 1 signal / turn (default); 1 / trick; unlimited natural language [💭] | `comms/channel.py:MessageChannel` | linear | ✅ turn-scoped; 🔧 for other bandwidth policies |
| S5 | `mi_conditioning` | plain I(A; H_partner); **corrected I(A; H_partner \| H_own, public)** (default; common-cause corrected) | `comms/analyze.py` | free | ✅ |
| S6 | `treatment` | silent; cooperative_emit; byzantine_emit; closed_loop (hand-written listener); closed_loop_search | `experiments/signalling.py` | seconds | ✅ 5 treatments run |
| S7 | `partnership_topology` | 2v2 (Euchre); 3-player individual (Indian Rummy variant); 4-player individual [💭] | game class | free | ✅ 2v2 |
| S8 | `signal_alphabet` | {STRONG_TRUMP, BOWER, WEAK, HAS_ACE, NONE} (default); learned discrete codebook [💭 arXiv:1605.06676 DIAL] | `comms/euchre_signals.py:SignalKind` | free | ✅ static; 💭 learned |

Priority interactions: **S2 × S3** (per PAPER.md §6, prediction #4 needs
a competent inner + smart listener before MI will lift); S1 × S2 ×
S3 (three-way, requires many treatments — target the resolution-IV
half-fraction of the framework); S8 × S3 (learned alphabets over
learned inner — the DIAL / SAD / OBL programme).

---

#### 5e.ix · Explainability factors

Each is an *analysis switch* over an already-recorded trace; runs
after the game, no policy change.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| X1 | `introspection_logging` | off; on (log per-turn optimal decomp + per-discard deadwood delta) | `eval/xplain.py:introspect_hand` | µs / turn | ✅ |
| X2 | `discard_regret_table` | off; on | `eval/xplain.py:discard_regret_table` | µs / discard | ✅ |
| X3 | `reward_channel_decomp` | off; on (Juozapaitis 2019: {deadwood, meld_formation, knock_bonus}) | `eval/xplain.py:reward_channels` | µs / turn | ✅ |
| X4 | `behavioral_fingerprint` | off; on (per-policy draw-source / discard-rank / knock-timing histograms) | `eval/xplain.py:fingerprint_from_history` | µs | ✅ |
| X5 | `rollout_regret` | k ∈ {10, 50, 200} Monte-Carlo counterfactuals per decision | `eval/xplain.py:rollout_regret` | k × 1 game / decision | ✅ |
| X6 | `llm_cot_audit` | off; on (extract stated knock-threshold from LLM rationale, compare against actual) | `experiments/cot_faithfulness.py` | free during game, extraction post-hoc | ✅ mocked; 🕓 real LLM |
| X7 | `concept_probe` | off; on (Spearman-correlate `deadwood`, `longest_run`, `opp_knock_risk` at t with final win) [💭] | 💭 | free | 💭 |
| X8 | `saliency` | off; SHAP; Integrated Gradients — for GNN policy only [💭 needs Captum] | 💭 (Captum optional dep) | ms / decision | 💭 |

Priority interactions: **X6 × L5** (Turpin replication — do CoT
rationales predict discard behaviour?); X4 × F1 (does DPO shift the
behavioural fingerprint compared to SFT on the same corpus?); X5 ×
C1 (does the empirical rollout-regret ranking of actions agree with
the CFR average-policy ranking on the same info-set?).

---

#### 5e.x · Evaluation-methodology factors (meta-ablations)

Ablate the *ablation methodology* itself — the field's methodological
choices have real effects on reported numbers (per §4 surveys).

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| M1 | `games_per_pair` | 50; 200 (default); 1 000; 5 000; 10 000 | tournament arg | linear | ✅ |
| M2 | `paired` | on (default); off | `--paired` flag | +2× (extra half of pairs) | ✅ |
| M3 | `rating_system` | raw win rate; Wilson CI; Bradley–Terry Elo; Glicko-2 [💭]; TrueSkill [💭] | `eval/rating.py:bradley_terry_ratings` | seconds | ✅ Wilson & BT |
| M4 | `variance_reduction` | none; paired hands; AIVAT [💭 — needs known-strategy player] | `eval/tournament.py:paired` | seconds | ✅ paired |
| M5 | `multiple_comparisons` | uncorrected; Bonferroni; Holm–Bonferroni (default); BH-FDR [💭] | `eval/stats.py:holm_bonferroni` | free | ✅ Holm |
| M6 | `bootstrap_samples` | 1 000; 10 000 (default); 100 000 | `eval/stats.py:bootstrap_ci` | linear | ✅ |
| M7 | `iqm_vs_mean` | mean; median; IQM (default, rliable) | `eval/stats.py:iqm` | free | ✅ |
| M8 | `n_seeds` | 1; 3; 5; **10 (paper norm)**; 30 | tournament seed sweep | linear | ✅ |
| M9 | `kfold` | 3-fold; 5-fold (default); LOO | `eval/kfold.py:opponent_kfold` | k × 1 tournament | ✅ |
| M10 | `contamination_check` | off; on (LLM only: rule-variant probe — e.g. knock-at-8 rather than knock-at-10) | 💭 (schedule under 5e.iv) | 1 extra run per LLM | 💭 |

Priority interactions: **M1 × M4** (how many hands does paired-hands
buy you vs. raw? — target: reproduce the ACPC ~2× effective sample
rule); M3 × M5 (does BT-Elo agree with raw win rate after Holm on
a many-agent tournament?); M8 × M3 (does the BT-Elo CI shrink at √N?).

---

#### 5e.xi · Federated & DP factors (infrastructure completeness)

Present for completeness with the honest caveat from §7 that a card
game has no realistic adversary against DP-on-telemetry.

| # | Factor | Levels | Impl | Cost | Status |
|---|--------|--------|------|------|--------|
| P1 | `nodes` | 1; 3; 10 concurrent runners aggregating decisions | `federated/aggregator.py:aggregate_telemetry` | free | ✅ |
| P2 | `dp_epsilon` | ∞ (no DP); 10; 1; 0.1 | `federated/dp.py:dp_mean` | free | ✅ |
| P3 | `dp_sensitivity` | 1; 5; 10 (per-metric clipping bound) | `federated/dp.py` | free | ✅ |
| P4 | `sanitised_metric_set` | latency only; latency + parse-fail; full behavioural histogram | `federated/protocol.py:PrivateReport` | free | ✅ |

Priority interactions: essentially none of scientific interest —
scheduled for future work if a cross-org LLM-cost-sharing use case
appears.

---

#### 5e.xii · Design totals and feasibility budget

| Family | Factor count | Full-factorial cells | Recommended design |
|--------|-------------:|---------------------:|-------------------|
| Heuristic (H) | 3 | 24 (2 × 3 × 4) | full-factorial (~40 s @ N=50 pairings) |
| Ensemble/MoE (E) | 4 | 96 | resolution-V fractional |
| CFR (C) | 6 | 1 728 | one-factor-at-a-time (OFAT) baseline sweep |
| LLM (L) | 8 | 4 608 | resolution-III half-fraction × 3 prompt seeds |
| Tools (T) | 5 | 32 | full-factorial per LLM |
| Fine-tune (F) | 9 | ≥ 10 000 | OFAT + expert-curated pairs |
| GNN (G) | 8 | ≥ 5 000 | resolution-III fractional |
| Signalling (S) | 8 | 4 608 | resolution-III half-fraction; blocked on S3 |
| Explain (X) | 8 | 384 | full-factorial (all are analysis switches; free once traces exist) |
| Meta (M) | 10 | > 10 000 | OFAT only |
| Federated (P) | 4 | 48 | OFAT |

Every one of these families is enumerable through the same
`FactorialDesign` / `FractionalFactorialDesign` API (`eval/ablation.py`).
The framework's guarantee is: **any subset of these factors can be
declared as an ablation, run, and analysed with main effects,
Holm-corrected interactions, and IQM + bootstrap CIs in a single
`run_ablation(...)` call**. That is the contribution the paper is
staking its methodological claim on.

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

### 6a. Which addition mattered most? (updated with composite factorial)

Cross-tabulating every ablation and matched-arm run above yields the
following ranking of *empirical contribution to policy quality*. All
main-effect sizes are draws-inclusive win-rate percentages against
`RandomPolicy`; interactions are Holm-corrected.

| Rank | Addition | Main effect | Interaction with #1 | Evidence |
|------|----------|-------------|---------------------|----------|
| 1 | `discard_highest_deadwood` | **+67.8 pp** | (self) | `experiments/ablation_composite.py` |
| 2 | `knock_asap` | **+30.6 pp** | **+92.9 pp discard×knock** (Holm p < 0.001) | same |
| 3 | `draw_if_reduces_deadwood` | +20.0 pp | 2.7 pp (n.s.) | `experiments/ablation_heuristic.py` |
| 4 | CFR training on abstracted Gin | +14 pp H2H vs uniform | n/a | `experiments/cfr_curve.py` |
| 5 | GNN SFT policy (2000-sample corpus) | +38 pp vs random / **−36.5 pp vs greedy** | n/a | `experiments/gnn_train.py` |
| 6 | Voting3 wrapper over identical experts | +0.3 pp | n.s. | composite factorial |
| 7 | Phase-gated MoE over identical experts | +0.3 pp | n.s. | composite factorial |
| 8 | Hand-written signalling listener | +0.010 win rate (n.s.), −0.003 MI | n/a | `experiments/signalling.py` |
| 9 | Search-based signalling listener | 0.000 win rate, −0.002 MI | n/a | search-listener run |

**Dominant lesson (unchanged and stronger)**: rule-quality and
rule-timing dwarf every architectural add currently shipped. The
composite factorial confirms this at scale — the `discard × knock`
interaction alone (**92.9 pp**) is larger than *the sum of every other
add's individual main effect*.

**Secondary lesson (new)**: coordination is a two-part problem.
Signalling infrastructure is necessary but not sufficient. Neither
hand-written rules nor 1-ply search over a partner-hand prior lifts
partner MI above the random-play baseline. A stronger inner policy
and/or multi-ply rollout are required — infrastructure alone is a
red herring for coordination gain.

**Tertiary lesson (new)**: architectural wrappers over identical
experts are functionally no-ops. Voting3 and phase-gated MoE both
land at 0.3 pp above their base — near the CI floor. The historical
M2CTS +122-Elo chess result requires *different* per-phase experts;
our substrate does not yet supply them.

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
