# AI-to-AI Information Absorption — Testing Framework

*Design document for a study of **how AI agents absorb information from
other AI agents**, using rummy / Euchre / Hanabi-mini as interchangeable
observation grounds. The card games are substrate, not subject.*

---

## 1. Thesis

**The interesting variable is not "did the team win" but "how much of
one agent's shared information actually leaked into another agent's
subsequent behaviour, and under what conditions does that leakage
happen, saturate, or break?"** This is the DIAL / SAD / OBL / CICERO
research programme reframed as a controlled counterfactual study
across matched agent architectures.

The card-game substrate is used because it gives clean *ground truth*
on the receiver's counterfactual action (the deck is finite, legal
moves are enumerable, the sender's message is separable from the game
state). Findings should transfer to any multi-agent LLM system.

---

## 2. Research questions (falsifiable)

| # | Question | Predicted answer |
|---|----------|------------------|
| **Q1** | Does *modality* dominate *architecture*, or vice versa, in absorption MI? | **Modality wins** — a well-designed structured signal outperforms free-form NL for any receiver |
| **Q2** | Is there a bandwidth saturation point above which more bits do not raise absorption MI? | **Yes; saturation ~5–8 bits/turn** for our substrate; more bits = same MI + more tokens/dollars |
| **Q3** | Does absorption depend on the receiver's architecture in a rank-order that mirrors capability (LLM > RL > heuristic)? | **No** — heuristics with hard-coded read-rules match or beat LLMs at low rate limits |
| **Q4** | Is absorption *linear* in adversarial-message fraction, or does it collapse past a threshold? | **Threshold collapse** — ~30 % Byzantine kills usable absorption (Byzantine Cheap Talk lineage) |
| **Q5** | Do recursive absorption loops (A→B→A→B) converge, oscillate, or destabilise? | **Depends on modality**: structured signals converge; NL LLM chains oscillate then collapse into agreement-cascades (echo chamber) |
| **Q6** | Do independently-trained agent pairs develop convergent shared codes when the channel is a discrete-vocabulary bottleneck? | **Yes, and the code is partner-specific** (OBL-style handshake problem) |
| **Q7** | Is absorption asymmetric — is one agent doing most of the teaching? | **Yes, systematically** — the higher-capacity agent absorbs less because the low-capacity one carries less mutual info |

Each question is testable in this framework. Predictions are pre-registered here.

---

## 3. Factors

The full factorial is **6-way**; we run resolution-III fractionals for
the routine sweep and full factorials for the interactions we care
about.

### F1 — Sender modality (5 levels)
| Level | Bits per message | Impl in tree |
|---|---|---|
| `raw_action` | log₂(action-space) ≈ 4 | free — every game already publishes actions |
| `structured_signal` | log₂(5) ≈ 2.3 | `comms/signals.py:SignalKind` |
| `natural_language` | ≈ 20-1000 (tokens × log₂ vocab) | `LLMPolicy` prompt-response |
| `learned_embedding` | 32-128 float dims ≈ log₂ resolution | `models/graph.py` card/meld embeddings |
| `posterior_dist` | H(state) ≈ 2-3 bits | `markov/opponent_hand_hmm.py:predict_hand_strength` |

### F2 — Receiver architecture (4 levels)
| Level | Impl |
|---|---|
| `heuristic_reader` | `ComposableHeuristicPolicy` with a hand-crafted `read_rule` that consumes messages |
| `rl_reader` | small policy net taking message vector as extra input (💭 needs training) |
| `llm_reader` | `LLMPolicy` with the message injected into the prompt |
| `gnn_reader` | `GNNPolicy` with message embedding fused into the node features |

### F3 — Absorption depth (3 levels)
| Level | Description |
|---|---|
| `last_only` | Receiver conditions on the most recent message only |
| `last_k=3` | Receiver conditions on last 3 messages |
| `full_history` | Receiver conditions on the entire message log |

### F4 — Bandwidth cap (5 levels)
1 message per (game / round / trick / turn / decision). Ties to F1 —
`natural_language` at "per decision" is dollars-per-second on a
frontier LLM.

### F5 — Adversarial fraction (5 levels)
0 %, 10 %, 20 %, 30 %, 50 % of sender messages are Byzantine-flipped
(`comms/byzantine.py:byzantine_swap` for structured; for NL, a
"lie-generator" prompt that inverts the sender's stated belief).

### F6 — Recursive depth (4 levels)
0 (one-shot), 1 (A→B), 2 (A→B→A), 3 (A→B→A→B) recursive
message-response loops per decision.

**Total design:** 5×4×3×5×5×4 = **6000 cells**. Full factorial is
absurd; we run:

- **Main-effects screen** (OFAT): 5+4+3+5+5+4 = **26 runs**, ~30 min total
- **Priority interactions** as 2^k full factorials:
  - F1 × F2 (modality × receiver arch): 20 cells
  - F5 × F6 (Byzantine × recursion): 20 cells
  - F1 × F4 (modality × bandwidth): 25 cells
- **Resolution-IV half-fraction** of the full 6-way: 3000 cells,
  ~40 hours of API + compute — the *outer envelope* if the priority
  interactions surface something the main-effects screen missed

---

## 4. Metrics — the actual research asset

**Every metric is defined counterfactually, not correlationally.**
Correlational MI can be spurious (the deck-arithmetic bug we already
fixed in §5e.viii of PAPER.md). Counterfactual absorption is the
gold standard.

### M1 — Counterfactual absorption
For each decision, run the receiver twice: once with the real
message, once with a *swapped* message from a matched hand. Measure
`P(action | real message) − P(action | swapped message)` in total
variation distance. **A receiver with zero absorption has zero swap
TVD by construction**; a receiver that fully conditions on the
message has swap TVD close to 1.

This is the *causal* absorption measurement. Everything else is a
correlational approximation.

### M2 — Conditional mutual information (correlational)
`I(receiver_action_t ; sender_message_t−1 | history_t−1)` estimated
via plug-in on coarse buckets, block-bootstrapped over games.
Cheap; use as a preliminary screen before M1.

### M3 — Behavioural entropy delta
`H(receiver_action | no message)` minus `H(receiver_action | with
message)` — how much does the message *reduce* the receiver's
decision entropy? A concentrated posterior implies genuine
absorption.

### M4 — Convergent-code entropy (over training)
For learned senders, measure how the entropy of the emitted message
distribution changes across training. Collapse to a low-entropy code
indicates emergent convention discovery (SAD/OBL signature).

### M5 — Absorption asymmetry
`AbsorptionMI(A→B) − AbsorptionMI(B→A)` on the same pair. A
non-zero score means one direction dominates.

### M6 — Real-time cost of absorption
tokens/decision, $/decision, ms/decision — a receiver that produces
+0.1 nats of absorption at 100 ms/decision is not comparable to one
that gets +0.05 nats at 1 ms/decision. Cost dashboard on every table
row (WRITING.md convention).

### M7 — Byzantine robustness curve
Absorption MI as a function of adversarial fraction — the shape of
the collapse is the deliverable, not any single number.

### M8 — Echo-chamber score (for recursive depth ≥ 1)
`|P(A's action at loop k) − P(A's action at loop 0)|` — did the
recursive loop change A's own behaviour? If yes and A wasn't given
new evidence, that's an echo chamber.

---

## 5. Substrates

Three interchangeable observation grounds, chosen per experiment:

| Substrate | Best for measuring | Why |
|---|---|---|
| **2v2 rummy team** | Sender-modality × receiver-arch main effects | Team score aggregation isolates the coordination effect from individual play; needs ~50 lines of team-scoring on `GinRummyGame` |
| **Euchre** | Byzantine × recursion; partnership signalling | The trump-calling phase is a natural pre-play message; partnerships are baked in |
| **Hanabi-mini** (2-player cooperative rummy: melt each other's melds) | Convergent-code emergence; asymmetry | Purely cooperative signal; no competitive noise; needs a small new engine (~200 lines) or import Farama's HLE |
| **Rummy team-within-seat** (1 seat = orchestrated ensemble) | Real-time orchestration cost | Same policy code, no game changes; measures "how much does the orchestrator's absorption cost per decision?" |

**Rule**: report *all four substrates in the same table wherever
possible* — the finding gains generality every time a result
replicates across substrates. Substrate becomes another factor;
call it **F7 · substrate** with 4 levels.

---

## 6. Experimental schedule (in dependency order)

### Phase A — Framework build (before any experiment)
- [ ] `absorption/counterfactual.py` — counterfactual-swap M1 runner
- [ ] `absorption/metrics.py` — M2-M8 implementations
- [ ] `absorption/senders.py` — 5 sender modalities as `SenderAdapter`s
- [ ] `absorption/receivers.py` — 4 receiver architectures as `ReceiverAdapter`s
- [ ] `absorption/experiment.py` — factorial runner with the framework's `FactorialDesign`
- [ ] `absorption/substrates.py` — 4 substrate adapters (2v2 rummy, Euchre, Hanabi-mini, orchestrated seat)
- [ ] `tests/test_absorption.py` — smoke tests + plant-recovery tests
- [ ] `experiments/absorption_main.py` — main entrypoint

### Phase B — Main-effects screen (OFAT, ~30 min)
Confirm every factor moves *any* absorption metric, so the fractional
factorial doesn't waste cells on inert factors. If a factor doesn't
move anything under OFAT, drop it from the fractional.

### Phase C — Priority interactions (~2 hours)
- **F1 × F2**: does modality-receiver pairing matter? (Q3)
- **F5 × F6**: does Byzantine damage compound recursively? (Q4, Q5)
- **F1 × F4**: does bandwidth-saturation depend on modality? (Q2)

### Phase D — Cross-substrate replication
Repeat Phase C on all 4 substrates. Any finding that replicates on
≥3 is the paper's headline.

### Phase E — Recursive loops + emergent codes (~10 hours + training)
- Convergent-code entropy over training (Q6)
- Absorption asymmetry across many pairings (Q7)

### Phase F — LLM arms
Costs money. Run only after Phases B–D isolate the interesting
interaction cells. Estimated $500–$2000 in API spend depending on
which frontier models are pooled.

---

## 7. Existing code the framework reuses

| Component | Existing file | Role in absorption framework |
|---|---|---|
| MessageChannel | `comms/channel.py` | Wire protocol between sender and receiver |
| SignalKind | `comms/signals.py` | F1 `structured_signal` level |
| TalkingPolicy / EuchreTalkingPolicy | `comms/(euchre_)signals.py` | Reference sender for structured modality |
| ByzantinePolicy | `comms/byzantine.py` | F5 adversarial mechanism |
| EuchreListeningPolicy / SearchListeningPolicy | `comms/euchre_(search_)listener.py` | Reference receivers for heuristic and search-based reading |
| LLMPolicy | `policies/llm.py` | F1 `natural_language` sender + F2 `llm_reader` receiver |
| GNNPolicy | `models/gnn_policy.py` | F1 `learned_embedding` sender + F2 `gnn_reader` receiver |
| OpponentHandHMM | `markov/opponent_hand_hmm.py` | F1 `posterior_dist` sender |
| BlackboardOrchestrator | `comms/orchestrator.py` | F6 recursive-loop coordinator |
| action_hand_mutual_information | `comms/metrics.py` | M2 backbone |
| FactorialDesign | `eval/ablation.py` | Runs the sweeps; main effects + Holm-corrected interactions |
| bootstrap_ci, wilson_ci | `eval/stats.py` | CIs on every metric |

**~85 % of the machinery already exists.** Framework work is
primarily the counterfactual-swap runner (M1), the sender/receiver
adapter interfaces, and the 4 substrate wrappers.

---

## 8. Novel infrastructure to build

### `absorption/counterfactual.py`
The heart. For each decision:
1. Snapshot the game state.
2. Sample a matched "swap" message from the marginal distribution over
   messages *at that game context* (rejection-sampling over recent
   games with similar public state).
3. Run the receiver on (real message | context) and (swap message |
   context); return the TVD between the two action distributions.
4. Bootstrap across games to get a CI.

### `absorption/senders.py`
`class SenderAdapter(Protocol)` with `.encode(state, hidden) -> Message`
and `.bits_per_message() -> float`. One implementation per F1 level.

### `absorption/receivers.py`
`class ReceiverAdapter(Protocol)` with `.integrate(message,
observation) -> Observation'` — the receiver's *conditioning* step.
The `Observation'` is what the wrapped Policy sees. Absorption is
literally "how much does the observation change?"

### `absorption/substrates.py`
Uniform interface: `.play_paired(sender, receiver, sender_policy,
receiver_policy, num_games) -> list[TrajectoryRecord]`. Each trajectory
record carries `(game_id, turn, sender_message, receiver_action,
counterfactual_swap_action, hidden_state)`.

### `experiments/absorption_main.py`
Orchestrates Phases B–F above. Dumps JSONL to
`experiments/results/absorption_*.jsonl`.

---

## 9. What we expect to find (pre-registered predictions from §2)

| Prediction | If confirmed, we claim | If falsified, we claim |
|---|---|---|
| **Q1** modality > architecture | "Design of the message vocabulary matters more than the receiver's model class" — a genuinely novel finding for the emergent-communication literature | Architecture-first framing wins; write up as a null of the modality-hypothesis |
| **Q2** ~5-8 bit saturation | Bandwidth ceiling; recommend against expensive NL modes | Bandwidth grows meaningfully with tokens; frontier LLMs earn their cost |
| **Q3** low-cap heuristic ties LLM | "Absorption is a substrate-conditioned property; capability rankings do not transfer from single-agent benchmarks" | LLM-first framing wins; consistent with capability-scaling laws |
| **Q4** ~30 % Byzantine collapse | Practical robustness curve; input to any multi-agent LLM deployment | Byzantine is either always fatal or never fatal; write up as a null with the specific decay curve |
| **Q5** structured converges, NL oscillates | "Modality determines whether recursive absorption stabilises" — direct extension of Byzantine Cheap Talk 2606.07790 | Recursion is universally stable or universally unstable; still publishable |
| **Q6** emergent codes are partner-specific | Reproduces OBL handshake problem on card games; a Hanabi-adjacent result | Codes generalise across partners; a *stronger* result — publish as "against the OBL claim" |
| **Q7** systematic asymmetry | Direction of asymmetry is the interesting variable — is the LLM the teacher or the student? | Bidirectional teaching, symmetric absorption; interesting null |

---

## 10. Threats to validity

- **Counterfactual matching is imperfect**: swapping messages
  requires matched hidden state, which we approximate via rejection
  sampling. Bias direction unclear; report the acceptance rate as a
  diagnostic.
- **Substrate contamination**: LLMs may have seen rummy strategy
  content in pretraining. Mitigate by running the same protocol on a
  rule variant (knock-at-8 instead of 10; already implemented as an
  ablation factor in PAPER.md §5e.iv).
- **Prompt sensitivity**: for the LLM sender/receiver, run ≥3 prompt
  variants per condition; report mean ± SD over both prompts and
  decoding seeds (WRITING.md convention).
- **Absorption ≠ improvement**: a receiver can absorb a message and
  make a *worse* decision because of it (the HMM-belief-tool result:
  25 % of legal knocks suppressed, −3 pp win rate). Absorption
  measurement is *orthogonal* to policy-quality measurement; the
  paper needs both axes.
- **Cost drift**: LLM prices drop monthly; log model + date + $/1M
  tokens on every LLM row.
- **Multiple comparisons**: 6000 factorial cells means multiple
  comparisons matters *a lot*. Report Holm–Bonferroni for the
  priority-interaction subset and Benjamini–Hochberg FDR for the
  full sweep. Every table row flagged as pre-specified or
  exploratory.

---

## 11. What this contributes if it works

1. A **counterfactual absorption metric** for AI-to-AI communication
   that is more rigorous than correlational MI.
2. A **factorial characterisation** of when absorption saturates,
   collapses under adversarial noise, and destabilises under
   recursion — parametrised, not anecdotal.
3. A **cost-per-nat table** that reframes LLM multi-agent claims: is
   the $10⁻² per NL message worth the +X nats of absorption over a
   free structured signal?
4. A **counter-example (or confirmation)** of the folk assumption
   that higher-capability agents absorb better.
5. A **card-game-agnostic** design that ports to any multi-agent LLM
   testbed (Werewolf, Diplomacy, Overcooked, code-review
   simulators). The rummy/Euchre substrate is the vehicle, not the
   claim.

---

## 12. Concrete first move

Build Phase A (framework skeleton — ~1000 lines across 6 files, ~30
tests). Then Phase B (main-effects screen — ~30 min compute) on a
single substrate (`2v2 rummy team`) to sanity-check that every
factor moves *any* metric. If a factor is inert, drop it before
committing to the full factorial.

Total time to first defensible finding: **~2 days of build + 1 day
of experiments**. Total time to a paper-shaped result across all
substrates: **~2 weeks + $500-2000 in API spend for the LLM arms**.

*If you want, fan out four subagents to build Phase A in parallel:
(1) counterfactual runner + metrics, (2) sender adapters + receiver
adapters, (3) substrate wrappers, (4) experiment orchestrator + tests
+ demo. That's the natural next commit.*
