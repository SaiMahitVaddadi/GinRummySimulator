# Writing conventions for this paper

*Synthesis of three surveys — canonical card-game AI papers,
experimental / behavioural economics, and modern (2022-2026)
ablation-heavy ML papers — distilled into concrete conventions to
follow when writing up the rummy + Euchre research programme.*

---

## Thesis

The paper is a **factorial ablation study** on a novel dual-substrate
(rummy + Euchre) with many honest null results. It should read
like **Bard et al.'s Hanabi Challenge in section shape**, like a
**Duflo/Banerjee RCT in table style**, and like **Agarwal et al.
`rliable` in statistical rigour**. The tone should be
Bowling-Camerer hedged (assertive on well-powered results, cautious
elsewhere), not Silicon-Valley-blogpost promotional.

---

## Section skeleton to steal (Hanabi-shape)

```
1. Introduction
    - real-world puzzle opener (econ convention)
    - one-paragraph mechanism story (econ)
    - intractability hook: "rummy has been under-studied in ML because…"
      (card-game convention)
    - what would falsify us (econ / pre-reg convention)
2. The Games — Gin Rummy + Euchre
    - substrate motivation: why *these* games (card-game convention)
3. The Benchmark
    - matchups, N per pairing, metrics, seeds, pre-registered vs exploratory
4. Agents (parallel subsections)
    - heuristic / MCCFR / LLM / GNN / hybrid — one voice per subsection
5. Empirical Results
    - 5a Headline matchup table
    - 5b Factorial ablation (heuristic decomposition; §5e.i-xiv)
    - 5c Cross-module interactions (Y×K, U×C, Y×L; §5e.xv)
6. Robustness & External Validity
    - Levitt-List five-factor frame (scrutiny, context, self-selection,
      stakes, horizon) mapped to game-AI: self-play vs cross-play vs
      human eval; opponent-pool composition; seed re-shuffles;
      scoring-rule variants; evaluation-length sweep
    - end with cost-per-outcome table ($ / +1pp win-rate,
      tokens / decision) — the RCT convention adapted to LLM agents
7. Related Work
8. Discussion + Conclusion
    - honest null summary
    - what would have to change for each null to flip
```

Ablations live **inside the empirical section, subsection by
subsection**, not in a separate "Analysis" appendix (Hanabi convention;
same choice the code makes with `experiments/ablation_*.py`).

---

## Table conventions

**Adopt the economics coefficient-table style** for the factorial
ablation. Rows are factors (H, E, C, L, T, F, G, S, X, U, Y, K, and
their pairwise interactions); columns are outcomes (win rate,
expected points, discard-error, tokens/decision, $/+1pp); each entry
is a point estimate with **SE in parentheses below**, plus stars for
significance:

```
                     win rate      $/+1pp      tokens/decision
knock_asap           +30.6 ***     n/a         n/a
                    (3.2)
discard_highest_dw   +67.8 ***     n/a         n/a
                    (2.9)
LLM tool: meld       +5.4 *        $12.30     147
                    (2.4)         (1.10)     (11)
```

Two parallel columns: **naive p-value** and **Holm–BH-corrected
p-value**. Star notation: `*` p<0.10, `**` p<0.05, `***` p<0.01,
consistent with the economics convention. Table notes flag each row
as **pre-specified** or **exploratory**.

For scalar aggregate reporting: **rliable-style IQM + stratified
bootstrap 95 % CI**, plotted as a **performance-profile CDF** for the
headline table. This is the single highest-rigour-per-page-count move.

---

## Statistical rigor

- **≥5 seeds** for exploratory, **≥10** for any headline claim.
- **Holm–Bonferroni** for a small fixed comparison set; **BH-FDR** for
  the full factor grid. Report *both* naive and corrected p-values.
- **Paired hands** on every 2-agent matchup (duplicate-poker
  variance reduction; already implemented as `--paired`).
- **`AdaStop`** if seed budget is tight and you want an adaptive
  stopping rule with FWER guarantees.
- **Prompt-variance** on every LLM arm: ≥3 paraphrases × ≥3 decoding
  seeds, reported as mean ± SD over both (the LLM analogue of RL
  seeds).

---

## Honest-reporting language

Language conventions (Bowling / Bard / Camerer):

| Situation | Language |
|---|---|
| Well-powered, large effect | *"greedy beats random 100 % [Wilson 95 % 99.8, 100.0]"* — assertive |
| Well-powered, small effect | *"we observe a 3 pp drop [95 % CI [1.2, 4.8]]; the effect is directionally consistent but of modest magnitude"* |
| Null result with power | *"we cannot reject the null; the 95 % CI rules out effects larger than 0.02 win-rate"* — the minimum-detectable-effect frame |
| Null with unclear power | *"the observed effect is small (0.7 pp) with wide CI [-1.5, 2.9]; a larger evaluation budget would be needed to distinguish these"* |
| Falsified prediction | Following Camerer 2016: *"observed effect is X % of the pre-registered prediction; the 95 % CI does/does not include the predicted point estimate"* |
| Failed technique | Following Kelidari 2026: *"we tried A, B, C; only C moved the metric; A and B are reported for completeness"* |

Reserve "we show", "we prove" for well-powered, effect-size-large,
Holm-surviving results. Everything else is "we observe",
"consistent with", "suggestive".

---

## Two things worth doing that no card-game paper currently does

1. **Explicit factorial interaction contrasts with CIs** (Chinchilla-era
   ML convention, formally standard in econ). Our `discard × knock =
   92.9 pp` interaction is a real result; reporting it *as* an
   interaction contrast — with the DoE-lineage citation to Box &
   Hunter — is a differentiator. Add an "Additive vs Non-Additive"
   subsection. Rebuffi/Karpathy-style **sensitivity-fan figure** with
   every factor overlaid on a normalised axis.

2. **Cost-per-decision + $/+1pp win-rate as a first-class column** in
   every headline table. LLM arms cost $10⁻³–$10⁻² per decision;
   heuristics cost ns and pennies; CFR µs at solve time. A reader
   can't judge "LLM + tool > LLM" without seeing the deployment
   economics. RCT convention (cost-per-QALY) adapted to ML.

---

## Pre-registration and robustness

Before running any expensive experiment (LLM arms, GPU training),
write a one-page **pre-analysis plan** that lists:

- The **primary outcome** (typically head-to-head win rate against a
  fixed external heuristic yardstick — the Kelidari 2026 convention).
- The **pre-specified hypotheses** as numbered predictions (already in
  PAPER.md §6 as `#1`–`#6`).
- The **exploratory hypotheses**, clearly labelled as such.
- The **primary comparisons** (which pairs are pre-specified, which
  are post-hoc).
- The **seed set** — commit the seed list before running.

Then run. In the write-up, **flag every table row as pre-specified or
exploratory** (star convention in table notes). This is the
Muralidharan-Romero-Wüthrich (REStat 2025) discipline for factorial
designs, and it's what our commits + JSONL results already support —
every experiment already ships its seed set, so we're pre-registered
by construction, we just need to *say so*.

---

## Concrete checklist before submission

- [ ] Every headline claim has: point estimate, 95 % CI, seed count,
      Wilson or bootstrap, and (if pairwise) Holm-corrected p.
- [ ] LLM arms report: mean tokens/decision, $/decision at listed API
      price, parse-fail rate, model+date stamp.
- [ ] Every ablation table row labelled pre-specified vs exploratory.
- [ ] Robustness section covers ≥3 of Levitt–List's five factors.
- [ ] Cost-per-outcome column in the headline table.
- [ ] One factorial interaction contrast with CI and Holm-p.
- [ ] One performance profile CDF as a hero figure.
- [ ] At least one honest null result *featured*, not buried.
- [ ] "What would falsify us" paragraph in intro.
- [ ] Real-world puzzle opener in first sentence.
- [ ] All numerics reproducible via `experiments/*.py` + committed
      seeds + `uv sync --extra dev`.

---

## Source pointers (one line each)

- **Bowling et al. 2015 (Science)** — HULHE "essentially solved"; exploitability-as-Nash-distance framing; hero exploitability curve.
- **Bard et al. 2020 (AIJ)** — Hanabi Challenge; self-play / cross-play split; negative results framed as contribution.
- **Brown & Sandholm 2019 (Pluribus)** — sample-size defense; scope disclaimer for n > 2.
- **Kahneman & Tversky 1979** — problem-by-problem table style.
- **Camerer et al. 2016 (Science replication)** — honest-null discourse; observed/expected effect ratios.
- **Muralidharan, Romero & Wüthrich 2025 (REStat)** — factorial-design rigour; short-model warnings.
- **List, Shaikh & Xu (NBER w21875)** — Romano–Wolf stepdown MHT for experiments.
- **Agarwal et al. 2021 (NeurIPS Outstanding, `rliable`)** — IQM, stratified bootstrap, performance profiles.
- **DeepSeek-V3 2024** — small→mid→full-scale cascade validation.
- **Kelidari et al. 2026 (arXiv:2607.06854)** — gold-standard external yardstick; five-things-that-failed reporting.
- **Levitt & List 2007 (JEP)** — five-factor external-validity frame.

Full URLs in the three survey reports (in commit history and in the
`/tmp/tasks/*.output` transcripts).
