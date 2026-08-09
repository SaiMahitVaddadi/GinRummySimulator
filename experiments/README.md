# Experiments

Runnable, reproducible scripts that populate the tables and predictions
in [`PAPER.md`](../PAPER.md). Each script is invocable both as a CLI
module and as an importable function so it composes into the
`run_all.py` dispatcher and into notebooks.

Artefacts land in `experiments/results/` unless `--out-dir` is passed.

## Prerequisites

```bash
uv sync                      # install the base package
# Optional extras only needed by future experiments:
uv sync --extra llm          # LiteLLM for the LLM arm
uv sync --extra gnn          # torch + torch_geometric for the GNN arm
```

## Registered experiments

| Script | Populates | Approx wall-clock (M-series laptop) |
|---|---|---|
| `matched_arm.py` | PAPER §5 Classic Gin row (5 matched-family entries + optional LLM) | ~10s at `--games-per-pair 200`; ~30s at `--games-per-pair 500`; ~5s at 100 |
| _TODO_ `euchre_coordination.py` | PAPER §5 Euchre row + §6 predictions 4, 5 | — |
| _TODO_ `cfr_minigin_exploitability.py` | PAPER §6 prediction 3 | — |
| _TODO_ `llm_tool_ablation.py` | PAPER §6 prediction 2 | — |
| _TODO_ `cot_fingerprint_audit.py` | PAPER §6 prediction 6 | — |

## How to run

Matched-arm at the paper resolution:

```bash
python -m experiments.matched_arm --games-per-pair 500 --seed 0
# artefacts:
#   experiments/results/matched_arm.jsonl
#   experiments/results/matched_arm.txt
```

Quick smoke:

```bash
python -m experiments.matched_arm --games-per-pair 20 --seed 0
```

Everything at once:

```bash
python -m experiments.run_all --games-per-pair 500 --seed 0
python -m experiments.run_all --only matched_arm --games-per-pair 200
```

## Outputs

Every experiment writes at least two files:

* `<name>.jsonl` — one JSON summary line followed by one JSON line per
  match. Suited for `pd.read_json(..., lines=True)` and for re-computing
  Bradley–Terry Elo / bootstrap CIs offline.
* `<name>.txt` — the human-readable `print_tournament` capture, plus a
  header naming the games-per-pair, seed, and any skipped entries.

The paper draft reads the `.txt` files directly for its § 5 table.

## The LLM arm

The `llm` entry is a **skipped-when-uncredentialled** placeholder. Its
factory checks `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`OPENROUTER_API_KEY` and raises `SkipError` if none are set. Even when
credentials *are* present the factory intentionally still skips, to keep
the default run offline-safe; edit
[`matched_arm.py`](matched_arm.py) `_llm_builder` to wire in a live
`LLMPolicy` (or a stubbed `completion_fn`) when you actually want that
row populated. The skip is logged to stderr and included in the header
of `matched_arm.txt`.

## Reproducibility

Every script accepts `--seed`. The paired-hands variance-reduction
option is enabled by default (`Tournament(paired=True)`), which halves
the number of distinct deals but doubles the sample size per deal.
Given the same seed and the same policy-build closures, results are
bit-identical across runs.
