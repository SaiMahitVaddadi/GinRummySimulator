# Gin Rummy Simulator

A fast, dependency-free, multi-variant rummy simulator. Designed to be a
substrate for research on card-game AI — from random baselines to
LLM-driven and RL agents.

- **Zero runtime dependencies** — pure Python 3.10+ standard library.
- **Exact optimal meld decomposition** via bitmask DP (~140 µs per
  10-card hand on a laptop).
- **Policy protocol** — the game engine is policy-agnostic, so a
  ``RandomPolicy``, an LLM-backed policy, or an RL agent all plug in
  identically.
- **Seedable RNG** — every game is reproducible.
- **Four variants shipped:** Classic Gin · Oklahoma Gin · Hollywood Gin
  · Indian Rummy (3/7/10/13/15-card hand sizes).

## Install

```bash
uv sync --extra dev        # installs the package + pytest
```

## CLI

```bash
uv run gin-rummy                                      # one Classic Gin hand
uv run gin-rummy --variant oklahoma --seed 42
uv run gin-rummy --variant hollywood --players 4 --hands 3
uv run gin-rummy --variant indian --players 4 --hand-size 13
uv run gin-rummy --variant classic --games 1000 --seed 0 --quiet   # batch stats
```

Or `python -m gin_rummy ...` if you'd rather not use the console script.

## Library

```python
from gin_rummy import ClassicGin, HollywoodGin, RandomPolicy

result = ClassicGin(num_players=2, seed=42).play()
print(result.outcome, result.winner_name, result.scores)

series = HollywoodGin(num_players=2, num_hands=3, seed=42).play()
print(series.totals)
```

### Plugging in a custom policy

The engine only needs three methods — draw source, discard choice, and
knock/no-knock — so you can slot in any decision-maker. Here's a
strawman "always draw from discard, always keep face cards" heuristic:

```python
from random import Random
from gin_rummy import ClassicGin, Observation

class KeepFacesPolicy:
    def __init__(self, rng): self._rng = rng
    def choose_draw_source(self, obs: Observation):
        return "discard" if obs.top_discard else "deck"
    def choose_discard(self, obs: Observation):
        low = [c for c in obs.hand if c.value < 8]
        return self._rng.choice(low) if low else self._rng.choice(obs.hand)
    def choose_to_knock(self, obs: Observation, deadwood_value: int):
        return True

rng = Random(0)
game = ClassicGin(2, seed=0, policies=[KeepFacesPolicy(rng), KeepFacesPolicy(rng)])
print(game.play().scores)
```

The same protocol is what an LLM-backed policy will implement (see the
roadmap below).

## Package layout

```
src/gin_rummy/
    cards.py         # Card, Deck, RANKS, SUITS
    meld.py          # optimal_decomposition (exact bitmask DP)
    player.py        # Player state
    policy.py        # Policy protocol + RandomPolicy baseline
    scoring.py       # ScoringRules
    game.py          # GinRummyGame — the shared engine
    variants/
        classic.py
        oklahoma.py
        hollywood.py
        indian.py
    cli.py
```

## Tests

```bash
uv run pytest
```

36 tests cover card fundamentals, exact meld decomposition on known
hands, seeded reproducibility, and end-to-end play for every variant.

## Variants

| Variant | Hand | Knock threshold | Ends when |
|---|---|---|---|
| Classic Gin | 10 | ≤ 10 deadwood | knock, gin, or undercut |
| Oklahoma Gin | 10 | rank of first upcard (A ⇒ gin only) | same |
| Hollywood Gin | 10 | ≤ 10 deadwood | after N hands (default 3), by total |
| Indian Rummy | 3 / 7 / 10 / 13 / 15 | ≤ 10 deadwood (approx.) | knock, gin |

The Indian Rummy variant is a rummy-family class exposing the
hand-size knob; it does **not** attempt to model canonical Indian Rummy
declaration rules (pure sequence requirement, joker wildcards, etc.).
For a rules-faithful implementation see
[IRumAI](https://arxiv.org/abs/2606.21975).

## Roadmap

- LLM-backed `Policy` via [LiteLLM](https://github.com/BerriAI/litellm)
  as a pluggable gateway.
- Euchre variant — the natural partnership-signaling counterpart to
  Gin Rummy's individual optimisation.
- MCCFR / Deep-MC baselines for a normative reference agent.
- Batch-metrics harness (game length, gin/knock/undercut/draw
  distributions per variant × player count).

## License

MIT.
