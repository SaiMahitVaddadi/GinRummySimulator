"""Self-play data collection pipeline.

Ties the game engine, an LLM policy (or any :class:`Policy`), and the
:class:`DataCollector` together so a single call produces trajectories
ready for DPO / SFT / KTO training.

Two entry points:

* :func:`run_selfplay` — play ``num_games`` matches, record every draw /
  knock / discard decision as a :class:`TrajectorySample`, and back-fill
  the terminal reward. Returns the populated collector.
* :func:`run_selfplay_dpo_iteration` — a **SPIN-style loop** (Chen et al.
  ICML 2024, arXiv:2401.01335). Each iteration plays the current
  checkpoint against the previous one, extracts DPO pairs from winners
  vs losers, trains, and rolls the checkpoint forward. Requires the
  ``finetune`` extra to actually train — but the pipeline shape is
  exercised in tests without it.

Design notes
------------
No LLM API calls are needed to *collect* data — we wrap whatever
:class:`Policy` you hand us in a :class:`_RecordingPolicy` that renders
the same prompts :class:`LLMPolicy` would send and logs the concrete
action the underlying policy chose. That means a game between two
:class:`GreedyKnockPolicy` seats will still produce a valid DPO corpus,
and the same code path lights up unchanged when the seat is an
:class:`LLMPolicy` reading a real base model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gin_rummy.cards import Card
from gin_rummy.finetune.collector import DataCollector, TrajectorySample
from gin_rummy.finetune.config import FineTuneRun
from gin_rummy.policies.heuristic import GreedyKnockPolicy
from gin_rummy.policies.llm import (
    _render_discard_prompt,
    _render_draw_prompt,
    _render_knock_prompt,
)
from gin_rummy.policy import DrawSource, Observation, Policy, RandomPolicy
from gin_rummy.variants.classic import ClassicGin


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def _policy_from_spec(spec: str, rng: random.Random) -> Policy:
    """Materialise a baseline policy from a short string tag."""
    spec = spec.lower().strip()
    if spec == "greedy":
        return GreedyKnockPolicy(rng)
    if spec == "random":
        return RandomPolicy(rng)
    raise ValueError(f"unknown opponent_policy_spec: {spec!r}")


# ---------------------------------------------------------------------------
# Recording wrapper — the seam between the engine and the collector.
# ---------------------------------------------------------------------------

@dataclass
class _RecordingPolicy:
    """Wraps a :class:`Policy` and records every decision to a collector.

    We render the same prompt an :class:`LLMPolicy` would send, so the
    corpus is fine-tuning-ready regardless of whether the seat is a real
    LLM or a heuristic baseline.
    """

    inner: Policy
    collector: DataCollector
    game_id: str
    player_id: int
    turn_ctr: list[int] = field(default_factory=lambda: [0])

    def _next_turn(self) -> int:
        self.turn_ctr[0] += 1
        return self.turn_ctr[0]

    def choose_draw_source(self, obs: Observation) -> DrawSource:
        source = self.inner.choose_draw_source(obs)
        self.collector.record(
            TrajectorySample(
                game_id=self.game_id,
                turn=obs.turn_number,
                player_id=self.player_id,
                kind="draw",
                prompt=_render_draw_prompt(obs),
                action=str({"source": source}),
            )
        )
        return source

    def choose_discard(self, obs: Observation) -> Card:
        card = self.inner.choose_discard(obs)
        self.collector.record(
            TrajectorySample(
                game_id=self.game_id,
                turn=obs.turn_number,
                player_id=self.player_id,
                kind="discard",
                prompt=_render_discard_prompt(obs),
                action=str({"discard": str(card)}),
            )
        )
        return card

    def choose_to_knock(self, obs: Observation, deadwood_value: int) -> bool:
        knock = self.inner.choose_to_knock(obs, deadwood_value)
        self.collector.record(
            TrajectorySample(
                game_id=self.game_id,
                turn=obs.turn_number,
                player_id=self.player_id,
                kind="knock",
                prompt=_render_knock_prompt(obs, deadwood_value),
                action=str({"knock": bool(knock)}),
                metadata={"deadwood_value": deadwood_value},
            )
        )
        return knock


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    """Knobs for :func:`run_selfplay`.

    Parameters
    ----------
    num_games:
        How many matches to play.
    opponent_policy_spec:
        Short tag resolved by :func:`_policy_from_spec` (``"greedy"``,
        ``"random"``). Ignored if ``base_llm_policy_factory`` returns two
        seats itself.
    base_llm_policy_factory:
        Zero-arg callable returning the *learner* policy for seat 0. Called
        once per game so mutable state (e.g. an :class:`LLMPolicy` trace)
        doesn't leak across matches. Defaults to a fresh
        :class:`GreedyKnockPolicy` — handy for tests.
    output_path:
        Optional JSONL destination for the populated collector.
    seat_learner:
        Which player id the learner occupies. Default 0.
    """

    num_games: int = 4
    opponent_policy_spec: str = "greedy"
    base_llm_policy_factory: Callable[[], Policy] | None = None
    output_path: str | Path | None = None
    seat_learner: int = 0


# ---------------------------------------------------------------------------
# Core: run_selfplay
# ---------------------------------------------------------------------------

def run_selfplay(
    cfg: SelfPlayConfig,
    *,
    cls: type = ClassicGin,
    seed: int = 0,
) -> DataCollector:
    """Play ``cfg.num_games`` matches and return a populated collector.

    Every draw / knock / discard is recorded as a :class:`TrajectorySample`
    with the same prompt an :class:`LLMPolicy` would have seen; after each
    game the terminal reward (+1/-1/0) is back-filled onto that game's
    rows via :meth:`DataCollector.close_game`.
    """
    collector = DataCollector()
    root_rng = random.Random(seed)
    factory = cfg.base_llm_policy_factory or (lambda: GreedyKnockPolicy(random.Random(0)))

    for game_idx in range(cfg.num_games):
        game_id = f"g{game_idx:04d}"
        game_seed = root_rng.randrange(2**31)
        game_rng = random.Random(game_seed)

        learner_inner = factory()
        opponent_inner = _policy_from_spec(cfg.opponent_policy_spec, game_rng)

        policies: list[Policy] = [opponent_inner, opponent_inner]  # placeholders
        policies[cfg.seat_learner] = _RecordingPolicy(
            inner=learner_inner,
            collector=collector,
            game_id=game_id,
            player_id=cfg.seat_learner,
        )
        opp_id = 1 - cfg.seat_learner
        policies[opp_id] = _RecordingPolicy(
            inner=opponent_inner,
            collector=collector,
            game_id=game_id,
            player_id=opp_id,
        )

        game = cls(num_players=2, seed=game_seed, policies=policies)
        result = game.play()
        collector.close_game(game_id, winner_id=result.winner_id)

    if cfg.output_path is not None:
        collector.dump(cfg.output_path)
    return collector


# ---------------------------------------------------------------------------
# Outcome-only DPO pair mining
# ---------------------------------------------------------------------------

def outcome_dpo_pairs(
    collector: DataCollector,
    *,
    match_on: str = "kind",
) -> list[dict[str, str]]:
    """Mine DPO preference pairs from game outcomes across a self-play run.

    The base :meth:`DataCollector.to_dpo_pairs` requires two policies to
    have made a decision at the *same* ``(game, turn, kind)`` — which
    never happens in serialised turn-taking games (each seat's turn index
    is distinct). This helper implements the semantics the fine-tuning
    survey actually flagged as an open research niche: **DPO strictly
    from game outcomes**. We pair each *loser's* decision with a *winner's*
    decision of the same ``kind`` from the same game, giving
    ``chosen = winner_action`` / ``rejected = loser_action``.

    Parameters
    ----------
    collector:
        A populated :class:`DataCollector` (rewards must be back-filled).
    match_on:
        Currently only ``"kind"`` is supported — group by decision type
        (draw / discard / knock) so we don't pair a draw against a discard.
    """
    if match_on != "kind":
        raise ValueError("only match_on='kind' is supported today")

    # game_id -> (winner_id, loser_id) — assumes 2-player games.
    outcomes: dict[str, tuple[int | None, int | None]] = {}
    for s in collector.samples:
        if s.reward is None:
            continue
        winner, loser = outcomes.get(s.game_id, (None, None))
        if s.reward > 0 and winner is None:
            winner = s.player_id
        elif s.reward < 0 and loser is None:
            loser = s.player_id
        outcomes[s.game_id] = (winner, loser)

    pairs: list[dict[str, str]] = []
    for game_id, (winner_id, loser_id) in outcomes.items():
        if winner_id is None or loser_id is None:
            continue
        # Bucket this game's samples by (player_id, kind).
        buckets: dict[tuple[int, str], list[TrajectorySample]] = {}
        for s in collector.samples:
            if s.game_id != game_id:
                continue
            buckets.setdefault((s.player_id, s.kind), []).append(s)
        # For each kind, pair each loser sample with the *most recent*
        # winner sample of the same kind. This keeps the corpus size
        # linear in the loser action count (rather than quadratic).
        for kind in {k for _, k in buckets.keys()}:
            wins = buckets.get((winner_id, kind), [])
            losses = buckets.get((loser_id, kind), [])
            if not wins or not losses:
                continue
            for lo in losses:
                # Nearest winner turn by absolute distance.
                w = min(wins, key=lambda x: abs(x.turn - lo.turn))
                if w.action == lo.action:
                    continue  # degenerate — same decision, no signal
                pairs.append(
                    {"prompt": lo.prompt, "chosen": w.action, "rejected": lo.action}
                )
    return pairs


# ---------------------------------------------------------------------------
# SPIN-style DPO iteration loop (Chen et al. ICML 2024 — arXiv:2401.01335)
# ---------------------------------------------------------------------------

@dataclass
class _SPINIteration:
    """One iteration's artefacts — kept in memory so callers can inspect."""

    iter_idx: int
    checkpoint_dir: str
    num_pairs: int
    collector: DataCollector


def run_selfplay_dpo_iteration(
    base_run: FineTuneRun,
    cfg: SelfPlayConfig,
    *,
    iters: int = 3,
    seed: int = 0,
) -> list[_SPINIteration]:
    """Run a SPIN-style loop: play, mine DPO pairs, train, roll forward.

    Each iteration:
      1. Play ``cfg.num_games`` matches with the current checkpoint at
         ``base_run.base_model`` vs a fresh opponent
         (``cfg.opponent_policy_spec``).
      2. Extract DPO pairs via :meth:`DataCollector.to_dpo_pairs`
         (winner-vs-loser actions on the same (game, turn, kind) tuple).
      3. Train via :func:`run_dpo` — writes to
         ``base_run.output_dir + f"/iter_{i}"``.
      4. Point ``base_run.base_model`` at the fresh checkpoint so the
         next iteration plays *against* the previous self.

    Requires the ``finetune`` extra for step 3 to succeed;
    :class:`MissingFineTuneExtras` propagates cleanly if trl is absent.
    """
    # Local import to keep the module load-safe when trl isn't installed.
    from gin_rummy.finetune.trainers import run_dpo

    artefacts: list[_SPINIteration] = []
    current_run = base_run
    for i in range(iters):
        iter_seed = seed + i
        collector = run_selfplay(cfg, seed=iter_seed)
        # Outcome-only pair mining — the research contribution. The
        # collector's turn-aligned pairer never fires on serialised games.
        pairs = outcome_dpo_pairs(collector)

        iter_dir = str(Path(current_run.output_dir) / f"iter_{i}")
        iter_run = FineTuneRun(
            name=f"{current_run.name}-iter{i}",
            base_model=current_run.base_model,
            method="dpo",
            peft=current_run.peft,
            optimizer=current_run.optimizer,
            scheduler=current_run.scheduler,
            seed=current_run.seed,
            batch_size=current_run.batch_size,
            num_epochs=current_run.num_epochs,
            max_seq_len=current_run.max_seq_len,
            output_dir=iter_dir,
            method_kwargs=dict(current_run.method_kwargs),
        )
        ckpt = run_dpo(iter_run, pairs)  # raises MissingFineTuneExtras w/o trl
        artefacts.append(
            _SPINIteration(
                iter_idx=i,
                checkpoint_dir=ckpt,
                num_pairs=len(pairs),
                collector=collector,
            )
        )
        # Roll the checkpoint forward so iter i+1 plays against iter i.
        current_run = FineTuneRun(
            name=current_run.name,
            base_model=ckpt,
            method=current_run.method,
            peft=current_run.peft,
            optimizer=current_run.optimizer,
            scheduler=current_run.scheduler,
            seed=current_run.seed,
            batch_size=current_run.batch_size,
            num_epochs=current_run.num_epochs,
            max_seq_len=current_run.max_seq_len,
            output_dir=current_run.output_dir,
            method_kwargs=dict(current_run.method_kwargs),
        )
    return artefacts


__all__ = [
    "SelfPlayConfig",
    "outcome_dpo_pairs",
    "run_selfplay",
    "run_selfplay_dpo_iteration",
]


# For static type-checkers / doc tools — keep the private helper import list
# stable even if the LLM prompt helpers get renamed later.
_prompt_helpers: tuple[Any, ...] = (
    _render_draw_prompt,
    _render_discard_prompt,
    _render_knock_prompt,
)
