"""Explainability primitives for card-game policies.

Following the ranked recommendations from the survey (Milani et al. 2023;
Juozapaitis et al. 2019; van der Waa et al. 2018; Amir & Amir 2018; Turpin
et al. NeurIPS 2023 for the LLM-CoT caveat), this module ships the
five techniques that are (i) defensible, (ii) tractable at rummy scale,
(iii) uniform across heuristic / RL / LLM / hybrid policies:

1. ``introspect_hand``           — optimal decomposition + per-discard deadwood delta.
2. ``discard_regret_table``      — contrastive "what if you had discarded X instead?"
3. ``reward_channels``           — Juozapaitis-style channel decomposition.
4. ``behavioral_fingerprint``    — draw-source / discard-rank / knock-timing histograms.
5. ``rollout_regret``            — Monte-Carlo counterfactual regret via re-rollouts.

The LLM CoT rationale is *logged, not trusted* — see ``LLMPolicy.trace``.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from gin_rummy.cards import Card
from gin_rummy.game import GinRummyGame, TurnRecord
from gin_rummy.meld import optimal_decomposition
from gin_rummy.policy import Observation, Policy


# ---------- 1. Optimal-decomposition introspection ----------

@dataclass(frozen=True)
class HandIntrospection:
    hand: tuple[Card, ...]
    melds: tuple[tuple[Card, ...], ...]
    deadwood: tuple[Card, ...]
    deadwood_value: int
    per_discard: dict[str, int]  # card -> resulting deadwood value


def introspect_hand(hand: Sequence[Card]) -> HandIntrospection:
    """Ground-truth analysis of a hand. No ML — pure game logic."""
    melds, deadwood, dv = optimal_decomposition(list(hand))
    per_discard: dict[str, int] = {}
    for i in range(len(hand)):
        remainder = list(hand[:i]) + list(hand[i + 1 :])
        _, _, dv_after = optimal_decomposition(remainder)
        per_discard[str(hand[i])] = dv_after
    return HandIntrospection(
        hand=tuple(hand),
        melds=tuple(tuple(m) for m in melds),
        deadwood=tuple(deadwood),
        deadwood_value=dv,
        per_discard=per_discard,
    )


# ---------- 2. Discard-regret contrastive table ----------

def discard_regret_table(hand: Sequence[Card], chosen: Card) -> dict[str, int]:
    """Return delta_deadwood for each legal discard vs the chosen one.

    Positive = alternative would leave *more* deadwood than the chosen card
    (chosen was better); negative = alternative would have been strictly
    better.
    """
    intro = introspect_hand(hand)
    chosen_dv = intro.per_discard.get(str(chosen))
    if chosen_dv is None:
        raise ValueError(f"{chosen} not in hand")
    return {c: dv - chosen_dv for c, dv in intro.per_discard.items()}


# ---------- 3. Reward-channel decomposition ----------

@dataclass(frozen=True)
class ChannelContribution:
    deadwood_reduction: int
    meld_formation_delta: int  # change in number of melds
    knock_bonus_progress: int  # deadwood_before - knock_limit if crossing threshold


def reward_channels(
    hand_before: Sequence[Card],
    hand_after: Sequence[Card],
    knock_limit: int,
) -> ChannelContribution:
    """Juozapaitis-style split of a turn's value into interpretable channels."""
    _, _, dv_before = optimal_decomposition(list(hand_before))
    melds_before, _, _ = optimal_decomposition(list(hand_before))
    melds_after, _, dv_after = optimal_decomposition(list(hand_after))

    crossed = 0
    if dv_before > knock_limit and dv_after <= knock_limit:
        crossed = dv_before - knock_limit
    return ChannelContribution(
        deadwood_reduction=dv_before - dv_after,
        meld_formation_delta=len(melds_after) - len(melds_before),
        knock_bonus_progress=crossed,
    )


# ---------- 4. Behavioral fingerprint ----------

@dataclass
class BehavioralFingerprint:
    policy_name: str
    draw_sources: Counter[str] = field(default_factory=Counter)
    discard_ranks: Counter[str] = field(default_factory=Counter)
    knock_turns: list[int] = field(default_factory=list)
    gin_turns: list[int] = field(default_factory=list)
    total_turns: int = 0

    def summary(self) -> str:
        n = self.total_turns or 1
        draws = "  ".join(
            f"{k}={v / n * 100:5.1f}%" for k, v in self.draw_sources.most_common()
        )
        top_discards = "  ".join(f"{k}:{v}" for k, v in self.discard_ranks.most_common(5))
        avg_knock = (
            sum(self.knock_turns) / len(self.knock_turns) if self.knock_turns else float("nan")
        )
        return (
            f"[{self.policy_name}] draws: {draws}\n"
            f"  top discarded ranks: {top_discards}\n"
            f"  mean knock turn: {avg_knock:.1f}  (n={len(self.knock_turns)} knocks, "
            f"{len(self.gin_turns)} gins)"
        )


def fingerprint_from_history(
    history: Sequence[TurnRecord], policy_name: str, policy_seat_id: int
) -> BehavioralFingerprint:
    """Extract behavioral stats for one seat from a game's turn history."""
    fp = BehavioralFingerprint(policy_name=policy_name)
    for rec in history:
        if rec.player_id != policy_seat_id:
            continue
        fp.total_turns += 1
        fp.draw_sources[rec.draw_source] += 1
        if rec.discarded is not None:
            fp.discard_ranks[rec.discarded.rank] += 1
        if rec.action == "knock":
            fp.knock_turns.append(rec.turn)
        elif rec.action == "gin":
            fp.gin_turns.append(rec.turn)
    return fp


# ---------- 5. Monte-Carlo rollout regret ----------

def rollout_regret(
    *,
    game_cls: type[GinRummyGame],
    baseline: Policy,
    variant: Policy,
    n_rollouts: int = 50,
    base_seed: int = 0,
    num_players: int = 2,
    **game_kwargs,
) -> tuple[float, int, int]:
    """Play ``n_rollouts`` games with ``baseline`` in both seats, then with
    ``variant`` in seat 1 vs. ``baseline`` in seat 2 on identical seeds.
    Returns (win-rate delta, variant wins, baseline wins) — a Monte-Carlo
    approximation of the regret of using ``variant`` over ``baseline``.
    """
    variant_wins = 0
    baseline_wins = 0
    for i in range(n_rollouts):
        rng_v = random.Random(base_seed + i)
        rng_b = random.Random(base_seed + i)
        g_v = game_cls(
            num_players,
            policies=[variant, baseline],
            seed=base_seed + i,
            **game_kwargs,
        )
        g_b = game_cls(
            num_players,
            policies=[baseline, baseline],
            seed=base_seed + i,
            **game_kwargs,
        )
        rv = g_v.play()
        rb = g_b.play()
        if rv.winner_id == 0:
            variant_wins += 1
        if rb.winner_id == 0:
            baseline_wins += 1
        # rng_v / rng_b are held for symmetry; policies pull from game.rng.
        del rng_v, rng_b
    n = variant_wins + baseline_wins
    delta = (variant_wins - baseline_wins) / n if n else 0.0
    return delta, variant_wins, baseline_wins
