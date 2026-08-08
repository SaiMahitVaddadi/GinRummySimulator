"""The core Gin Rummy game engine.

One class, ``GinRummyGame``, drives every variant. Rulesets differ in
knock threshold, hand size, and scoring — all injected via constructor
arguments — but the turn structure and end-of-game handling live here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from gin_rummy.cards import Card, Deck
from gin_rummy.meld import optimal_decomposition
from gin_rummy.player import Player
from gin_rummy.policy import Observation, Policy, RandomPolicy
from gin_rummy.scoring import ScoringRules


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    player_id: int
    draw_source: str  # "deck" or "discard"
    drawn: Card | None
    discarded: Card | None
    action: str  # "play", "gin", "knock", "undercut", "draw"
    deadwood_value: int


@dataclass
class GameResult:
    winner_id: int | None
    scores: dict[str, int]
    turns: int
    outcome: str  # "gin", "knock", "undercut", "draw"
    history: list[TurnRecord] = field(default_factory=list)

    @property
    def winner_name(self) -> str | None:
        if self.winner_id is None:
            return None
        return f"Player {self.winner_id + 1}"


class GinRummyGame:
    """Two-to-N-player Gin Rummy. Classic rules by default.

    Parameters
    ----------
    num_players : int, default 2
        Number of players (2..12).
    policies : sequence of ``Policy``, optional
        One policy per player. Defaults to ``RandomPolicy`` for every seat.
    seed : int, optional
        Seed for reproducibility. When ``None``, uses OS entropy.
    knock_limit : int, default 10
        Maximum deadwood at which a knock is legal. ``0`` means only gin ends
        the hand (Oklahoma's ace-upcard rule).
    scoring : ScoringRules
        Bonus/undercut point rules.
    hand_size : int, default 10
        Cards dealt to each player. Adaptively reduced if the deck is small.
    max_turns : int, default 200
        Safety cap on turn count so pathological games terminate.
    """

    def __init__(
        self,
        num_players: int = 2,
        *,
        policies: Sequence[Policy] | None = None,
        seed: int | None = None,
        knock_limit: int = 10,
        scoring: ScoringRules = ScoringRules(),
        hand_size: int = 10,
        max_turns: int = 200,
    ) -> None:
        if not 2 <= num_players <= 12:
            raise ValueError("num_players must be between 2 and 12")
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.deck = Deck(Deck.optimal_deck_count(num_players), rng=self.rng)
        self.players: list[Player] = [
            Player(f"Player {i + 1}", i) for i in range(num_players)
        ]
        self.discard_pile: list[Card] = []
        self.knock_limit = knock_limit
        self.scoring = scoring
        self.hand_size = self._resolve_hand_size(hand_size)
        self.max_turns = max_turns

        default_policies = [RandomPolicy(self.rng) for _ in range(num_players)]
        self.policies: list[Policy] = list(policies) if policies else default_policies
        if len(self.policies) != num_players:
            raise ValueError("Expected one policy per player")

        self._turn = 0

    # ---------- setup ----------

    def _resolve_hand_size(self, requested: int) -> int:
        """Shrink the hand if the shoe can't cover num_players × requested + 5."""
        total = self.deck.num_decks * 52
        needed = self.num_players * requested + 5
        if total >= needed:
            return requested
        return max(3, (total - 5) // self.num_players)

    def _deal(self) -> None:
        for _ in range(self.hand_size):
            for p in self.players:
                card = self.deck.draw()
                assert card is not None, "Deck exhausted during deal"
                p.add(card)
        upcard = self.deck.draw()
        assert upcard is not None, "Deck exhausted for upcard"
        self.discard_pile.append(upcard)

    # ---------- observations ----------

    def _observation(self, player_id: int) -> Observation:
        return Observation(
            hand=tuple(self.players[player_id].hand),
            top_discard=self.discard_pile[-1] if self.discard_pile else None,
            discard_pile_size=len(self.discard_pile),
            deck_size=len(self.deck),
            turn_number=self._turn,
            knock_limit=self.knock_limit,
            player_id=player_id,
            num_players=self.num_players,
            other_hand_sizes=tuple(len(p.hand) for p in self.players),
        )

    # ---------- scoring ----------

    def _score_gin(self, winner_id: int) -> dict[str, int]:
        scores = {p.name: 0 for p in self.players}
        opp_deadwood = 0
        for p in self.players:
            if p.player_id == winner_id:
                continue
            _, _, dv = optimal_decomposition(p.hand)
            opp_deadwood += dv
        scores[self.players[winner_id].name] = self.scoring.gin_bonus + opp_deadwood
        return scores

    def _score_knock(
        self, knocker_id: int, knocker_deadwood: int
    ) -> tuple[dict[str, int], int, str]:
        scores = {p.name: 0 for p in self.players}
        # Best opponent's deadwood determines undercut.
        best_opp_id = -1
        best_opp_dw = -1
        for p in self.players:
            if p.player_id == knocker_id:
                continue
            _, _, dv = optimal_decomposition(p.hand)
            if best_opp_id == -1 or dv < best_opp_dw:
                best_opp_id, best_opp_dw = p.player_id, dv
        if best_opp_dw <= knocker_deadwood:
            scores[self.players[best_opp_id].name] = (
                self.scoring.undercut_bonus + (knocker_deadwood - best_opp_dw)
            )
            return scores, best_opp_id, "undercut"
        gain = self.scoring.knock_bonus
        for p in self.players:
            if p.player_id == knocker_id:
                continue
            _, _, dv = optimal_decomposition(p.hand)
            gain += max(0, dv - knocker_deadwood)
        scores[self.players[knocker_id].name] = gain
        return scores, knocker_id, "knock"

    # ---------- main loop ----------

    def play(self) -> GameResult:
        self._deal()
        history: list[TurnRecord] = []
        scores: dict[str, int] = {p.name: 0 for p in self.players}

        while self._turn < self.max_turns:
            for player_id in range(self.num_players):
                self._turn += 1
                player = self.players[player_id]
                policy = self.policies[player_id]

                # ----- draw -----
                obs = self._observation(player_id)
                source = policy.choose_draw_source(obs)
                if source == "discard" and self.discard_pile:
                    drawn = self.discard_pile.pop()
                else:
                    source = "deck"
                    drawn = self.deck.draw()
                    if drawn is None:
                        return GameResult(None, scores, self._turn, "draw", history)
                player.add(drawn)

                # ----- evaluate hand -----
                _, _, dw_value = optimal_decomposition(player.hand)

                if dw_value == 0:
                    scores = self._score_gin(player_id)
                    history.append(
                        TurnRecord(self._turn, player_id, source, drawn, None, "gin", 0)
                    )
                    return GameResult(player_id, scores, self._turn, "gin", history)

                if self.knock_limit > 0 and dw_value <= self.knock_limit:
                    obs_after = self._observation(player_id)
                    if policy.choose_to_knock(obs_after, dw_value):
                        scores, winner_id, outcome = self._score_knock(
                            player_id, dw_value
                        )
                        history.append(
                            TurnRecord(
                                self._turn, player_id, source, drawn, None, outcome, dw_value
                            )
                        )
                        return GameResult(winner_id, scores, self._turn, outcome, history)

                # ----- discard -----
                obs_after = self._observation(player_id)
                discard = policy.choose_discard(obs_after)
                player.remove(discard)
                self.discard_pile.append(discard)
                history.append(
                    TurnRecord(
                        self._turn, player_id, source, drawn, discard, "play", dw_value
                    )
                )

        return GameResult(None, scores, self._turn, "draw", history)
