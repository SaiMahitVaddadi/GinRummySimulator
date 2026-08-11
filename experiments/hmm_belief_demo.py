"""Scripted-LLM demo of the Y × L interaction (HMM belief tool).

Wires an :class:`LLMPolicy` — with a scripted ``completion_fn`` that
never touches the network — against a :class:`GreedyKnockPolicy`. The
scripted policy behaves as follows on every knock decision:

  1. Emit a tool call for ``opponent_hmm_belief`` (no args).
  2. On the re-ask, inspect the returned posterior and knock iff
     ``P(strong) + P(gin_ready) < 0.3``. The idea: if we're not
     confident the opponent is close to winning, we press the knock;
     if we suspect they're loaded, we hold off (avoiding an undercut).

Draws and discards fall back to the default random policy — this demo
is only about the *knock threshold gate* and the effect of belief.

Everything happens through the injected completion function; no real
LLM API call is made.
"""

from __future__ import annotations

import json
import random
import statistics
from typing import Any

from gin_rummy import ClassicGin, GreedyKnockPolicy
from gin_rummy.game import GameResult, TurnRecord
from gin_rummy.markov.opponent_hand_hmm import OpponentHandHMM
from gin_rummy.policies.heuristic import GreedyKnockPolicy as _GreedyKnockPolicy
from gin_rummy.policies.llm import LLMPolicy
from gin_rummy.policies.tools import bind_hmm_belief_tool


TRAIN_GAMES = 200
EVAL_GAMES = 100
RNG_SEED = 20260810
KNOCK_HOLD_THRESHOLD = 0.3


# --------------------------------------------------------- training --------


def _train_hmm(seed: int, num_games: int) -> OpponentHandHMM:
    """Train an OpponentHandHMM on ``num_games`` self-play ClassicGin
    games from the perspective of seat 1 (the opponent, from the LLM
    seat's POV)."""
    games: list[GameResult] = []
    for i in range(num_games):
        s = seed + i
        rng = random.Random(s)
        engine = ClassicGin(
            num_players=2,
            policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
            seed=s,
        )
        games.append(engine.play())
    hmm = OpponentHandHMM(rng=random.Random(seed))
    hmm.train(games, target_player_id=1, iterations=15, tol=1e-4)
    return hmm


# ------------------------------ engine subclass that exposes history ------


class _HistoryExposingClassicGin(ClassicGin):
    """ClassicGin subclass that accumulates ``TurnRecord``s on ``self.history``
    as they're appended, so tools can read the mid-game trace.

    We override :meth:`play` because the base class only surfaces the
    history at the *end*. Re-implements the loop verbatim but assigns
    into ``self.history``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[TurnRecord] = []

    def play(self) -> GameResult:  # type: ignore[override]
        from gin_rummy.meld import optimal_decomposition

        self._deal()
        self.history = []
        scores: dict[str, int] = {p.name: 0 for p in self.players}

        while self._turn < self.max_turns:
            for player_id in range(self.num_players):
                self._turn += 1
                player = self.players[player_id]
                policy = self.policies[player_id]

                obs = self._observation(player_id)
                source = policy.choose_draw_source(obs)
                if source == "discard" and self.discard_pile:
                    drawn = self.discard_pile.pop()
                else:
                    source = "deck"
                    drawn = self.deck.draw()
                    if drawn is None:
                        return GameResult(
                            None, scores, self._turn, "draw", self.history
                        )
                player.add(drawn)

                _, _, dw_value = optimal_decomposition(player.hand)

                if dw_value == 0:
                    scores = self._score_gin(player_id)
                    self.history.append(
                        TurnRecord(
                            self._turn, player_id, source, drawn, None, "gin", 0
                        )
                    )
                    return GameResult(
                        player_id, scores, self._turn, "gin", self.history
                    )

                if self.knock_limit > 0 and dw_value <= self.knock_limit:
                    obs_after = self._observation(player_id)
                    if policy.choose_to_knock(obs_after, dw_value):
                        scores, winner_id, outcome = self._score_knock(
                            player_id, dw_value
                        )
                        self.history.append(
                            TurnRecord(
                                self._turn,
                                player_id,
                                source,
                                drawn,
                                None,
                                outcome,
                                dw_value,
                            )
                        )
                        return GameResult(
                            winner_id, scores, self._turn, outcome, self.history
                        )

                obs_after = self._observation(player_id)
                discard = policy.choose_discard(obs_after)
                player.remove(discard)
                self.discard_pile.append(discard)
                self.history.append(
                    TurnRecord(
                        self._turn,
                        player_id,
                        source,
                        drawn,
                        discard,
                        "play",
                        dw_value,
                    )
                )

        return GameResult(None, scores, self._turn, "draw", self.history)


# ----------------------- scripted completion function ---------------------


def _build_scripted_completion_fn(hold_threshold: float) -> Any:
    """Return a ``completion_fn(model, messages, **kwargs) -> str`` that
    plays the tool-then-decide protocol for knock decisions and returns
    strategically-sane answers for draw and discard so that the LLM
    actually reaches knockable states.

    Draw/discard heuristic:
        * Draw from discard iff picking it up strictly reduces deadwood
          under the optimal decomposition (the same rule as
          ``draw_if_reduces_deadwood``).
        * Discard the card whose removal minimises the resulting
          deadwood (the same rule as ``discard_lowest_deadwood``).
    These are the exact heuristics used by ``GreedyKnockPolicy``, so the
    only thing the scripted LLM *actually decides* — and where the HMM
    belief has room to matter — is the knock/no-knock gate.

    Knock protocol:
        * Fresh decision → emit the HMM tool call (no args).
        * Tool-result echo → knock iff
          ``P(strong) + P(gin_ready) < hold_threshold``.

    Returns a callable with a ``.stats`` attribute for later inspection.
    """
    import re

    from gin_rummy.cards import Card
    from gin_rummy.meld import optimal_decomposition

    stats = {
        "tool_calls_emitted": 0,
        "knock_decisions": 0,
        "knock_true": 0,
        "knock_false": 0,
        "draws_from_discard": 0,
        "draws_from_deck": 0,
        "discards_chosen": 0,
    }

    _CARD_RE = re.compile(r"(10|[A2-9JQK])([♠♥♦♣])")

    def _parse_hand_line(prompt: str) -> list[Card]:
        """Extract the hand from the ``Hand: ...`` line rendered by
        :func:`_describe_hand` in ``llm.py``. Card glyphs only."""
        for line in prompt.splitlines():
            if line.startswith("Hand:"):
                return [Card(rank, suit) for rank, suit in _CARD_RE.findall(line)]
        return []

    def _parse_top_discard(prompt: str) -> Card | None:
        m = re.search(r"Top of discard: (10|[A2-9JQK])([♠♥♦♣])", prompt)
        if not m:
            return None
        return Card(m.group(1), m.group(2))

    def _completion(model: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        last_user = ""
        for m in messages:
            if m["role"] == "user":
                last_user = m["content"]

        # Was the last user message a tool_result echo?
        is_tool_echo = last_user.lstrip().startswith("{") and "tool_result" in last_user

        if is_tool_echo:
            try:
                payload = json.loads(last_user)
                result = payload.get("tool_result", {})
                posterior = result.get("posterior", {})
            except (json.JSONDecodeError, AttributeError):
                posterior = {}

            danger = float(posterior.get("strong", 0.0)) + float(
                posterior.get("gin_ready", 0.0)
            )
            knock = danger < hold_threshold
            stats["knock_decisions"] += 1
            if knock:
                stats["knock_true"] += 1
            else:
                stats["knock_false"] += 1
            return json.dumps({"knock": knock})

        # Fresh decision — dispatch by prompt shape.
        if "may knock" in last_user:
            stats["tool_calls_emitted"] += 1
            return json.dumps(
                {"tool_call": {"name": "opponent_hmm_belief", "args": {}}}
            )

        if "Legal discards" in last_user:
            hand = _parse_hand_line(last_user)
            if not hand:
                return "{}"
            # Discard-lowest-deadwood.
            best_card = hand[0]
            best_dv = None
            for c in hand:
                remaining = [x for x in hand if x is not c]
                _, _, dv = optimal_decomposition(remaining)
                if best_dv is None or dv < best_dv:
                    best_dv = dv
                    best_card = c
            stats["discards_chosen"] += 1
            return json.dumps({"discard": str(best_card)})

        if "Top of discard" in last_user and (
            '"source"' in last_user or "deck" in last_user
        ):
            hand = _parse_hand_line(last_user)
            top = _parse_top_discard(last_user)
            if not hand or top is None:
                stats["draws_from_deck"] += 1
                return json.dumps({"source": "deck"})
            _, _, base_dv = optimal_decomposition(hand)
            _, _, hyp_dv = optimal_decomposition(hand + [top])
            if hyp_dv < base_dv:
                stats["draws_from_discard"] += 1
                return json.dumps({"source": "discard"})
            stats["draws_from_deck"] += 1
            return json.dumps({"source": "deck"})

        return "{}"

    _completion.stats = stats  # type: ignore[attr-defined]
    return _completion


# ------------------------------- driver -----------------------------------


def _play_eval(
    hmm: OpponentHandHMM,
    seed_base: int,
    num_games: int,
    hold_threshold: float,
) -> dict[str, Any]:
    llm_wins = 0
    opponent_wins = 0
    draws = 0
    knock_events = 0
    knock_deadwoods: list[int] = []
    tool_calls = 0
    aggregate_stats = {"knock_true": 0, "knock_false": 0}

    for i in range(num_games):
        seed = seed_base + i
        rng = random.Random(seed)

        # Build engine first so the LLM can close over its history via
        # attribute reference (the reference is stable — `engine.history`
        # gets *replaced* by the play() call, so we look it up lazily).
        engine = _HistoryExposingClassicGin(
            num_players=2, seed=seed
        )

        completion_fn = _build_scripted_completion_fn(hold_threshold)

        # Wire the HMM belief tool over the engine's live history.
        hmm_tool = bind_hmm_belief_tool(
            hmm,
            get_history=lambda e=engine: e.history,
            get_target_id=lambda: 1,
        )

        llm_policy = LLMPolicy(
            model="scripted",
            completion_fn=completion_fn,
            tools=[hmm_tool],
            max_tool_calls=2,
            fallback=_GreedyKnockPolicy(rng),
        )
        opponent = GreedyKnockPolicy(rng)
        engine.policies = [llm_policy, opponent]

        result = engine.play()
        if result.winner_id == 0:
            llm_wins += 1
        elif result.winner_id == 1:
            opponent_wins += 1
        else:
            draws += 1

        # Collect per-game stats.
        for call in llm_policy.trace:
            if call.kind == "knock" and not call.fell_back:
                if call.parsed and call.parsed.get("knock") is True:
                    knock_events += 1
                    ctx = call.context or {}
                    dv = ctx.get("deadwood_value")
                    if isinstance(dv, int):
                        knock_deadwoods.append(dv)
            if call.kind == "tool_call":
                tool_calls += 1

        cstats = getattr(completion_fn, "stats", {})
        aggregate_stats["knock_true"] += cstats.get("knock_true", 0)
        aggregate_stats["knock_false"] += cstats.get("knock_false", 0)

    total_decisions = aggregate_stats["knock_true"] + aggregate_stats["knock_false"]
    knock_freq = (
        aggregate_stats["knock_true"] / total_decisions if total_decisions else 0.0
    )

    return {
        "games": num_games,
        "llm_wins": llm_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "llm_win_rate": llm_wins / num_games if num_games else 0.0,
        "knock_events": knock_events,
        "mean_knock_deadwood": (
            statistics.mean(knock_deadwoods) if knock_deadwoods else 0.0
        ),
        "tool_calls": tool_calls,
        "knock_decisions": total_decisions,
        "knock_yes_fraction": knock_freq,
    }


def main() -> None:
    print(
        f"[hmm-belief-demo] training OpponentHandHMM on {TRAIN_GAMES} ClassicGin games "
        f"(seed {RNG_SEED})..."
    )
    hmm = _train_hmm(seed=RNG_SEED, num_games=TRAIN_GAMES)

    print(
        f"[hmm-belief-demo] evaluating scripted LLM vs GreedyKnockPolicy for "
        f"{EVAL_GAMES} games..."
    )
    stats_with_belief = _play_eval(
        hmm,
        seed_base=RNG_SEED + 10_000,
        num_games=EVAL_GAMES,
        hold_threshold=KNOCK_HOLD_THRESHOLD,
    )

    # For contrast: same LLM protocol but "belief" collapsed to always
    # knock (threshold = infinity). This is what an *un-informed* scripted
    # LLM looks like — knock-whenever-legal, exactly like GreedyKnockPolicy.
    print("[hmm-belief-demo] control run (belief-ignored, always knock)...")
    stats_control = _play_eval(
        hmm,
        seed_base=RNG_SEED + 10_000,
        num_games=EVAL_GAMES,
        hold_threshold=float("inf"),
    )

    def _print(tag: str, s: dict[str, Any]) -> None:
        print(
            f"[{tag}] games={s['games']}  llm_wins={s['llm_wins']}  "
            f"opp_wins={s['opponent_wins']}  draws={s['draws']}  "
            f"win_rate={s['llm_win_rate']:.3f}"
        )
        print(
            f"[{tag}] knock_events={s['knock_events']}  "
            f"mean_knock_deadwood={s['mean_knock_deadwood']:.2f}  "
            f"tool_calls={s['tool_calls']}  "
            f"knock_yes_fraction={s['knock_yes_fraction']:.3f}"
        )

    print()
    _print("belief-gated  ", stats_with_belief)
    _print("belief-ignored", stats_control)


if __name__ == "__main__":
    main()
