"""HMM-conditioned cascade response models and comparison harness.

This module wires the trained :class:`OpponentHandHMM` into the
probabilistic cascade of :mod:`gin_rummy.markov.cascade`. The baseline
:func:`greedy_response_model` treats every opponent identically; here we
let the *belief* over the opponent's latent hand-strength class modulate
their discard distribution.

Design summary
--------------
The core primitive is
:func:`hmm_conditioned_response_model(hmm, history, target_player_id)`
which returns a callable with the same ``(state, actor_id) ->
{card: prob}`` signature as :func:`greedy_response_model`. For the
*target* actor we:

1. Extract the observation trace for ``target_player_id`` from
   ``history`` and, as the cascade descends, extend it with each
   subsequent target-player discard the cascade explores. Draw source
   inside the cascade is unknown so we default to ``"deck"``.
2. Ask the HMM for the posterior over its 5 latent states at the last
   turn of the current observation trace.
3. Map that posterior onto the semantic hand-strength classes (identity
   mapping by default; callers can pass an explicit alignment when the
   Baum-Welch permutation matters).
4. Scale the softmax temperature by ``1 + β · P(strong ∪ gin_ready)`` —
   a "strong opponent" belief widens the softmax so mass shifts toward
   the higher-value discards a strong hand is more likely to shed.

For non-target actors we fall back to the ordinary
:func:`greedy_response_model` at ``base_temperature``. This keeps the
cascade honest: only the *belief* over the modelled opponent gets
plugged in; every other seat runs the same baseline logic.

Comparison harness
------------------
:func:`run_cascade_comparison` replays a batch of games up to a fixed
mid-game turn, snapshots the hands and the top-of-discard, and enumerates
top-k cascades under both the baseline and the HMM-conditioned model.
It reports:

* top-1 predictive accuracy against the *actually observed* next discard
* fraction of games where the 3-step observed continuation appears in
  the top-5 cascade
* per-game records for downstream JSONL dumping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from gin_rummy.cards import Card
from gin_rummy.game import GameResult, TurnRecord
from gin_rummy.markov.cascade import (
    CascadeState,
    CascadeStep,
    ResponseModel,
    cascade_value,
    greedy_response_model,
    probabilistic_cascade,
)
from gin_rummy.markov.opponent_hand_hmm import (
    HAND_STRENGTH_CLASSES,
    OpponentHandHMM,
    extract_observations,
)
from gin_rummy.meld import optimal_decomposition


# ------------------------------------------------------------- constants ---

_STRONG_CLASS_IDS: tuple[int, ...] = tuple(
    HAND_STRENGTH_CLASSES.index(n) for n in ("strong", "gin_ready")
)
_WEAK_CLASS_IDS: tuple[int, ...] = tuple(
    HAND_STRENGTH_CLASSES.index(n) for n in ("very_weak", "weak")
)


# ------------------------------------------------------- helper primitives -


def _softmax(scores: dict[Card, float], temperature: float) -> dict[Card, float]:
    """Numerically-stable softmax; ``temperature <= 0`` collapses to argmax."""
    if not scores:
        return {}
    if temperature <= 0.0:
        best = max(scores.values())
        winners = [c for c, s in scores.items() if s == best]
        share = 1.0 / len(winners)
        return {c: (share if c in winners else 0.0) for c in scores}
    m = max(scores.values())
    exps = {c: math.exp((s - m) / temperature) for c, s in scores.items()}
    z = sum(exps.values())
    if z <= 0.0:
        n = len(scores)
        return {c: 1.0 / n for c in scores}
    return {c: v / z for c, v in exps.items()}


def _score_discards_by_deadwood(hand: Sequence[Card]) -> dict[Card, float]:
    """For each candidate discard, compute -post-discard-deadwood.

    Higher scores prefer the greedy-legal discard (least remaining
    deadwood). Mirrors what :func:`greedy_response_model` does under the
    hood so both models share the same score surface — only the softmax
    temperature differs.
    """
    scores: dict[Card, float] = {}
    for c in hand:
        remainder = [x for x in hand if x is not c]
        _, _, dw = optimal_decomposition(remainder)
        scores[c] = -float(dw)
    return scores


def _class_posterior_from_states(
    state_posterior: Sequence[float],
    state_to_class_mapping: Sequence[int] | None,
    n_classes: int,
) -> list[float]:
    """Fold a per-hidden-state posterior into a per-semantic-class posterior."""
    if state_to_class_mapping is None:
        # Identity mapping — assumes hmm.n_states == n_classes and the
        # learned states happen to line up in order (or the caller is happy
        # with an arbitrary alignment for their downstream diagnostic).
        out = [0.0] * n_classes
        for i, p in enumerate(state_posterior):
            if 0 <= i < n_classes:
                out[i] += p
        # Normalise (mapping may drop mass if len mismatch).
        s = sum(out)
        if s > 0:
            out = [v / s for v in out]
        else:
            out = [1.0 / n_classes] * n_classes
        return out

    out = [0.0] * n_classes
    for state_idx, class_idx in enumerate(state_to_class_mapping):
        if state_idx >= len(state_posterior):
            break
        if 0 <= class_idx < n_classes:
            out[class_idx] += state_posterior[state_idx]
    s = sum(out)
    if s > 0:
        out = [v / s for v in out]
    else:
        out = [1.0 / n_classes] * n_classes
    return out


# ---------------------------------------------------- response model ------


def hmm_conditioned_response_model(
    hmm: OpponentHandHMM,
    history: Sequence[TurnRecord],
    target_player_id: int,
    *,
    base_temperature: float = 1.0,
    beta: float = 0.5,
    state_to_class_mapping: Sequence[int] | None = None,
) -> ResponseModel:
    """Build a :type:`ResponseModel` conditioned on an HMM belief.

    Parameters
    ----------
    hmm : OpponentHandHMM
        Trained HMM over opponent hand-strength classes.
    history : Sequence[TurnRecord]
        Public game history observed *before* the cascade root. Only the
        target player's turns contribute to the observation trace.
    target_player_id : int
        The seat whose hand-strength belief we condition on.
    base_temperature : float, default 1.0
        Softmax temperature for the baseline (non-target) response and
        the un-tilted starting point of the target response.
    beta : float, default 0.5
        Strength of the tilt. Effective temperature for the target is
        ``base_temperature * (1 + beta * P(strong ∪ gin_ready))``.
    state_to_class_mapping : sequence of int, optional
        Mapping ``learned_state_idx -> semantic_class_idx`` (see
        :func:`gin_rummy.markov.opponent_hand_hmm.HAND_STRENGTH_CLASSES`).
        Defaults to identity, which is correct when the caller has no
        alignment or when the HMM was trained with an ordering that
        already matches ``HAND_STRENGTH_CLASSES``.

    Semantics
    ---------
    * ``strong`` / ``gin_ready`` belief *sharpens* the softmax → mass
      concentrates on the argmin-deadwood picks, which are exactly the
      highest-value non-meld cards (a strong hand's leftover deadwood).
    * ``very_weak`` / ``weak`` belief does not tilt (weak hands shed high
      cards regardless — the baseline temperature already predicts
      that).
    * ``medium`` belief also does not tilt: it degenerates to baseline.

    Concretely: only the *strong* mass factors into the temperature
    scale; weak/medium leave temperature at ``base_temperature``.

    Note the *division*: ``T_eff = T_base / (1 + β · P(strong))``.
    Sharpening under strong belief makes the model more decisive about
    the specific high-value non-meld discard the opponent is most likely
    to shed. The design-doc "shift mass toward high-value non-meld
    cards" is realised as concentration on the argmin-deadwood tail
    which, by construction of the greedy score, *is* the high-value
    non-meld tail.
    """
    base_obs = list(extract_observations(history, target_player_id))
    n_classes = len(HAND_STRENGTH_CLASSES)

    # Extended observation cache: keyed by the number of *cascade*
    # discards the target has made so far along a path. We compute the
    # posterior lazily and memoise per-cascade-node inside the closure.
    posterior_cache: dict[tuple[int, ...], list[float]] = {}

    def _posterior_for_extension(extra_obs: tuple[int, ...]) -> list[float]:
        key = extra_obs
        cached = posterior_cache.get(key)
        if cached is not None:
            return cached
        full = base_obs + list(extra_obs)
        if not full:
            uniform = [1.0 / n_classes] * n_classes
            posterior_cache[key] = uniform
            return uniform
        gamma = hmm.posteriors(full)
        # We care about the belief at the *most recent* turn.
        state_post = gamma[-1] if gamma else [1.0 / hmm.n_states] * hmm.n_states
        class_post = _class_posterior_from_states(
            state_post, state_to_class_mapping, n_classes
        )
        posterior_cache[key] = class_post
        return class_post

    def _extension_from_state(state: CascadeState) -> tuple[int, ...]:
        """Reconstruct the target's cascade-added observation sequence.

        We rely on the fact that :class:`CascadeState.extra` may carry an
        opaque ``"target_cascade_obs"`` key with a tuple of ints. If not
        present (top-of-cascade), we return an empty tuple.
        """
        extra = state.extra or {}
        seq = extra.get("target_cascade_obs")
        if seq is None:
            return ()
        return tuple(seq)

    def _model(state: CascadeState, actor_id: int) -> dict[Card, float]:
        hand = list(state.hands.get(actor_id, []))
        if not hand:
            return {}

        if actor_id != target_player_id:
            # Non-target: baseline.
            return greedy_response_model(
                state, actor_id, temperature=base_temperature
            )

        # Target actor: fetch belief over classes, tilt temperature.
        extra_obs = _extension_from_state(state)
        class_post = _posterior_for_extension(extra_obs)
        p_strong = sum(class_post[i] for i in _STRONG_CLASS_IDS)

        # Sharpen (divide) under strong belief so mass concentrates on
        # the argmin-deadwood picks — which are the high-value non-meld
        # cards by construction of the score.
        temp = base_temperature / (1.0 + beta * p_strong)
        scores = _score_discards_by_deadwood(hand)
        return _softmax(scores, temp)

    return _model


# ---------------------------------------------------- config bundle ------


@dataclass
class HMMCascadeConfig:
    """Bundle of hyperparameters + trained HMM for reproducibility."""

    hmm: OpponentHandHMM
    beta: float = 0.5
    base_temperature: float = 1.0
    cascade_depth: int = 3
    top_k: int = 5
    state_to_class_mapping: Sequence[int] | None = None


# ---------------------------------------------------- comparison harness -


@dataclass
class CascadeComparisonRow:
    """Per-game record from :func:`run_cascade_comparison`."""

    game_index: int
    turn: int
    baseline_top1_correct: bool
    hmm_top1_correct: bool
    baseline_top5_covers_observed: bool
    hmm_top5_covers_observed: bool
    baseline_top1_prob: float
    hmm_top1_prob: float
    baseline_top1_value: float
    hmm_top1_value: float
    observed_next_discard: str | None = None


def replay_with_snapshots(
    engine_factory: Callable[[int], "object"],
    game_seed: int,
    target_player_id: int,
    target_turn: int,
) -> tuple[CascadeState, GameResult, TurnRecord | None] | None:
    """Play a game and snapshot the cascade state just before ``target_turn``.

    Parameters
    ----------
    engine_factory : callable
        ``(seed) -> game engine``. The engine must expose ``players``,
        ``discard_pile``, ``_turn`` and a ``play()`` -> :class:`GameResult`.
    game_seed : int
        Seed passed to the engine.
    target_player_id : int
        Seat whose next-discard we'll try to predict from the snapshot.
    target_turn : int
        Turn number at which to snapshot (1-indexed to match
        ``TurnRecord.turn``).

    Returns
    -------
    (state, result, observed_next_record) or None
        ``state`` is the reconstructed :class:`CascadeState` right before
        the target player's discard at that turn; ``result`` is the full
        played :class:`GameResult`; ``observed_next_record`` is the
        :class:`TurnRecord` at ``target_turn`` for the target player (or
        ``None`` if that turn belongs to another player or was terminal
        without a discard).
    """
    # Delegate to a specialised instrumented engine that snapshots hands.
    engine = engine_factory(game_seed)
    snap = {"state": None, "record": None}

    from gin_rummy.game import TurnRecord as _TR
    from gin_rummy.meld import optimal_decomposition as _dw

    # We monkey-patch the engine's play loop rather than requiring a
    # subclass so callers can point us at any engine variant. We rely on
    # its public attributes and the standard rummy turn structure.
    #
    # Standard turn: choose_draw_source -> draw -> gin? knock? discard.
    # We snapshot *just before the discard* at target_turn for
    # target_player_id.

    orig_play = engine.play

    def _instrumented_play() -> GameResult:
        # Re-implement the loop with a snapshot hook. This duplicates
        # some of :meth:`GinRummyGame.play` but keeps the module free of
        # engine-subclass boilerplate.
        engine._deal()
        history: list[_TR] = []
        scores = {p.name: 0 for p in engine.players}

        while engine._turn < engine.max_turns:
            for player_id in range(engine.num_players):
                engine._turn += 1
                player = engine.players[player_id]
                policy = engine.policies[player_id]

                obs = engine._observation(player_id)
                source = policy.choose_draw_source(obs)
                if source == "discard" and engine.discard_pile:
                    drawn = engine.discard_pile.pop()
                else:
                    source = "deck"
                    drawn = engine.deck.draw()
                    if drawn is None:
                        return GameResult(None, scores, engine._turn, "draw", history)
                player.add(drawn)

                _, _, dw_value = _dw(player.hand)

                if dw_value == 0:
                    scores = engine._score_gin(player_id)
                    history.append(
                        _TR(engine._turn, player_id, source, drawn, None, "gin", 0)
                    )
                    return GameResult(player_id, scores, engine._turn, "gin", history)

                if engine.knock_limit > 0 and dw_value <= engine.knock_limit:
                    obs_after = engine._observation(player_id)
                    if policy.choose_to_knock(obs_after, dw_value):
                        scores, winner_id, outcome = engine._score_knock(
                            player_id, dw_value
                        )
                        history.append(
                            _TR(
                                engine._turn,
                                player_id,
                                source,
                                drawn,
                                None,
                                outcome,
                                dw_value,
                            )
                        )
                        return GameResult(
                            winner_id, scores, engine._turn, outcome, history
                        )

                obs_after = engine._observation(player_id)
                discard = policy.choose_discard(obs_after)

                # ---- snapshot hook ----
                if (
                    engine._turn == target_turn
                    and player_id == target_player_id
                    and snap["state"] is None
                ):
                    hands = {
                        pid: list(engine.players[pid].hand)
                        for pid in range(engine.num_players)
                    }
                    snap["state"] = CascadeState(
                        hands=hands,
                        top_discard=(
                            engine.discard_pile[-1] if engine.discard_pile else None
                        ),
                        turn=engine._turn,
                        extra={},
                    )
                    # We'll fill in snap["record"] below once we know the
                    # discard the policy would actually make.

                player.remove(discard)
                engine.discard_pile.append(discard)
                rec = _TR(
                    engine._turn, player_id, source, drawn, discard, "play", dw_value
                )
                history.append(rec)

                if (
                    engine._turn == target_turn
                    and player_id == target_player_id
                    and snap["record"] is None
                ):
                    snap["record"] = rec

        return GameResult(None, scores, engine._turn, "draw", history)

    # Swap in the instrumented loop for this one play(). We don't
    # restore since the engine is single-use.
    engine.play = _instrumented_play  # type: ignore[assignment]
    result = engine.play()
    engine.play = orig_play  # type: ignore[assignment]

    if snap["state"] is None:
        return None
    return snap["state"], result, snap["record"]


def _path_matches_observed(
    path: list[CascadeStep],
    observed: list[TurnRecord],
    target_player_id: int,
) -> bool:
    """True iff the cascade's actions match the observed continuation.

    Compares each cascade step's action (a :class:`Card` discard) to the
    corresponding turn record's ``discarded`` card in order. We stop
    once we run out of either.
    """
    obs_iter = iter(r for r in observed if r.discarded is not None)
    for step in path:
        rec = next(obs_iter, None)
        if rec is None:
            return False
        if not isinstance(step.action, Card):
            return False
        if step.action != rec.discarded:
            return False
    return True


def run_cascade_comparison(
    games_and_snapshots: Sequence[
        tuple[int, CascadeState, GameResult, TurnRecord | None]
    ],
    target_player_id: int,
    hmm: OpponentHandHMM,
    *,
    depth: int = 3,
    top_k: int = 5,
    beta: float = 0.5,
    base_temperature: float = 1.0,
    state_to_class_mapping: Sequence[int] | None = None,
) -> dict:
    """Compare the baseline and HMM-conditioned cascades on a batch.

    Parameters
    ----------
    games_and_snapshots : sequence of ``(game_index, state, result, record)``
        Output of :func:`replay_with_snapshots` batched over many games.
        Games where snapshotting failed (returned ``None``) should be
        filtered out by the caller.
    target_player_id : int
        Seat of interest.
    hmm : OpponentHandHMM
        Trained HMM used by the conditioned model.
    depth, top_k : int
        Cascade parameters.
    beta, base_temperature : float
        Passed to :func:`hmm_conditioned_response_model`.

    Returns
    -------
    dict
        ``{"rows": [...], "baseline": {"top1_correct": int, "top1_total":
        int, "top5_cover": int, "top5_total": int}, "hmm": {...}, ...}``.
    """
    rows: list[CascadeComparisonRow] = []

    b_top1_correct = 0
    b_top1_total = 0
    b_cover = 0
    b_cover_total = 0
    h_top1_correct = 0
    h_top1_total = 0
    h_cover = 0
    h_cover_total = 0

    actor_order = [target_player_id, 1 - target_player_id]  # 2p only

    for game_index, state, result, record in games_and_snapshots:
        # History up to (but excluding) the target turn.
        prefix = [r for r in result.history if r.turn < state.turn]
        # Observed continuation: target's subsequent discards.
        continuation = [
            r
            for r in result.history
            if r.turn >= state.turn
            and r.player_id == target_player_id
            and r.discarded is not None
        ]

        # ---- baseline cascade
        baseline_model = lambda s, a: greedy_response_model(  # noqa: E731
            s, a, temperature=base_temperature
        )
        baseline_paths = probabilistic_cascade(
            state,
            baseline_model,
            depth=depth,
            top_k=top_k,
            actor_order=actor_order,
        )

        # ---- HMM-conditioned cascade
        hmm_model = hmm_conditioned_response_model(
            hmm,
            prefix,
            target_player_id,
            base_temperature=base_temperature,
            beta=beta,
            state_to_class_mapping=state_to_class_mapping,
        )
        hmm_paths = probabilistic_cascade(
            state,
            hmm_model,
            depth=depth,
            top_k=top_k,
            actor_order=actor_order,
        )

        observed_next = record.discarded if record is not None else None

        # Top-1 next-discard accuracy: is the first action of the top-1
        # path (which belongs to the target since actor_order[0] == target)
        # equal to the observed next discard?
        def _top1_action(paths: list[list[CascadeStep]]) -> Card | None:
            if not paths or not paths[0]:
                return None
            act = paths[0][0].action
            return act if isinstance(act, Card) else None

        b_top1 = _top1_action(baseline_paths)
        h_top1 = _top1_action(hmm_paths)

        if observed_next is not None:
            b_top1_total += 1
            h_top1_total += 1
            b_correct = b_top1 == observed_next
            h_correct = h_top1 == observed_next
            if b_correct:
                b_top1_correct += 1
            if h_correct:
                h_top1_correct += 1
        else:
            b_correct = False
            h_correct = False

        # Top-5 coverage of the observed 3-step continuation:
        # does any of the top-k paths' action sequence for the *target*
        # match the first ``depth``-many observed target discards? (We
        # only compare target actions because the cascade alternates.)
        def _target_actions(path: list[CascadeStep]) -> list[Card]:
            out: list[Card] = []
            for step in path:
                if step.actor_id == target_player_id and isinstance(step.action, Card):
                    out.append(step.action)
            return out

        obs_target_seq = [r.discarded for r in continuation if r.discarded is not None]
        target_steps_per_path = max(1, depth // 2 + depth % 2)
        obs_prefix = obs_target_seq[:target_steps_per_path]

        cover_eligible = len(obs_prefix) == target_steps_per_path
        b_cov = False
        h_cov = False
        if cover_eligible:
            b_cover_total += 1
            h_cover_total += 1
            for path in baseline_paths:
                if _target_actions(path)[:target_steps_per_path] == obs_prefix:
                    b_cov = True
                    break
            for path in hmm_paths:
                if _target_actions(path)[:target_steps_per_path] == obs_prefix:
                    h_cov = True
                    break
            if b_cov:
                b_cover += 1
            if h_cov:
                h_cover += 1

        # Path values
        def _val(paths: list[list[CascadeStep]]) -> tuple[float, float]:
            if not paths or not paths[0]:
                return (0.0, 0.0)
            p = 1.0
            for step in paths[0]:
                p *= step.probability
            v = cascade_value(
                paths[0],
                perspective_actor_id=target_player_id,
                initial_hand=list(state.hands.get(target_player_id, [])),
            )
            return p, v

        b_p, b_v = _val(baseline_paths)
        h_p, h_v = _val(hmm_paths)

        rows.append(
            CascadeComparisonRow(
                game_index=game_index,
                turn=state.turn,
                baseline_top1_correct=b_correct,
                hmm_top1_correct=h_correct,
                baseline_top5_covers_observed=b_cov,
                hmm_top5_covers_observed=h_cov,
                baseline_top1_prob=b_p,
                hmm_top1_prob=h_p,
                baseline_top1_value=b_v,
                hmm_top1_value=h_v,
                observed_next_discard=str(observed_next) if observed_next else None,
            )
        )

    return {
        "rows": rows,
        "baseline": {
            "top1_correct": b_top1_correct,
            "top1_total": b_top1_total,
            "top5_cover": b_cover,
            "top5_total": b_cover_total,
        },
        "hmm": {
            "top1_correct": h_top1_correct,
            "top1_total": h_top1_total,
            "top5_cover": h_cover,
            "top5_total": h_cover_total,
        },
    }


__all__ = [
    "CascadeComparisonRow",
    "HMMCascadeConfig",
    "hmm_conditioned_response_model",
    "replay_with_snapshots",
    "run_cascade_comparison",
]
