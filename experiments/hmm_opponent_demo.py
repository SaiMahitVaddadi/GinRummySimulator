"""End-to-end demo for the Gin Rummy opponent-hand HMM.

Runs the following pipeline:

1. Play ``TRAIN_GAMES`` ClassicGin hands with two ``GreedyKnockPolicy``
   seats, capturing both the standard turn history *and* per-turn
   snapshots of the target seat's ground-truth hand-strength class.
2. Fit an :class:`OpponentHandHMM` (5 latent states, 8 observation
   symbols) on the training histories.
3. Play ``HELDOUT_GAMES`` fresh hands with the same instrumentation.
4. Score the trained HMM on held-out data:
     * Final log-likelihood (train and held-out, per turn).
     * Viterbi accuracy against ground truth after we align the HMM's
       arbitrarily-permuted latent state labels with the semantic
       hand-strength classes by max-count matching on the training
       data. This alignment step is critical — Baum-Welch has no way
       to know which learned state is "very_weak" vs "gin_ready".

The point is not to build a strong predictor — it's to see whether the
public draw/discard signal is strong enough for a 5-state HMM to
recover any meaningful structure at all under a fixed, weak opponent
policy.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Sequence

from gin_rummy import ClassicGin, GreedyKnockPolicy
from gin_rummy.game import GameResult
from gin_rummy.markov.opponent_hand_hmm import (
    HAND_STRENGTH_CLASSES,
    N_OBS,
    OBSERVATION_ALPHABET,
    OpponentHandHMM,
    encode_observation,
    hand_strength_class,
    hand_strength_index,
)
from gin_rummy.meld import optimal_decomposition


TRAIN_GAMES = 200
HELDOUT_GAMES = 50
TARGET_SEAT = 1
N_STATES = len(HAND_STRENGTH_CLASSES)
EM_ITERATIONS = 25
RNG_SEED = 20250809


# ---------------------------------------------------- instrumented player --


class _InstrumentedClassicGin(ClassicGin):
    """Subclass that snapshots the target seat's hand-strength label at
    exactly the same points where the standard turn record is appended.

    We can't recompute hand strength from ``TurnRecord`` alone because
    the record doesn't carry the full hand — only the drawn/discarded
    cards. So we override :meth:`play` isn't possible cleanly; instead
    we install a hook on ``optimal_decomposition``... no — the cleanest
    path is to just replay the game with the extra bookkeeping.

    In practice we do it by peeking at the player's hand from
    ``self.players[target].hand`` inside a post-turn wrapper. We
    override :meth:`_observation` to piggy-back on the engine's own
    call sequence.
    """

    def __init__(self, *args, target_seat: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target_seat = target_seat
        self.target_obs_seq: list[int] = []
        self.target_truth_seq: list[int] = []

    def play(self) -> GameResult:  # type: ignore[override]
        # Reproduce the base loop but capture snapshots per target turn.
        self._deal()
        history = []
        scores = {p.name: 0 for p in self.players}

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
                        return GameResult(None, scores, self._turn, "draw", history)
                player.add(drawn)

                _, _, dw_value = optimal_decomposition(player.hand)

                if dw_value == 0:
                    scores = self._score_gin(player_id)
                    if player_id == self._target_seat:
                        self.target_obs_seq.append(encode_observation(source, None))
                        self.target_truth_seq.append(
                            hand_strength_index(player.hand)
                        )
                    from gin_rummy.game import TurnRecord

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
                        if player_id == self._target_seat:
                            self.target_obs_seq.append(
                                encode_observation(source, None)
                            )
                            self.target_truth_seq.append(
                                hand_strength_index(player.hand)
                            )
                        from gin_rummy.game import TurnRecord

                        history.append(
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
                            winner_id, scores, self._turn, outcome, history
                        )

                obs_after = self._observation(player_id)
                discard = policy.choose_discard(obs_after)
                player.remove(discard)
                self.discard_pile.append(discard)

                if player_id == self._target_seat:
                    self.target_obs_seq.append(encode_observation(source, discard))
                    self.target_truth_seq.append(
                        hand_strength_index(player.hand)
                    )

                from gin_rummy.game import TurnRecord

                history.append(
                    TurnRecord(
                        self._turn, player_id, source, drawn, discard, "play", dw_value
                    )
                )

        return GameResult(None, scores, self._turn, "draw", history)


# ---------------------------------------------------- driver / evaluation --


def _play_batch(num_games: int, seed_base: int, target_seat: int):
    games: list[GameResult] = []
    obs_seqs: list[list[int]] = []
    truth_seqs: list[list[int]] = []
    for i in range(num_games):
        seed = seed_base + i
        rng = random.Random(seed)
        engine = _InstrumentedClassicGin(
            num_players=2,
            policies=[GreedyKnockPolicy(rng), GreedyKnockPolicy(rng)],
            seed=seed,
            target_seat=target_seat,
        )
        result = engine.play()
        games.append(result)
        if engine.target_obs_seq and len(engine.target_obs_seq) == len(
            engine.target_truth_seq
        ):
            obs_seqs.append(engine.target_obs_seq)
            truth_seqs.append(engine.target_truth_seq)
    return games, obs_seqs, truth_seqs


def _align_states_by_confusion(
    hmm: OpponentHandHMM,
    obs_seqs: Sequence[Sequence[int]],
    truth_seqs: Sequence[Sequence[int]],
) -> list[int]:
    """Return a mapping learned_state_idx -> semantic_class_idx that
    maximises training-set overlap. Greedy Hungarian-lite: pick the
    (learned, semantic) cell with the largest count, lock it in, remove
    that row and column, repeat.

    Ties are broken by insertion order (deterministic).
    """
    counts = [[0] * N_STATES for _ in range(hmm.n_states)]
    for obs, truth in zip(obs_seqs, truth_seqs):
        pred = hmm.viterbi(obs)
        for p, t in zip(pred, truth):
            counts[p][t] += 1

    mapping = [-1] * hmm.n_states
    used_learned: set[int] = set()
    used_semantic: set[int] = set()
    cells = [
        (counts[i][j], i, j)
        for i in range(hmm.n_states)
        for j in range(N_STATES)
    ]
    cells.sort(reverse=True)
    for _, i, j in cells:
        if i in used_learned or j in used_semantic:
            continue
        mapping[i] = j
        used_learned.add(i)
        used_semantic.add(j)
    # Fill any leftover learned states with unused semantic ids
    # (happens when n_states > N_STATES; here they're equal, but be safe).
    leftover_semantic = [j for j in range(N_STATES) if j not in used_semantic]
    for i in range(hmm.n_states):
        if mapping[i] == -1:
            mapping[i] = leftover_semantic.pop() if leftover_semantic else 0
    return mapping


def _viterbi_accuracy(
    hmm: OpponentHandHMM,
    mapping: Sequence[int],
    obs_seqs: Sequence[Sequence[int]],
    truth_seqs: Sequence[Sequence[int]],
) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for obs, truth in zip(obs_seqs, truth_seqs):
        pred = hmm.viterbi(obs)
        for p, t in zip(pred, truth):
            if mapping[p] == t:
                correct += 1
            total += 1
    return (correct / total if total else 0.0, correct, total)


def _per_turn_ll(hmm: OpponentHandHMM, seqs: Sequence[Sequence[int]]) -> float:
    total_ll = 0.0
    total_len = 0
    for s in seqs:
        _, ll = hmm.forward(s)
        total_ll += ll
        total_len += len(s)
    return total_ll / total_len if total_len else 0.0


def _majority_class_baseline(truth_seqs: Sequence[Sequence[int]]) -> float:
    counts = Counter()
    for s in truth_seqs:
        counts.update(s)
    total = sum(counts.values())
    if not total:
        return 0.0
    return max(counts.values()) / total


def _topk_emissions(hmm: OpponentHandHMM, k: int = 3) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    for row in hmm.emit:
        indexed = sorted(enumerate(row), key=lambda t: t[1], reverse=True)
        out.append([(OBSERVATION_ALPHABET[i], p) for i, p in indexed[:k]])
    return out


def main() -> None:
    print(f"[hmm-demo] playing {TRAIN_GAMES} training games...")
    train_games, train_obs, train_truth = _play_batch(
        TRAIN_GAMES, seed_base=RNG_SEED, target_seat=TARGET_SEAT
    )
    train_obs = [s for s in train_obs if len(s) >= 2]
    train_truth = [t for s, t in zip(train_obs, train_truth) if len(s) >= 2]

    print(f"[hmm-demo] playing {HELDOUT_GAMES} held-out games...")
    _, held_obs, held_truth = _play_batch(
        HELDOUT_GAMES,
        seed_base=RNG_SEED + 10_000,
        target_seat=TARGET_SEAT,
    )
    held_obs = [s for s in held_obs if len(s) >= 2]
    held_truth = [t for s, t in zip(held_obs, held_truth) if len(s) >= 2]

    print(
        f"[hmm-demo] usable training sequences: {len(train_obs)} "
        f"(total turns: {sum(len(s) for s in train_obs)})"
    )
    print(
        f"[hmm-demo] usable held-out sequences: {len(held_obs)} "
        f"(total turns: {sum(len(s) for s in held_obs)})"
    )

    ll_trace: list[float] = []
    hmm = OpponentHandHMM(n_states=N_STATES, rng=random.Random(RNG_SEED))
    hmm.baum_welch(
        train_obs,
        iterations=EM_ITERATIONS,
        tol=1e-5,
        callback=lambda it, ll: (
            ll_trace.append(ll),
            print(f"[hmm-demo] EM iter {it:>2d}  train_ll={ll:.3f}"),
        ),
    )

    train_ll_per_turn = _per_turn_ll(hmm, train_obs)
    held_ll_per_turn = _per_turn_ll(hmm, held_obs)

    print("\n[hmm-demo] learned start probabilities:")
    for i, p in enumerate(hmm.start):
        print(f"  s{i}  π={p:.3f}")

    print("\n[hmm-demo] learned transition matrix (rows sum to 1):")
    header = "        " + "  ".join(f"s{j}    " for j in range(hmm.n_states))
    print(header)
    for i, row in enumerate(hmm.trans):
        print(f"  s{i} -> " + "  ".join(f"{v:.3f}" for v in row))

    print("\n[hmm-demo] top-3 emissions per learned state:")
    for i, top in enumerate(_topk_emissions(hmm, k=3)):
        pretty = ", ".join(f"{sym}={p:.3f}" for sym, p in top)
        print(f"  s{i}: {pretty}")

    mapping = _align_states_by_confusion(hmm, train_obs, train_truth)
    print("\n[hmm-demo] learned-state -> semantic-class alignment (by train-set overlap):")
    for i, sem in enumerate(mapping):
        print(f"  s{i} -> {HAND_STRENGTH_CLASSES[sem]}")

    train_acc, tc, tt = _viterbi_accuracy(hmm, mapping, train_obs, train_truth)
    held_acc, hc, ht = _viterbi_accuracy(hmm, mapping, held_obs, held_truth)
    baseline = _majority_class_baseline(train_truth)

    print("\n[hmm-demo] --- summary ---")
    print(f"  train per-turn log-likelihood   : {train_ll_per_turn:.4f}")
    print(f"  heldout per-turn log-likelihood : {held_ll_per_turn:.4f}")
    print(f"  train Viterbi accuracy vs truth : {train_acc:.3%}  ({tc}/{tt})")
    print(f"  heldout Viterbi accuracy        : {held_acc:.3%}  ({hc}/{ht})")
    print(f"  majority-class baseline (train) : {baseline:.3%}")


if __name__ == "__main__":
    main()
