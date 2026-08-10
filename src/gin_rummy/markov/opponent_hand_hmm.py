"""HMM specialisation for opponent hand-strength inference.

The idea: an opponent's hand-strength class is a *latent* variable that
evolves slowly over the game (Markovian). Public observations — what
they drew from, what they discarded — are *emitted* from that latent
state. Once we've fit the HMM on many trajectories, Viterbi gives us a
best-guess opponent-hand-strength sequence from the public history
alone.

Hand-strength alphabet (5 buckets) mirrors the coarsening philosophy in
``comms.analyze``: keep the alphabet small so plug-in estimates and EM
statistics remain well-conditioned on a few hundred games.

The observation alphabet is intentionally tiny (8 symbols) — the
Cartesian product of {draw source} × {discard rank class}. Fewer
symbols means Baum-Welch has enough data per (state, symbol) cell to
avoid degenerate rows.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from gin_rummy.cards import Card
from gin_rummy.game import GameResult, TurnRecord
from gin_rummy.markov.hmm import HMM, ProgressCallback
from gin_rummy.meld import optimal_decomposition


# --------------------------------------------------------------- alphabets --

HAND_STRENGTH_CLASSES: tuple[str, ...] = (
    "very_weak",
    "weak",
    "medium",
    "strong",
    "gin_ready",
)

DRAW_SOURCES: tuple[str, ...] = ("deck", "discard")
DISCARD_RANK_CLASSES: tuple[str, ...] = ("ace", "low", "mid", "face")

# Encoded observations = draw_source_idx * len(DISCARD_RANK_CLASSES) + rank_idx.
OBSERVATION_ALPHABET: tuple[str, ...] = tuple(
    f"{src}|{rc}" for src in DRAW_SOURCES for rc in DISCARD_RANK_CLASSES
)
N_OBS = len(OBSERVATION_ALPHABET)  # 8


def _rank_class(card: Card | None) -> str:
    """Bucket a discarded card into 4 coarse classes.

    A missing discard (turn ended with knock/gin, no discard) still needs
    a symbol; we treat "no discard" as ``mid`` — the class most
    consistent with a random draw — so it does not push the state
    posterior toward any extreme. Callers who want to exclude
    knock/gin turns should filter them upstream.
    """
    if card is None:
        return "mid"
    r = card.rank
    if r == "A":
        return "ace"
    if r in ("2", "3", "4"):
        return "low"
    if r in ("J", "Q", "K"):
        return "face"
    return "mid"


def encode_observation(draw_source: str, discard: Card | None) -> int:
    """Turn a (draw_source, discard) pair into an integer 0..N_OBS-1."""
    src_idx = DRAW_SOURCES.index(draw_source if draw_source in DRAW_SOURCES else "deck")
    rc = _rank_class(discard)
    rc_idx = DISCARD_RANK_CLASSES.index(rc)
    return src_idx * len(DISCARD_RANK_CLASSES) + rc_idx


def decode_observation(symbol: int) -> tuple[str, str]:
    """Inverse of :func:`encode_observation`, useful for interpretability."""
    src_idx, rc_idx = divmod(symbol, len(DISCARD_RANK_CLASSES))
    return DRAW_SOURCES[src_idx], DISCARD_RANK_CLASSES[rc_idx]


# ---------------------------------------------------------- extraction API --


def extract_observations(
    history: Sequence[TurnRecord], target_player_id: int
) -> list[int]:
    """Walk a ``GameResult.history`` and pull out the observation sequence
    for ``target_player_id`` — one integer per turn that player took.

    Only ``play`` and terminal (``gin`` / ``knock``) turns for the target
    player contribute. Terminal turns without a discard are still
    included so the sequence length matches the number of visible turns.
    """
    out: list[int] = []
    for rec in history:
        if rec.player_id != target_player_id:
            continue
        out.append(encode_observation(rec.draw_source, rec.discarded))
    return out


def hand_strength_class(hand: Sequence[Card]) -> str:
    """Ground-truth hand-strength bucket derived from actual deadwood value.

    Used only for evaluation — this is *not* observed by the HMM. The
    thresholds mirror common practitioner intuition: ≤0 = gin, <=5 =
    strong, <=15 = medium, <=25 = weak, else very_weak. The buckets are
    coarse on purpose so the 5-symbol Viterbi output has any chance of
    matching them at meaningful rates.
    """
    _, _, dw = optimal_decomposition(list(hand))
    if dw == 0:
        return "gin_ready"
    if dw <= 5:
        return "strong"
    if dw <= 15:
        return "medium"
    if dw <= 25:
        return "weak"
    return "very_weak"


def hand_strength_index(hand: Sequence[Card]) -> int:
    return HAND_STRENGTH_CLASSES.index(hand_strength_class(hand))


def describe_state(state_idx: int) -> str:
    """Map an HMM state index back to its human-readable class name.

    Note: after training the *permutation* between learned states and
    the semantic labels is not guaranteed; callers who want a
    semantically-aligned mapping should re-label states by matching
    emissions or by post-hoc alignment against ground-truth
    trajectories (see the demo).
    """
    if not 0 <= state_idx < len(HAND_STRENGTH_CLASSES):
        raise ValueError(f"state_idx out of range: {state_idx}")
    return HAND_STRENGTH_CLASSES[state_idx]


# ------------------------------------------------------- the concrete HMM --


class OpponentHandHMM(HMM):
    """HMM tailored to the opponent-hand-strength problem.

    Fixes ``n_obs`` to :data:`N_OBS` (8) and defaults ``n_states`` to 5
    (matching :data:`HAND_STRENGTH_CLASSES`) but allows overriding it —
    a caller who wants a coarser latent alphabet may pass e.g.
    ``n_states=3``.
    """

    def __init__(self, n_states: int = 5, *, rng: random.Random | None = None) -> None:
        r = rng if rng is not None else random.Random(0)
        seed_model = HMM.random(n_states=n_states, n_obs=N_OBS, rng=r)
        super().__init__(
            n_states=n_states,
            n_obs=N_OBS,
            start=seed_model.start,
            trans=seed_model.trans,
            emit=seed_model.emit,
        )

    # ------------------------------------------------------------- training

    def train(
        self,
        games: Iterable[GameResult],
        target_player_id: int,
        *,
        iterations: int = 20,
        tol: float = 1e-4,
        callback: ProgressCallback | None = None,
    ) -> list[list[int]]:
        """Fit the HMM to per-game observation sequences for one seat.

        Returns the list of observation sequences that were actually
        used, so callers can inspect coverage.
        """
        sequences: list[list[int]] = []
        for g in games:
            seq = extract_observations(g.history, target_player_id)
            if len(seq) >= 2:
                sequences.append(seq)
        self.baum_welch(sequences, iterations=iterations, tol=tol, callback=callback)
        return sequences

    # -------------------------------------------------------- inference API

    def predict_hand_strength(
        self, observations: Sequence[int]
    ) -> tuple[list[int], list[list[float]]]:
        """Given an observation sequence, return the Viterbi state path
        and the per-turn posterior distribution over states."""
        return self.viterbi(observations), self.posteriors(observations)


__all__ = [
    "DISCARD_RANK_CLASSES",
    "DRAW_SOURCES",
    "HAND_STRENGTH_CLASSES",
    "N_OBS",
    "OBSERVATION_ALPHABET",
    "OpponentHandHMM",
    "decode_observation",
    "describe_state",
    "encode_observation",
    "extract_observations",
    "hand_strength_class",
    "hand_strength_index",
]
