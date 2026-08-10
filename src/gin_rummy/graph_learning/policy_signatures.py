"""Behavioural fingerprints for policies — the research asset.

Given a bag of :class:`~gin_rummy.game.GameResult` objects, we extract a
fixed-length feature vector per policy that summarises *how* it plays,
independently of *how well*. The features follow the structure of
:func:`gin_rummy.eval.xplain.fingerprint_from_history` but are
serialised as a dense vector suitable for unsupervised clustering:

    [ draw_deck_frac,
      draw_discard_frac,
      <13 discard-rank fractions>,     # A, 2, 3, ..., K
      knock_turn_mean_norm,             # normalised to max_turns
      knock_turn_std_norm,
      knock_frac, gin_frac, undercut_frac, draw_frac,
      meld_set_frac, meld_run_frac,
      meld_avg_length_norm,             # normalised to 5 (max useful meld len)
      avg_deadwood_value_norm ]         # normalised to 100

The :func:`graph_augmented_signature` helper concatenates that
behavioural summary with the *average card embedding over the cards the
policy chose to discard* — a graph-conditioned view of which regions of
the card lattice the policy prefers to shed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from gin_rummy.cards import RANKS
from gin_rummy.game import GameResult
from gin_rummy.graph_learning.embeddings import (
    HashedCoocEmbeddings,
    SkipgramEmbeddings,
    average_embedding,
)
from gin_rummy.meld import optimal_decomposition
from gin_rummy.models.graph import card_node_id


# Feature block lengths — kept as constants so tests can assert on them.
_N_DRAW = 2
_N_RANK = len(RANKS)  # 13
_N_KNOCK = 2  # mean, std of knock turn
_N_OUTCOME = 4  # knock, gin, undercut, draw
_N_MELD = 3  # set_frac, run_frac, avg_length_norm
_N_HAND = 1  # avg deadwood value normalised
FEATURE_LEN = _N_DRAW + _N_RANK + _N_KNOCK + _N_OUTCOME + _N_MELD + _N_HAND


@dataclass(frozen=True)
class PolicyFingerprint:
    policy_name: str
    seat_id: int
    n_games: int
    features: list[float]
    discarded_card_ids: list[int]  # for graph augmentation

    @property
    def dim(self) -> int:
        return len(self.features)


class PolicyFingerprintExtractor:
    """Turn a collection of :class:`GameResult`s into per-policy vectors.

    A single instance can process multiple ``(policy_name, seat_id,
    game_results)`` triples via :meth:`extract`. The dense feature layout
    is documented in the module docstring and its total length is exposed
    as :data:`FEATURE_LEN`.
    """

    feature_len: int = FEATURE_LEN

    def __init__(self, max_turns: int = 200) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        self.max_turns = max_turns

    def extract(
        self,
        policy_name: str,
        seat_id: int,
        results: Sequence[GameResult],
    ) -> PolicyFingerprint:
        draw_counts: Counter[str] = Counter()
        rank_counts: Counter[str] = Counter()
        knock_turns: list[int] = []
        outcome_counts: Counter[str] = Counter()
        deadwood_values: list[int] = []
        discarded_ids: list[int] = []
        # Meld-usage histogram is derived from a mid-game snapshot: for each
        # game we grab the winner's final hand melds (via the recorded turn
        # right before termination). Since ``TurnRecord`` doesn't carry the
        # final hand, we approximate meld usage using the drawn-card stream
        # from the same seat — every drawn card that later ended up in a
        # run vs. set gets counted. Cheap and label-free.
        meld_kind_counts: Counter[str] = Counter()  # "set" | "run"
        meld_lengths: list[int] = []

        n_relevant = 0
        for game in results:
            history = game.history
            saw_seat = False
            for rec in history:
                if rec.player_id != seat_id:
                    continue
                saw_seat = True
                draw_counts[rec.draw_source] += 1
                if rec.discarded is not None:
                    rank_counts[rec.discarded.rank] += 1
                    try:
                        discarded_ids.append(card_node_id(rec.discarded))
                    except KeyError:
                        pass
                deadwood_values.append(rec.deadwood_value)
                if rec.action == "knock":
                    knock_turns.append(rec.turn)
                    outcome_counts["knock"] += 1
                elif rec.action == "gin":
                    outcome_counts["gin"] += 1
                elif rec.action == "undercut":
                    outcome_counts["undercut"] += 1
            if saw_seat:
                n_relevant += 1
            if game.outcome == "draw":
                outcome_counts["draw"] += 1

            # Cheap meld-shape sample from the final discarded stream for
            # this seat: if the seat drew any card that we can link to a
            # meld in a hypothetical hand, tally its shape. This is a
            # coarse proxy — the important property is that different
            # discard policies leave visibly different residues here.
            hand_proxy = [rec.drawn for rec in history
                          if rec.player_id == seat_id and rec.drawn is not None]
            if len(hand_proxy) >= 3:
                melds, _dw, _dv = optimal_decomposition(hand_proxy)
                for m in melds:
                    if len(m) >= 3:
                        rank_set = {c.rank for c in m}
                        suit_set = {c.suit for c in m}
                        kind = "set" if len(rank_set) == 1 else "run" if len(suit_set) == 1 else "set"
                        meld_kind_counts[kind] += 1
                        meld_lengths.append(len(m))

        total_draws = sum(draw_counts.values()) or 1
        total_ranks = sum(rank_counts.values()) or 1
        total_outcomes = sum(outcome_counts.values()) or 1
        total_melds = sum(meld_kind_counts.values()) or 1

        # -- draw sources --
        f_draw = [
            draw_counts.get("deck", 0) / total_draws,
            draw_counts.get("discard", 0) / total_draws,
        ]
        # -- discard ranks --
        f_rank = [rank_counts.get(r, 0) / total_ranks for r in RANKS]
        # -- knock timing --
        if knock_turns:
            mean_kt = sum(knock_turns) / len(knock_turns)
            var_kt = sum((k - mean_kt) ** 2 for k in knock_turns) / len(knock_turns)
            std_kt = var_kt ** 0.5
        else:
            mean_kt = 0.0
            std_kt = 0.0
        f_knock = [mean_kt / self.max_turns, std_kt / self.max_turns]
        # -- outcome mixture --
        f_out = [
            outcome_counts.get("knock", 0) / total_outcomes,
            outcome_counts.get("gin", 0) / total_outcomes,
            outcome_counts.get("undercut", 0) / total_outcomes,
            outcome_counts.get("draw", 0) / total_outcomes,
        ]
        # -- meld usage --
        avg_meld_len = (
            (sum(meld_lengths) / len(meld_lengths) / 5.0)
            if meld_lengths else 0.0
        )
        f_meld = [
            meld_kind_counts.get("set", 0) / total_melds,
            meld_kind_counts.get("run", 0) / total_melds,
            min(1.0, avg_meld_len),
        ]
        # -- hand quality --
        avg_dw = (sum(deadwood_values) / len(deadwood_values)) if deadwood_values else 0.0
        f_hand = [min(1.0, avg_dw / 100.0)]

        features = f_draw + f_rank + f_knock + f_out + f_meld + f_hand
        assert len(features) == FEATURE_LEN, (
            f"feature length mismatch: {len(features)} != {FEATURE_LEN}"
        )
        return PolicyFingerprint(
            policy_name=policy_name,
            seat_id=seat_id,
            n_games=n_relevant,
            features=features,
            discarded_card_ids=discarded_ids,
        )

    def extract_many(
        self,
        entries: Iterable[tuple[str, int, Sequence[GameResult]]],
    ) -> dict[str, PolicyFingerprint]:
        return {name: self.extract(name, seat, res) for name, seat, res in entries}


def graph_augmented_signature(
    policy_name: str,
    fingerprint: PolicyFingerprint,
    embeddings: HashedCoocEmbeddings | SkipgramEmbeddings,
) -> list[float]:
    """Behavioural vector concatenated with mean discard-card embedding.

    The trailing embedding block captures *which regions of the graph*
    the policy tends to shed cards from — a much richer view than the
    rank histogram alone, because it reflects each card's structural
    neighbourhood (adjacent runs, competing sets) inside a real hand.
    """
    del policy_name  # kept as an arg so callers can log/route — not used here.
    tail = average_embedding(fingerprint.discarded_card_ids, embeddings)
    return list(fingerprint.features) + list(tail)
