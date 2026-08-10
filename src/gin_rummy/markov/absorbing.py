"""Absorbing Markov chain specialisation.

An absorbing state ``a`` satisfies ``P[a][a] = 1``. The classical results
(fundamental matrix ``N = (I - Q)^{-1}``, expected steps to absorption
``t = N 1``, absorption probabilities ``B = N R``) are all wired up here
using the pure-Python Gauss elimination in :mod:`gin_rummy.markov.chain`.

Applied helper :func:`expected_turns_to_gin` builds a small absorbing
chain that models a rummy hand as a coarse "deadwood remaining" bucket
with ``gin`` and ``knock`` as terminal states — enough for a
back-of-envelope answer to "how many more turns should this hand take?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from gin_rummy.markov.chain import MarkovChain, _gauss_solve, _invert


@dataclass
class AbsorbingChain(MarkovChain):
    """A Markov chain with a distinguished set of absorbing states.

    The absorbing rows must be self-loops (``P[a][a] = 1``); constructor
    :meth:`validate_absorbing` verifies that.

    Parameters
    ----------
    absorbing_states : Iterable[str]
        Which states are absorbing. Every one must appear in ``states``.
    """

    absorbing_states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        # De-duplicate but preserve order.
        seen: set[str] = set()
        ordered: list[str] = []
        for s in self.absorbing_states:
            if s not in seen:
                if s not in self._index:
                    raise ValueError(
                        f"Absorbing state {s!r} is not in the state list"
                    )
                seen.add(s)
                ordered.append(s)
        self.absorbing_states = ordered

    # ---- validation ----

    def validate_absorbing(self, *, tol: float = 1e-6) -> None:
        """Verify each absorbing row is a self-loop."""
        for a in self.absorbing_states:
            row = self.transition.get(a, {})
            self_p = row.get(a, 0.0)
            other = sum(p for t, p in row.items() if t != a)
            if abs(self_p - 1.0) > tol or other > tol:
                raise ValueError(
                    f"Absorbing state {a!r} must have P[{a},{a}]=1; got "
                    f"self={self_p}, other={other}"
                )

    # ---- partitions ----

    def _partitioned_states(self) -> tuple[list[str], list[str]]:
        absorbing = list(self.absorbing_states)
        absorbing_set = set(absorbing)
        transient = [s for s in self.states if s not in absorbing_set]
        return transient, absorbing

    def canonical_form(self) -> tuple[list[list[float]], list[list[float]]]:
        """Return ``(Q, R)`` from the canonical partition ``P = [[Q, R], [0, I]]``.

        ``Q`` is the transient-to-transient block, ``R`` the
        transient-to-absorbing block.
        """
        transient, absorbing = self._partitioned_states()
        t_idx = {s: i for i, s in enumerate(transient)}
        a_idx = {s: i for i, s in enumerate(absorbing)}
        Q = [[0.0] * len(transient) for _ in range(len(transient))]
        R = [[0.0] * len(absorbing) for _ in range(len(transient))]
        for i, s in enumerate(transient):
            row = self.transition.get(s, {})
            for t, p in row.items():
                if t in t_idx:
                    Q[i][t_idx[t]] = p
                elif t in a_idx:
                    R[i][a_idx[t]] = p
        return Q, R

    # ---- fundamental matrix ----

    def fundamental_matrix(self) -> list[list[float]]:
        """``N = (I - Q)^{-1}``; ``N[i][j]`` = expected # visits to transient
        state ``j`` starting from transient state ``i`` before absorption."""
        Q, _ = self.canonical_form()
        n = len(Q)
        if n == 0:
            return []
        im_q = [
            [(1.0 if i == j else 0.0) - Q[i][j] for j in range(n)]
            for i in range(n)
        ]
        return _invert(im_q)

    def expected_absorption(self, start: str) -> float:
        """Expected number of steps until absorption starting from ``start``.

        Returns ``0.0`` if ``start`` is already absorbing.
        """
        if start in self.absorbing_states:
            return 0.0
        transient, _ = self._partitioned_states()
        if start not in transient:
            raise KeyError(f"{start!r} is not a state in the chain")
        # Solve (I - Q) t = 1 directly (avoids inverting the whole matrix
        # when the caller only wants one start row).
        Q, _ = self.canonical_form()
        n = len(Q)
        a = [
            [(1.0 if i == j else 0.0) - Q[i][j] for j in range(n)]
            for i in range(n)
        ]
        b = [1.0] * n
        t = _gauss_solve(a, b)
        idx = transient.index(start)
        return t[idx]

    def absorption_probabilities(self) -> list[list[float]]:
        """``B = N R``; ``B[i][k]`` = P(absorbed in absorbing state ``k``
        | start in transient state ``i``).

        Returned as a dense list-of-lists. Row ordering matches
        ``[s for s in states if s not in absorbing_states]``; column
        ordering matches ``absorbing_states``.
        """
        Q, R = self.canonical_form()
        n = len(Q)
        k = len(R[0]) if R else 0
        if n == 0 or k == 0:
            return [[0.0] * k for _ in range(n)]
        N = self.fundamental_matrix()
        B = [[0.0] * k for _ in range(n)]
        for i in range(n):
            for j in range(k):
                s = 0.0
                for m in range(n):
                    s += N[i][m] * R[m][j]
                B[i][j] = s
        return B


# ----------------------------------------------------------- applied helper


def expected_turns_to_gin(state_encoding: dict) -> float:
    """Rough answer to "how many more turns for this hand?" from a coarse
    deadwood-bucket absorbing chain.

    Parameters
    ----------
    state_encoding : dict
        Must contain at least ``"start_bucket"`` (one of the bucket names
        the chain is defined over). Optional key ``"transition"`` can override
        the default per-bucket transition rows; otherwise we use a default
        MLE-flavoured chain that biases high-deadwood hands to trickle
        downward toward ``knock`` and gives a small per-turn gin
        probability that increases as deadwood drops.

    Notes
    -----
    This is a *back-of-envelope* model — the coarse buckets and defaults
    are pedagogical, not tuned against real self-play data. Empirical
    experiments should refit the transition matrix from actual game
    trajectories via
    :func:`gin_rummy.markov.chain.estimate_chain_from_sequences`.
    """
    states = ["hi", "mid", "lo", "knock", "gin"]
    default_trans = {
        # Deadwood tends to drop as the hand improves; small gin prob.
        "hi":  {"hi": 0.55, "mid": 0.40, "lo": 0.03, "knock": 0.01, "gin": 0.01},
        "mid": {"hi": 0.10, "mid": 0.55, "lo": 0.28, "knock": 0.05, "gin": 0.02},
        "lo":  {"hi": 0.02, "mid": 0.15, "lo": 0.53, "knock": 0.25, "gin": 0.05},
        "knock": {"knock": 1.0},
        "gin":   {"gin": 1.0},
    }
    transition = state_encoding.get("transition", default_trans)
    start = state_encoding.get("start_bucket", "mid")
    if start not in states:
        raise ValueError(f"start_bucket must be one of {states}; got {start!r}")
    chain = AbsorbingChain(
        states=states,
        transition=transition,
        absorbing_states=["knock", "gin"],
    )
    chain.validate()
    chain.validate_absorbing()
    return chain.expected_absorption(start)


__all__ = [
    "AbsorbingChain",
    "expected_turns_to_gin",
]
