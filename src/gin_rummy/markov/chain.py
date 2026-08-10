"""General discrete-time Markov chains, pure-Python.

Sparse dict-of-dicts transition representation keeps the API friendly for
small-alphabet chains (hand-strength buckets, action-space summaries) and
avoids a NumPy dependency. The dense linear-algebra routines used for
mean first-passage times and the fundamental matrix are hand-rolled Gauss
elimination with partial pivoting — good enough for the low-dimensional
chains we build in this project (dozens of states, not thousands).

All row-stochastic invariants are checked lazily via :meth:`validate` so
callers can build a chain incrementally, then verify once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# --------------------------------------------------------------------- solver


def _gauss_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``A x = b`` for square ``A`` via Gauss elimination with partial
    pivoting. Raises ``ValueError`` if ``A`` is singular. Pure Python; O(n^3).
    """
    n = len(a)
    if n == 0:
        return []
    # Copy so we don't mutate caller data.
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        # Partial pivot: swap the row with the largest |value| in this column.
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            if abs(m[r][col]) > best:
                best = abs(m[r][col])
                pivot = r
        if best < 1e-14:
            raise ValueError("Matrix is singular (or nearly so) at column " f"{col}")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        # Eliminate below.
        piv_val = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / piv_val
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    # Back-substitute.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n]
        for j in range(i + 1, n):
            s -= m[i][j] * x[j]
        x[i] = s / m[i][i]
    return x


def _invert(a: list[list[float]]) -> list[list[float]]:
    """Invert a square matrix via repeated ``_gauss_solve`` on identity cols."""
    n = len(a)
    inv: list[list[float]] = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [1.0 if i == col else 0.0 for i in range(n)]
        x = _gauss_solve(a, e)
        for r in range(n):
            inv[r][col] = x[r]
    return inv


# ----------------------------------------------------------------- MarkovChain


@dataclass
class MarkovChain:
    """A time-homogeneous, discrete-state Markov chain.

    Parameters
    ----------
    states : list[str]
        Ordered state alphabet. Duplicates are rejected.
    transition : dict[str, dict[str, float]]
        Sparse row-stochastic matrix. Missing entries are treated as 0.
        Each row (``transition[s]``) must sum to ~1; call :meth:`validate`
        to check.
    """

    states: list[str]
    transition: dict[str, dict[str, float]] = field(default_factory=dict)

    # ---- construction sanity ----

    def __post_init__(self) -> None:
        if len(set(self.states)) != len(self.states):
            raise ValueError("MarkovChain states must be unique")
        self._index = {s: i for i, s in enumerate(self.states)}

    # ---- validation ----

    def validate(self, *, tol: float = 1e-6) -> None:
        """Raise ``ValueError`` if any row does not sum to ~1 or references
        an unknown state."""
        for s in self.states:
            row = self.transition.get(s, {})
            for t in row:
                if t not in self._index:
                    raise ValueError(
                        f"Transition {s!r} -> {t!r} references unknown state"
                    )
            total = sum(row.values())
            # Allow all-zero rows only for absorbing states set externally;
            # baseline MarkovChain rejects them.
            if abs(total - 1.0) > tol:
                raise ValueError(
                    f"Row {s!r} sums to {total:.6f}, expected 1.0 (+/-{tol})"
                )

    # ---- simulation ----

    def step(self, current: str, rng: random.Random) -> str:
        """One Monte-Carlo transition from ``current``.

        Uses inverse-CDF sampling over the sparse row. Falls back to staying
        put if the row is empty (defensive; :meth:`validate` catches this).
        """
        row = self.transition.get(current, {})
        if not row:
            return current
        u = rng.random()
        acc = 0.0
        last = current
        for t, p in row.items():
            acc += p
            last = t
            if u <= acc:
                return t
        return last  # numerical drift: return final key

    def simulate(self, start: str, n: int, rng: random.Random) -> list[str]:
        """Return the ``n``-step trajectory starting from ``start`` (inclusive)."""
        if n <= 0:
            return []
        traj = [start]
        current = start
        for _ in range(n - 1):
            current = self.step(current, rng)
            traj.append(current)
        return traj

    # ---- dense matrix form ----

    def transition_matrix(self) -> list[list[float]]:
        """Dense row-stochastic matrix in the ``states`` ordering."""
        n = len(self.states)
        m = [[0.0] * n for _ in range(n)]
        for i, s in enumerate(self.states):
            row = self.transition.get(s, {})
            for t, p in row.items():
                m[i][self._index[t]] = p
        return m

    # ---- stationary distribution ----

    def stationary_distribution(
        self, *, tol: float = 1e-9, max_iter: int = 1000
    ) -> dict[str, float]:
        """Power iteration on ``pi_{k+1} = pi_k P``.

        For irreducible aperiodic chains this converges to the unique
        stationary distribution. If the chain is reducible (e.g. absorbing
        states), the iteration still returns a fixed point of ``pi P = pi``;
        callers with absorbing chains should use
        :class:`absorbing.AbsorbingChain` instead.
        """
        n = len(self.states)
        if n == 0:
            return {}
        pi = [1.0 / n] * n
        P = self.transition_matrix()
        for _ in range(max_iter):
            new = [0.0] * n
            for i in range(n):
                pi_i = pi[i]
                if pi_i == 0.0:
                    continue
                row = P[i]
                for j in range(n):
                    if row[j]:
                        new[j] += pi_i * row[j]
            s = sum(new)
            if s > 0:
                new = [v / s for v in new]
            diff = max(abs(new[i] - pi[i]) for i in range(n))
            pi = new
            if diff < tol:
                break
        return {self.states[i]: pi[i] for i in range(n)}

    # ---- expected first-passage times ----

    def expected_hits(self, start: str, target: str) -> float:
        """Expected number of steps to first reach ``target`` starting from
        ``start``.

        Solves the linear system ``h_i = 1 + sum_{j != target} P[i][j] h_j``
        for all non-target states via pure-Python Gauss elimination. Returns
        ``0.0`` when ``start == target``. Returns ``float('inf')`` if the
        target is unreachable from ``start``.
        """
        if start == target:
            return 0.0
        if start not in self._index or target not in self._index:
            raise KeyError("start / target must be states in the chain")

        # Reachability check via BFS on non-zero transitions.
        reachable = {start}
        frontier = [start]
        while frontier:
            s = frontier.pop()
            for t, p in self.transition.get(s, {}).items():
                if p > 0 and t not in reachable:
                    reachable.add(t)
                    frontier.append(t)
        if target not in reachable:
            return float("inf")

        # Build linear system over states != target.
        others = [s for s in self.states if s != target]
        idx = {s: i for i, s in enumerate(others)}
        n = len(others)
        a = [[0.0] * n for _ in range(n)]
        b = [1.0] * n
        for i, s in enumerate(others):
            a[i][i] = 1.0
            row = self.transition.get(s, {})
            for t, p in row.items():
                if t == target:
                    continue
                a[i][idx[t]] -= p
        try:
            h = _gauss_solve(a, b)
        except ValueError:
            return float("inf")
        return h[idx[start]]

    def mean_first_passage_matrix(self) -> dict[str, dict[str, float]]:
        """Full ``m[s][t] = E[first-hit t | start s]`` matrix.

        Diagonal entries are ``0.0``. Unreachable pairs are ``inf``.
        """
        out: dict[str, dict[str, float]] = {}
        for s in self.states:
            row: dict[str, float] = {}
            for t in self.states:
                row[t] = 0.0 if s == t else self.expected_hits(s, t)
            out[s] = row
        return out


# ------------------------------------------------------------- helpers


def estimate_chain_from_sequences(
    sequences: Iterable[Sequence[str]],
    states: Sequence[str],
    *,
    laplace_alpha: float = 1.0,
) -> MarkovChain:
    """Fit a ``MarkovChain`` from observed state sequences by MLE with
    additive smoothing.

    ``laplace_alpha`` puts a symmetric Dirichlet prior on each row; the
    default of 1.0 (add-one) keeps every transition strictly positive and
    avoids all-zero rows when a state is rare in the sample.
    """
    states_list = list(states)
    counts: dict[str, dict[str, float]] = {
        s: {t: laplace_alpha for t in states_list} for s in states_list
    }
    for seq in sequences:
        for a, b in zip(seq, list(seq)[1:]):
            if a in counts and b in counts[a]:
                counts[a][b] += 1.0
    transition: dict[str, dict[str, float]] = {}
    for s in states_list:
        row = counts[s]
        total = sum(row.values())
        transition[s] = {t: c / total for t, c in row.items()}
    return MarkovChain(states=states_list, transition=transition)


__all__ = [
    "MarkovChain",
    "_gauss_solve",
    "_invert",
    "estimate_chain_from_sequences",
]
