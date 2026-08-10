"""A pure-Python discrete-observation Hidden Markov Model.

Zero external dependencies. All algorithms run in **log-space** to avoid
underflow on realistically-long observation sequences (Gin Rummy games
routinely produce 30-60 turn traces; without log-space the joint
probability underflows quickly).

Design choices:

* ``start``, ``trans``, ``emit`` are kept in **linear space** on the
  ``HMM`` dataclass for readability and easy inspection from callers /
  tests. Every algorithm converts to log-space internally on entry.
* ``baum_welch`` supports a list of sequences and re-estimates from
  pooled statistics — this is important for the rummy application where
  each game is a separate observation sequence.
* Random initialisation uses a supplied ``random.Random`` so callers can
  make experiments deterministic; the factory adds a small amount of
  jitter around uniform so EM does not get stuck on the exact fixed
  point at initialisation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from gin_rummy.markov.utils import (
    NEG_INF,
    log_normalise,
    logsumexp,
    normalise,
    safe_log,
)


ProgressCallback = Callable[[int, float], None]


@dataclass
class HMM:
    """Discrete-observation Hidden Markov Model.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_obs : int
        Size of the observation alphabet (integers 0..n_obs-1).
    start : list[float]
        Initial state distribution; ``start[i] = P(state_0 = i)``.
    trans : list[list[float]]
        Row-stochastic transition matrix; ``trans[i][j] = P(s_{t+1}=j | s_t=i)``.
    emit : list[list[float]]
        Row-stochastic emission matrix; ``emit[i][o] = P(o | s=i)``.
    """

    n_states: int
    n_obs: int
    start: list[float] = field(default_factory=list)
    trans: list[list[float]] = field(default_factory=list)
    emit: list[list[float]] = field(default_factory=list)

    # ------------------------------------------------------------ construction

    def __post_init__(self) -> None:
        if not self.start:
            self.start = [1.0 / self.n_states] * self.n_states
        if not self.trans:
            self.trans = [
                [1.0 / self.n_states] * self.n_states for _ in range(self.n_states)
            ]
        if not self.emit:
            self.emit = [[1.0 / self.n_obs] * self.n_obs for _ in range(self.n_states)]
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if len(self.start) != self.n_states:
            raise ValueError("start must have length n_states")
        if len(self.trans) != self.n_states or any(
            len(row) != self.n_states for row in self.trans
        ):
            raise ValueError("trans must be n_states x n_states")
        if len(self.emit) != self.n_states or any(
            len(row) != self.n_obs for row in self.emit
        ):
            raise ValueError("emit must be n_states x n_obs")

    @classmethod
    def random(cls, n_states: int, n_obs: int, rng: random.Random) -> "HMM":
        """Random initialisation, uniform + jitter, then row-normalised.

        The jitter (uniform in [0.5, 1.5)) prevents EM from starting at
        the symmetric fixed point where all states are indistinguishable.
        """
        def _row(k: int) -> list[float]:
            return normalise([0.5 + rng.random() for _ in range(k)])

        return cls(
            n_states=n_states,
            n_obs=n_obs,
            start=_row(n_states),
            trans=[_row(n_states) for _ in range(n_states)],
            emit=[_row(n_obs) for _ in range(n_states)],
        )

    # -------------------------------------------------------- log-space views

    def _log_start(self) -> list[float]:
        return [safe_log(p) for p in self.start]

    def _log_trans(self) -> list[list[float]]:
        return [[safe_log(p) for p in row] for row in self.trans]

    def _log_emit(self) -> list[list[float]]:
        return [[safe_log(p) for p in row] for row in self.emit]

    # ---------------------------------------------------------------- forward

    def forward(self, obs_seq: Sequence[int]) -> tuple[list[list[float]], float]:
        """Forward algorithm in log-space.

        Returns
        -------
        log_alpha : list[list[float]]
            ``log_alpha[t][i] = log P(obs[:t+1], s_t=i | params)``.
        log_likelihood : float
            ``log P(obs | params)`` — the marginal likelihood.
        """
        T = len(obs_seq)
        if T == 0:
            return [], 0.0
        N = self.n_states
        log_start = self._log_start()
        log_trans = self._log_trans()
        log_emit = self._log_emit()

        alpha: list[list[float]] = [[NEG_INF] * N for _ in range(T)]
        o0 = obs_seq[0]
        for i in range(N):
            alpha[0][i] = log_start[i] + log_emit[i][o0]

        for t in range(1, T):
            ot = obs_seq[t]
            prev = alpha[t - 1]
            for j in range(N):
                # α_t(j) = e_j(o_t) · Σ_i α_{t-1}(i) · a_{ij}
                terms = [prev[i] + log_trans[i][j] for i in range(N)]
                alpha[t][j] = logsumexp(terms) + log_emit[j][ot]

        log_lik = logsumexp(alpha[T - 1])
        return alpha, log_lik

    # --------------------------------------------------------------- backward

    def backward(self, obs_seq: Sequence[int]) -> list[list[float]]:
        """Backward algorithm in log-space.

        ``log_beta[t][i] = log P(obs[t+1:] | s_t=i, params)``. By
        convention ``log_beta[T-1][i] = 0`` for all i.
        """
        T = len(obs_seq)
        if T == 0:
            return []
        N = self.n_states
        log_trans = self._log_trans()
        log_emit = self._log_emit()

        beta: list[list[float]] = [[NEG_INF] * N for _ in range(T)]
        for i in range(N):
            beta[T - 1][i] = 0.0

        for t in range(T - 2, -1, -1):
            ot1 = obs_seq[t + 1]
            nxt = beta[t + 1]
            for i in range(N):
                terms = [log_trans[i][j] + log_emit[j][ot1] + nxt[j] for j in range(N)]
                beta[t][i] = logsumexp(terms)
        return beta

    # ---------------------------------------------------------------- viterbi

    def viterbi(self, obs_seq: Sequence[int]) -> list[int]:
        """Most likely hidden state sequence via Viterbi in log-space."""
        T = len(obs_seq)
        if T == 0:
            return []
        N = self.n_states
        log_start = self._log_start()
        log_trans = self._log_trans()
        log_emit = self._log_emit()

        delta: list[list[float]] = [[NEG_INF] * N for _ in range(T)]
        psi: list[list[int]] = [[0] * N for _ in range(T)]

        o0 = obs_seq[0]
        for i in range(N):
            delta[0][i] = log_start[i] + log_emit[i][o0]

        for t in range(1, T):
            ot = obs_seq[t]
            prev = delta[t - 1]
            for j in range(N):
                best_val = NEG_INF
                best_i = 0
                for i in range(N):
                    v = prev[i] + log_trans[i][j]
                    if v > best_val:
                        best_val = v
                        best_i = i
                delta[t][j] = best_val + log_emit[j][ot]
                psi[t][j] = best_i

        # Backtrack.
        last = delta[T - 1]
        best_final = 0
        best_final_val = last[0]
        for i in range(1, N):
            if last[i] > best_final_val:
                best_final_val = last[i]
                best_final = i

        path: list[int] = [0] * T
        path[T - 1] = best_final
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t][path[t]]
        return path

    # ------------------------------------------------------------- posteriors

    def posteriors(self, obs_seq: Sequence[int]) -> list[list[float]]:
        """γ_t(i) = P(state_t=i | obs, params); returned in **linear** space."""
        T = len(obs_seq)
        if T == 0:
            return []
        alpha, log_lik = self.forward(obs_seq)
        beta = self.backward(obs_seq)
        N = self.n_states
        gamma: list[list[float]] = [[0.0] * N for _ in range(T)]
        for t in range(T):
            for i in range(N):
                gamma[t][i] = math.exp(alpha[t][i] + beta[t][i] - log_lik) if log_lik != NEG_INF else 0.0
            # Numerical cleanup: renormalise so each row sums to exactly 1.
            gamma[t] = normalise(gamma[t])
        return gamma

    # ----------------------------------------------------------- baum-welch

    def baum_welch(
        self,
        obs_sequences: Sequence[Sequence[int]],
        iterations: int = 20,
        tol: float = 1e-4,
        callback: ProgressCallback | None = None,
    ) -> None:
        """In-place EM re-estimation across multiple sequences.

        Standard multi-sequence Baum-Welch: E-step computes γ_t(i) and
        ξ_t(i,j) per sequence in log-space; M-step pools numerators /
        denominators across sequences before renormalising. Stops early
        once the per-iteration log-likelihood improvement drops below
        ``tol``.
        """
        if iterations <= 0:
            return
        prev_ll = None
        for it in range(iterations):
            ll = self._em_step(obs_sequences)
            if callback is not None:
                callback(it, ll)
            if prev_ll is not None and abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

    def _em_step(self, obs_sequences: Sequence[Sequence[int]]) -> float:
        """One EM iteration; returns the pooled log-likelihood *before*
        the M-step update (i.e. the LL under the incoming parameters)."""
        N = self.n_states
        M = self.n_obs

        # Log-space accumulators (initialised to log 0 = -inf).
        log_start_acc = [NEG_INF] * N
        log_trans_num = [[NEG_INF] * N for _ in range(N)]
        log_trans_den = [NEG_INF] * N
        log_emit_num = [[NEG_INF] * M for _ in range(N)]
        log_emit_den = [NEG_INF] * N

        log_trans = self._log_trans()
        log_emit = self._log_emit()

        total_ll = 0.0
        num_seqs = 0

        for obs_seq in obs_sequences:
            T = len(obs_seq)
            if T == 0:
                continue
            alpha, log_lik = self.forward(obs_seq)
            beta = self.backward(obs_seq)
            if log_lik == NEG_INF:
                # Degenerate: skip contributing this sequence.
                continue
            total_ll += log_lik
            num_seqs += 1

            # log γ_t(i) = α_t(i) + β_t(i) - log P(obs)
            log_gamma: list[list[float]] = [[0.0] * N for _ in range(T)]
            for t in range(T):
                for i in range(N):
                    log_gamma[t][i] = alpha[t][i] + beta[t][i] - log_lik

            # ------- accumulate initial-state stats -------
            for i in range(N):
                log_start_acc[i] = logsumexp([log_start_acc[i], log_gamma[0][i]])

            # ------- accumulate emission stats -------
            for t in range(T):
                ot = obs_seq[t]
                for i in range(N):
                    log_emit_num[i][ot] = logsumexp(
                        [log_emit_num[i][ot], log_gamma[t][i]]
                    )
                    log_emit_den[i] = logsumexp([log_emit_den[i], log_gamma[t][i]])

            # ------- accumulate transition stats -------
            # log ξ_t(i,j) = α_t(i) + log a_ij + log b_j(o_{t+1}) + β_{t+1}(j) - log P(obs)
            for t in range(T - 1):
                ot1 = obs_seq[t + 1]
                for i in range(N):
                    ai = alpha[t][i]
                    if ai == NEG_INF:
                        continue
                    for j in range(N):
                        lx = (
                            ai
                            + log_trans[i][j]
                            + log_emit[j][ot1]
                            + beta[t + 1][j]
                            - log_lik
                        )
                        log_trans_num[i][j] = logsumexp([log_trans_num[i][j], lx])
                # Denominator for transitions is the expected number of
                # transitions out of state i, which equals sum over t of γ_t(i).
                for i in range(N):
                    log_trans_den[i] = logsumexp(
                        [log_trans_den[i], log_gamma[t][i]]
                    )

        if num_seqs == 0:
            return NEG_INF

        # ---------------- M-step ----------------
        # start: pooled γ_0 / (# sequences).
        log_num_seqs = math.log(num_seqs)
        new_start_log = [v - log_num_seqs for v in log_start_acc]
        self.start = [math.exp(v) for v in log_normalise(new_start_log)]

        new_trans: list[list[float]] = []
        for i in range(N):
            if log_trans_den[i] == NEG_INF:
                new_trans.append([1.0 / N] * N)
            else:
                row_log = [log_trans_num[i][j] - log_trans_den[i] for j in range(N)]
                new_trans.append([math.exp(v) for v in log_normalise(row_log)])
        self.trans = new_trans

        new_emit: list[list[float]] = []
        for i in range(N):
            if log_emit_den[i] == NEG_INF:
                new_emit.append([1.0 / M] * M)
            else:
                row_log = [log_emit_num[i][o] - log_emit_den[i] for o in range(M)]
                new_emit.append([math.exp(v) for v in log_normalise(row_log)])
        self.emit = new_emit

        return total_ll


__all__ = ["HMM", "ProgressCallback"]
