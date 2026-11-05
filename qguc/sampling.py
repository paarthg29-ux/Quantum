"""Exact sampling metrics from the full output distribution (no Monte Carlo).

The simulator gives |psi_k|^2 for all 2^n states, so the distribution of the
best of K independent measurements is known in closed form. Sort states by
energy ascending (ties: by state number), with probabilities p_j in that
order, and let T_j = sum_{k >= j} p_k (the probability that one sample is
state j or worse). The best of K samples is state j exactly when all K are
j-or-worse but not all are (j+1)-or-worse:

    P(best of K = state j) = T_j^K - T_{j+1}^K,     T_{2^n} = 0.

"Best" means lowest QUBO energy, which is what a QAOA user would pick. Per
instance we tabulate for every state its operating cost, adequacy, and the
operating cost after :func:`qguc.repair.repair`; any per-state quantity's
exact expectation under best-of-K is then a dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact import iter_bitstrings
from .repair import repair

KS = (1, 128, 1024)


@dataclass
class StateTable:
    """Per-state data of one instance, in sorted-energy order."""

    order: np.ndarray  # state numbers sorted by energy ascending (stable)
    n_ground: int  # ground states occupy order[:n_ground]
    gap_raw: np.ndarray  # (C(x) - C_MILP) / C_MILP, sorted order
    adequate: np.ndarray  # bool, sorted order
    gap_repaired: np.ndarray  # same after repair(x), sorted order

    @classmethod
    def build(cls, instance, E, ground, milp_cost: float, chunk_size: int = 1 << 16) -> "StateTable":
        n = instance.n_vars
        cost = np.empty(1 << n)
        rcost = np.empty(1 << n)
        adequate = np.empty(1 << n, bool)
        for start, X in iter_bitstrings(n, chunk_size):
            sl = slice(start, start + X.shape[0])
            cost[sl] = instance.operating_cost(X)
            adequate[sl] = instance.is_adequate(X)
            rcost[sl] = instance.operating_cost(repair(instance, X))
        order = np.argsort(E, kind="stable")
        return cls(order=order, n_ground=len(ground), gap_raw=((cost - milp_cost) / milp_cost)[order],
                   adequate=adequate[order], gap_repaired=((rcost - milp_cost) / milp_cost)[order])


def best_of_k_weights(p_sorted, K: int) -> np.ndarray:
    """P(best of K samples is state j), states in sorted-energy order."""
    p = np.asarray(p_sorted, np.float64)
    T = np.cumsum(p[::-1])[::-1]
    T /= T[0]  # exact normalisation (removes float drift)
    T_next = np.r_[T[1:], 0.0]
    return np.power(T, K) - np.power(T_next, K)


def exact_sampling_metrics(probs, table: StateTable, Ks=KS) -> dict:
    """{K: {p_ground, adequate, gap_raw, gap_repaired}} for the best of K
    samples from ``probs`` (all 2^n basis-state probabilities, state order).
    gap_raw is meaningful only together with ``adequate`` (see
    qguc.metrics.encoding_gap); gap_repaired is always adequate."""
    p_sorted = np.asarray(probs)[table.order]
    out = {}
    for K in Ks:
        w = best_of_k_weights(p_sorted, K)
        out[K] = dict(p_ground=float(w[: table.n_ground].sum()), adequate=float(w @ table.adequate),
                      gap_raw=float(w @ table.gap_raw), gap_repaired=float(w @ table.gap_repaired))
    return out
