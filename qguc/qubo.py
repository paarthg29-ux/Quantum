"""QUBO formulation of the multi-period unit-commitment problem.

The energy is stored as an upper-triangular matrix Q plus a scalar offset:

    E(x) = offset + sum_k Q[k,k] x_k + sum_{k<l} Q[k,l] x_k x_l

for binary x, using x_k^2 = x_k. Below, (i,t) stands for the flat index
``instance.var_index(i, t)``; this module never assumes the layout itself.

Terms
-----
1. Fuel cost, sum_{i,t} (a_i + b_i P_i) x[i,t]:
       diag[(i,t)] += a_i + b_i P_i

2. Demand penalty, lambda_D * sum_t (sum_i P_i x[i,t] - D'_t)^2 with the
   target D'_t = D_t + B (``instance.target_mw``; B = ``demand_bias_mw``):
       diag[(i,t)]               += lambda_D (P_i^2 - 2 D'_t P_i)
       Q[(i,t),(j,t)], i<j       += 2 lambda_D P_i P_j
       offset                    += lambda_D D'_t^2
   The penalty is two-sided: surplus is penalised exactly like a shortfall
   of the same size. With fixed blocks the target can rarely be met
   exactly, so the QUBO minimiser need not be adequate (generation >= D_t).
   The bias B > 0 shifts the target up to improve adequacy; it is not a
   reserve requirement. Check with ``UCInstance.is_adequate``. The cost of
   this encoding is measured by :func:`qguc.metrics.encoding_gap`.

3. Start-up cost, S_i x[i,t] (1 - x[i,t-1]). This is exactly quadratic, so it
   enters without approximation:
       t = 0:  diag[(i,0)]               += S_i (1 - initial_on[i])
       t >= 1: diag[(i,t)]               += S_i
               Q[(i,t-1),(i,t)]          += -S_i

4. Switching penalty, lambda_switch * sum_{i,t} (x[i,t] - x[i,t-1])^2.
   OFF by default (lambda_switch = 0): start-up cost already discourages
   cycling, and the penalty would confound the main study. It stays
   configurable so its effect can be reported.
       t = 0:  diag[(i,0)]               += lambda_switch (1 - 2 initial_on[i])
               offset                    += lambda_switch initial_on[i]^2
       t >= 1: diag[(i,t-1)], diag[(i,t)] += lambda_switch
               Q[(i,t-1),(i,t)]          += -2 lambda_switch

   LIMITATION: this is a QUADRATIC RELAXATION of a minimum up/down-time
   constraint, not the constraint itself. It charges every on/off transition
   and so discourages rapid cycling, but it does not forbid any schedule.
   Exact minimum up-time of two hours must forbid the pattern
   (1 - x[t-1]) x[t] (1 - x[t+1]), which is CUBIC; reducing it to a QUBO
   needs ancilla qubits. That is not done here, and the paper must say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instance import UCInstance


@dataclass
class QUBO:
    Q: np.ndarray  # (n_vars, n_vars), upper triangular
    offset: float
    n_vars: int
    params: dict = field(default_factory=dict)


def build_qubo(
    instance: UCInstance,
    lambda_demand: float,
    lambda_switch: float = 0.0,
    include_startup: bool = True,
) -> QUBO:
    n = instance.n_vars
    N, T = instance.n_plants, instance.n_hours
    P = instance.capacity
    D = instance.target_mw
    S = instance.startup_cost
    cost = instance.hourly_cost
    init = instance.initial_on.astype(float)
    k = instance.var_index

    Q = np.zeros((n, n))
    offset = 0.0

    def add(p: int, q: int, value: float) -> None:
        if p > q:
            p, q = q, p
        Q[p, q] += value

    # 1. fuel
    for t in range(T):
        for i in range(N):
            add(k(i, t), k(i, t), cost[i])

    # 2. per-hour demand penalty
    for t in range(T):
        offset += lambda_demand * D[t] ** 2
        for i in range(N):
            add(k(i, t), k(i, t), lambda_demand * (P[i] ** 2 - 2.0 * D[t] * P[i]))
            for j in range(i + 1, N):
                add(k(i, t), k(j, t), 2.0 * lambda_demand * P[i] * P[j])

    # 3. start-up cost
    if include_startup:
        for i in range(N):
            add(k(i, 0), k(i, 0), S[i] * (1.0 - init[i]))
            for t in range(1, T):
                add(k(i, t), k(i, t), S[i])
                add(k(i, t - 1), k(i, t), -S[i])

    # 4. switching penalty (quadratic relaxation of min up/down time)
    if lambda_switch:
        for i in range(N):
            add(k(i, 0), k(i, 0), lambda_switch * (1.0 - 2.0 * init[i]))
            offset += lambda_switch * init[i] ** 2
            for t in range(1, T):
                add(k(i, t - 1), k(i, t - 1), lambda_switch)
                add(k(i, t), k(i, t), lambda_switch)
                add(k(i, t - 1), k(i, t), -2.0 * lambda_switch)

    return QUBO(
        Q=Q,
        offset=float(offset),
        n_vars=n,
        params={
            "lambda_demand": float(lambda_demand),
            "lambda_switch": float(lambda_switch),
            "include_startup": bool(include_startup),
            "demand_bias_mw": instance.demand_bias_mw,
            "instance_label": instance.label,
        },
    )


def qubo_energy(qubo: QUBO, x):
    """Energy of one state (n,) -> float, or of a batch (m, n) -> (m,)."""
    x = np.asarray(x)
    if x.ndim == 1:
        xf = x.astype(float)
        return float(qubo.offset + xf @ qubo.Q @ xf)
    return qubo_energy_batch(qubo, x)


def qubo_energy_batch(qubo: QUBO, X, chunk_size: int = 1 << 16) -> np.ndarray:
    """Energies of the rows of X, shape (m, n) -> (m,).

    Rows are converted to float one chunk at a time, so X can be a compact
    integer array (e.g. all 2^20 bitstrings as uint8) without a large
    float copy.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != qubo.n_vars:
        raise ValueError(f"X has shape {X.shape}, expected (m, {qubo.n_vars})")
    out = np.empty(X.shape[0])
    for start in range(0, X.shape[0], chunk_size):
        xc = X[start : start + chunk_size].astype(float)
        out[start : start + chunk_size] = np.einsum("mi,mi->m", xc @ qubo.Q, xc)
    return out + qubo.offset
