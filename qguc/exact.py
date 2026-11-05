"""Exact reference solutions. Two DIFFERENT references live here:

(A) :func:`brute_force` - the exact ground state of a QUBO, by enumerating
    all 2^n states. This is the reference for QAOA approximation ratios,
    because the QUBO (penalty term included) is the objective the circuit
    minimises.

(B) :func:`solve_milp` - the OPERATIONAL optimum: minimum fuel + start-up
    cost subject to generation >= demand in every hour, solved as a MILP
    with HiGHS. This is the classical benchmark. It optimises a different
    objective from (A): no penalty term, no demand bias, and a hard
    constraint instead of a soft one. The two need not share a schedule;
    the difference in operating cost is :func:`qguc.metrics.encoding_gap`.

:func:`brute_force_operational` solves (B) by enumeration; it exists to
cross-check the MILP formulation on small instances.

State numbering
---------------
State k (0 <= k < 2^n) is the bit vector with x_j = (k >> j) & 1, i.e. x_0
is the least significant bit. Energies returned with ``return_energies``
are in this order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .instance import UCInstance
from .qubo import QUBO, qubo_energy_batch

MAX_BRUTE_FORCE_VARS = 24


def iter_bitstrings(n: int, chunk_size: int = 1 << 16):
    """Yield (start, X) with X the uint8 bit matrix of states start..start+m-1."""
    shifts = np.arange(n, dtype=np.uint64)
    total = 1 << n
    for start in range(0, total, chunk_size):
        k = np.arange(start, min(start + chunk_size, total), dtype=np.uint64)
        yield start, ((k[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)


def index_to_bits(k, n: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.uint64)
    return ((k[..., None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


@dataclass
class BruteForceResult:
    ground_energy: float
    ground_states: np.ndarray  # (g, n) uint8, all states within tolerance of the minimum
    ground_indices: np.ndarray  # (g,) state numbers
    max_energy: float
    n_vars: int
    elapsed_s: float
    spectrum: np.ndarray | None = None  # all 2^n energies, sorted ascending
    energies: np.ndarray | None = None  # all 2^n energies, indexed by state number

    @property
    def degeneracy(self) -> int:
        return self.ground_states.shape[0]

    @property
    def is_degenerate(self) -> bool:
        return self.degeneracy > 1

    @property
    def ground_state(self) -> np.ndarray:
        return self.ground_states[0]


def brute_force(
    qubo: QUBO,
    return_spectrum: bool = False,
    return_energies: bool = False,
    chunk_size: int = 1 << 16,
    degeneracy_rtol: float = 1e-12,
) -> BruteForceResult:
    """Exact ground state(s) of ``qubo`` by full enumeration (n <= 24).

    States whose energy is within ``degeneracy_rtol * max|E|`` of the
    minimum are reported as degenerate ground states.
    """
    n = qubo.n_vars
    if n > MAX_BRUTE_FORCE_VARS:
        raise ValueError(f"n_vars={n} exceeds brute-force limit {MAX_BRUTE_FORCE_VARS}")
    t0 = time.perf_counter()
    energies = np.empty(1 << n)
    for start, X in iter_bitstrings(n, chunk_size):
        energies[start : start + X.shape[0]] = qubo_energy_batch(qubo, X, chunk_size=chunk_size)
    e_min = float(energies.min())
    e_max = float(energies.max())
    tol = degeneracy_rtol * max(abs(e_min), abs(e_max), 1.0)
    idx = np.flatnonzero(energies <= e_min + tol)
    idx = idx[np.argsort(energies[idx], kind="stable")]
    elapsed = time.perf_counter() - t0
    return BruteForceResult(
        ground_energy=e_min,
        ground_states=index_to_bits(idx, n),
        ground_indices=idx,
        max_energy=e_max,
        n_vars=n,
        elapsed_s=elapsed,
        spectrum=np.sort(energies) if return_spectrum else None,
        energies=energies if return_energies else None,
    )


@dataclass
class OperationalResult:
    x: np.ndarray  # (n_vars,) int schedule, flat layout
    operating_cost: float  # fuel + start-up, recomputed from the instance
    fuel_cost: float
    startup_cost: float
    solve_time_s: float  # wall clock of the solver call only
    build_time_s: float = 0.0
    status: int = 0
    message: str = ""
    solver_objective: float = float("nan")
    mip_gap: float = float("nan")


def _milp_matrices(instance: UCInstance):
    """Objective c and constraints lb <= A [x; y] <= ub of the operational
    problem (see :func:`solve_milp`); variables are x (n_vars) then y (n_vars)."""
    n, N, T = instance.n_vars, instance.n_plants, instance.n_hours
    k = instance.var_index
    P = instance.capacity

    c = np.zeros(2 * n)
    A = np.zeros((T + n, 2 * n))
    lb = np.zeros(T + n)
    ub = np.full(T + n, np.inf)
    for t in range(T):
        lb[t] = instance.demand_mw[t]
        for i in range(N):
            c[k(i, t)] = instance.hourly_cost[i]
            c[n + k(i, t)] = instance.startup_cost[i]
            A[t, k(i, t)] = P[i]
            row = T + k(i, t)
            A[row, n + k(i, t)] = 1.0  # y[i,t]
            A[row, k(i, t)] = -1.0  # - x[i,t]
            if t == 0:
                lb[row] = -float(instance.initial_on[i])
            else:
                A[row, k(i, t - 1)] = 1.0  # + x[i,t-1]
    return c, A, lb, ub


def solve_milp(instance: UCInstance, time_limit: float | None = None) -> OperationalResult:
    """Operational optimum by MILP (scipy.optimize.milp / HiGHS).

        min  sum_{i,t} (a_i + b_i P_i) x[i,t] + S_i y[i,t]
        s.t. sum_i P_i x[i,t] >= D_t                        for all t
             y[i,t] >= x[i,t] - x[i,t-1],  x[i,-1] = initial_on[i]
             x binary, 0 <= y <= 1

    D_t is the true demand; ``demand_bias_mw`` plays no part. y is
    continuous; it equals the start-up indicator at the optimum because
    S_i > 0. The relative MIP gap is set to 0, so the result is optimal.
    """
    t_build = time.perf_counter()
    c, A, lb, ub = _milp_matrices(instance)
    n = instance.n_vars
    integrality = np.r_[np.ones(n), np.zeros(n)]
    options = {"disp": False, "mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    build_time = time.perf_counter() - t_build

    t0 = time.perf_counter()
    res = milp(
        c,
        constraints=LinearConstraint(A, lb, ub),
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        options=options,
    )
    solve_time = time.perf_counter() - t0
    if res.x is None:
        raise RuntimeError(f"MILP failed for {instance.label}: status {res.status}, {res.message}")

    x = np.round(res.x[:n]).astype(int)
    fuel = float(instance.fuel_cost(x))
    startup = float(instance.startup_expense(x))
    if not np.isclose(fuel + startup, res.fun, rtol=1e-9):
        raise RuntimeError(
            f"MILP objective {res.fun} != recomputed cost {fuel + startup} for {instance.label}"
        )
    return OperationalResult(
        x=x,
        operating_cost=fuel + startup,
        fuel_cost=fuel,
        startup_cost=startup,
        solve_time_s=solve_time,
        build_time_s=build_time,
        status=int(res.status),
        message=str(res.message),
        solver_objective=float(res.fun),
        mip_gap=float(getattr(res, "mip_gap", np.nan)),
    )


def brute_force_operational(instance: UCInstance, chunk_size: int = 1 << 16) -> OperationalResult:
    """Operational optimum by enumeration, for cross-checking :func:`solve_milp`."""
    n = instance.n_vars
    if n > MAX_BRUTE_FORCE_VARS:
        raise ValueError(f"n_vars={n} exceeds brute-force limit {MAX_BRUTE_FORCE_VARS}")
    t0 = time.perf_counter()
    best_cost, best_k = np.inf, -1
    for start, X in iter_bitstrings(n, chunk_size):
        cost = np.where(instance.is_adequate(X), instance.operating_cost(X), np.inf)
        j = int(np.argmin(cost))
        if cost[j] < best_cost:
            best_cost, best_k = float(cost[j]), start + j
    elapsed = time.perf_counter() - t0
    if best_k < 0:
        raise RuntimeError(f"no schedule meets demand for {instance.label}")
    x = index_to_bits(best_k, n).astype(int)
    return OperationalResult(
        x=x,
        operating_cost=best_cost,
        fuel_cost=float(instance.fuel_cost(x)),
        startup_cost=float(instance.startup_expense(x)),
        solve_time_s=elapsed,
    )


@dataclass
class LPRelaxation:
    x: np.ndarray  # (n_vars,) fractional commitments in [0, 1], flat layout
    objective: float  # LP lower bound on the operating cost, INR
    solve_time_s: float  # solver call only
    build_time_s: float
    status: int = 0
    message: str = ""

    @property
    def total_time_s(self) -> float:
        return self.solve_time_s + self.build_time_s


def solve_lp_relaxation(instance: UCInstance) -> LPRelaxation:
    """Continuous relaxation of :func:`solve_milp`: the same objective and
    constraints with x in [0, 1] instead of {0, 1}, by scipy.optimize.linprog
    (HiGHS). Its optimum is a lower bound on the MILP optimum."""
    t_build = time.perf_counter()
    c, A, lb, ub = _milp_matrices(instance)
    finite = np.isfinite(ub)
    A_ub = np.vstack([-A, A[finite]])  # lb <= A z  ->  -A z <= -lb ;  A z <= ub where finite
    b_ub = np.r_[-lb, ub[finite]]
    build_time = time.perf_counter() - t_build
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0.0, 1.0), method="highs")
    solve_time = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"LP relaxation failed for {instance.label}: {res.message}")
    return LPRelaxation(x=np.clip(res.x[: instance.n_vars], 0.0, 1.0), objective=float(res.fun),
                        solve_time_s=solve_time, build_time_s=build_time, status=int(res.status),
                        message=str(res.message))
