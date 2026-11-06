"""QAOA runs on the fast simulator, and the per-run metrics.

Every arm is the same loop -- p layers of (cost phase, mixer), objective
<H_C>, COBYLA with the same stopping rule -- and differs only in the
INITIAL STATE and the MIXER (see qguc.arms). The initial PARAMETERS are
drawn from (seed, p) alone, so runs are paired across arms.

Parameter scaling
-----------------
H_C is the normalised Ising Hamiltonian (max|h, J| = 1). Its spread over
the uniform distribution is sigma = sqrt(sum h^2 + sum J^2); the optimiser
works on theta_k = sigma * gamma_k (dimensionless, O(1)) and beta_k. Initial
theta_k ~ U(0, 1), beta_k ~ U(-pi/4, pi/4), independently per layer.

Stopping rule (identical for every arm -- a measurement instrument)
--------------------------------------------------------------------
COBYLA, rhobeg 0.5, rhoend 1e-3, cap = 200 evaluations per parameter
(400 p; raised from 100 to de-censor total evaluation counts). External patience stop: the run ends once the best APPROXIMATION
RATIO has not improved by >= 1e-4 within the last 50 evaluations. A run
that ends at the cap is CENSORED ("cap hit") and is counted as such.
The reported state is the best one seen.

Metrics
-------
* r = (E_max - <H_C>) / (E_max - E_min), the exact-spectrum ratio.
* Common-reference gain g = (r - r_unif) / (1 - r_unif), r_unif = ratio of
  the uniform superposition (depends only on the instance, so it is the
  same reference for every arm). g = 0: no better than a random bitstring;
  g = 1: the ground state.
* overlap0 = |<ground|psi_0>|^2 (summed over degenerate ground states): the
  initial state's weight on the exact ground state(s), before any layer.
* g_state0: g of the initial state itself (no layers) -- the head start of
  the warm start alone; g_init: g at the first evaluation (initial
  parameters); g_final: best g.
* evals_to_g(t): first evaluation at which the running best g >= t (None =
  never, i.e. censored).
* Exact best-of-K (qguc.sampling) for K in {1, 128, 1024}: P(ground),
  adequacy, expected operating-cost gap vs MILP raw and after repair.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .config import FROZEN, OperatingPoint
from .exact import OperationalResult, solve_milp
from .fastsim import EnergyVector, QAOASimulator, energy_vector, expectation, probabilities
from .instance import UCInstance
from .sampling import KS, StateTable, exact_sampling_metrics

G_TARGETS = (0.9, 0.95, 0.99)


@dataclass
class QAOAProblem:
    instance: UCInstance
    ev: EnergyVector
    milp: OperationalResult
    point: OperatingPoint
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.ev.n

    @property
    def uniform_ratio(self) -> float:
        """r_unif: ratio of the uniform superposition (= a uniformly random bitstring)."""
        return float(self.ev.ratio(self.ev.E.mean()))

    def gain(self, ratio):
        r0 = self.uniform_ratio
        return (np.asarray(ratio, float) - r0) / (1.0 - r0)

    @property
    def table(self) -> StateTable:
        if "table" not in self.cache:
            self.cache["table"] = StateTable.build(self.instance, self.ev.E, self.ev.ground,
                                                   self.milp.operating_cost)
        return self.cache["table"]


def make_problem(instance: UCInstance, point: OperatingPoint = FROZEN) -> QAOAProblem:
    """Applies ``point``'s demand bias to ``instance`` and builds E and the MILP."""
    inst = instance
    if instance.demand_bias_mw != point.demand_bias_mw:
        inst = dataclasses.replace(instance, demand_bias_mw=point.demand_bias_mw)
    ev = energy_vector(inst, point.lambda_demand, point.lambda_switch)  # cached on inst
    return QAOAProblem(inst, ev, solve_milp(inst), point)


@dataclass(frozen=True)
class OptimiserSettings:
    rhobeg: float = 0.5
    rhoend: float = 1e-3
    cap_per_param: int = 200
    patience: int = 50
    min_delta: float = 1e-4  # on the approximation ratio

    def max_evals(self, n_params: int) -> int:
        return self.cap_per_param * n_params


@dataclass
class RunResult:
    arm: str
    p: int
    seed: int
    gammas: np.ndarray
    betas: np.ndarray
    energies: np.ndarray  # <H_C> at every evaluation, in order
    ratios: np.ndarray
    gains: np.ndarray
    stop_reason: str
    early_stopped: bool
    cap_hit: bool
    max_evals: int
    g_state0: float
    overlap0: float  # |<ground|psi_0>|^2 summed over all degenerate ground states
    sampling: dict  # K -> {p_ground, adequate, gap_raw, gap_repaired}, exact
    wall_s: float  # optimisation + metrics
    source_s: float = 0.0  # time to produce the warm start (e.g. LP solve)
    extra: dict = field(default_factory=dict)

    @property
    def n_evals(self) -> int:
        return len(self.energies)

    @property
    def best_ratio(self) -> float:
        return float(self.ratios.max())

    @property
    def initial_ratio(self) -> float:
        return float(self.ratios[0])

    @property
    def g_init(self) -> float:
        return float(self.gains[0])

    @property
    def g_final(self) -> float:
        return float(self.gains.max())

    @property
    def p_ground(self) -> float:
        return self.sampling[1]["p_ground"]

    @property
    def total_wall_s(self) -> float:
        return self.wall_s + self.source_s

    def evals_to_g(self, target: float) -> int | None:
        hit = np.flatnonzero(np.maximum.accumulate(self.gains) >= target)
        return int(hit[0]) + 1 if hit.size else None


class _EarlyStop(Exception):
    pass


def initial_parameters(p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(theta, beta) from (seed, p) only -- identical for every arm."""
    rng = np.random.default_rng([seed, p])
    return rng.uniform(0.0, 1.0, p), rng.uniform(-np.pi / 4, np.pi / 4, p)


def run_qaoa(
    problem: QAOAProblem,
    p: int,
    seed: int,
    *,
    arm: str = "A",
    initial_state=None,
    mixer=None,
    init_params=None,
    settings: OptimiserSettings = OptimiserSettings(),
    dtype=np.complex128,
    source_s: float = 0.0,
    extra: dict | None = None,
    Ks=KS,
    shots: int | None = None,
    method: str = "COBYLA",
) -> RunResult:
    """One optimisation run; ``init_params`` = (theta, beta) with theta = sigma * gamma.

    ``shots``: if given, the optimiser sees a FINITE-SHOT estimate of <H_C> (the
    mean energy of ``shots`` bitstrings drawn from the exact distribution;
    multinomial counts, rng from (seed, p, shots)) instead of the exact value.
    The patience stop and the choice of the reported parameters (best measured
    value) use what the optimiser saw; every reported metric is then evaluated
    EXACTLY at those parameters. ``energies`` holds the estimates; the exact
    trajectory and the exact gain at the chosen parameters are in ``extra``.
    ``method``: 'COBYLA' (default) or 'Nelder-Mead', with the same evaluation
    cap and patience stop."""
    t0 = time.perf_counter()
    ev = problem.ev
    sim = QAOASimulator(ev.E, initial_state=initial_state, mixer=mixer, dtype=dtype)
    sigma = ev.sigma
    theta0, beta0 = initial_parameters(p, seed) if init_params is None else init_params
    x0 = np.concatenate([np.asarray(theta0, float), np.asarray(beta0, float)])
    max_evals = settings.max_evals(2 * p)
    energies, xs, exact = [], [], []
    state = {"best": -np.inf, "best_at": 0}
    shot_rng = np.random.default_rng([seed, p, int(shots)]) if shots else None

    def f(x):
        if shots:
            pr = probabilities(sim.state(x[:p] / sigma, x[p:])).astype(np.float64)
            pr /= pr.sum()
            exact.append(float(pr @ ev.E))
            e = float(shot_rng.multinomial(int(shots), pr) @ ev.E) / shots
        else:
            e = sim.energy(x[:p] / sigma, x[p:])
        energies.append(e)
        xs.append(np.array(x))
        r = float(ev.ratio(e))
        if r >= state["best"] + settings.min_delta:
            state["best"], state["best_at"] = r, len(energies)
        if len(energies) - state["best_at"] >= settings.patience:
            raise _EarlyStop()
        return e

    early = False
    try:
        if method == "COBYLA":
            res = minimize(f, x0, method="COBYLA",
                           options=dict(rhobeg=settings.rhobeg, tol=settings.rhoend, maxiter=max_evals))
        elif method == "Nelder-Mead":
            res = minimize(f, x0, method="Nelder-Mead",
                           options=dict(maxfev=max_evals, xatol=settings.rhoend, fatol=0.0))
        else:
            raise ValueError(f"unknown method {method!r}")
        reason = f"{method.lower()}: " + str(res.message)
    except _EarlyStop:
        early, reason = True, f"patience ({settings.patience} evals without +{settings.min_delta:g} in ratio)"
    energies = np.array(energies)
    cap_hit = (not early) and len(energies) >= max_evals

    best = int(np.argmin(energies))
    gammas, betas = xs[best][:p] / sigma, xs[best][p:]
    probs = probabilities(sim.state(gammas, betas))
    ratios = np.asarray(ev.ratio(energies), float)
    g0 = float(problem.gain(ev.ratio(expectation(sim.initial_state, ev.E))))
    overlap0 = float(probabilities(sim.initial_state)[ev.ground].sum())
    extra = dict(extra or {})
    extra["method"] = method
    if shots:
        ex = np.asarray(exact)
        extra.update(shots=int(shots), exact_g_chosen=float(problem.gain(ev.ratio(ex[best]))),
                     exact_g_init=float(problem.gain(ev.ratio(ex[0]))),
                     exact_g_trajectory_max=float(problem.gain(ev.ratio(ex.min()))))
    return RunResult(
        arm=arm, p=p, seed=seed, gammas=gammas, betas=betas, energies=energies, ratios=ratios,
        gains=problem.gain(ratios), stop_reason=reason, early_stopped=early, cap_hit=cap_hit,
        max_evals=max_evals, g_state0=g0, overlap0=overlap0, sampling=exact_sampling_metrics(probs, problem.table, Ks),
        wall_s=time.perf_counter() - t0, source_s=source_s, extra=extra or {},
    )


def arm_a(problem: QAOAProblem, p: int, seed: int, **kw) -> RunResult:
    """Arm A, cold start: |+>^n, X mixer."""
    return run_qaoa(problem, p, seed, arm="A", **kw)
