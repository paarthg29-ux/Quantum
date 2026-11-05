"""QAOA arms on the fast simulator, and the per-run metrics.

Every arm is the same loop -- p layers of (cost phase, mixer), objective
<H_C>, COBYLA with early stopping -- and differs only in the INITIAL STATE,
the MIXER and the initial PARAMETERS. Arm A (cold start) is:

    |psi_0> = |+>^n, mixer = prod_j RX(2 beta_k), (gamma, beta) random.

Parameter scaling
-----------------
H_C is the normalised Ising Hamiltonian (max|h, J| = 1). Its spread over
the uniform distribution is sigma = sqrt(sum h^2 + sum J^2), so the
optimiser works on theta_k = sigma * gamma_k (dimensionless, O(1)) and
beta_k directly; one COBYLA trust radius then suits both. Cold start draws
theta_k ~ U(0, 1) and beta_k ~ U(-pi/4, pi/4), independently per layer,
from the run's seed. (gamma -> -gamma, beta -> -beta conjugates the state
and leaves <H_C> unchanged, so gamma >= 0 loses nothing.)

Optimiser and early stopping (identical for every arm)
------------------------------------------------------
COBYLA (scipy), rhobeg 0.25, rhoend 1e-4, at most ``max_evals`` energy
evaluations; additionally stopped once the best approximation ratio has not
improved by ``min_delta`` within the last ``patience`` evaluations. The
reported state is the best one seen.

Metrics (per run)
-----------------
* approximation ratio r = (E_max - <H_C>) / (E_max - E_min), exact spectrum;
* p_ground: probability of measuring an exact ground state (all degenerate
  ground states summed);
* evals_to(0.95, ref): first evaluation at which the running best ratio
  reaches 95% of ``ref`` (default: the run's own best; for paired arm
  comparisons pass the best over all arms and seeds of that scenario and p);
* best sampled bitstring: lowest-energy state among ``shots`` samples of the
  final state, its adequacy (true demand) and its operating-cost gap vs the
  MILP optimum (:func:`qguc.metrics.encoding_gap`; meaningful only if
  adequate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .config import FROZEN, OperatingPoint
from .exact import OperationalResult, index_to_bits, solve_milp
from .fastsim import EnergyVector, QAOASimulator, energy_vector, probabilities, sample
from .instance import UCInstance
from .metrics import encoding_gap


@dataclass
class QAOAProblem:
    instance: UCInstance
    ev: EnergyVector
    milp: OperationalResult
    point: OperatingPoint

    @property
    def n(self) -> int:
        return self.ev.n

    @property
    def uniform_ratio(self) -> float:
        """Ratio of the uniform superposition (= a uniformly random bitstring)."""
        return float(self.ev.ratio(self.ev.E.mean()))


def make_problem(instance: UCInstance, point: OperatingPoint = FROZEN) -> QAOAProblem:
    """Applies ``point``'s demand bias to ``instance`` and builds E and the MILP."""
    import dataclasses

    inst = instance
    if instance.demand_bias_mw != point.demand_bias_mw:
        inst = dataclasses.replace(instance, demand_bias_mw=point.demand_bias_mw)
    ev = energy_vector(inst, point.lambda_demand, point.lambda_switch)  # cached on inst
    return QAOAProblem(inst, ev, solve_milp(inst), point)


@dataclass(frozen=True)
class OptimiserSettings:
    rhobeg: float = 0.25
    rhoend: float = 1e-4
    max_evals_per_param: int = 150
    patience_per_param: int = 20
    min_patience: int = 30
    min_delta: float = 1e-5  # in approximation-ratio units
    shots: int = 1024

    def max_evals(self, n_params: int) -> int:
        return self.max_evals_per_param * n_params

    def patience(self, n_params: int) -> int:
        return max(self.min_patience, self.patience_per_param * n_params)


@dataclass
class RunResult:
    arm: str
    p: int
    seed: int
    gammas: np.ndarray
    betas: np.ndarray
    energies: np.ndarray  # <H_C> at every evaluation, in order
    ratios: np.ndarray
    stop_reason: str
    p_ground: float
    sample_bits: np.ndarray  # best sampled bitstring
    sample_ratio: float
    sample_adequate: bool
    sample_gap: float  # operating-cost gap vs MILP (fraction)
    sample_is_ground: bool
    wall_s: float
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

    def evals_to(self, frac: float = 0.95, reference: float | None = None) -> int | None:
        ref = self.best_ratio if reference is None else reference
        hit = np.flatnonzero(np.maximum.accumulate(self.ratios) >= frac * ref)
        return int(hit[0]) + 1 if hit.size else None


class _EarlyStop(Exception):
    pass


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
) -> RunResult:
    """One optimisation run. ``init_params`` = (theta (p,), beta (p,)) with
    theta = sigma * gamma; None draws the cold-start distribution."""
    t0 = time.perf_counter()
    ev = problem.ev
    sim = QAOASimulator(ev.E, initial_state=initial_state, mixer=mixer, dtype=dtype)
    sigma = ev.sigma
    rng = np.random.default_rng([seed, p])
    if init_params is None:
        x0 = np.concatenate([rng.uniform(0.0, 1.0, p), rng.uniform(-np.pi / 4, np.pi / 4, p)])
    else:
        x0 = np.concatenate([np.asarray(init_params[0], float), np.asarray(init_params[1], float)])
    n_params = 2 * p
    max_evals, patience = settings.max_evals(n_params), settings.patience(n_params)
    energies, xs = [], []
    state = {"best": -np.inf, "best_at": 0}

    def f(x):
        e = sim.energy(x[:p] / sigma, x[p:])
        energies.append(e)
        xs.append(np.array(x))
        r = float(ev.ratio(e))
        if r > state["best"] + settings.min_delta:
            state["best"], state["best_at"] = r, len(energies)
        if len(energies) - state["best_at"] >= patience:
            raise _EarlyStop("no improvement")
        return e

    try:
        res = minimize(f, x0, method="COBYLA",
                       options=dict(rhobeg=settings.rhobeg, tol=settings.rhoend, maxiter=max_evals))
        reason = "cobyla: " + str(res.message)
    except _EarlyStop as exc:
        reason = f"early stop ({exc}, patience {patience})"

    energies = np.array(energies)
    best = int(np.argmin(energies))
    x = xs[best]
    gammas, betas = x[:p] / sigma, x[p:]
    psi = sim.state(gammas, betas)
    p_ground = float(probabilities(psi)[ev.ground].sum())
    shots = sample(psi, settings.shots, np.random.default_rng([seed, p, 1]))
    k = int(shots[np.argmin(ev.E[shots])])
    bits = index_to_bits(np.array([k]), problem.n)[0].astype(int)
    adequate = bool(problem.instance.is_adequate(bits))
    return RunResult(
        arm=arm, p=p, seed=seed, gammas=gammas, betas=betas, energies=energies,
        ratios=np.asarray(ev.ratio(energies), float), stop_reason=reason, p_ground=p_ground,
        sample_bits=bits, sample_ratio=float(ev.ratio(ev.E[k])), sample_adequate=adequate,
        sample_gap=float(encoding_gap(problem.instance, bits, problem.milp)),
        sample_is_ground=bool(np.isin(k, ev.ground)), wall_s=time.perf_counter() - t0,
    )


def arm_a(problem: QAOAProblem, p: int, seed: int, **kw) -> RunResult:
    """Arm A, cold start: |+>^n, X mixer, random (gamma, beta)."""
    return run_qaoa(problem, p, seed, arm="A", initial_state=None, mixer=None, init_params=None, **kw)
