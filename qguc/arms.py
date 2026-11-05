"""Arm registry. An arm = a source of the warm-start vector c plus two
switches (warm initial state? warm mixer?). Adding an arm is supplying a new
``c_source``; everything else (optimiser, stopping rule, metrics, initial
parameters) is shared, and runs are paired on (scenario, seed).

    A  cold start      no c: |+>^n and the standard X mixer
    D  LP relaxation   c = fractional commitments of the MILP's continuous
                       relaxation (scipy linprog / HiGHS); solve time recorded
    E  random control  c_i ~ U(eps, 1 - eps), drawn from (scenario, seed)
    F  merit order     greedy per hour: commit by cost per MW until D_t + B
                       is met; committed -> 0.9, uncommitted -> 0.1
    (next step)
    B  GNN c + standard X mixer   (washout ablation: warm_mixer=False)
    C  GNN c + warm mixer

All warm arms regularise c -> clip(c, eps, 1 - eps) before use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .exact import solve_lp_relaxation
from .fastsim import XMixer
from .qaoa import OptimiserSettings, QAOAProblem, RunResult, run_qaoa
from .repair import cost_per_mw_order
from .warmstart import DEFAULT_EPS, WarmStartMixer, regularise, warm_initial_state


@dataclass(frozen=True)
class CVector:
    c: np.ndarray  # raw, before regularisation
    source_s: float  # wall time to produce it, charged to every run that uses it
    info: dict


CSource = Callable[[QAOAProblem, int, float], CVector]  # (problem, seed, eps) -> c


def c_lp_relaxation(problem: QAOAProblem, seed: int, eps: float) -> CVector:
    """Per-instance LP relaxation (solved once per problem, its time charged to each run)."""
    if "lp" not in problem.cache:
        t0 = time.perf_counter()
        lp = solve_lp_relaxation(problem.instance)
        problem.cache["lp"] = (lp, time.perf_counter() - t0)
    lp, wall = problem.cache["lp"]
    frac = int(((lp.x > 1e-9) & (lp.x < 1 - 1e-9)).sum())
    return CVector(lp.x, wall, dict(lp_solve_s=lp.solve_time_s, lp_build_s=lp.build_time_s,
                                    lp_bound=lp.objective, n_fractional=frac))


def c_random(problem: QAOAProblem, seed: int, eps: float) -> CVector:
    inst = problem.instance
    key = [seed, int(inst.meta.get("scenario", 0)), int(inst.meta.get("set_seed", 0)), 0xE]
    t0 = time.perf_counter()
    c = np.random.default_rng(key).uniform(eps, 1.0 - eps, inst.n_vars)
    return CVector(c, time.perf_counter() - t0, {})


def c_merit_order(problem: QAOAProblem, seed: int, eps: float, on: float = 0.9, off: float = 0.1) -> CVector:
    inst = problem.instance
    t0 = time.perf_counter()
    U = np.zeros((inst.n_plants, inst.n_hours), int)
    for t in range(inst.n_hours):
        gen = 0.0
        for i in cost_per_mw_order(inst):
            if gen >= inst.target_mw[t]:
                break
            U[i, t], gen = 1, gen + inst.capacity[i]
    x = inst.flatten(U)
    return CVector(np.where(x == 1, on, off), time.perf_counter() - t0, dict(schedule=x))


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    c_source: CSource | None  # None: cold start (no c)
    warm_state: bool = True
    warm_mixer: bool = True


ARMS: dict[str, Arm] = {
    "A": Arm("A", "cold start: uniform + X mixer", None, warm_state=False, warm_mixer=False),
    "D": Arm("D", "LP relaxation + warm mixer", c_lp_relaxation),
    "E": Arm("E", "random c + warm mixer (control)", c_random),
    "F": Arm("F", "merit order + warm mixer", c_merit_order),
}


def run_arm(arm: Arm | str, problem: QAOAProblem, p: int, seed: int, *, eps: float = DEFAULT_EPS,
            settings: OptimiserSettings = OptimiserSettings(), dtype=np.complex128) -> RunResult:
    arm = ARMS[arm] if isinstance(arm, str) else arm
    if arm.c_source is None:
        return run_qaoa(problem, p, seed, arm=arm.name, settings=settings, dtype=dtype)
    cv = arm.c_source(problem, seed, eps)
    c = regularise(cv.c, eps)
    psi0 = warm_initial_state(c, dtype) if arm.warm_state else None
    mixer = WarmStartMixer(c) if arm.warm_mixer else XMixer(problem.n)
    return run_qaoa(problem, p, seed, arm=arm.name, initial_state=psi0, mixer=mixer, settings=settings,
                    dtype=dtype, source_s=cv.source_s, extra=dict(c=c, eps=eps, **cv.info))
