"""Arm registry. An arm = a source of the warm-start vector c plus two
switches (warm initial state? warm mixer?). Adding an arm is supplying a new
``c_source``; everything else (optimiser, stopping rule, metrics, initial
parameters) is shared, and runs are paired on (scenario, seed).

    A  cold start      no c: |+>^n and the standard X mixer
    D  LP relaxation   c = fractional commitments of the continuous relaxation
                       with generation >= D_t + B, the QUBO's own target
                       (scipy linprog / HiGHS); solve time recorded
    D-dem              the same at generation >= D_t (the pilot-v1 arm D,
                       which relaxed a different problem from the circuit's)
    E  random control  c_i ~ U(eps, 1 - eps), drawn from (scenario, seed)
    F  merit order     greedy per hour: commit by cost per MW until D_t + B
                       is met; committed -> 0.9, uncommitted -> 0.1
    B         GNN (MILP labels) c + standard X mixer (washout ablation)
    C-milp    GNN trained on MILP commitments + warm mixer (scalable labels)
    C-oracle  GNN trained on exact QUBO ground states + warm mixer (upper
              bound for a learned start; labels need brute force)

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


def c_lp_relaxation(problem: QAOAProblem, seed: int, eps: float, target: str = "qubo") -> CVector:
    """Per-instance LP relaxation (solved once per problem, its time charged to
    each run). Default target: D_t + demand bias, the level the QUBO targets."""
    key = ("lp", target)
    if key not in problem.cache:
        t0 = time.perf_counter()
        lp = solve_lp_relaxation(problem.instance, target=target)
        problem.cache[key] = (lp, time.perf_counter() - t0)
    lp, wall = problem.cache[key]
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


_GNN_CACHE: dict = {}


def _gnn_model(variant: str, instance):
    """The trained GCN for this instance's block set (models/gnn_<variant>_L<k>.pt), cached per process."""
    from .gnn import MODEL_DIR, WarmStartModel
    from .instance import REFINEMENT_LEVELS, refinement_fleet

    names = tuple(p.name for p in instance.plants)
    level = next((k for k in REFINEMENT_LEVELS if tuple(p.name for p in refinement_fleet(k)) == names), None)
    key = (variant, level)
    if key not in _GNN_CACHE:
        import torch

        torch.set_num_threads(1)
        model = WarmStartModel.load(MODEL_DIR / f"gnn_{variant}_L{level}.pt")
        model.predict(instance)  # warm-up, so the charged time is steady-state inference
        _GNN_CACHE[key] = model
    return _GNN_CACHE[key]


def c_gnn(variant: str) -> CSource:
    """c from the trained GCN (labels: 'milp' or 'oracle'); the charged time is
    feature construction + forward pass for this instance."""

    def source(problem: QAOAProblem, seed: int, eps: float) -> CVector:
        model = _gnn_model(variant, problem.instance)
        t0 = time.perf_counter()
        c = model.predict_flat(problem.instance)
        return CVector(c, time.perf_counter() - t0, dict(gnn=variant))

    return source


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    c_source: CSource | None  # None: cold start (no c)
    warm_state: bool = True
    warm_mixer: bool = True


ARMS: dict[str, Arm] = {
    "A": Arm("A", "cold start: uniform + X mixer", None, warm_state=False, warm_mixer=False),
    "D": Arm("D", "LP relaxation at the QUBO target D + B + warm mixer", c_lp_relaxation),
    "D-dem": Arm("D-dem", "LP relaxation at true demand D (pilot v1; inconsistent target) + warm mixer",
                 lambda pb, seed, eps: c_lp_relaxation(pb, seed, eps, target="demand")),
    "E": Arm("E", "random c + warm mixer (control)", c_random),
    "F": Arm("F", "merit order + warm mixer", c_merit_order),
    "B": Arm("B", "GNN (MILP labels) warm state + standard X mixer (washout ablation)", c_gnn("milp"),
             warm_state=True, warm_mixer=False),
    "C-milp": Arm("C-milp", "GNN trained on MILP labels + warm mixer", c_gnn("milp")),
    "C-oracle": Arm("C-oracle", "GNN trained on exact QUBO ground states + warm mixer", c_gnn("oracle")),
}


PIN_TAU = 0.1


def frozen_bits(problem: QAOAProblem, c) -> tuple[int, int]:
    """(pinned, pinned_wrong) for a raw warm-start vector c: a variable is
    PINNED if c <= PIN_TAU or c >= 1 - PIN_TAU, and pinned WRONG if its
    rounded value disagrees with the exact ground state (the degenerate
    ground state that disagrees least is used)."""
    from .exact import index_to_bits

    if "ground_bits" not in problem.cache:
        problem.cache["ground_bits"] = index_to_bits(problem.ev.ground, problem.n).astype(int)
    G = problem.cache["ground_bits"]
    c = np.asarray(c, float)
    pinned = (c <= PIN_TAU) | (c >= 1 - PIN_TAU)
    wrong = min(int((pinned & (np.round(c) != g)).sum()) for g in G)
    return int(pinned.sum()), wrong


def run_arm(arm: Arm | str, problem: QAOAProblem, p: int, seed: int, *, eps: float = DEFAULT_EPS,
            settings: OptimiserSettings = OptimiserSettings(), dtype=np.complex128, shots: int | None = None,
            method: str = "COBYLA") -> RunResult:
    """One run of ``arm``; ``shots`` / ``method`` as in :func:`qguc.qaoa.run_qaoa` (defaults: exact, COBYLA)."""
    arm = ARMS[arm] if isinstance(arm, str) else arm
    if arm.c_source is None:
        return run_qaoa(problem, p, seed, arm=arm.name, settings=settings, dtype=dtype, shots=shots, method=method,
                        extra=dict(eps=eps, n_pinned=0, n_pinned_wrong=0))
    cv = arm.c_source(problem, seed, eps)
    c = regularise(cv.c, eps)
    psi0 = warm_initial_state(c, dtype) if arm.warm_state else None
    mixer = WarmStartMixer(c) if arm.warm_mixer else XMixer(problem.n)
    n_pin, n_wrong = frozen_bits(problem, cv.c)
    return run_qaoa(problem, p, seed, arm=arm.name, initial_state=psi0, mixer=mixer, settings=settings,
                    dtype=dtype, source_s=cv.source_s, shots=shots, method=method,
                    extra=dict(c=c, eps=eps, n_pinned=n_pin, n_pinned_wrong=n_wrong, **cv.info))
