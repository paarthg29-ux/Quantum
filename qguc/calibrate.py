"""Grid evaluation of the penalty encoding over (lambda_D, demand bias B).

For every grid cell and scenario we find the exact QUBO ground state(s) and
measure, against the MILP optimum (generation >= D_t):

* adequacy of the ground state,
* the encoding gap (:func:`qguc.metrics.encoding_gap`),
* whether it is the MILP schedule (up to swapping identical blocks),
* the Ising coefficient dynamic range max|c| / min nonzero |c| over (h, J).

With lambda_switch = 0 the QUBO energy splits exactly as

    E(x) = C(x) + lambda_D * M_B(x),
    C = fuel + start-up,   M_B = sum_t (g_t - D_t - B)^2,

so C and the hourly generation g are enumerated once per scenario (C via
brute_force(build_qubo(instance, 0))) and each cell is an argmin over the
same 2^n vectors. :func:`cross_check` re-solves random cells through the
full brute_force(build_qubo(...)) path and requires identical ground states.

If several states tie for the ground state, a cell counts as adequate only
if all of them are, and the WORST encoding gap among them is reported.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .exact import OperationalResult, brute_force, index_to_bits, iter_bitstrings, solve_milp
from .instance import UCInstance
from .ising import qubo_to_ising
from .metrics import encoding_gap
from .qubo import build_qubo


def dynamic_range(*arrays, rtol: float = 1e-12) -> float:
    """max|c| / min nonzero |c| over all coefficients; float dust below
    rtol * max|c| is treated as zero."""
    vals = np.abs(np.concatenate([np.ravel(a) for a in arrays]))
    top = vals.max()
    nz = vals[vals > rtol * top]
    return float(top / nz.min())


@dataclass
class Enumeration:
    """Operating cost and hourly generation of all 2^n states of one instance."""

    cost: np.ndarray  # (2^n,)
    generation: np.ndarray  # (2^n, T)
    elapsed_s: float


def enumerate_states(instance: UCInstance) -> Enumeration:
    t0 = time.perf_counter()
    base = dataclasses.replace(instance, demand_bias_mw=0.0)
    cost = brute_force(build_qubo(base, 0.0), return_energies=True).energies
    gen = np.concatenate([instance.generation(X) for _, X in iter_bitstrings(instance.n_vars)])
    return Enumeration(cost, gen, time.perf_counter() - t0)


def _ground(energy: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Indices within the brute_force degeneracy tolerance of the minimum."""
    e_min, e_max = energy.min(), energy.max()
    tol = rtol * max(abs(e_min), abs(e_max), 1.0)
    return np.flatnonzero(energy <= e_min + tol)


@dataclass
class Grid:
    """Results indexed [bias, lambda, scenario]."""

    lambdas: np.ndarray
    biases: np.ndarray
    labels: list[str]
    adequate: np.ndarray  # bool
    gap: np.ndarray  # worst encoding gap among ground states (fraction)
    match: np.ndarray  # bool: a ground state equals the MILP schedule up to symmetry
    degeneracy: np.ndarray  # number of ground states
    classes: np.ndarray  # distinct ground states up to symmetry
    dyn_range: np.ndarray  # Ising (h, J)
    ground: dict = field(default_factory=dict)  # (b, l, s) -> ground state indices
    milp_times: np.ndarray | None = None
    enum_times: np.ndarray | None = None

    @property
    def adequacy_rate(self) -> np.ndarray:
        return self.adequate.mean(axis=-1)

    def _gap_over_adequate(self, reduce) -> np.ndarray:
        g = np.where(self.adequate, self.gap, np.nan)
        with warnings.catch_warnings():  # all-NaN rows are masked below
            warnings.simplefilter("ignore", RuntimeWarning)
            out = reduce(g, axis=-1)
        return np.where(self.adequate.any(axis=-1), out, np.nan)

    @property
    def gap_mean(self) -> np.ndarray:
        """Mean encoding gap over scenarios with an adequate ground state."""
        return self._gap_over_adequate(np.nanmean)

    @property
    def gap_worst(self) -> np.ndarray:
        return self._gap_over_adequate(np.nanmax)

    @property
    def match_rate(self) -> np.ndarray:
        return self.match.mean(axis=-1)

    @property
    def dyn_worst(self) -> np.ndarray:
        return self.dyn_range.max(axis=-1)


def sweep_grid(
    instances: list[UCInstance], lambdas, biases, milps: list[OperationalResult] | None = None, progress=None
) -> Grid:
    lambdas, biases = np.asarray(lambdas, float), np.asarray(biases, float)
    shape = (len(biases), len(lambdas), len(instances))
    adequate = np.zeros(shape, bool)
    gap = np.zeros(shape)
    match = np.zeros(shape, bool)
    degeneracy = np.zeros(shape, int)
    classes = np.zeros(shape, int)
    dyn = np.zeros(shape)
    ground = {}
    milp_times, enum_times = [], []

    for s, inst in enumerate(instances):
        milp = milps[s] if milps is not None else solve_milp(inst)
        milp_times.append(milp.solve_time_s)
        milp_canon = inst.canonical(milp.x)
        enum = enumerate_states(inst)
        enum_times.append(enum.elapsed_s)
        for b, bias in enumerate(biases):
            biased = dataclasses.replace(inst, demand_bias_mw=bias)
            penalty = ((enum.generation - biased.target_mw) ** 2).sum(axis=1)
            for li, lam in enumerate(lambdas):
                idx = _ground(enum.cost + lam * penalty)
                ground[(b, li, s)] = idx
                states = index_to_bits(idx, inst.n_vars).astype(int)
                canon = {tuple(inst.canonical(x)) for x in states}
                reps = [np.array(c) for c in canon]
                adequate[b, li, s] = all(inst.is_adequate(x) for x in reps)
                gap[b, li, s] = max(encoding_gap(inst, x, milp) for x in reps)
                match[b, li, s] = tuple(milp_canon) in canon
                degeneracy[b, li, s] = len(idx)
                classes[b, li, s] = len(canon)
                ising = qubo_to_ising(build_qubo(biased, lam))
                dyn[b, li, s] = dynamic_range(ising.h, ising.J)
        if progress is not None:
            progress(s, inst, enum)

    return Grid(
        lambdas=lambdas, biases=biases, labels=[i.label for i in instances],
        adequate=adequate, gap=gap, match=match, degeneracy=degeneracy, classes=classes,
        dyn_range=dyn, ground=ground, milp_times=np.array(milp_times), enum_times=np.array(enum_times),
    )


def _sweep_chunk(args):
    return sweep_grid(*args)


def merge_grids(grids: list[Grid]) -> Grid:
    """Concatenate grids over disjoint, ordered scenario chunks."""
    ground, off = {}, 0
    for g in grids:
        ground.update({(b, li, s + off): v for (b, li, s), v in g.ground.items()})
        off += len(g.labels)

    def cat(name):
        return np.concatenate([getattr(g, name) for g in grids], axis=-1)

    return Grid(
        lambdas=grids[0].lambdas, biases=grids[0].biases, labels=[lab for g in grids for lab in g.labels],
        adequate=cat("adequate"), gap=cat("gap"), match=cat("match"), degeneracy=cat("degeneracy"),
        classes=cat("classes"), dyn_range=cat("dyn_range"), ground=ground,
        milp_times=cat("milp_times"), enum_times=cat("enum_times"),
    )


def sweep_grid_parallel(instances: list[UCInstance], lambdas, biases, workers: int) -> Grid:
    """sweep_grid over contiguous scenario chunks in worker processes; the
    merged result is identical to the sequential sweep."""
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1 or len(instances) < 2:
        return sweep_grid(instances, lambdas, biases)
    bounds = np.linspace(0, len(instances), min(workers, len(instances)) + 1).astype(int)
    jobs = [(instances[a:b], lambdas, biases) for a, b in zip(bounds[:-1], bounds[1:])]
    with ProcessPoolExecutor(len(jobs)) as pool:
        return merge_grids(list(pool.map(_sweep_chunk, jobs)))


def cross_check(instances: list[UCInstance], grid: Grid, n_checks: int, rng) -> float:
    """Re-solve random cells with brute_force(build_qubo(...)); raise on any
    disagreement in the ground-state set. Returns the mean brute-force time."""
    times = []
    for _ in range(n_checks):
        b, li, s = (int(rng.integers(len(grid.biases))), int(rng.integers(len(grid.lambdas))),
                    int(rng.integers(len(instances))))
        inst = dataclasses.replace(instances[s], demand_bias_mw=grid.biases[b])
        bf = brute_force(build_qubo(inst, grid.lambdas[li]))
        times.append(bf.elapsed_s)
        if not np.array_equal(np.sort(bf.ground_indices), np.sort(grid.ground[(b, li, s)])):
            raise AssertionError(f"ground-state mismatch at bias={grid.biases[b]}, "
                                 f"lambda={grid.lambdas[li]}, scenario={instances[s].label}")
    return float(np.mean(times))


def select_operating_point(grid: Grid, plateau: int = 1) -> tuple[int, int] | None:
    """Cell (bias index, lambda index) with 100% adequacy and the smallest
    worst-case encoding gap; ties go to the smaller Ising dynamic range,
    then the smaller bias. None if no cell qualifies.

    With ``plateau = k`` a cell qualifies only if the k lambda grid points
    on either side (same bias, where they exist) are also fully adequate.
    Picking the best of many cells on one scenario set otherwise favours a
    cell that is fully adequate only by luck, at the edge of the adequate
    region, and it then fails on unseen scenarios.
    """
    full = grid.adequacy_rate == 1.0
    n_lam = len(grid.lambdas)
    best, best_key = None, None
    for b in range(len(grid.biases)):
        for li in range(n_lam):
            window = full[b, max(0, li - plateau): min(n_lam, li + plateau + 1)]
            if not window.all() or li - plateau < 0:
                continue
            key = (round(grid.gap_worst[b, li], 9), round(grid.gap_mean[b, li], 9),
                   grid.dyn_worst[b, li], grid.biases[b])
            if best_key is None or key < best_key:
                best, best_key = (b, li), key
    return best
