"""Does band share generalise beyond unit commitment? 3-regular MaxCut, p = 1.

The UC protocol (scripts/s8_eps_main.py, phase 1) is repeated unchanged on a
different problem class: the same warm state and warm mixer (qguc.warmstart),
the same optimiser, stopping rule and initial parameters (qguc.qaoa.run_qaoa),
the same eps sweep, the same exact best-of-K metric (qguc.sampling) and the
same initial-state features (scripts/s8_mass.features). Only the cost
Hamiltonian and the sources of the warm-start vector c differ.

Everything below was fixed BEFORE any run and is not tuned afterwards.

Instances   random connected 3-regular graphs (networkx.random_regular_graph,
            seed GRAPH_SEED + 1000 n + i, redrawn with the next seed if
            disconnected), N_INST = 30 per size, n in {12, 16, 20}.
Hamiltonian H_C = sum_{(i,j) in E} Z_i Z_j  (already normalised: max|J| = 1),
            so E = |E| - 2 cut and the ground states are the maximum cuts.
            They are at least two-fold degenerate (Z2); overlap0 and
            P(ground) sum over all of them, as for UC.
Sources     A      cold start: |+>^n, X mixer
            GW     Goemans-Williamson: the MaxCut SDP (solved by a full-rank
                   Burer-Monteiro factorisation, optimality certified by the
                   dual slack matrix) rounded with ONE random hyperplane;
                   c = the cut bitstring
            GW-LS  the same cut after best-improvement single-flip local search
            E      random control, c_i ~ U(eps, 1 - eps) (the UC arm E)
            BADk   deliberately badly pinned, k = 0..4: one optimal cut with
                   the first k vertices of a fixed random permutation flipped
                   (nested doses). BAD0 is the optimum itself: an oracle, the
                   zero-dose reference only.
            All warm arms: c -> clip(c, eps, 1 - eps), warm state, warm mixer.
Wrong pins  as qguc.arms.frozen_bits: pinned if c <= 0.1 or c >= 0.9, wrong if
            the rounded value disagrees with the NEAREST optimal cut.
Sweep       eps in (0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5), 3 seeds, p = 1.

One resumable store, results/s11_maxcut_runs.jsonl, keyed by
(n, inst, arm, eps, seed). Each (n, inst, seed) job runs every (arm, eps) from
identical initial parameters, so comparisons are paired.

    python scripts/s11_maxcut.py --probe    # timing probe only (cost estimate)
    python scripts/s11_maxcut.py --plan     # task counts only
    python scripts/s11_maxcut.py            # run / resume
Analysis: scripts/s11_maxcut_analysis.py, scripts/s11_ranking.py.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SIZES = (12, 16, 20)
N_INST = 30
GRAPH_SEED = 11000
EPS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5)
SEEDS = (0, 1, 2)
P = 1
BAD_K = (0, 1, 2, 3, 4)
WARM_ARMS = ("GW", "GW-LS", "E") + tuple(f"BAD{k}" for k in BAD_K)
WORKERS = 8
STORE = ROOT / "results" / "s11_maxcut_runs.jsonl"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_PROBLEMS = {}


# ---- instance and problem ---------------------------------------------------


def make_graph(n: int, inst: int):
    import networkx as nx

    seed = GRAPH_SEED + 1000 * n + inst
    while True:
        g = nx.random_regular_graph(3, n, seed=seed)
        if nx.is_connected(g):
            return sorted((min(u, v), max(u, v)) for u, v in g.edges()), seed
        seed += 100_000  # redraw; never collides with another instance's first seed


@dataclass
class MaxCutEV:
    """The parts of qguc.fastsim.EnergyVector that run_qaoa and the features use."""

    E: np.ndarray
    e_min: float
    e_max: float
    ground: np.ndarray
    sigma: float
    n: int

    def ratio(self, energy):
        return (self.e_max - np.asarray(energy)) / (self.e_max - self.e_min)


@dataclass
class MaxCutProblem:
    n: int
    inst: int
    edges: list
    graph_seed: int
    ev: MaxCutEV
    table: object  # qguc.sampling.StateTable
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def uniform_ratio(self) -> float:
        return float(self.ev.ratio(self.ev.E.mean()))

    def gain(self, ratio):
        r0 = self.uniform_ratio
        return (np.asarray(ratio, float) - r0) / (1.0 - r0)

    def cut(self, x) -> int:
        x = np.asarray(x, int)
        return int(sum(x[i] != x[j] for i, j in self.edges))


def get_problem(n: int, inst: int) -> MaxCutProblem:
    from qguc.sampling import StateTable

    if (n, inst) not in _PROBLEMS:
        edges, gseed = make_graph(n, inst)
        k = np.arange(1 << n, dtype=np.int64)
        z = [1 - 2 * ((k >> j) & 1) for j in range(n)]  # spin of qubit j on basis state k
        E = np.zeros(1 << n)
        for i, j in edges:
            E += z[i] * z[j]
        e_min = float(E.min())
        ground = np.flatnonzero(E == e_min)
        ev = MaxCutEV(E=E, e_min=e_min, e_max=float(E.max()), ground=ground, sigma=float(np.sqrt(len(edges))), n=n)
        N = 1 << n
        table = StateTable(order=np.argsort(E, kind="stable"), n_ground=len(ground), gap_raw=np.zeros(N),
                           adequate=np.ones(N, bool), gap_repaired=np.zeros(N))  # no UC notions for MaxCut
        _PROBLEMS[(n, inst)] = MaxCutProblem(n, inst, edges, gseed, ev, table)
    return _PROBLEMS[(n, inst)]


def bits(index: int, n: int) -> np.ndarray:
    return (int(index) >> np.arange(n)) & 1


# ---- warm-start sources -----------------------------------------------------


def solve_sdp(pb: MaxCutProblem):
    """MaxCut SDP min <A, X>, X_ii = 1, X >= 0 by the full-rank factorisation X = V V^T with unit
    rows (L-BFGS on the unnormalised rows). Returns (V, sdp_cut_bound, min eigenvalue of the dual
    slack A - diag((A X)_ii)); the slack being PSD certifies global optimality."""
    from scipy.optimize import minimize

    n = pb.n
    A = np.zeros((n, n))
    for i, j in pb.edges:
        A[i, j] = A[j, i] = 1.0

    def f(y):
        Y = y.reshape(n, n)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        V = Y / nrm
        G = A @ V  # d/dV of 0.5 <A, V V^T>
        G = (G - (G * V).sum(1, keepdims=True) * V) / nrm  # through the row normalisation
        return 0.5 * float((A * (V @ V.T)).sum()), G.ravel()

    best = None
    for r in range(5):
        y0 = np.random.default_rng([pb.graph_seed, r, 0x5D]).standard_normal(n * n)
        res = minimize(f, y0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
        V = res.x.reshape(n, n)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        X = V @ V.T
        slack = float(np.linalg.eigvalsh(A - np.diag((A @ X).diagonal())).min())
        val = 0.5 * float((A * X).sum())  # sum over edges of v_i . v_j
        if best is None or slack > best[2]:
            best = (V, (len(pb.edges) - val) / 2.0, slack)
        if slack > -1e-6:
            break
    return best


def local_search(pb: MaxCutProblem, x) -> np.ndarray:
    """Best-improvement single-flip local search to a 1-flip-optimal cut."""
    x = np.array(x, int)
    nbrs = [[] for _ in range(pb.n)]
    for i, j in pb.edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    while True:
        gain = np.array([sum(1 if x[v] == x[u] else -1 for u in nbrs[v]) for v in range(pb.n)])
        v = int(np.argmax(gain))
        if gain[v] <= 0:
            return x
        x[v] ^= 1


def sources(pb: MaxCutProblem) -> dict:
    """{arm: (raw c, info)} for the pinned sources (arm E depends on seed and eps: see c_random)."""
    if "sources" not in pb.cache:
        V, bound, slack = solve_sdp(pb)
        r = np.random.default_rng([pb.graph_seed, 0x6A]).standard_normal(pb.n)
        x_gw = (V @ r < 0).astype(int)
        x_ls = local_search(pb, x_gw)
        opt = (len(pb.edges) - pb.ev.e_min) / 2.0
        info = dict(sdp_bound=bound, sdp_slack_min_eig=slack, cut_opt=opt)
        out = {"GW": (x_gw.astype(float), dict(info, cut_source=pb.cut(x_gw))),
               "GW-LS": (x_ls.astype(float), dict(info, cut_source=pb.cut(x_ls)))}
        x_opt = bits(pb.ev.ground[0], pb.n)
        perm = np.random.default_rng([pb.graph_seed, 0xBAD]).permutation(pb.n)
        for k in BAD_K:
            x = x_opt.copy()
            x[perm[:k]] ^= 1
            out[f"BAD{k}"] = (x.astype(float), dict(cut_opt=opt, cut_source=pb.cut(x), k_flipped=k))
        pb.cache["sources"] = out
    return pb.cache["sources"]


def c_random(pb: MaxCutProblem, seed: int, eps: float) -> np.ndarray:
    return np.random.default_rng([seed, pb.inst, pb.graph_seed, 0xE]).uniform(eps, 1.0 - eps, pb.n)


def wrong_pins(pb: MaxCutProblem, c, tau: float = 0.1) -> tuple[int, int]:
    """(pinned, pinned wrong against the nearest optimal cut); the rule of qguc.arms.frozen_bits."""
    if "ground_bits" not in pb.cache:
        pb.cache["ground_bits"] = np.array([bits(g, pb.n) for g in pb.ev.ground])
    c = np.asarray(c, float)
    pinned = (c <= tau) | (c >= 1 - tau)
    wrong = int(((np.round(c)[None, :] != pb.cache["ground_bits"]) & pinned[None, :]).sum(1).min())
    return int(pinned.sum()), wrong


# ---- runs -------------------------------------------------------------------


def one_run(pb: MaxCutProblem, arm: str, eps: float, seed: int) -> dict:
    import s8_mass

    from qguc.fastsim import probabilities, uniform_state
    from qguc.qaoa import run_qaoa
    from qguc.warmstart import WarmStartMixer, regularise, warm_initial_state

    info = {}
    if arm == "A":
        r = run_qaoa(pb, P, seed, arm="A")
        psi0, n_pin, n_wrong = uniform_state(pb.n), 0, 0
    else:
        if arm == "E":
            raw = c_random(pb, seed, eps)
        else:
            raw, info = sources(pb)[arm]
        c = regularise(raw, eps)
        psi0 = warm_initial_state(c)
        n_pin, n_wrong = wrong_pins(pb, raw)
        r = run_qaoa(pb, P, seed, arm=arm, initial_state=psi0, mixer=WarmStartMixer(c))
    feat = s8_mass.features(probabilities(psi0), pb.table, pb.ev)
    assert abs(feat["overlap0"] - r.overlap0) <= 1e-9 * max(r.overlap0, 1e-300) + 1e-15
    cut = (len(pb.edges) - pb.ev.E[pb.table.order]) / 2.0  # sorted: best cuts first
    p0 = probabilities(psi0)[pb.table.order]
    rec = dict(n=pb.n, inst=pb.inst, arm=arm, eps=eps, seed=seed, p=P, n_ground=len(pb.ev.ground),
               n_edges=len(pb.edges), r_unif=pb.uniform_ratio, n_evals=r.n_evals, max_evals=r.max_evals,
               cap_hit=r.cap_hit, early=r.early_stopped, overlap0=r.overlap0, n_pinned=n_pin,
               n_pinned_wrong=n_wrong, g_state0=r.g_state0, g_init=r.g_init, g_final=r.g_final,
               ratio=r.best_ratio, wall_s=r.wall_s,
               m_cut1=float(p0[cut >= cut[0] - 1].sum()), m_cut2=float(p0[cut >= cut[0] - 2].sum()), **info)
    rec.update({k: v for k, v in feat.items() if k not in ("overlap0", "m_adeq")})
    for K, m in r.sampling.items():
        rec[f"p_ground_K{K}"] = m["p_ground"]
    return rec


def job(args):
    n, inst, seed, combos = args
    os.environ.update(THREAD_ENV)
    pb = get_problem(n, inst)
    return [one_run(pb, arm, eps, seed) for arm, eps in combos]


def tasks():
    t = {(n, i, a, e, s) for n in SIZES for i in range(N_INST) for a in WARM_ARMS for e in EPS for s in SEEDS}
    return t | {(n, i, "A", 0.1, s) for n in SIZES for i in range(N_INST) for s in SEEDS}


def load():
    if not STORE.exists():
        return []
    out = {}
    for line in STORE.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by an interrupted run; that job is redone
            continue
        out[(r["n"], r["inst"], r["arm"], r["eps"], r["seed"])] = r
    return list(out.values())


def probe():
    """Time a fixed, small sample of the real jobs' runs at each size (results are discarded)."""
    for n in SIZES:
        t0 = time.perf_counter()
        pb = get_problem(n, 0)
        sources(pb)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        combos = [("A", 0.1), ("GW", 0.1), ("E", 0.25), ("BAD2", 0.05), ("BAD4", 0.35), ("GW-LS", 0.0)]
        ev = sum(one_run(pb, a, e, 0)["n_evals"] for a, e in combos)
        t = (time.perf_counter() - t0) / len(combos)
        per_size = len([1 for x in tasks() if x[0] == n])
        print(f"n = {n}: build {t_build:.1f} s; {t:.2f} s per run ({ev / len(combos):.0f} evaluations on average); "
              f"{per_size} runs -> {per_size * t / 60:.0f} core-minutes, about {per_size * t / 60 / WORKERS:.0f} min "
              f"on {WORKERS} workers", flush=True)


def main():
    if "--probe" in sys.argv:
        return probe()
    todo = tasks()
    done = {(r["n"], r["inst"], r["arm"], r["eps"], r["seed"]) for r in load()}
    left = todo - done
    groups = {}
    for n, i, a, e, s in sorted(left):
        groups.setdefault((n, i, s), []).append((a, e))
    jobs = sorted(((n, i, s, c) for (n, i, s), c in groups.items()), key=lambda j: (-j[0], j[1], j[2]))
    print(f"{len(todo)} runs in total ({len(done & todo)} done, {len(left)} left) in {len(jobs)} jobs; "
          f"{WORKERS} workers", flush=True)
    if "--plan" in sys.argv or not jobs:
        return
    t0 = time.perf_counter()
    text = STORE.read_text() if STORE.exists() else ""
    if text and not text.endswith("\n"):  # drop a partial last line before appending
        STORE.write_text(text[: text.rfind("\n") + 1])
    failed = 0
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        futs = {pool.submit(job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                recs = fut.result()
            except Exception as exc:  # log and carry on; the job stays undone and is retried on resume
                failed += 1
                print(f"  FAILED job {futs[fut][:3]}: {exc!r}", flush=True)
                continue
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} jobs, {failed} failed, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
